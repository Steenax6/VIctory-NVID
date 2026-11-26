//! How much shader machine code the host's pipelines hold in the driver.
//!
//! NVIDIA's driver keeps the machine code of every pipeline that is in use
//! in a heap of at most 512 MB, and a submission that needs more removes the
//! device (0x887a0005, nothing in DRED). That was measured elsewhere on nine
//! NVIDIA cards, among them an RTX 4070 SUPER 12 GB; in that title the
//! removal came after 261 to 303 seconds. Code counts from a pipeline's
//! first use until the pipeline is released; a pipeline that was made and
//! never used does not count. Our own 22 minutes in the open world ended
//! without a removal, but nothing said how close they came: this is the
//! measure.
//!
//! The number kept here is not the driver's. It is the size of the cached
//! blob (`ID3D12PipelineState::GetCachedBlob`), which holds the machine code
//! and other state besides it, so it is a proxy that may read higher than
//! the code itself. For orientation, the test that measured the removal
//! elsewhere had about 448 MB of cached blob at the point of removal, in 262
//! pipelines of 1.7 MB each. The size is asked once per pipeline, where it
//! is made or loaded (a pipeline worker, `crate::pipeline::host`), not when
//! it is bound. Pipelines of our own passes are not counted: they are few
//! and small, and made outside that place.
//!
//! A pipeline is in use from its first bind in a list of ours
//! ([`Code::bound`]) until its object is destroyed: the object carries a
//! ticket as private data, so the code counts as long as a list that still
//! holds the pipeline (`HostList::keep`) does, which is as long as the
//! driver does. `VICTORY_CODE_IN_USE=0` turns the measuring off; the cached
//! blob is then never asked for.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;
use windows::core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface};

const MB: u64 = 1 << 20;
/// The sums (in MB) a warning is logged once for when the code in use first
/// passes them.
const WARN_MB: [u64; 3] = [300, 400, 450];

/// What the pipelines of a process hold, in bytes of cached blob.
pub(crate) struct Books {
    in_use: AtomicU64,
    in_use_pipelines: AtomicU64,
    most: AtomicU64,
    unbound: AtomicU64,
    unbound_pipelines: AtomicU64,
    /// Pipelines alive whose size could not be asked for.
    unknown: AtomicU64,
    warned: [AtomicBool; 3],
}

impl Books {
    const fn new() -> Books {
        Books {
            in_use: AtomicU64::new(0),
            in_use_pipelines: AtomicU64::new(0),
            most: AtomicU64::new(0),
            unbound: AtomicU64::new(0),
            unbound_pipelines: AtomicU64::new(0),
            unknown: AtomicU64::new(0),
            warned: [const { AtomicBool::new(false) }; 3],
        }
    }

    /// A pipeline made or loaded, of `size` bytes if that is known.
    fn made(&'static self, size: Option<u64>) -> Arc<Code> {
        match size {
            Some(n) => {
                self.unbound.fetch_add(n, Relaxed);
                self.unbound_pipelines.fetch_add(1, Relaxed);
            }
            None => {
                self.unknown.fetch_add(1, Relaxed);
            }
        }
        Arc::new(Code { books: self, size, bound: AtomicBool::new(false) })
    }

    /// Moves a pipeline of `size` bytes from "made" to "in use"; the sums
    /// (MB) this made the code in use pass for the first time.
    fn first_bind(&self, size: u64) -> Vec<u64> {
        self.unbound.fetch_sub(size, Relaxed);
        self.unbound_pipelines.fetch_sub(1, Relaxed);
        let now = self.in_use.fetch_add(size, Relaxed) + size;
        self.in_use_pipelines.fetch_add(1, Relaxed);
        self.most.fetch_max(now, Relaxed);
        let mut passed = Vec::new();
        for (warned, mb) in self.warned.iter().zip(WARN_MB) {
            if now > mb * MB && !warned.swap(true, Relaxed) {
                passed.push(mb);
            }
        }
        passed
    }

    /// The pipeline's code is gone.
    fn gone(&self, size: Option<u64>, bound: bool) {
        match (size, bound) {
            (Some(n), true) => {
                self.in_use.fetch_sub(n, Relaxed);
                self.in_use_pipelines.fetch_sub(1, Relaxed);
            }
            (Some(n), false) => {
                self.unbound.fetch_sub(n, Relaxed);
                self.unbound_pipelines.fetch_sub(1, Relaxed);
            }
            (None, _) => {
                self.unknown.fetch_sub(1, Relaxed);
            }
        }
    }

    /// In use (bytes, pipelines), the most in use so far (bytes), made and
    /// not yet bound (bytes, pipelines), size unknown (pipelines).
    fn read(&self) -> [u64; 6] {
        [
            self.in_use.load(Relaxed),
            self.in_use_pipelines.load(Relaxed),
            self.most.load(Relaxed),
            self.unbound.load(Relaxed),
            self.unbound_pipelines.load(Relaxed),
            self.unknown.load(Relaxed),
        ]
    }
}

static BOOKS: Books = Books::new();

/// One pipeline's share of the books.
pub(crate) struct Code {
    books: &'static Books,
    /// Its cached blob's bytes; `None` if the host did not give the blob.
    size: Option<u64>,
    /// Whether it was bound in a list of ours yet.
    bound: AtomicBool,
}

impl Code {
    /// The sums (MB) the first bind of this pipeline made the code in use
    /// pass for the first time; nothing for any later bind.
    fn bind(&self) -> Vec<u64> {
        if self.bound.load(Relaxed) || self.bound.swap(true, Relaxed) {
            return Vec::new();
        }
        match self.size {
            Some(size) => self.books.first_bind(size),
            None => Vec::new(),
        }
    }

    /// The pipeline is bound in a list: counts the first time only.
    pub(crate) fn bound(&self) {
        for mb in self.bind() {
            let [in_use, pipelines, ..] = self.books.read();
            tracing::warn!(
                "shader code of pipelines in use passed {mb} MB: {} MB in {pipelines} pipelines (size of their cached blobs, which \
                 hold more than the machine code; NVIDIA drivers were measured to remove the device at 512 MB of code in use, \
                 after 448 MB of blob in 262 pipelines in one test)",
                in_use >> 20
            );
        }
    }
}

impl Drop for Code {
    fn drop(&mut self) {
        self.books.gone(self.size, self.bound.load(Relaxed));
    }
}

/// Whether the measuring is on (`VICTORY_CODE_IN_USE=0` turns it off).
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CODE_IN_USE").is_none_or(|v| v != "0"))
}

/// Asks `pso` for its size and books it as made; `None` with the measuring
/// off. The result goes into the host pipeline, and a ticket with it into
/// the pipeline object, so the code is let go of when the driver's object
/// is destroyed.
pub(crate) fn measure(pso: &ID3D12PipelineState) -> Option<Arc<Code>> {
    if !on() {
        return None;
    }
    // SAFETY: a live pipeline; the blob is only asked its size and released.
    let size = unsafe { pso.GetCachedBlob() }.ok().map(|blob| unsafe { blob.GetBufferSize() } as u64);
    let code = BOOKS.made(size);
    let ticket = ticket(code.clone());
    // SAFETY: a live object and a ticket it holds from here on.
    if let Err(e) = unsafe { pso.SetPrivateDataInterface(&TICKET_ID, &ticket) } {
        // The code is then let go of when the host pipeline is.
        tracing::debug!(error = %e, "no ticket on a pipeline for its code in use");
    }
    Some(code)
}

/// The line with the others every few seconds; nothing while the measuring
/// is off or no pipeline was made.
pub(crate) fn line() {
    if !on() {
        return;
    }
    let [in_use, in_use_n, most, unbound, unbound_n, unknown] = BOOKS.read();
    if in_use_n + unbound_n + unknown == 0 && most == 0 {
        return;
    }
    tracing::info!(
        "shader code of pipelines in use: {} MB in {in_use_n} pipelines (the most so far: {} MB); made and not yet bound: \
         {} MB in {unbound_n} pipelines; size unknown: {unknown} (sizes are the cached blobs', a proxy for the driver's machine \
         code; NVIDIA drivers remove the device at 512 MB of code in use; VICTORY_CODE_IN_USE=0 turns this off)",
        in_use >> 20,
        most >> 20,
        unbound >> 20,
    );
}

/// What a pipeline object holds as private data: `IUnknown` and no more.
#[repr(C)]
struct Ticket {
    vtable: &'static IUnknown_Vtbl,
    references: AtomicU32,
    _code: Arc<Code>,
}

const TICKET_ID: GUID = GUID::from_u128(0x6f0c_4b1e_8d3a_4e57_9a21_c5b7_3d90_e1a4);

static TICKET: IUnknown_Vtbl = IUnknown_Vtbl {
    QueryInterface: ticket_query,
    AddRef: ticket_add,
    Release: ticket_release,
};

fn ticket(code: Arc<Code>) -> IUnknown {
    let ticket = Box::new(Ticket { vtable: &TICKET, references: AtomicU32::new(1), _code: code });
    // SAFETY: an object whose first field points to the three methods, with
    // one reference, which the caller gets.
    unsafe { IUnknown::from_raw(Box::into_raw(ticket).cast()) }
}

unsafe extern "system" fn ticket_query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: the caller passes a live ticket, an identifier and an out
    // pointer.
    unsafe {
        if *iid != IUnknown::IID {
            *out = std::ptr::null_mut();
            return E_NOINTERFACE;
        }
        ticket_add(this);
        *out = this;
    }
    HRESULT(0)
}

unsafe extern "system" fn ticket_add(this: *mut c_void) -> u32 {
    // SAFETY: a live ticket.
    unsafe { &*this.cast::<Ticket>() }.references.fetch_add(1, Relaxed) + 1
}

unsafe extern "system" fn ticket_release(this: *mut c_void) -> u32 {
    use std::sync::atomic::Ordering::AcqRel;
    let this = this.cast::<Ticket>();
    // SAFETY: a live ticket.
    let left = unsafe { &*this }.references.fetch_sub(1, AcqRel) - 1;
    if left == 0 {
        // SAFETY: made by `ticket` from a box, and the last reference.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    fn books() -> &'static Books {
        Box::leak(Box::new(Books::new()))
    }

    #[test]
    fn a_pipeline_counts_from_its_first_bind_once() {
        let books = books();
        let a = books.made(Some(100 * MB));
        assert_eq!(books.read(), [0, 0, 0, 100 * MB, 1, 0]);
        a.bound();
        a.bound();
        assert_eq!(books.read(), [100 * MB, 1, 100 * MB, 0, 0, 0]);
        let b = books.made(Some(50 * MB));
        assert_eq!(books.read()[3..5], [50 * MB, 1]);
        b.bound();
        assert_eq!(books.read(), [150 * MB, 2, 150 * MB, 0, 0, 0]);
    }

    #[test]
    fn a_released_pipeline_is_subtracted_and_the_most_stays() {
        let books = books();
        let a = books.made(Some(200 * MB));
        let b = books.made(Some(30 * MB));
        a.bound();
        drop(a);
        assert_eq!(books.read(), [0, 0, 200 * MB, 30 * MB, 1, 0]);
        // Released without ever being bound: leaves the "made" sum.
        drop(b);
        assert_eq!(books.read(), [0, 0, 200 * MB, 0, 0, 0]);
    }

    #[test]
    fn a_pipeline_of_unknown_size_is_counted_apart() {
        let books = books();
        let a = books.made(None);
        a.bound();
        assert_eq!(books.read(), [0, 0, 0, 0, 0, 1]);
        drop(a);
        assert_eq!(books.read(), [0; 6]);
    }

    #[test]
    fn each_threshold_fires_once() {
        let books = books();
        let mut pipelines: Vec<_> = (0..10).map(|_| books.made(Some(50 * MB))).collect();
        let mut fired = Vec::new();
        for p in &pipelines {
            fired.extend(p.bind());
            assert!(p.bind().is_empty());
        }
        // 50 MB each: 300 is passed by the seventh (350 MB), 400 by the
        // ninth (450 > 400), 450 by the tenth (500 > 450).
        assert_eq!(fired, [300, 400, 450]);
        // Down below and up again: nothing fires twice.
        pipelines.truncate(6);
        for _ in 0..6 {
            assert!(books.made(Some(50 * MB)).bind().is_empty());
        }
    }

    #[test]
    fn one_step_can_pass_several() {
        let books = books();
        assert_eq!(books.made(Some(460 * MB)).bind(), [300, 400, 450]);
    }
}
