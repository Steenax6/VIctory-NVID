//! Who touches what a landing wrote (`VICTORY_READ_CENSUS=<n>`, a
//! diagnostic; off unless set): a count of the first access to pages of
//! private title memory after a landing put shader-written bytes into them.
//!
//! About half of what shaders write to copies in the card's memory and the
//! landing thread then puts into title memory, some 49 MB a frame in the
//! first room, lies in three buffers the shaders rewrite whole every frame.
//! Whether anything on the CPU's side ever reads those bytes was not known:
//! the buffer census counts our own fills and copies and the title's
//! writes, not the title's reads. This counts them.
//!
//! One page in `n` of every stretch a landing wrote into private memory
//! (no guard there; view memory is left out) is made inaccessible once the
//! landing is through with it. The first access faults into the write
//! watch's handler, which gives the page back and counts the access by who
//! made it: a read or a write, on a thread of the layer's or one of the
//! title's, by code of the layer's or other code; the landing thread's next
//! write to a page nobody touched meanwhile is counted as that. A write of
//! ours through the kernel (which would fail on such a page, not fault)
//! gives the pages back first (`crate::watch::writing_as`).
//!
//! Run it with `VICTORY_LAND_TAKEN=1`, so that the copy's own refresh does
//! not read the landing's pages again (that echo is known and counted
//! elsewhere). It costs two changes of protection and a fault per page
//! sampled: for a census run, not for frame rate.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

const PAGE: usize = 4096;

/// One page in this many is sampled; 0: off.
pub(crate) fn every() -> usize {
    static EVERY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let n = std::env::var("VICTORY_READ_CENSUS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
        if n > 0 {
            tracing::info!(n, "read census: one page in n of what landings write into private title memory is made inaccessible until its next access (VICTORY_READ_CENSUS)");
        }
        n
    })
}

/// The pages made inaccessible and not touched since.
static ARMED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// The lowest and highest address ever armed: a fault outside is not ours.
static SPAN: [AtomicUsize; 2] = [AtomicUsize::new(usize::MAX), AtomicUsize::new(0)];

/// Since the last line: pages armed; first touched by: a read on a title
/// thread by other code than the layer's, such a write, a read on a title
/// thread by the layer's code, such a write, a read on a layer thread other
/// than the landing one, such a write, a read on the landing thread, the
/// landing thread's next write (nobody touched the page between two
/// landings); pages given back before a kernel write of ours on another
/// thread; pages found armed still when armed again.
static COUNTS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

thread_local! {
    /// 0: a thread of the title's; 1: a thread of the layer's; 2: the
    /// thread that lands flights.
    static KIND: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// The stretches this thread's landing wrote, armed when it is through.
    static WROTE: std::cell::RefCell<Vec<(usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// This thread is one of the layer's own (`landing`: the one that lands
/// flights).
pub(crate) fn layer_thread(landing: bool) {
    KIND.set(if landing { 2 } else { 1 });
}

/// A landing wrote `[at, at + len)` of title memory: noted for [`arm`].
pub(crate) fn wrote(at: usize, len: usize) {
    if every() == 0 || crate::inplace::view_at(at).is_some() {
        return;
    }
    WROTE.with(|w| w.borrow_mut().push((at, len)));
}

/// The landing is through with what it wrote ([`wrote`]): one page in
/// [`every`] of it is made inaccessible.
pub(crate) fn arm() {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect};
    let n = every();
    if n == 0 {
        return;
    }
    let stretches = WROTE.with(|w| std::mem::take(&mut *w.borrow_mut()));
    if stretches.is_empty() {
        return;
    }
    let mut armed = ARMED.lock().unwrap_or_else(|e| e.into_inner());
    let armed = armed.get_or_insert_with(HashSet::new);
    for (at, len) in stretches {
        let (first, last) = (at.div_ceil(PAGE), (at + len) / PAGE);
        for page in (first..last).filter(|p| p % n == 0) {
            let address = page * PAGE;
            if armed.contains(&address) {
                COUNTS[10].fetch_add(1, Relaxed);
                continue;
            }
            let mut old = 0;
            // SAFETY: a page of title memory a landing has just written;
            // only its protection changes, and the handler gives it back.
            if unsafe { VirtualProtect(address as *const _, PAGE, PAGE_NOACCESS, &mut old) } == 0 {
                continue;
            }
            if old != PAGE_READWRITE {
                // Not plain memory of the title's: left as it was.
                // SAFETY: as above.
                unsafe { VirtualProtect(address as *const _, PAGE, old, &mut old) };
                continue;
            }
            armed.insert(address);
            SPAN[0].fetch_min(address, Relaxed);
            SPAN[1].fetch_max(address + PAGE, Relaxed);
            COUNTS[0].fetch_add(1, Relaxed);
        }
    }
}

/// Whether `rip` lies in the layer's own image.
fn ours(rip: usize) -> bool {
    static IMAGE: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    let (base, end) = *IMAGE.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
        // SAFETY: the process's own image; its headers are mapped.
        unsafe {
            let base = GetModuleHandleW(std::ptr::null()) as usize;
            if base == 0 {
                return (0, 0);
            }
            let pe = base + (base as *const u8).add(0x3c).cast::<u32>().read_unaligned() as usize;
            // SizeOfImage of the optional header (PE32+).
            let size = (pe as *const u8).add(0x18 + 0x38).cast::<u32>().read_unaligned() as usize;
            (base, base + size)
        }
    });
    rip >= base && rip < end
}

/// From the fault handler: an access violation at `at` (`write`: by a
/// write) with the instruction at `rip`. Whether the page was one of ours
/// and is accessible again (the instruction runs again).
pub(crate) fn fault(at: usize, write: bool, rip: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualProtect, VirtualQuery};
    if every() == 0 || at < SPAN[0].load(Relaxed) || at >= SPAN[1].load(Relaxed) {
        return false;
    }
    let page = at & !(PAGE - 1);
    let mut armed = ARMED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(set) = armed.as_mut() else {
        return false;
    };
    if !set.remove(&page) {
        // Another thread's fault on the same page gave it back while this
        // one waited for the lock: accessible now, if it is.
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(page as *const _, &mut info, size_of_val(&info)) };
        return n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE;
    }
    let mut old = 0;
    // SAFETY: a page this census made inaccessible.
    if unsafe { VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old) } == 0 {
        return false;
    }
    let place = match (KIND.get(), ours(rip), write) {
        (2, _, true) => 8,
        (2, _, false) => 7,
        (1, _, false) => 5,
        (1, _, true) => 6,
        (_, true, false) => 3,
        (_, true, true) => 4,
        (_, false, false) => 1,
        (_, false, true) => 2,
    };
    COUNTS[place].fetch_add(1, Relaxed);
    true
}

/// Before a write of ours into `[at, at + len)` through the kernel, which
/// does not fault: the armed pages there are given back, and counted as
/// the access they stand for.
pub(crate) fn disarm(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if every() == 0 || len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    if end <= SPAN[0].load(Relaxed) || at >= SPAN[1].load(Relaxed) {
        return;
    }
    let mut armed = ARMED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(set) = armed.as_mut() else {
        return;
    };
    // (The range may be the title's, as long as it likes: the armed pages
    // are looked through where they are fewer than its pages.)
    let first = at & !(PAGE - 1);
    let pages: Vec<usize> = if (end - first) / PAGE > set.len() {
        set.iter().copied().filter(|&p| p >= first && p < end).collect()
    } else {
        (first..end).step_by(PAGE).filter(|p| set.contains(p)).collect()
    };
    for page in pages {
        set.remove(&page);
        let mut old = 0;
        // SAFETY: a page this census made inaccessible.
        unsafe { VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old) };
        COUNTS[if KIND.get() == 2 { 8 } else { 9 }].fetch_add(1, Relaxed);
    }
}

/// With the frame line.
pub(crate) fn line(frames: f64) {
    let n = every();
    if n == 0 {
        return;
    }
    let c: Vec<f64> = COUNTS.iter().map(|c| c.swap(0, Relaxed) as f64 / frames).collect();
    let still = ARMED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, HashSet::len);
    tracing::info!(
        "read census, per frame (VICTORY_READ_CENSUS={n}: one 4 KiB page in {n} of what landings wrote into private title memory): \
         {:.1} pages armed; first touched by: other code than the layer's on a title thread reading {:.1}, writing {:.1}; the \
         layer's code on a title thread reading {:.1}, writing {:.1}; a layer thread other than the landing one reading {:.1}, \
         writing {:.1}; the landing thread reading {:.1}; the landing thread's next write with nothing between {:.1}; given back \
         before a kernel write of ours on another thread {:.1}; found armed still when armed again {:.1}; {still} armed now",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10]
    );
}
