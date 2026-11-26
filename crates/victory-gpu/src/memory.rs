//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! With the structure builder asked for (`crate::pipeline::builder_allowed`)
//! the directory follows the title's buffers: a filled page in memory whose
//! buffer the title registered, released or replaced since
//! ([`ownership_changed`]) is given the copy of what holds it now, or none
//! ([`Memory::follow`]), before the next dispatch that reads the directory.
//! Without it a builder stage stored through such a page into the copy of a
//! buffer the title had released, and the programs reading the buffer now
//! there never saw it (ka1qux/BRVIctory-ka1qux 4c9dff93d).
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every list with such a shader brings all numbered copies up to
//! date and, if its shader stores by address, declares all of them written
//! (the bytes are compared afterwards, as for any unordered access view),
//! each copy once per list: a dirty-page mask written by the shader would
//! bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

/// Spans of title memory (first page, page past the end) whose buffers
/// changed since the directory last followed them ([`Memory::follow`]), and
/// whether more changed than are kept (then every filled page is looked
/// at). Collected only once the directory exists with the builder asked for
/// ([`FOLLOW`]).
static CHANGES: Mutex<(Vec<(u64, u64)>, bool)> = Mutex::new((Vec::new(), false));
/// See [`CHANGES`].
static FOLLOW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Spans [`CHANGES`] keeps.
const CHANGES_KEPT: usize = 4096;

/// The title registered, released or replaced a buffer over
/// `[start, start + len)`: filled pages of the directory there may now
/// belong to another copy.
pub(crate) fn ownership_changed(start: u64, len: u64) {
    if !FOLLOW.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut changes = CHANGES.lock().unwrap_or_else(|e| e.into_inner());
    if changes.0.len() == CHANGES_KEPT {
        changes.1 = true;
        return;
    }
    changes.0.push((start / PAGE, start.saturating_add(len).div_ceil(PAGE)));
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        if crate::pipeline::builder_allowed() {
            FOLLOW.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return;
        }
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if let Some((n, offset)) = self.place(page) {
            self.set(page, n, offset);
        }
    }

    /// Directory entry `page`: copy `n` (0: none), at byte `offset` in it.
    fn set(&mut self, page: u64, n: u32, offset: u32) {
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping (`page` is below its last
        // entry).
        unsafe {
            let entry = self.directory.1.add(page as usize * 2);
            entry.add(1).write_volatile(offset);
            entry.write_volatile(n);
        }
    }

    /// The copy that holds `page` now (its number, from 1, and the page's
    /// byte in it), numbering it if it has no number yet; None if the
    /// memory is not the title's, no buffer with unordered access holds the
    /// page whole, or no number is left.
    fn place(&mut self, page: u64) -> Option<(u32, u32)> {
        let start = page * PAGE;
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return None;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let shadow = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB. Not a page that
                // lies whole in another title buffer with a copy of its
                // own (one inside this one, or any inside a window outside
                // buffers): the programs that bind a view of that buffer
                // read and write the buffer's copy, so through this one a
                // stage read stale bytes and its stores never reached them
                // (alexanderskyequinn/victory-chapter-ii 8e68eb4ac: every
                // one of 600 logged reload dispatches read its vertices
                // so). Such a page is left empty: a shader that touches it
                // notes it, and the next frame gives it the buffer's copy.
                // shortcut: a page of a buffer whose copy is made later, or
                // that is registered later, keeps this copy.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    let elsewhere = || {
                        crate::shadow::existing(at, PAGE).is_some_and(|b| {
                            // SAFETY: a live buffer.
                            let flags = unsafe { b.buffer.GetDesc() }.Flags;
                            !Arc::ptr_eq(&b, &shadow) && flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
                        })
                    };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 && !elsewhere() {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return None;
            }
        };
        Some((n as u32 + 1, shadow.offset_in(start).unwrap_or(0) as u32))
    }

    /// Follows the buffers the title registered, released or replaced since
    /// the last look ([`ownership_changed`]): a filled page there whose copy
    /// is no longer what holds it gets the copy of what does, or none (a
    /// shader that touches it then notes it, as an empty page). In place:
    /// the title gives up memory only once the work that uses it has run,
    /// so a list recorded before reads no page that changed.
    fn follow(&mut self) {
        let (spans, all) = {
            let mut changes = CHANGES.lock().unwrap_or_else(|e| e.into_inner());
            if changes.0.is_empty() && !changes.1 {
                return;
            }
            let all = std::mem::take(&mut changes.1);
            (std::mem::take(&mut changes.0), all)
        };
        let last = u64::from(MEMORY_PAGES) - 1;
        let pages: Vec<u64> = if all {
            (0..last).collect()
        } else {
            let mut pages: Vec<u64> =
                spans.iter().flat_map(|&(first, end)| first.min(last)..end.min(last)).collect();
            pages.sort_unstable();
            pages.dedup();
            pages
        };
        let mut moved = 0u64;
        for page in pages {
            // SAFETY: inside the directory's mapping.
            let (n, offset) = unsafe {
                let entry = self.directory.1.add(page as usize * 2);
                (entry.read(), entry.add(1).read())
            };
            if n == 0 {
                continue;
            }
            self.refused.remove(&page);
            let now = self.place(page).unwrap_or((0, 0));
            if now != (n, offset) {
                self.set(page, now.0, now.1);
                moved += 1;
            }
        }
        if moved > 0 {
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
                tracing::info!(moved, "title memory by address: pages whose buffer changed follow it to its copy");
            }
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }

    /// The copies (indices into `copies`) that [`Memory::name`] of
    /// `address` gave or found: the directory's entries for its two pages.
    fn named(&self, address: u64, out: &mut Vec<usize>) {
        for at in [address, (address & !(CELL - 1)) + CELL] {
            let page = at / PAGE;
            if page >= u64::from(MEMORY_PAGES) - 1 {
                continue;
            }
            // SAFETY: inside the directory's mapping.
            let n = unsafe { self.directory.1.add(page as usize * 2).read() };
            if n != 0 && !out.contains(&(n as usize - 1)) {
                out.push(n as usize - 1);
            }
        }
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    m.follow();
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    // The copies this dispatch names (`crate::card_fine::named_writes`).
    let mut named: Vec<usize> = Vec::new();
    for &address in addresses {
        m.name(address);
        m.named(address, &mut named);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            let address = u64::from(c) | u64::from(high) << 32;
            m.name(address);
            m.named(address, &mut named);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
            m.named(u64::from(c) << 8, &mut named);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Each copy once per list, whichever list numbered it: the builder
    // alone dispatches hundreds of times a frame. The title records its
    // lists on several threads, so a copy can be numbered by another list
    // (or by the frame's fault notes) after this list's first dispatch;
    // this list's shaders reach it through the directory all the same, and
    // a copy it had not brought up to date or declared written was read as
    // it stood and its stores never looked for (16 dispatches in the
    // builder's first 2 s saw such copies, fxceless/victory-fork#370).
    // Copies are only ever added and a list takes all it has not taken at
    // each dispatch, so the ones it has not taken yet are the last ones.
    const READ: usize = usize::MAX - 0x6d;
    const READ_COPY: usize = usize::MAX - 0x1_0000;
    const WRITE_COPY: usize = usize::MAX - 0x2_0000;
    let first_read = list.first_use(READ);
    for (n, shadow) in m.copies.iter().enumerate().rev() {
        if !list.first_use(READ_COPY - n) {
            break;
        }
        let (start, len) = shadow.span();
        shadow.refresh(list, start, len);
    }
    // With `VICTORY_NAMED_WRITES=1`, but for every few frames: only the
    // copies a dispatch names are declared written (`card_fine::named_writes`
    // says what that risks).
    let narrow = writes && crate::card_fine::named_writes() && !frame.is_multiple_of(crate::card_fine::named_net());
    let declare = |shadow: &Arc<Shadow>| {
        let (start, len) = shadow.span();
        shadow.gpu_writes(list, start, len);
        list.address_declared(shadow);
    };
    if writes {
        for &n in &named {
            // Noted for the count of flags in copies no dispatch named.
            let first = list.address_named(&m.copies[n]);
            if narrow && first {
                declare(&m.copies[n]);
            }
        }
    }
    if writes && !narrow {
        for (n, shadow) in m.copies.iter().enumerate().rev() {
            if !list.first_use(WRITE_COPY - n) {
                break;
            }
            declare(shadow);
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}
