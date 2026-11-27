//! With `VICTORY_LAND_SAMPLE=1` (a check; off by default): the layer's own
//! writes into title memory, kept for a moment, so that the sample at a
//! landing's telling (`crate::host`'s `land_jobs`) can say who wrote a
//! stretch after the merge that landed it.
//!
//! The sample finds stretches a merge wrote whose title memory holds other
//! bytes at the telling, a few milliseconds later. Three things can have
//! happened: the title stored there (through any view of the pages), the
//! bytes never were at the title's address, or the layer itself wrote there
//! afterwards (another merge's job, a title buffer copy, a read-back). The
//! third is told here: each write of the layer's is noted by the physical
//! place it goes to (a section of the title's pages and the offset in it,
//! whichever view it was written through; the address for private memory),
//! with its thread and its time.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Whether the sample is taken, and the writes kept for it.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_SAMPLE").is_some_and(|v| v == "1"))
}

/// Microseconds since the first asking.
pub(crate) fn now() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// The title's views are of 64 KiB pages: a write is noted piece by piece.
const K64: usize = 64 << 10;
const MOST: usize = 1 << 16;

/// What, the section (0: private memory), the offset in it (or the
/// address), bytes, the thread, microseconds.
type Write = (u8, usize, u64, u32, u32, u64);

static WRITES: Mutex<Vec<Write>> = Mutex::new(Vec::new());
static NEXT: AtomicUsize = AtomicUsize::new(0);

/// Where title memory at `at` lies: its section and the offset in it, or
/// 0 and the address for memory that is no view.
fn place(at: usize) -> (usize, u64) {
    match crate::inplace::view_at(at) {
        Some((base, _, section, offset)) => (section, offset + (at - base) as u64),
        None => (0, at as u64),
    }
}

/// The layer writes `[at, at + len)` of title memory now. `what`: `b` a
/// merge beside the guards, `w` a merge at the title's address, `c` a copy
/// into title memory (a title buffer copy, a read-back), `k` a write
/// through the kernel, `p` a bottom-of-pipe write, `d` and `r` what the
/// runtime told it wrote (into pages, to a plain address).
pub(crate) fn wrote(what: u8, at: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    let us = now();
    // SAFETY: no arguments.
    let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let end = at.saturating_add(len);
    let mut writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let (mut a, mut pieces) = (at, 0);
    while a < end && pieces < 256 {
        let e = ((a / K64 + 1) * K64).min(end);
        let (section, offset) = place(a);
        let write = (what, section, offset, (e - a) as u32, thread, us);
        let slot = NEXT.fetch_add(1, Relaxed) % MOST;
        if writes.len() < MOST {
            writes.push(write);
        } else {
            writes[slot] = write;
        }
        (a, pieces) = (e, pieces + 1);
    }
}

/// The writes noted over `[at, at + len)` (by where it lies now) from
/// `from` microseconds on, oldest first: what, bytes, microseconds after
/// `from`, thread; up to sixteen, and how many there were.
pub(crate) fn since(at: usize, len: usize, from: u64) -> String {
    let (section, offset) = place(at);
    let end = offset + len as u64;
    let writes = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let mut found: Vec<&Write> = writes
        .iter()
        .filter(|w| w.1 == section && w.2 < end && w.2 + u64::from(w.3) > offset && w.5 >= from)
        .collect();
    found.sort_unstable_by_key(|w| w.5);
    let said: Vec<String> = found.iter().take(16).map(|w| format!("{}:{}@+{}us/t{}", w.0 as char, w.3, w.5 - from, w.4)).collect();
    format!("{} [{}]", found.len(), said.join(" "))
}
