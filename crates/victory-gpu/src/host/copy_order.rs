//! Title buffer copies recorded behind a write of their own submission's
//! shaders, counted and looked at: an order this layer does not keep.
//!
//! A copy is made on the CPU when its submission is prepared, before any
//! of its lists runs. Recorded behind a dispatch whose shaders write its
//! source, it takes the bytes from before they ran; recorded behind one
//! whose shaders write its destination, their bytes land over the copy's
//! with the copy-back, where the list's order has the copy over them.
//! (Against flights in the air the order is kept: [`blocked_by`],
//! [`copy_goes_late`], [`late_among`].)
//!
//! Marvisak/VIctory pull request 2810 leaves such a copy until its
//! submission's copy-backs are done, by the ranges declared. Here a
//! declaration says what shaders may write, not what they do (the title
//! has an unordered view over the whole of its 174 MB buffer), and a copy
//! left to its landing makes the later submissions that read its
//! destination wait for it ([`late_order`]). So the numbers come first:
//! how many copies a frame are made behind such a declaration, out of the
//! range and into it; and of how many of them the source and the
//! destination no longer hold the same bytes once their submission's
//! copy-back is done. Those are the copies that went wrong, and with them
//! the ones the title, a later copy's landing or a shader recorded behind
//! the copy wrote under since: an upper bound.
//!
//! Moved out of `host.rs`, which the asset policy keeps under 512 KiB;
//! nothing in it changed.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// Since the last frame line: copies made out of a declared range;
/// into one; looked at when their submission's copy-back was done; of
/// those, the ones whose two sides differed.
static COUNT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// The copies still to be looked at: submission, destination, source,
/// bytes.
static MADE: Mutex<Vec<Made>> = Mutex::new(Vec::new());
/// How many `MADE` holds (read without the lock: nearly always none).
static WAITING: AtomicUsize = AtomicUsize::new(0);
/// Kept at most, and compared of each at most.
const MOST: usize = 256;
const MOST_BYTES: usize = 64 << 10;

type Made = (u64, usize, usize, usize);

fn over(a: usize, a_len: usize, b: usize, b_len: usize) -> bool {
    a < b.saturating_add(b_len) && b < a.saturating_add(a_len)
}

/// Notes a range a submission's shaders are declared to write among
/// those declared so far (address, end).
pub(super) fn declare(declared: &mut Vec<(usize, usize)>, address: usize, len: usize) {
    let range = (address, address.saturating_add(len));
    // The same few ranges, dispatch after dispatch.
    if range.0 < range.1 && !declared.iter().rev().take(64).any(|w| w.0 <= range.0 && range.1 <= w.1) {
        declared.push(range);
    }
}

/// Whether a copy of `len` bytes from `src` to `dst` reads ([0]) or
/// writes ([1]) a range in `declared`.
pub(super) fn behind(declared: &[(usize, usize)], dst: usize, src: usize, len: usize) -> [bool; 2] {
    let hit = |at: usize| len > 0 && declared.iter().any(|w| w.0 < at.saturating_add(len) && at < w.1);
    [hit(src), hit(dst)]
}

/// `made` without the copies a write of `len` bytes at `dst` reaches:
/// their sides differ by that write from here on, whatever shaders did.
fn written_over(made: &mut Vec<Made>, dst: usize, len: usize) {
    made.retain(|&(_, d, s, n)| !over(d, n, dst, len) && !over(s, n, dst, len));
}

/// The copies of `made` whose submission is `upto` or one before it,
/// taken out of it.
fn take_landed(made: &mut Vec<Made>, upto: u64) -> Vec<Made> {
    let mut due = Vec::new();
    made.retain(|copy| {
        let landed = copy.0 <= upto;
        if landed {
            due.push(*copy);
        }
        !landed
    });
    due
}

/// Whether the two sides of a copy hold the same bytes now (the first
/// [`MOST_BYTES`] of them); `None`: one of them is not there to read.
fn same(dst: usize, src: usize, len: usize) -> Option<bool> {
    let n = len.min(MOST_BYTES);
    let (mut from, mut to) = (vec![0u8; n], vec![0u8; n]);
    (super::read_into(src as *const u8, &mut from) && super::read_into(dst as *const u8, &mut to)).then(|| from == to)
}

/// A copy of submission `submission` is made now, with `declared` what
/// the steps before it declared.
pub(super) fn made(declared: &[(usize, usize)], submission: u64, dst: usize, src: usize, len: usize) {
    let [out_of, into] = if declared.is_empty() { [false; 2] } else { behind(declared, dst, src, len) };
    if !(out_of || into) && WAITING.load(Relaxed) == 0 {
        return;
    }
    COUNT[0].fetch_add(u64::from(out_of), Relaxed);
    COUNT[1].fetch_add(u64::from(into), Relaxed);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    written_over(&mut made, dst, len);
    // (A copy that overlaps itself has two sides that differ anyway.)
    if (out_of || into) && made.len() < MOST && !over(dst, len, src, len) {
        made.push((submission, dst, src, len));
    }
    WAITING.store(made.len(), Relaxed);
}

/// The copy-backs of the submissions up to `upto` are done, and nothing
/// of them has been told to the title yet.
pub(super) fn landed(upto: u64) {
    if WAITING.load(Relaxed) == 0 {
        return;
    }
    let due = {
        let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
        let due = take_landed(&mut made, upto);
        WAITING.store(made.len(), Relaxed);
        due
    };
    for (_, dst, src, len) in due {
        if let Some(same) = same(dst, src, len) {
            COUNT[2].fetch_add(1, Relaxed);
            COUNT[3].fetch_add(u64::from(!same), Relaxed);
        }
    }
}

/// With the frame line.
pub(super) fn line(frames: f64) {
    let [out_of, into, looked, differed] = [0, 1, 2, 3].map(|i| COUNT[i].swap(0, Relaxed));
    if out_of == 0 && into == 0 && looked == 0 {
        return;
    }
    tracing::info!(
        "title buffer copies made at their submission behind a range its shaders are declared to write before them: \
         {:.1} a frame out of such a range, {:.1} into one (the copy is made before the shaders run, an order not kept: \
         Marvisak/VIctory pull request 2810); looked at once the copy-back was done: {looked}, of them {differed} whose \
         source and destination no longer held the same bytes",
        out_of as f64 / frames,
        into as f64 / frames
    );
}

#[cfg(test)]
mod tests {
    use super::{Made, behind, declare, same, take_landed, written_over};

    /// What shaders are declared to write is kept once, and a copy is
    /// behind it by its source, by its destination, or not at all.
    #[test]
    fn a_copy_is_behind_a_declared_range_by_either_side() {
        let mut declared = Vec::new();
        declare(&mut declared, 0x1000, 0x1000);
        declare(&mut declared, 0x1800, 0x100);
        declare(&mut declared, 0x9000, 0);
        declare(&mut declared, 0x4000, 0x10);
        assert_eq!(declared, [(0x1000, 0x2000), (0x4000, 0x4010)]);
        // (destination, source, bytes)
        assert_eq!(behind(&declared, 0x8000, 0x1ff0, 0x20), [true, false]);
        assert_eq!(behind(&declared, 0x400f, 0x8000, 0x20), [false, true]);
        assert_eq!(behind(&declared, 0x1000, 0x4000, 0x10), [true, true]);
        // Next to a range is not in it, and nothing is copied by none.
        assert_eq!(behind(&declared, 0x2000, 0x0ff0, 0x10), [false, false]);
        assert_eq!(behind(&declared, 0x1000, 0x1000, 0), [false, false]);
        assert_eq!(behind(&[], 0x1000, 0x1000, 0x10), [false, false]);
        // Ranges that are nonsense do not panic.
        declare(&mut declared, usize::MAX - 4, usize::MAX);
        assert_eq!(behind(&declared, usize::MAX - 2, 0, usize::MAX), [true, true]);
    }

    /// A copy is looked at when its submission has landed, unless a
    /// later copy wrote over one of its sides; then its two sides say
    /// whether anything changed under it.
    #[test]
    fn a_copy_is_looked_at_once_its_submission_has_landed() {
        let mut made: Vec<Made> = vec![(3, 0x1000, 0x5000, 0x100), (4, 0x2000, 0x6000, 0x100), (5, 0x3000, 0x7000, 0x100)];
        // A write over the second one's source, and one next to the
        // first one's destination.
        written_over(&mut made, 0x60ff, 0x10);
        written_over(&mut made, 0x1100, 0x100);
        assert_eq!(made, [(3, 0x1000, 0x5000, 0x100), (5, 0x3000, 0x7000, 0x100)]);
        assert!(take_landed(&mut made, 2).is_empty());
        assert_eq!(take_landed(&mut made, 4), [(3, 0x1000, 0x5000, 0x100)]);
        assert_eq!(made, [(5, 0x3000, 0x7000, 0x100)]);
        // The two sides, as they are in memory.
        let from = vec![7u8; 4096];
        let mut to = from.clone();
        let (src, dst) = (from.as_ptr() as usize, to.as_ptr() as usize);
        assert_eq!(same(dst, src, 4096), Some(true));
        to[4095] = 8;
        let dst = to.as_ptr() as usize;
        assert_eq!(same(dst, src, 4096), Some(false));
        assert_eq!(same(dst, src, 4095), Some(true));
        // Memory that is not there says nothing.
        assert_eq!(same(dst, 16, 64), None);
    }
}
