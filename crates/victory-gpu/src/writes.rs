//! The order between the title's writes that no guard sees (DirectStorage's
//! reads into physical pages, at every address the title shows them, and
//! its reads to plain addresses) and the uploads of the textures that hold
//! their bytes.
//!
//! The title maps physical pages that held other data, writes a texture's
//! view, has the pages filled and waits for the read's status before it
//! uses the texture. The texture pass may upload the texture in between:
//! its host copy then holds what the pages held before, page by page, and
//! nothing the title does later says so. The title's decoration bake picks
//! its sources by index once, at load: a source with such a copy is baked
//! as noise for the run (`crate::draw` `early_textures`).
//!
//! So every such write gets a number once its bytes are there ([`next`],
//! given under the lock of the ranges kept for the texture pass), and every
//! upload of the pass the number its read began at ([`stamp`]): an upload
//! whose stamp is below the number of a write under its texture read
//! before that write's bytes were there ([`holds`]), whatever flags of the
//! queue say, and is stale (`crate::descriptor::host` `slots_written_under`).
//!
//! Ranges the title has submitted reads for ([`title_memory_will_write`],
//! from its Submit until the read is done) let the pass leave a texture
//! for after its write instead of uploading the bytes from before and again
//! after it, and say when a bake samples memory a read is still writing.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};

/// The stamp of an upload whose read has not begun: it reads when its
/// list is submitted, and takes every write numbered by then.
pub(crate) const NOT_READ: u64 = u64::MAX;

/// The last number given to a write ([`next`]).
static LAST: AtomicU64 = AtomicU64::new(0);

/// A new number for a write whose bytes are there. To be called under the
/// lock of the ranges the pass takes, so that a pass that takes them takes
/// every number up to the last.
pub(crate) fn next() -> u64 {
    LAST.fetch_add(1, SeqCst) + 1
}

/// The number a read of title memory that begins now begins at: it sees
/// the bytes of every write numbered up to it. Taken before the read.
pub(crate) fn stamp() -> u64 {
    LAST.load(SeqCst)
}

/// Whether an upload with read stamp `stamp` holds the bytes of the write
/// numbered `number`: it read after the write was numbered, or has not
/// read yet (and will).
pub(crate) fn holds(stamp: u64, number: u64) -> bool {
    stamp == NOT_READ || stamp >= number
}

/// Title memory about to be written by reads the title has submitted
/// ([`title_memory_will_write`]): by start and a number of its own, the
/// end; and the longest range so far, which bounds a look for the ranges
/// that overlap one.
struct Pending {
    ranges: BTreeMap<(u64, u64), u64>,
    longest: u64,
    next: u64,
}

static PENDING: Mutex<Pending> = Mutex::new(Pending { ranges: BTreeMap::new(), longest: 0, next: 0 });

/// Ranges in [`PENDING`], for those who ask without its lock.
static PENDING_ANY: AtomicUsize = AtomicUsize::new(0);

/// Since the start: reads the title submitted that write under texture
/// views; their ranges there.
pub(crate) static PENDING_COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

impl Pending {
    fn overlaps(&self, start: u64, end: u64) -> bool {
        let from = (start.saturating_sub(self.longest), 0);
        self.ranges.range(from..(end, 0)).any(|(&(at, _), &until)| at < end && until > start)
    }
}

/// Whether a read the title submitted is about to write some of title
/// memory `[start, end)`. Locks one mutex and calls nothing.
pub(crate) fn pending_over(start: u64, end: u64) -> bool {
    if PENDING_ANY.load(Relaxed) == 0 || end <= start {
        return false;
    }
    PENDING.lock().unwrap_or_else(|e| e.into_inner()).overlaps(start, end)
}

/// Ranges of title memory a read is about to write, until dropped: from
/// the title's Submit until the read's bytes are there and told
/// (`title_memory_written`), or it failed, or its queue closed.
pub struct PendingWrite {
    keys: Vec<(u64, u64)>,
}

/// A read the title submitted is about to write title memory at `ranges`
/// (address, bytes), as long as what this returns lives: the texture pass
/// leaves the textures there for after the write. Ranges outside every
/// texture view are left out. Locks one mutex and calls nothing.
pub fn title_memory_will_write(ranges: &[(usize, usize)]) -> PendingWrite {
    let ranges: Vec<(u64, u64)> = ranges
        .iter()
        .map(|&(at, len)| (at as u64, at.saturating_add(len) as u64))
        .filter(|&(start, end)| end > start && crate::descriptor::host::spanned(start, end))
        .collect();
    if ranges.is_empty() {
        return PendingWrite { keys: Vec::new() };
    }
    PENDING_COUNTS[0].fetch_add(1, Relaxed);
    PENDING_COUNTS[1].fetch_add(ranges.len() as u64, Relaxed);
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let keys = ranges.into_iter().map(|(start, end)| pending.add(start, end)).collect();
    PENDING_ANY.store(pending.ranges.len(), Relaxed);
    PendingWrite { keys }
}

impl Pending {
    fn add(&mut self, start: u64, end: u64) -> (u64, u64) {
        self.next += 1;
        let key = (start, self.next);
        self.ranges.insert(key, end);
        self.longest = self.longest.max(end - start);
        key
    }

    fn remove(&mut self, keys: &[(u64, u64)]) {
        for key in keys {
            self.ranges.remove(key);
        }
        if self.ranges.is_empty() {
            self.longest = 0;
        }
    }
}

impl Drop for PendingWrite {
    fn drop(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&self.keys);
        PENDING_ANY.store(pending.ranges.len(), Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upload holds a write's bytes if it read after the write was
    /// numbered, or reads later; one that read before does not, whatever
    /// else is known of it.
    #[test]
    fn an_upload_holds_the_writes_numbered_before_its_read() {
        assert!(holds(7, 7));
        assert!(holds(8, 7));
        assert!(!holds(6, 7));
        assert!(holds(NOT_READ, 1 << 40));
        // An upload whose read could not be had counts as read at 0.
        assert!(!holds(0, 1));
    }

    /// A read that begins after a write is numbered stamps that number or
    /// a higher one; numbers only grow.
    #[test]
    fn stamps_follow_the_numbers() {
        let before = stamp();
        let n = next();
        assert!(n > before);
        assert!(!holds(before, n));
        assert!(holds(stamp(), n));
    }

    /// Pending ranges are found by any range that overlaps them, however
    /// long, and go when removed.
    #[test]
    fn pending_ranges_overlap_and_go() {
        let mut p = Pending { ranges: BTreeMap::new(), longest: 0, next: 0 };
        let long = p.add(0x1_0000, 0x3_0000);
        let short = p.add(0x8_0000, 0x8_1000);
        assert!(p.overlaps(0x2_f000, 0x4_0000));
        assert!(p.overlaps(0x0_8000, 0x1_0001));
        assert!(!p.overlaps(0x3_0000, 0x8_0000));
        assert!(!p.overlaps(0x0_0000, 0x1_0000));
        assert!(p.overlaps(0x8_0fff, 0x9_0000));
        p.remove(&[long]);
        assert!(!p.overlaps(0x2_f000, 0x4_0000));
        assert!(p.overlaps(0x8_0000, 0x8_0001));
        p.remove(&[short]);
        assert!(p.ranges.is_empty() && p.longest == 0);
    }
}
