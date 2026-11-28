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
//! given under the lock of the ranges kept for the texture pass; so is
//! every copy of the title's that the layer does in title memory itself,
//! `crate::descriptor::host::title_copied`), and every
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

/// Who tells the texture pass that title memory changes or changed
/// (`crate::descriptor::host` `written`, `remapped`). The first three are
/// told when the bytes are there and numbered ([`next`]); a copy of the
/// title's is also told before it is done, as the others are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Cause {
    /// DirectStorage wrote physical pages through a view of the runtime's
    /// own: told at every title address that shows them.
    Pages = 0,
    /// A DirectStorage read to a plain address finished.
    Plain,
    /// A copy of the title's that the layer does in title memory (a buffer
    /// copy at its submission or when its submission lands,
    /// `CopyMemoryToMemoryX`).
    Copy,
    /// What shaders wrote to a buffer, copied back into title memory.
    Back,
    /// The texels of a texture shaders wrote, or of one the title reads
    /// back into a buffer, copied into title memory.
    Texels,
    /// The runtime is about to write through the kernel (a file read, a
    /// DirectStorage read to a plain address before it begins).
    Kernel,
    /// The runtime mapped or unmapped the memory.
    Mapped,
}

/// The causes there are.
pub(crate) const CAUSES: usize = 7;

impl Cause {
    pub(crate) const ALL: [Cause; CAUSES] =
        [Cause::Pages, Cause::Plain, Cause::Copy, Cause::Back, Cause::Texels, Cause::Kernel, Cause::Mapped];

    /// The cause a `u8` stands for (one past the last: the last).
    pub(crate) fn of(code: u8) -> Cause {
        Cause::ALL[usize::from(code).min(CAUSES - 1)]
    }

    /// For the lines.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Cause::Pages => "DirectStorage into pages",
            Cause::Plain => "DirectStorage to plain addresses",
            Cause::Copy => "copies of the title's done by the layer",
            Cause::Back => "shader writes to buffers copied back",
            Cause::Texels => "texels copied back",
            Cause::Kernel => "reads through the kernel",
            Cause::Mapped => "mapped or unmapped",
        }
    }
}

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

/// The pages of [`TOLD`]'s chunks (64 KiB each: 64 MiB a chunk), and the
/// chunks: the title's addresses (40 bits).
const TOLD_PAGE: u64 = 64 << 10;
const TOLD_CHUNK: u64 = 1024;
const TOLD_CHUNKS: usize = ((1u64 << 40) / TOLD_PAGE / TOLD_CHUNK) as usize;

/// The highest number of a tell to the texture pass (memory mapped,
/// written by DirectStorage, by the layer's copies or copied back, about
/// to be written through the kernel; `crate::descriptor::host` `remapped`
/// and `written`) by 64 KiB page, in chunks made where a page is first
/// told: whether anything was told of a texture's part after a read is a
/// look at its pages without a lock ([`told_after`]).
static TOLD: [std::sync::OnceLock<Box<[AtomicU64]>>; TOLD_CHUNKS] = [const { std::sync::OnceLock::new() }; TOLD_CHUNKS];

/// `[start, end)` was told under the number `number` ([`next`]). Called
/// under the lock of the ranges the pass takes, after the number is given;
/// calls nothing that locks.
pub(crate) fn told(start: u64, end: u64, number: u64) {
    if end <= start {
        return;
    }
    let (first, last) = (start / TOLD_PAGE, end.div_ceil(TOLD_PAGE).min(TOLD_CHUNKS as u64 * TOLD_CHUNK));
    for page in first..last {
        let chunk = TOLD[(page / TOLD_CHUNK) as usize]
            .get_or_init(|| (0..TOLD_CHUNK).map(|_| AtomicU64::new(0)).collect());
        chunk[(page % TOLD_CHUNK) as usize].fetch_max(number, SeqCst);
    }
}

/// Whether something was told of `[start, end)` under a number above
/// `stamp`: a read that began at `stamp` may not hold what is there now.
/// Memory above the title's addresses counts as told.
pub(crate) fn told_after(start: u64, end: u64, stamp: u64) -> bool {
    if end <= start {
        return false;
    }
    let (first, last) = (start / TOLD_PAGE, end.div_ceil(TOLD_PAGE));
    if last > TOLD_CHUNKS as u64 * TOLD_CHUNK {
        return true;
    }
    (first..last).any(|page| {
        TOLD[(page / TOLD_CHUNK) as usize].get().is_some_and(|chunk| chunk[(page % TOLD_CHUNK) as usize].load(SeqCst) > stamp)
    })
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

/// With the refresh check (`crate::shadow::refresh_checked`): every range
/// the title's submitted reads are about to write, under a texture view or
/// not ([`in_flight_over`]). A read writes its destination's pages through
/// a view of the runtime's own for as long as it takes, and the guards are
/// told when it is done: until then every host copy over them holds the
/// bytes from before, as it should, and the check must know it.
static IN_FLIGHT: Mutex<Pending> = Mutex::new(Pending { ranges: BTreeMap::new(), longest: 0, next: 0 });

/// Whether a read the title submitted is about to write, or is writing,
/// some of title memory `[start, end)`; known only with the refresh check
/// on, and only for reads whose pages were known at their Submit.
pub(crate) fn in_flight_over(start: u64, end: u64) -> bool {
    end > start && IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).overlaps(start, end)
}

/// Ranges of title memory a read is about to write, until dropped: from
/// the title's Submit until the read's bytes are there and told
/// (`title_memory_written`), or it failed, or its queue closed.
pub struct PendingWrite {
    keys: Vec<(u64, u64)>,
    /// Its ranges in [`IN_FLIGHT`].
    all: Vec<(u64, u64)>,
}

/// A read the title submitted is about to write title memory at `ranges`
/// (address, bytes), as long as what this returns lives: the texture pass
/// leaves the textures there for after the write. Ranges outside every
/// texture view are left out. Locks one mutex and calls nothing.
pub fn title_memory_will_write(ranges: &[(usize, usize)]) -> PendingWrite {
    let ranges: Vec<(u64, u64)> =
        ranges.iter().map(|&(at, len)| (at as u64, at.saturating_add(len) as u64)).filter(|&(start, end)| end > start).collect();
    // For the refresh check: all of them.
    let all = if crate::shadow::refresh_checked() && !ranges.is_empty() {
        let mut flying = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
        ranges.iter().map(|&(start, end)| flying.add(start, end)).collect()
    } else {
        Vec::new()
    };
    let ranges: Vec<(u64, u64)> = ranges.into_iter().filter(|&(start, end)| crate::descriptor::host::spanned(start, end)).collect();
    if ranges.is_empty() {
        return PendingWrite { keys: Vec::new(), all };
    }
    PENDING_COUNTS[0].fetch_add(1, Relaxed);
    PENDING_COUNTS[1].fetch_add(ranges.len() as u64, Relaxed);
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let keys = ranges.into_iter().map(|(start, end)| pending.add(start, end)).collect();
    PENDING_ANY.store(pending.ranges.len(), Relaxed);
    PendingWrite { keys, all }
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
        if !self.all.is_empty() {
            IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.all);
        }
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

    /// Every cause has a code of its own and a name for the lines.
    #[test]
    fn causes_are_told_apart() {
        for (code, cause) in Cause::ALL.into_iter().enumerate() {
            assert_eq!(cause as usize, code);
            assert_eq!(Cause::of(code as u8), cause);
        }
        assert_eq!(Cause::of(200), Cause::Mapped);
        let names: std::collections::BTreeSet<&str> = Cause::ALL.iter().map(|c| c.name()).collect();
        assert_eq!(names.len(), CAUSES);
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
