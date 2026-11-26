//! Diagnostics: a census of what the layer moves for buffers and why, with
//! the frame-rate line. Every count is taken where the work is done; none
//! changes what is done.
//!
//! - **Read again** ([`refreshed_block`]): of the 4 KiB blocks a refresh of
//!   a copy in the card's memory read from title memory, those equal to the
//!   copy's CPU side and those that differ, each split by whether a landing
//!   (the copy-back of what shaders wrote) put bytes into the block since a
//!   refresh last read it, and by private and view memory. An equal block a
//!   landing wrote is the layer reading its own write.
//! - **Guards opened** ([`opened`]): guarded blocks opened by the layer's
//!   own writes into title memory, by what wrote; the rest of the guards'
//!   "blocks opened" are the title's faults.
//! - **Who touches what shaders write** ([`flagged`], [`title_wrote`],
//!   [`cpu_read`], [`lines`]): per copy, the 4 KiB blocks a compare on the
//!   card flagged since the last line, and of them the ones the title's CPU
//!   wrote, the ones a CPU step of ours read (a fill of upload memory, a
//!   buffer copy of the title's) and the ones another host copy holds too.
//! - **Title bytes under flights** ([`under_page`], [`under_block`],
//!   [`gave_up`], [`joined`]): the pages `Shadow::merge_under_writes` looks
//!   at by their changed runs, why it leaves the flights to be settled, and
//!   the blocks that took a run that is not byte-exact and that a flight
//!   then in the air turned out to have written.
//! - **Who asks** ([`asked`], [`updated`]): refreshes recorded and copies
//!   brought up to date, by who asked for them, and the copies asked about
//!   that had nothing written.
//!
//! Ranges come from the title and from shaders: every one is cut to what
//! the tables hold, and one whose end lies before its start counts nothing.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// The kernel's page: the unit of every bit below.
const KP: usize = 4096;
/// Title address space per chunk of a [`Bits`], and how much of it has
/// bits at all (the title's addresses have 40 bits).
const CHUNK: usize = 1 << 30;
const CHUNKS: usize = 1 << 10;
const WORDS: usize = CHUNK / KP / 64;
/// The most pages one call sets or counts: a range longer than any buffer
/// is cut there (16 GiB).
const MOST_PAGES: usize = (16 << 30) / KP;

/// One bit per 4 KiB page of title address space, in chunks of 1 GiB made
/// when a bit of theirs is first set. No lock: bits are set, taken and
/// counted with atomic words, by any thread.
pub(crate) struct Bits {
    chunks: [OnceLock<Box<[AtomicU64]>>; CHUNKS],
}

impl Bits {
    pub(crate) const fn new() -> Bits {
        Bits { chunks: [const { OnceLock::new() }; CHUNKS] }
    }

    /// The pages `[first, last)` of `[at, at + len)`, cut to what a call
    /// may walk and to the address space the table has.
    fn pages(at: usize, len: usize) -> (usize, usize) {
        let first = at / KP;
        let last = at.saturating_add(len).div_ceil(KP).min(CHUNKS * WORDS * 64);
        (first, last.min(first.saturating_add(MOST_PAGES)))
    }

    /// The chunk of `page`'s word, made if `make` says so.
    fn chunk(&self, page: usize, make: bool) -> Option<&[AtomicU64]> {
        let slot = self.chunks.get(page / (WORDS * 64))?;
        if make {
            Some(slot.get_or_init(|| (0..WORDS).map(|_| AtomicU64::new(0)).collect()))
        } else {
            slot.get().map(|c| &**c)
        }
    }

    /// The word that holds `page`'s bit, if its chunk is there.
    fn word(&self, page: usize) -> Option<&AtomicU64> {
        self.chunk(page, false).and_then(|c| c.get(page / 64 % WORDS))
    }

    /// Calls `f` with each word that holds pages of `[first, last)`: the
    /// first of those pages, the word, and the mask of those pages in it.
    /// Words of chunks that are not there are passed over unless `make`.
    fn words(&self, first: usize, last: usize, make: bool, mut f: impl FnMut(usize, &AtomicU64, u64)) {
        let mut page = first;
        while page < last {
            let bit = page % 64;
            let n = (64 - bit).min(last - page);
            let mask = mask_below(bit + n) & mask_from(bit);
            if let Some(word) = self.chunk(page, make).and_then(|c| c.get(page / 64 % WORDS)) {
                f(page, word, mask);
            }
            page += n;
        }
    }

    /// Sets the bits of every page `[at, at + len)` touches.
    pub(crate) fn set(&self, at: usize, len: usize) {
        if len == 0 {
            return;
        }
        let (first, last) = Self::pages(at, len);
        self.words(first, last, true, |_, word, mask| {
            if word.load(Relaxed) & mask != mask {
                word.fetch_or(mask, Relaxed);
            }
        });
    }

    /// Whether the page of `at` has its bit set.
    pub(crate) fn get(&self, at: usize) -> bool {
        let page = at / KP;
        self.word(page).is_some_and(|word| word.load(Relaxed) & (1 << (page % 64)) != 0)
    }

    /// Clears the bit of `at`'s page; whether it was set.
    pub(crate) fn take(&self, at: usize) -> bool {
        let page = at / KP;
        let bit = 1u64 << (page % 64);
        self.word(page)
            .is_some_and(|word| word.load(Relaxed) & bit != 0 && word.fetch_and(!bit, Relaxed) & bit != 0)
    }

    /// The pages of `[at, at + len)` whose bit is set here and, if given,
    /// in `other` too.
    pub(crate) fn count(&self, other: Option<&Bits>, at: usize, len: usize) -> u64 {
        if len == 0 {
            return 0;
        }
        let (first, last) = Self::pages(at, len);
        let mut n = 0u64;
        self.words(first, last, false, |page, word, mask| {
            let mut bits = word.load(Relaxed) & mask;
            if bits != 0
                && let Some(other) = other
            {
                bits &= other.word(page).map_or(0, |w| w.load(Relaxed));
            }
            n += u64::from(bits.count_ones());
        });
        n
    }

    /// The 64 KiB pieces (their first address) of `[at, at + len)` that
    /// hold a set bit of the range, `most` of them at most.
    fn pieces(&self, at: usize, len: usize, most: usize, out: &mut Vec<usize>) {
        if len == 0 {
            return;
        }
        let (first, last) = Self::pages(at, len);
        let mut page = first;
        while page < last && out.len() < most {
            // 16 pages a piece, so a piece never lies across two words.
            let end = (page / 16 * 16 + 16).min(last);
            let bit = page % 64;
            let mask = mask_below(bit + (end - page)) & mask_from(bit);
            if self.word(page).is_some_and(|w| w.load(Relaxed) & mask != 0) {
                out.push(page / 16 * 16 * KP);
            }
            page = end;
        }
    }

    /// Clears every bit.
    pub(crate) fn clear(&self) {
        for chunk in self.chunks.iter().filter_map(|c| c.get()) {
            for word in chunk.iter() {
                if word.load(Relaxed) != 0 {
                    word.store(0, Relaxed);
                }
            }
        }
    }
}

/// The bits of a word from bit `from` on.
fn mask_from(from: usize) -> u64 {
    if from >= 64 { 0 } else { u64::MAX << from }
}

/// The bits of a word below bit `below`.
fn mask_below(below: usize) -> u64 {
    if below >= 64 { u64::MAX } else { (1u64 << below) - 1 }
}

/// Pages a landing put bytes into that no refresh has read since.
static LANDED: Bits = Bits::new();
/// Pages the guards answered for: view memory (no write watch there).
static VIEW: Bits = Bits::new();
/// Since the last line: pages a compare on the card flagged; pages the
/// title's CPU wrote; pages a CPU step of ours read.
static FLAGGED: Bits = Bits::new();
static TITLE: Bits = Bits::new();
static READ: Bits = Bits::new();

/// Bytes a refresh read, by `equal * 4 + landed * 2 + view`.
static READ_AGAIN: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// A landing wrote `len` bytes of title memory at `at`.
pub(crate) fn landed(at: usize, len: usize) {
    LANDED.set(at, len);
}

/// The guards answered for `[at, at + len)`: view memory.
pub(crate) fn view(at: usize, len: usize) {
    VIEW.set(at, len);
}

/// A refresh of a copy in the card's memory read the 4 KiB block of `len`
/// bytes at `at` and found it equal to the copy's CPU side, or not.
pub(crate) fn refreshed_block(at: usize, len: usize, equal: bool) {
    let ours = LANDED.take(at);
    let view = VIEW.get(at);
    let class = usize::from(equal) * 4 + usize::from(ours) * 2 + usize::from(view);
    if let Some(n) = READ_AGAIN.get(class) {
        n.fetch_add(len as u64, Relaxed);
    }
    // The title's own write: every block read that no landing wrote (the
    // write watch or a guard said written), and one a landing wrote that
    // holds other bytes than the landing's.
    if !(ours && equal) {
        TITLE.set(at, len);
    }
}

/// The title's CPU wrote `[at, at + len)` (bytes found changed under a
/// flight's shader writes).
pub(crate) fn title_wrote(at: usize, len: usize) {
    TITLE.set(at, len);
}

/// A CPU step of ours read `[at, at + len)` of title memory to stand in
/// for the card: a fill of upload memory, a buffer copy of the title's.
pub(crate) fn cpu_read(at: usize, len: usize) {
    READ.set(at, len);
}

/// Pages that took a joined run (first to last changed byte, not
/// byte-exact) under flights' shader writes, and for each the submission
/// that was being prepared when it last did: the run went up behind every
/// flight sent before that one.
static JOINED: Bits = Bits::new();
static JOINED_AT: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// The most pages [`JOINED_AT`] holds before the old ones go.
const JOINED_MOST: usize = 1 << 15;

/// While `submission` was prepared, the title's bytes of `[at, at + len)`
/// went up as one run under flights' shader writes.
pub(crate) fn joined(at: usize, len: usize, submission: u64) {
    if len == 0 {
        return;
    }
    JOINED.set(at, len);
    let (first, last) = Bits::pages(at, len);
    let mut map = JOINED_AT.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(Default::default);
    if map.len() > JOINED_MOST {
        // Flights land within a frame or two: what was noted hundreds of
        // submissions ago is asked about by no landing still to come.
        map.retain(|_, noted| noted.saturating_add(512) > submission);
    }
    // (A run is one 64 KiB page at most.)
    for page in (first..last).take(64) {
        map.insert(page, submission);
    }
}

/// At the landing of flight `flight` (its submission's number): its
/// compare on the card flagged the blocks `mask` (bit `b`: the 4 KiB block
/// `b`) of the 64 KiB piece of title memory at `piece`.
pub(crate) fn flagged(flight: u64, piece: usize, mask: u32) {
    let (mut looked, mut hit) = (0u64, 0u64);
    for b in 0..16usize {
        if mask & (1 << b) == 0 {
            continue;
        }
        let at = piece.saturating_add(b * KP);
        FLAGGED.set(at, 1);
        looked += 1;
        if JOINED.get(at) {
            let map = JOINED_AT.lock().unwrap_or_else(|e| e.into_inner());
            let noted = map.as_ref().and_then(|m| m.get(&(at / KP)).copied());
            hit += u64::from(noted.is_some_and(|noted| noted > flight));
        }
    }
    JOINED_FLAGGED[0].fetch_add(looked, Relaxed);
    JOINED_FLAGGED[1].fetch_add(hit, Relaxed);
}

/// The causes of `crate::writes::Cause`, by their number, for [`opened`].
const CAUSES: [&str; 7] = [
    "DirectStorage into pages",
    "DirectStorage to plain addresses",
    "copies of the title's done by the layer",
    "landings of what shaders wrote",
    "texels copied back",
    "the runtime's writes through the kernel",
    "mappings",
];

/// Guarded blocks opened by a write of the layer's own, by cause.
static OPENED: [AtomicU64; CAUSES.len()] = [const { AtomicU64::new(0) }; CAUSES.len()];

/// A write of the layer's into title memory for `cause` (the number of a
/// `crate::writes::Cause`) opened `blocks` guarded blocks.
pub(crate) fn opened(cause: usize, blocks: u64) {
    if let Some(n) = OPENED.get(cause) {
        n.fetch_add(blocks, Relaxed);
    }
}

/// Why `Shadow::merge_under_writes` left the flights to be settled.
pub(crate) const UNWATCHED: usize = 0;
pub(crate) const NO_JOURNAL: usize = 1;
pub(crate) const NO_SNAPSHOT: usize = 2;
pub(crate) const RUN_CAP: usize = 3;
const REASONS: [&str; 4] = [
    "the copy's memory is neither watched nor guarded",
    "nothing can say what was written there",
    "the copy's bytes from before are missing",
    "more changed runs than a submission may send up",
];

/// Since the last line, per reason: how often, and the last case (the
/// copy's start and length, the page looked at, the runs found on it and
/// the runs the submission had left).
static GAVE_UP: [AtomicU64; REASONS.len()] = [const { AtomicU64::new(0) }; REASONS.len()];
type Case = (u64, u64, usize, usize, usize);
static LAST_CASE: Mutex<[Option<Case>; REASONS.len()]> = Mutex::new([None; REASONS.len()]);

/// The flights are settled for `reason`: the copy `[start, start + len)`,
/// the page at `page` with `runs` changed runs, `left` runs the submission
/// could still send.
pub(crate) fn gave_up(reason: usize, start: u64, len: u64, page: usize, runs: usize, left: usize) {
    let Some(n) = GAVE_UP.get(reason) else {
        return;
    };
    n.fetch_add(1, Relaxed);
    let mut last = LAST_CASE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(case) = last.get_mut(reason) {
        *case = Some((start, len, page, runs, left));
    }
}

/// The upper bounds of the classes pages and blocks are counted in by
/// their changed runs (the last class has none).
const RUN_CLASSES: [usize; 5] = [0, 16, 64, 512, 4096];

fn run_class(runs: usize) -> usize {
    RUN_CLASSES.iter().position(|&most| runs <= most).unwrap_or(RUN_CLASSES.len())
}

/// Since the last line: pages looked at under flights' writes by the exact
/// runs found on them, and those runs; 4 KiB blocks a compare ever flagged
/// by the runs in them, and those runs.
static UNDER_PAGES: [[AtomicU64; 2]; RUN_CLASSES.len() + 1] = [const { [const { AtomicU64::new(0) }; 2] }; RUN_CLASSES.len() + 1];
static UNDER_BLOCKS: [[AtomicU64; 2]; RUN_CLASSES.len() + 1] = [const { [const { AtomicU64::new(0) }; 2] }; RUN_CLASSES.len() + 1];

/// A page looked at under flights' writes had `runs` exact changed runs.
pub(crate) fn under_page(runs: usize) {
    if let Some(class) = UNDER_PAGES.get(run_class(runs)) {
        class[0].fetch_add(1, Relaxed);
        class[1].fetch_add(runs as u64, Relaxed);
    }
}

/// A 4 KiB block a compare ever flagged had `runs` changed runs when its
/// page was looked at under flights' writes.
pub(crate) fn under_block(runs: usize) {
    if let Some(class) = UNDER_BLOCKS.get(run_class(runs)) {
        class[0].fetch_add(1, Relaxed);
        class[1].fetch_add(runs as u64, Relaxed);
    }
}

/// Since the last line: the blocks flights' compares flagged at their
/// landings, and those of them that had taken a joined run (not
/// byte-exact) after their flight was sent ([`joined`], [`flagged`]): a
/// shader's bytes there may have been put back to what the copy held
/// before.
static JOINED_FLAGGED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the last line: refreshes recorded and their bytes, by who asked
/// (`crate::shadow::ASKED_BY_DRAW`, `ASKED_BY_PICKED`, `ASKED_BY_ADDRESS`,
/// in that order).
static ASKED: [[AtomicU64; 2]; 3] = [const { [const { AtomicU64::new(0) }; 2] }; 3];
/// Since the last line, by the askers a copy had when it was brought up to
/// date (0 to 7, bits as above): updates, the bytes they named, and the
/// updates that found nothing written.
static UPDATED: [[AtomicU64; 3]; 8] = [const { [const { AtomicU64::new(0) }; 3] }; 8];

/// A refresh of `len` bytes was recorded for `who` (one of the bits).
pub(crate) fn asked(who: u8, len: u64) {
    let at = match who {
        crate::shadow::ASKED_BY_PICKED => 1,
        crate::shadow::ASKED_BY_ADDRESS => 2,
        _ => 0,
    };
    ASKED[at][0].fetch_add(1, Relaxed);
    ASKED[at][1].fetch_add(len, Relaxed);
}

/// A copy asked for by `askers` was brought up to date over `len` bytes;
/// `copied` of them were read from title memory.
pub(crate) fn updated(askers: u8, len: usize, copied: usize) {
    if let Some(class) = UPDATED.get(usize::from(askers & 7)) {
        class[0].fetch_add(1, Relaxed);
        class[1].fetch_add(len as u64, Relaxed);
        class[2].fetch_add(u64::from(copied == 0), Relaxed);
    }
}

/// The most flagged pieces a line looks at for other copies of their
/// memory (each is a question to the tables of views and of copies).
const MOST_PIECES: usize = 2048;

/// Whether a host copy other than `[start, start + len)` holds a byte of
/// the 64 KiB at `piece`: `spans` is every live copy (start, bytes), in
/// order of start.
fn other_copy(spans: &[(u64, u64)], start: u64, len: u64, piece: u64) -> bool {
    let end = piece.saturating_add(1 << 16);
    // Spans start in order; none is longer than `MOST_PAGES` pages, so the
    // ones that can reach the piece start no further back than that.
    let from = spans.partition_point(|s| s.0 < piece.saturating_sub((MOST_PAGES * KP) as u64));
    spans
        .get(from..)
        .unwrap_or(&[])
        .iter()
        .take_while(|s| s.0 < end)
        .any(|&(s, n)| s.saturating_add(n) > piece && (s, n) != (start, len))
}

/// The census's lines, with the frame-rate line. `frames`: the frames
/// since the last one.
pub(crate) fn lines(frames: f64) {
    let frames = frames.max(1.0);
    let per = |a: &AtomicU64| a.swap(0, Relaxed) as f64 / frames;
    let mb = |v: f64| v / (1 << 20) as f64;

    let r: Vec<f64> = READ_AGAIN.iter().map(per).collect();
    tracing::info!(
        "census, title memory read by refreshes of copies in the card's memory, per frame: equal to the copy and written by a \
         landing since it was last read (the layer reading its own write) {:.1} MB in view memory, {:.1} MB in private memory; \
         equal and written by no landing (the title wrote what was there) {:.1} MB and {:.1} MB; other bytes than the copy's \
         after a landing {:.1} MB and {:.1} MB; other bytes, no landing {:.1} MB and {:.1} MB",
        mb(r[7]),
        mb(r[6]),
        mb(r[5]),
        mb(r[4]),
        mb(r[3]),
        mb(r[2]),
        mb(r[1]),
        mb(r[0]),
    );

    let opened: Vec<String> = CAUSES
        .iter()
        .zip(&OPENED)
        .map(|(what, n)| (what, per(n)))
        .filter(|(_, n)| *n > 0.0)
        .map(|(what, n)| format!("{n:.1} by {what}"))
        .collect();
    tracing::info!(
        "census, guarded blocks opened by the layer's own writes per frame (the rest of the guards' \"blocks opened\" are the \
         title's faults): {}",
        if opened.is_empty() { "none".to_owned() } else { opened.join(", ") }
    );

    let gave_up: Vec<f64> = GAVE_UP.iter().map(per).collect();
    let cases = std::mem::take(&mut *LAST_CASE.lock().unwrap_or_else(|e| e.into_inner()));
    let why: Vec<String> = REASONS
        .iter()
        .zip(&gave_up)
        .zip(&cases)
        .filter(|((_, n), _)| **n > 0.0)
        .map(|((what, n), case)| match case {
            Some((start, len, page, runs, left)) => format!(
                "{n:.2} because {what} (the last: copy {start:#x}+{len:#x}, the page at {page:#x}, {runs} runs on it, {left} left)"
            ),
            None => format!("{n:.2} because {what}"),
        })
        .collect();
    let classes = |table: &[[AtomicU64; 2]; RUN_CLASSES.len() + 1]| -> String {
        let names = ["none", "1-16", "17-64", "65-512", "513-4096", "more"];
        table
            .iter()
            .zip(names)
            .map(|(class, name)| format!("{name}: {:.1} ({:.0} runs)", per(&class[0]), per(&class[1])))
            .collect::<Vec<_>>()
            .join(", ")
    };
    tracing::info!(
        "census, title bytes under flights' shader writes per frame: flights left to be settled {}; pages looked at by their \
         exact changed runs: {}; blocks a compare ever flagged, by the runs in them: {}; at landings {:.1} flagged blocks \
         looked at, {:.2} of them had taken a joined run while their flight was in the air (a shader's bytes there may be lost)",
        if why.is_empty() { "0 times".to_owned() } else { why.join("; ") },
        classes(&UNDER_PAGES),
        classes(&UNDER_BLOCKS),
        per(&JOINED_FLAGGED[0]),
        per(&JOINED_FLAGGED[1]),
    );

    let a: Vec<[f64; 2]> = ASKED.iter().map(|c| [per(&c[0]), per(&c[1])]).collect();
    let u: Vec<[f64; 3]> = UPDATED.iter().map(|c| [per(&c[0]), per(&c[1]), per(&c[2])]).collect();
    // Copies only the pass over picked views asked for; only draws; both;
    // any with memory by address.
    let sum = |which: &[usize]| -> [f64; 3] {
        which.iter().fold([0.0; 3], |acc, &i| [acc[0] + u[i][0], acc[1] + u[i][1], acc[2] + u[i][2]])
    };
    let (picked, draw, both, address, none) = (sum(&[2]), sum(&[1]), sum(&[3]), sum(&[4, 5, 6, 7]), sum(&[0]));
    tracing::info!(
        "census, who asks for refreshes, per frame: recorded {:.0} for draws' own views ({:.0} MB), {:.0} by the pass over \
         picked buffer views ({:.0} MB), {:.0} for memory by address ({:.0} MB); copies brought up to date: {:.0} only the \
         pass asked for ({:.0} MB, {:.0} of them with nothing written), {:.0} only draws ({:.0} MB, {:.0} nothing written), \
         {:.0} both ({:.0} MB, {:.0}), {:.0} with memory by address ({:.0} MB, {:.0}), {:.0} asked by none of these \
         ({:.0} MB, {:.0})",
        a[0][0],
        mb(a[0][1]),
        a[1][0],
        mb(a[1][1]),
        a[2][0],
        mb(a[2][1]),
        picked[0],
        mb(picked[1]),
        picked[2],
        draw[0],
        mb(draw[1]),
        draw[2],
        both[0],
        mb(both[1]),
        both[2],
        address[0],
        mb(address[1]),
        address[2],
        none[0],
        mb(none[1]),
        none[2],
    );

    // Per copy in the card's memory: what the card wrote since the last
    // line, and who else touched it.
    let cards = crate::shadow::card_spans();
    let mut all = crate::shadow::copy_spans();
    all.sort_unstable();
    let mut rows: Vec<(u64, u64, u64, u64, u64, u64)> = Vec::new();
    let mut total = [0u64; 4];
    let mut pieces_left = MOST_PIECES;
    let mut pieces: Vec<usize> = Vec::new();
    for &(start, len) in &cards {
        let (at, n) = (start as usize, len as usize);
        let flagged = FLAGGED.count(None, at, n);
        if flagged == 0 {
            continue;
        }
        let title = FLAGGED.count(Some(&TITLE), at, n);
        let read = FLAGGED.count(Some(&READ), at, n);
        // The flagged blocks in pieces another copy holds too: one that
        // overlaps this one's memory, or one over another address of the
        // same physical pages.
        pieces.clear();
        FLAGGED.pieces(at, n, pieces_left, &mut pieces);
        pieces_left = pieces_left.saturating_sub(pieces.len());
        let mut shared = 0u64;
        for &piece in &pieces {
            let elsewhere = other_copy(&all, start, len, piece as u64)
                || crate::inplace::pages_of(piece)
                    .into_iter()
                    .any(|alias| alias != piece && other_copy(&all, u64::MAX, 0, alias as u64));
            if elsewhere {
                let lo = piece.max(at);
                let hi = piece.saturating_add(1 << 16).min(at.saturating_add(n));
                shared += FLAGGED.count(None, lo, hi.saturating_sub(lo));
            }
        }
        total = [total[0] + flagged, total[1] + title, total[2] + read, total[3] + shared];
        rows.push((flagged, start, len, title, read, shared));
    }
    rows.sort_unstable_by_key(|r| std::cmp::Reverse(r.0));
    let listed: Vec<String> = rows
        .iter()
        .take(16)
        .map(|&(flagged, start, len, title, read, shared)| {
            format!("{start:#x}+{len:#x}: {flagged} blocks, {title} the title wrote too, {read} a step of ours read, {shared} another copy holds")
        })
        .collect();
    tracing::info!(
        "census, what shaders wrote since the last such line (4 KiB blocks a compare on the card flagged, counted once however \
         often): {} copies, {} blocks; of them {} the title's CPU wrote too in that time, {} a fill or a title buffer copy of \
         ours read, {} lie in memory another host copy holds too ({}); the most written: {}",
        rows.len(),
        total[0],
        total[1],
        total[2],
        total[3],
        if pieces_left == 0 { "not all pieces looked at" } else { "every piece looked at" },
        if listed.is_empty() { "none".to_owned() } else { listed.join("; ") }
    );
    FLAGGED.clear();
    TITLE.clear();
    READ.clear();

    let used = crate::shadow::card_use();
    tracing::info!(
        "census, copies in the card's memory by when they were last asked for: {} this frame or the one before {:.0} MB, {} \
         within 60 frames {:.0} MB, {} within 600 frames {:.0} MB, {} longer ago {:.0} MB",
        used[0][0],
        mb(used[0][1] as f64),
        used[1][0],
        mb(used[1][1] as f64),
        used[2][0],
        mb(used[2][1] as f64),
        used[3][0],
        mb(used[3][1] as f64),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_set_counted_and_taken_by_page() {
        let bits = Bits::new();
        // Two pages, the second touched by one byte.
        bits.set(0x1000, 0x1001);
        assert!(bits.get(0x1000) && bits.get(0x2fff) && !bits.get(0x3000) && !bits.get(0xfff));
        assert_eq!(bits.count(None, 0, 0x10000), 2);
        assert_eq!(bits.count(None, 0x2000, 0x1000), 1);
        assert!(bits.take(0x1234) && !bits.take(0x1234));
        assert_eq!(bits.count(None, 0, 0x10000), 1);
        // Across a word and across a chunk.
        bits.set(63 * KP, 2 * KP);
        assert_eq!(bits.count(None, 62 * KP, 4 * KP), 2);
        bits.set(CHUNK - KP, 2 * KP);
        assert_eq!(bits.count(None, CHUNK - 2 * KP, 4 * KP), 2);
        bits.clear();
        assert_eq!(bits.count(None, 0, 2 * CHUNK), 0);
    }

    #[test]
    fn two_tables_are_counted_where_both_are_set() {
        let (a, b) = (Bits::new(), Bits::new());
        a.set(0x10_0000, 0x8000);
        b.set(0x10_4000, 0x8000);
        assert_eq!(a.count(Some(&b), 0x10_0000, 0x10_0000), 4);
        assert_eq!(a.count(Some(&b), 0x10_0000, 0x5000), 1);
        // A table with no chunk there: nothing in common.
        assert_eq!(a.count(Some(&Bits::new()), 0x10_0000, 0x10_0000), 0);
    }

    /// The ranges come from the title and from shaders.
    #[test]
    fn ranges_that_are_nonsense_count_nothing_and_do_not_panic() {
        let bits = Bits::new();
        bits.set(usize::MAX - 5, 100);
        bits.set(usize::MAX, usize::MAX);
        bits.set(0x5000, 0);
        // Past the address space the table has.
        bits.set(CHUNKS * CHUNK + 0x1000, 0x1000);
        assert!(!bits.get(usize::MAX) && !bits.take(usize::MAX));
        assert_eq!(bits.count(None, usize::MAX - 5, usize::MAX), 0);
        assert_eq!(bits.count(None, 0x5000, 0), 0);
        // A range longer than a call walks is cut, not walked to its end.
        bits.set(0, usize::MAX);
        assert_eq!(bits.count(None, 0, usize::MAX), MOST_PAGES as u64);
        let mut out = Vec::new();
        bits.pieces(usize::MAX - 1, usize::MAX, 8, &mut out);
        assert!(out.is_empty());
        refreshed_block(usize::MAX, usize::MAX, true);
        flagged(0, usize::MAX, u32::MAX);
        joined(usize::MAX, usize::MAX, 0);
        joined(0x10, 0, 0);
        gave_up(99, 0, 0, 0, 0, 0);
        opened(99, 1);
        updated(255, usize::MAX, 0);
        asked(255, u64::MAX);
    }

    /// A block that took a joined run while submission 10 was prepared:
    /// flight 9 was in the air then, flight 10 was not sent yet.
    #[test]
    fn a_joined_run_counts_against_the_flights_sent_before_it() {
        let at = 0x7_1234_0000usize;
        joined(at + 0x1100, 0x200, 10);
        let hits = || JOINED_FLAGGED[1].load(Relaxed);
        let before = hits();
        // Block 1 of the piece took the run; block 0 did not.
        flagged(9, at, 0b01);
        assert_eq!(hits(), before);
        flagged(9, at, 0b10);
        assert_eq!(hits(), before + 1);
        flagged(10, at, 0b10);
        flagged(11, at, 0b10);
        assert_eq!(hits(), before + 1);
    }

    #[test]
    fn flagged_pieces_are_found_by_their_first_address() {
        let bits = Bits::new();
        bits.set(0x2_3000, 1);
        bits.set(0x5_f000, 0x2000);
        let mut out = Vec::new();
        bits.pieces(0x1_0000, 0x10_0000, 16, &mut out);
        assert_eq!(out, [0x2_0000, 0x5_0000, 0x6_0000]);
        // Cut to the range asked for: a bit below it is not the range's.
        out.clear();
        bits.pieces(0x2_4000, 0x1000, 16, &mut out);
        assert!(out.is_empty());
        out.clear();
        bits.pieces(0x1_0000, 0x10_0000, 2, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn another_copy_is_one_that_holds_a_byte_of_the_piece() {
        let spans = [(0x10_0000, 0x10_0000), (0x18_0000, 0x1000), (0x40_0000, 0x1_0000)];
        // The copy itself is not another one; the small copy inside it is.
        assert!(!other_copy(&spans, 0x10_0000, 0x10_0000, 0x10_0000));
        assert!(other_copy(&spans, 0x10_0000, 0x10_0000, 0x18_0000));
        assert!(other_copy(&spans, 0x18_0000, 0x1000, 0x18_0000));
        assert!(!other_copy(&spans, 0x40_0000, 0x1_0000, 0x40_0000));
        assert!(!other_copy(&spans, 0, 0, 0x30_0000));
        assert!(other_copy(&spans, u64::MAX, 0, 0x40_0000));
    }

    #[test]
    fn runs_fall_into_their_classes() {
        assert_eq!([0, 1, 16, 17, 512, 513, 4096, 4097].map(run_class), [0, 1, 1, 2, 3, 4, 4, 5]);
    }
}
