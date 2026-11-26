//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved. Each page's entry says
/// which of its 4 KiB pages were written ([`Written`]), so that a refresh
/// can copy less than the page (`crate::shadow`'s `VICTORY_REFRESH_4K`).
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

/// What the journal knows of a page's writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Written {
    /// The asking that last found the page written.
    pub(crate) at: u64,
    /// The asking that found it written before that one (0: none), and the
    /// kernel's 4 KiB pages of it (a bit each, from the page's first) that
    /// every write found after `before`, up to `at`, lies in. [`FULL`]
    /// when that is not known (view memory, our own writes, a late answer).
    pub(crate) before: u64,
    pub(crate) mask: u16,
}

/// [`Written::mask`] of a page written anywhere, or not known where.
pub(crate) const FULL: u16 = u16::MAX;

/// Notes in `written` that asking `now` found `page` written in the kernel
/// pages `mask`.
fn note(written: &mut BTreeMap<usize, Written>, page: usize, now: u64, mask: u16) {
    match written.get_mut(&page) {
        None => {
            written.insert(page, Written { at: now, before: 0, mask });
        }
        Some(w) if w.at < now => *w = Written { at: now, before: w.at, mask },
        // The same asking again (another range of it): its pages together.
        Some(w) if w.at == now => w.mask |= mask,
        // An answer numbered before the last one noted (an asking that
        // stood while another was made): kept at the later number, which
        // is the safe side, and where is no longer known.
        Some(w) => w.mask = FULL,
    }
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written (and where).
    written: BTreeMap<usize, Written>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
    /// Numbers the submissions' askings ([`ask_once`]): what the guards
    /// learn while one stands is kept for it ([`absent_once`]).
    once_id: u64,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
    once_id: 0,
});

/// Notes the kernel's answer (`j.found[..count]`, addresses of written
/// 4 KiB pages) under asking `now`, a mask per 64 KiB page.
fn note_found(j: &mut Journal, count: usize, now: u64) {
    let mut k = 0;
    while k < count {
        let page = j.found[k] / PAGE;
        let mut mask = 0u16;
        while k < count && j.found[k] / PAGE == page {
            mask |= 1 << ((j.found[k] % PAGE) / KERNEL_PAGE);
            k += 1;
        }
        note(&mut j.written, page, now, mask);
    }
}

/// The journal's entries for pages `[first, last)`.
fn entries(j: &Journal, first: usize, last: usize) -> Vec<(usize, Written)> {
    j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()
}

/// Whether ranges whose first block the guards hold are left out of the
/// submission's kernel askings ([`ask_ahead`], [`ask_once`]):
/// `VICTORY_ASK_SKIP_VIEWS=0` asks about them, as before. View memory is
/// not write-watched, so the asking can only fail; in a run of ranges it
/// failed the whole run, which was then asked again by halves, and each
/// such range was asked once more on its own. [`writes`] guards them as it
/// did.
fn skip_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_SKIP_VIEWS").is_none_or(|v| v != "0"))
}

/// `ranges` without those [`known_view`] says are view memory (with
/// [`skip_views`]).
fn watchable(ranges: &[(usize, usize)]) -> std::borrow::Cow<'_, [(usize, usize)]> {
    if !skip_views() || view_ask() {
        return std::borrow::Cow::Borrowed(ranges);
    }
    let g = guards();
    std::borrow::Cow::Owned(ranges.iter().copied().filter(|r| !held(&g, r.0 / BLOCK)).collect())
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let ranges = watchable(ranges);
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_end();
    let ranges = &*watchable(ranges);
    ask_ahead(ranges);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.once_id += 1;
    if j.ahead.is_empty() || j.asked != j.ahead_asked {
        // `ask_ahead` asked nothing (fewer than two ranges), or another
        // asking was made since it let go of the journal: the askings below
        // are numbered after that one, so that what they find is not noted
        // as found before it.
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        note(&mut j.written, page, now, FULL);
    }
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    note_found(j, count, now);
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, w)| (p, w.at)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    writes_fine(start, end, guarded).map(|(now, pages)| (now, pages.into_iter().map(|(p, w)| (p, w.at)).collect()))
}

/// [`writes`], with what the journal knows of where in each page.
pub(crate) fn writes_fine(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, Written)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((j.ahead_asked, entries(j, first, last)));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    let known_view = guarded && known_view(first * PAGE);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known_view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        // While the submission's one asking stands, what the guards found
        // unmapped stays so for it (see [`absent_once`]).
        let epoch = once().then_some(Epoch { submission: j.once_id, standing: j.kept });
        let written = guard_fine(first * PAGE, last * PAGE, epoch);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    note_found(j, count, now);
    for (page, mask) in guarded {
        note(&mut j.written, page, now, mask);
    }
    Some((now, entries(j, first, last)))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory: the journal's pages, one to a block.
const BLOCK: usize = PAGE;
const _: () = assert!(BLOCK == PAGE && BLOCK / KERNEL_PAGE == 16);

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    !view_ask() && held(&guards(), start / BLOCK)
}

/// `VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1` (see [`known_view`]).
fn view_ask() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// Whether the guards hold `block` as view memory: read-only by our hand,
/// noted absent, or (with [`views_known`]) guarded by us since the runtime
/// last mapped or unmapped it.
fn held(g: &Guards, block: usize) -> bool {
    g.protected.contains(&block) || g.absent.contains(&block) || (views_known() && g.views.contains(&block))
}

/// Whether a block the guards made read-only stays known as view memory
/// after a write opened it, until the runtime maps or unmaps it
/// (`VICTORY_VIEW_KNOWN=0`: only while it is read-only or absent, as
/// before).
///
/// A write opens the block, and the next refresh of a copy that begins
/// there asked the write watch (which fails for a view) and then Windows
/// whether the memory is a view (`VirtualQuery`, which walks to the end of
/// the region): the refreshes after each of a frame's guard faults. The
/// block is a view still: what makes it something else is a mapping, which
/// the runtime tells ([`title_memory_remapped`]). Taken for a view, memory
/// is guarded, which sees writes to any memory; it is not write-watched.
fn views_known() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_KNOWN").is_none_or(|v| v != "0"))
}

/// Whether what the refreshes find of unmapped blocks holds for the rest of
/// the frame (`VICTORY_ABSENT_ONCE=0`: every refresh asks Windows again, as
/// before).
///
/// A block found wholly uncommitted is copied as zeros once and noted
/// absent; every later refresh over it asked Windows whether it still is
/// (`VirtualQuery`), since an ordinary commit is not told to us. The
/// refreshes cover the same memory many times over (heap and window copies
/// overlap, 29 submissions a frame), so the same question was asked
/// thousands of times a frame: one per absent run per submission at best
/// (round 2: 422 a submission, the placeholders between views split into
/// runs of one block).
///
/// With the switch it is asked once a frame for each run. The title's view
/// memory is committed only by the runtime's mappings, which tell us
/// ([`title_memory_remapped`] forgets the runs they touch); what else could
/// commit such a block unseen (an ordinary reservation or freed memory
/// inside a range taken for view memory, committed by the title) is seen
/// at the next frame. Also: committed blocks that cannot be made read-only
/// are not tried again within a submission.
///
/// Round 3 asked once a frame (about 7,000 questions a frame: thousands of
/// distinct runs, the holes between streamed views being one block each).
/// Now the answer holds for [`absent_frames`] frames, and the runtime's
/// allocations of title memory (`XMemVirtualAlloc`, which is how freed
/// address space comes back as title GPU memory) count as remappings.
fn absent_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ABSENT_ONCE").is_none_or(|v| v != "0"))
}

/// Frames an answer of [`absent_once`] holds for: `VICTORY_ABSENT_FRAMES`,
/// 30 unless set (1: a frame, as in round 3). What can commit an absent
/// block without our being told is the title committing part of a
/// reservation of its own (not one of the runtime's placeholders, which
/// only the runtime's mappings fill): such memory is seen this many frames
/// late at most.
fn absent_frames() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_ABSENT_FRAMES").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&n| n > 0).unwrap_or(30)
    })
}

/// Whether blocks the title writes again and again are left writable and
/// counted written once a submission instead of being guarded again after
/// each write (`VICTORY_HOT_BLOCKS=1`; off by default: round 2 reported 160
/// blocks a submission written, 112 MB a frame more copied, for 0.3 blocks
/// made hot a submission; with [`mark_only`] a write costs one re-arming).
///
/// A guarded block the title writes faults once (the handler gives write
/// access back to it and to every other view of its pages), and the next
/// refresh makes it and those views read-only again: in the intro 378 such
/// faults a frame, each re-armed with its other views, most of the
/// submitting thread's 11 ms a frame of `VirtualProtect` (each call flushes
/// the address on every processor running the process), and a fault on
/// the title's thread for each. A block re-armed in [`HOT_TIMES`] of the
/// last [`HOT_WINDOW`] submissions is left open, with all the views of its
/// pages, for [`HOT_FOR`] submissions; meanwhile every check reports it
/// written, once a submission while the submission's one asking stands
/// (what its lists read was written before they were submitted, as
/// [`ask_once`] has it), at every check otherwise. Reporting a block
/// written that was not only copies it again: the copies hold what they
/// would hold.
fn hot_blocks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HOT_BLOCKS").is_some_and(|v| v == "1"))
}

/// See [`hot_blocks`]. The first arming counts as one.
const HOT_TIMES: u32 = 3;
const HOT_WINDOW: u64 = 32;
const HOT_FOR: u64 = 512;

/// What a check of the guards is told of the submission it is made for.
#[derive(Clone, Copy, Debug)]
struct Epoch {
    /// The submission's number ([`Journal::once_id`]).
    submission: u64,
    /// Whether the submission's one asking stands ([`ask_once`]).
    standing: bool,
}

/// A set of blocks as bits, 64 blocks to a word: whether a range of a few
/// thousand blocks is all in it takes some fifty lookups, where a tree of
/// blocks took one step per block (most of the guards' own time).
struct BlockSet {
    words: BTreeMap<usize, u64>,
}

impl BlockSet {
    const fn new() -> BlockSet {
        BlockSet { words: BTreeMap::new() }
    }

    fn word(&self, chunk: usize) -> u64 {
        self.words.get(&chunk).copied().unwrap_or(0)
    }

    fn contains(&self, block: &usize) -> bool {
        self.word(block / 64) & (1 << (block % 64)) != 0
    }

    fn insert(&mut self, block: usize) {
        *self.words.entry(block / 64).or_insert(0) |= 1 << (block % 64);
    }

    fn remove(&mut self, block: &usize) {
        if let std::collections::btree_map::Entry::Occupied(mut e) = self.words.entry(block / 64) {
            *e.get_mut() &= !(1u64 << (block % 64));
            if *e.get() == 0 {
                e.remove();
            }
        }
    }

    fn extend(&mut self, blocks: std::ops::Range<usize>) {
        for block in blocks {
            self.insert(block);
        }
    }

    /// The bits of chunk `chunk` that lie in `[first, last)`.
    fn span(chunk: usize, first: usize, last: usize) -> u64 {
        let lo = first.saturating_sub(chunk * 64).min(64);
        let hi = last.saturating_sub(chunk * 64).min(64);
        if lo >= hi {
            return 0;
        }
        (if hi == 64 { u64::MAX } else { (1u64 << hi) - 1 }) & !((1u64 << lo) - 1)
    }

    /// Whether every block of `[first, last)` is in the set.
    fn all(&self, first: usize, last: usize) -> bool {
        if first >= last {
            return true;
        }
        let (c0, c1) = (first / 64, (last - 1) / 64);
        let mut next = c0;
        for (&chunk, &word) in self.words.range(c0..=c1) {
            if chunk != next {
                return false;
            }
            let want = Self::span(chunk, first, last);
            if word & want != want {
                return false;
            }
            next = chunk + 1;
        }
        next == c1 + 1
    }

    /// The blocks of `[first, last)` in the set, in order.
    #[cfg(test)]
    fn range(&self, first: usize, last: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if first >= last {
            return out;
        }
        for (&chunk, &word) in self.words.range(first / 64..=(last - 1) / 64) {
            let mut bits = word & Self::span(chunk, first, last);
            while bits != 0 {
                out.push(chunk * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        out
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BlockSet,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BTreeSet<usize>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// Blocks made read-only or noted absent since the runtime last mapped
    /// or unmapped them: view memory ([`views_known`]).
    views: BTreeSet<usize>,
    /// With [`absent_once`]: runs of blocks found wholly uncommitted, first
    /// block → end, while the submission's asking numbered `absent_for`
    /// stands.
    absent_runs: BTreeMap<usize, usize>,
    absent_for: u64,
    /// With [`absent_once`]: committed blocks that could not be made
    /// read-only in the submission `unguardable_for` (they are reported
    /// written at every check, as before, without trying again).
    unguardable: BTreeSet<usize>,
    unguardable_for: u64,
    /// With [`hot_blocks`]: blocks re-armed lately, the times (in distinct
    /// submissions), the submission of the first of them and of the last.
    rearmed: BTreeMap<usize, (u32, u64, u64)>,
    /// Blocks left open: the submission that made them so, and the last
    /// submission whose standing asking they were reported written to.
    hot: BTreeMap<usize, (u64, u64)>,
    /// With [`guard_pages`]: guarded blocks written since they were last
    /// reported, and their kernel pages (a bit each) that may have been;
    /// a block not clean without an entry may have been written anywhere.
    /// A block here that is still guarded may have pages a fault made
    /// writable (from the faulting page to the block's end).
    dirty: BTreeMap<usize, u16>,
    /// Of those, the guarded ones a fault made writable in part.
    partly: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BlockSet::new(),
    absent: BTreeSet::new(),
    aliased: BTreeSet::new(),
    views: BTreeSet::new(),
    absent_runs: BTreeMap::new(),
    absent_for: 0,
    unguardable: BTreeSet::new(),
    unguardable_for: 0,
    rearmed: BTreeMap::new(),
    hot: BTreeMap::new(),
    dirty: BTreeMap::new(),
    partly: BTreeSet::new(),
});

/// Diagnostics, since the last [`guard_line`]: kernel calls the guards
/// made, by what for: `VirtualQuery` for whether a range is a view, for
/// whether absent blocks are still unmapped, after a failed protection;
/// `VirtualProtect` arming runs of blocks, arming blocks one by one (a run
/// that failed), arming other views, opening blocks; blocks made hot, hot
/// blocks reported written, checks that found their range all clean, all
/// checks; absent runs found in the cache, times the cache was emptied for
/// a new epoch, runs a remapping took out of it; faults that opened part
/// of a block ([`guard_pages`]), blocks open in part armed again; faults on
/// blocks a check made read-only after they were opened ([`reopen_stray`]).
#[cfg(feature = "diagnostics")]
pub(crate) static GUARD_CALLS: [std::sync::atomic::AtomicU64; 17] = [const { std::sync::atomic::AtomicU64::new(0) }; 17];

#[cfg(feature = "diagnostics")]
fn count(i: usize) {
    GUARD_CALLS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(not(feature = "diagnostics"))]
fn count(_: usize) {}

/// Diagnostics: logs [`GUARD_CALLS`] every five seconds or more, per
/// submission (the guards know no frames).
#[cfg(feature = "diagnostics")]
fn guard_line(submission: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: Mutex<Option<(std::time::Instant, u64)>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let Some((at, from)) = *last else {
        *last = Some((std::time::Instant::now(), submission));
        return;
    };
    if at.elapsed() < std::time::Duration::from_secs(5) || submission <= from {
        return;
    }
    *last = Some((std::time::Instant::now(), submission));
    let n = (submission - from) as f64;
    let c: Vec<f64> = GUARD_CALLS.iter().map(|a| a.swap(0, Relaxed) as f64 / n).collect();
    let same = [0, 1].map(|i| crate::shadow::SENT_SAME[i].swap(0, Relaxed) as f64 / n);
    let (hot, runs, absent, partly) = {
        let g = guards();
        (g.hot.len(), g.absent_runs.len(), g.absent.len(), g.dirty.len())
    };
    tracing::info!(
        "guards per submission: {:.1} checks, {:.1} all clean; VirtualQuery {:.1} is-view, {:.1} absent, {:.1} after a failed \
         protection; VirtualProtect {:.1} runs armed, {:.1} blocks armed one by one, {:.1} other views armed, {:.1} blocks opened; \
         {:.2} blocks made hot, {:.1} hot blocks reported written, {hot} hot now (VICTORY_HOT_BLOCKS=1 leaves blocks written often open); \
         absent runs: {:.1} found in the cache, cache emptied {:.2} times, {:.1} runs dropped by remappings, {runs} cached, {absent} \
         absent blocks; faults that opened part of a block {:.1}, such blocks armed again {:.1}, {partly} blocks with 4 KiB pages \
         noted written now, {:.2} faults on blocks armed after they were opened; refreshes of copies in the card's memory: {:.1} 4 KiB blocks ({:.2} MB) equal to the CPU side, not sent \
         up (VICTORY_SEND_CHANGED=0 sends them)",
        c[10], c[9], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[11], c[12], c[13], c[14], c[15], c[16], same[0], same[1] / 1e6
    );
}

impl Guards {
    /// The blocks that could not be made read-only in `epoch`'s submission.
    fn unguardable_in(&mut self, epoch: u64) -> &mut BTreeSet<usize> {
        if self.unguardable_for != epoch {
            self.unguardable_for = epoch;
            self.unguardable.clear();
        }
        &mut self.unguardable
    }

    /// The end of the run of uncommitted blocks from `block` on, as found
    /// while the submission's asking `epoch` stands, if it was.
    fn absent_until(&mut self, block: usize, epoch: u64) -> Option<usize> {
        if self.absent_for != epoch {
            count(12);
            self.absent_for = epoch;
            self.absent_runs.clear();
            return None;
        }
        let found = self
            .absent_runs
            .range(..=block)
            .next_back()
            .and_then(|(_, &until)| (until > block).then_some(until));
        if found.is_some() {
            count(11);
        }
        found
    }
}

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`guard_fine`] without the kernel pages: the pages written.
#[cfg(test)]
fn guard(start: usize, end: usize, epoch: Option<Epoch>) -> Option<Vec<usize>> {
    guard_fine(start, end, epoch).map(|w| w.into_iter().map(|(page, _)| page).collect())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
/// `epoch`: the submission the check is made for, with [`once`]
/// ([`absent_once`], [`hot_blocks`]).
fn guard_fine(start: usize, end: usize, epoch: Option<Epoch>) -> Option<Vec<(usize, u16)>> {
    count(10);
    #[cfg(feature = "diagnostics")]
    if let Some(e) = epoch {
        guard_line(e.submission);
    }
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    install();
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    // One taking of the guards' lock for a range they hold (the lock is
    // also the fault handler's, on the title's threads).
    let mut g = guards();
    if ask || !held(&g, start / BLOCK) {
        drop(g);
        if !is_view(start, end) {
            return None;
        }
        g = guards();
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The range's clean blocks, in order: read off the set in one pass,
    // not looked up one block at a time. A frame asks about some 10 GB of
    // view memory, 150,000 blocks, nearly all clean: the lookups were most
    // of the 60-90 ms a frame of "guards" (run 54). What the loop adds to
    // the set meanwhile is the block it stands at or ones before it.
    if g.clean.all(first, last) {
        // Nothing written, mapped or new since the range was last taken.
        count(9);
        return Some(written);
    }
    // What the loop adds to the clean blocks meanwhile is the block it
    // stands at or ones before it: the word read for a chunk stays right
    // for the blocks after.
    let (mut chunk, mut word) = (usize::MAX, 0u64);
    let mut uncommitted_until = first;
    // Unmapped blocks found so stay so for the frame ([`absent_once`]);
    // blocks that would not be protected, for the submission.
    let absent_epoch = epoch.filter(|_| absent_once()).map(|_| crate::frame::current() / absent_frames());
    let unguardable_epoch = epoch.filter(|_| absent_once()).map(|e| e.submission);
    let hot_epoch = epoch.filter(|_| hot_blocks());
    // With `VICTORY_GUARD_UNLOCKED`: the changes of protection this check
    // decides are made after the guards' lock is let go ([`Deferred`]).
    let defer = unlocked_guards() && hot_epoch.is_none();
    let mut later = Deferred::default();
    for block in first..=last {
        if block / 64 != chunk {
            chunk = block / 64;
            word = g.clean.word(chunk);
        }
        let is_clean = word & (1 << (block % 64)) != 0;
        let mut open = block < last && !is_clean;
        if open && g.absent.contains(&block) {
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if block >= uncommitted_until {
                uncommitted_until = match absent_epoch.and_then(|e| g.absent_until(block, e)) {
                    Some(until) => until,
                    None => {
                        count(1);
                        let until = uncommitted_end(block);
                        if absent_epoch.is_some() && until > block {
                            g.absent_runs.insert(block, until);
                        }
                        until
                    }
                };
            }
            open = block >= uncommitted_until;
            if open {
                g.absent.remove(&block);
            }
        }
        if open {
            if let Some(e) = hot_epoch
                && let Some(hot) = g.hot.get_mut(&block)
            {
                if e.submission < hot.0 + HOT_FOR {
                    // Left open: written, as far as anyone can tell; once a
                    // submission while its asking stands.
                    if !(e.standing && hot.1 == e.submission) {
                        if e.standing {
                            hot.1 = e.submission;
                        }
                        count(8);
                        written.push((block, FULL));
                    }
                    if let Some(from) = run.take() {
                        protect(&mut g, from, block, epoch);
                    }
                    continue;
                }
                // Its time is up: guarded again, and watched whether it
                // is still written that often.
                g.hot.remove(&block);
                g.rearmed.remove(&block);
            }
            let rearmed = g.protected.contains(&block) && {
                // Written through the runtime or another view, still
                // read-only; or opened by a fault from its kernel page on
                // ([`guard_pages`]), armed again whole now. Reported with
                // the kernel pages that may have been written.
                let mask = g.dirty.remove(&block).unwrap_or(FULL);
                written.push((block, mask));
                if defer {
                    if g.partly.remove(&block) {
                        later.rearm.push(block);
                    }
                    true
                } else {
                    rearm_tail(&mut g, block)
                }
            };
            if !g.protected.contains(&block) && !written.last().is_some_and(|&(b, _)| b == block) {
                written.push((block, FULL));
            }
            if rearmed {
                g.clean.insert(block);
            } else if unguardable_epoch.is_some_and(|e| g.unguardable_in(e).contains(&block)) {
                // Could not be made read-only in this submission: reported
                // written at every check, as it would be after trying.
            } else if let Some(e) = hot_epoch
                && rearmed_often(&mut g, block, e.submission)
            {
                make_hot(&mut g, block, e.submission);
                // Reported written just now (its other views are at their
                // next check).
                if e.standing
                    && let Some(hot) = g.hot.get_mut(&block)
                {
                    hot.1 = e.submission;
                }
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            if defer {
                plan_protect(&mut g, from, block, &mut later, epoch);
            } else {
                protect(&mut g, from, block, epoch);
            }
        }
    }
    drop(g);
    if defer && !later.is_empty() {
        let failed = later.apply();
        if !failed.is_empty() {
            settle(&mut guards(), &failed, epoch);
        }
    }
    Some(written)
}

/// Whether the guards' changes of protection are made with their lock let
/// go (`VICTORY_GUARD_UNLOCKED=0`: under it, as before).
///
/// A check held the lock across every `VirtualProtect` it made (and the
/// lookups of a block's other views), and the lock is the fault handler's:
/// the title's threads that wrote a guarded block waited for it, 15 ms a
/// frame on the recording threads (round 5's profile). Now a check notes
/// the blocks guarded first (so that a fault on one once it is read-only
/// finds it the guards'), lets the lock go, makes the calls, and takes the
/// lock again only for those that failed. A write landing on a block
/// between the note and the call is one before the check's copy reads the
/// block, and the check reports the block written into the journal, whose
/// lock it holds throughout (every other copy learns of it there).
fn unlocked_guards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_UNLOCKED").is_none_or(|v| v != "0"))
}

/// The changes of protection a check decided ([`unlocked_guards`]): runs
/// of blocks to make read-only (noted guarded and clean already), other
/// views of their pages (noted guarded), and blocks a fault opened in part
/// to make read-only whole again (noted clean).
#[derive(Default)]
struct Deferred {
    runs: Vec<(usize, usize)>,
    aliases: Vec<usize>,
    rearm: Vec<usize>,
}

/// The blocks whose protection could not be changed ([`Deferred::apply`]).
#[derive(Default)]
struct Failed {
    runs: Vec<usize>,
    aliases: Vec<usize>,
    rearm: Vec<usize>,
}

impl Failed {
    fn is_empty(&self) -> bool {
        self.runs.is_empty() && self.aliases.is_empty() && self.rearm.is_empty()
    }
}

impl Deferred {
    fn is_empty(&self) -> bool {
        self.runs.is_empty() && self.aliases.is_empty() && self.rearm.is_empty()
    }

    /// Makes the calls, without the guards' lock.
    fn apply(mut self) -> Failed {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        let set = |from: usize, to: usize| {
            let mut old = 0;
            // SAFETY: title view memory the guards hold; only its
            // protection changes.
            unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) != 0 }
        };
        let mut failed = Failed::default();
        for &(from, to) in &self.runs {
            count(if to - from > 1 { 3 } else { 4 });
            if set(from, to) {
                continue;
            }
            for block in from..to {
                count(4);
                if !set(block, block + 1) {
                    failed.runs.push(block);
                }
            }
        }
        self.aliases.sort_unstable();
        self.aliases.dedup();
        for (first, last) in block_runs(&self.aliases) {
            count(5);
            if last - first > 1 && set(first, last) {
                continue;
            }
            for block in first..last {
                if last - first > 1 {
                    count(5);
                }
                if !set(block, block + 1) {
                    failed.aliases.push(block);
                }
            }
        }
        for &block in &self.rearm {
            count(15);
            if !set(block, block + 1) {
                failed.rearm.push(block);
            }
        }
        failed
    }
}

/// [`protect`] for a check with [`unlocked_guards`]: notes the blocks of
/// `[from, to)` that may be guarded ([`view_run`], as [`protect_here`])
/// guarded and clean and their pages' other views guarded, and leaves the
/// calls to `later`.
fn plan_protect(g: &mut Guards, from: usize, to: usize, later: &mut Deferred, epoch: Option<Epoch>) {
    let mut at = from;
    while at < to {
        let (end, view) = kind_run(g, at, to);
        if view {
            plan_protect_run(g, at, end, later);
        } else {
            for block in at..end {
                not_guardable(g, block, epoch);
            }
        }
        at = end;
    }
}

/// [`plan_protect`] for a run of blocks that may be guarded.
fn plan_protect_run(g: &mut Guards, from: usize, to: usize, later: &mut Deferred) {
    for block in from..to {
        g.protected.insert(block);
        g.clean.insert(block);
        g.views.insert(block);
        g.dirty.remove(&block);
        g.partly.remove(&block);
    }
    later.runs.push((from, to));
    if !view_guard() {
        return;
    }
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block && !g.protected.contains(&alias) {
                    // Not clean: its copies have not taken these pages.
                    g.protected.insert(alias);
                    g.views.insert(alias);
                    g.dirty.remove(&alias);
                    later.aliases.push(alias);
                }
            }
        }
    }
}

/// For a write that faulted at `at` on a block of view memory the guards
/// know but do not hold guarded ([`unlocked_guards`]): if the block is
/// read-only (a check made it so after something opened it), makes it
/// writable and notes it written; true if that was done.
fn reopen_stray(g: &mut Guards, at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
    };
    let block = at / BLOCK;
    if g.protected.contains(&block) || !g.views.contains(&block) {
        return false;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT || info.Protect != PAGE_READONLY {
        return false;
    }
    count(16);
    let mut old = 0;
    // SAFETY: a block of title view memory we made read-only.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) } == 0 {
        return false;
    }
    g.clean.remove(&block);
    g.dirty.remove(&block);
    g.partly.remove(&block);
    true
}

/// Undoes the notes of the blocks whose protection could not be changed
/// ([`Deferred::apply`]), as [`protect_here`] would have noted them.
fn settle(g: &mut Guards, failed: &Failed, epoch: Option<Epoch>) {
    for &block in &failed.runs {
        g.protected.remove(&block);
        g.clean.remove(&block);
        count(2);
        if block < uncommitted_end(block) {
            g.absent.insert(block);
        } else {
            g.views.remove(&block);
            if let Some(e) = epoch.filter(|_| absent_once()) {
                g.unguardable_in(e.submission).insert(block);
            }
        }
    }
    for &block in &failed.aliases {
        g.protected.remove(&block);
    }
    for &block in &failed.rearm {
        // As `rearm_tail`: no longer guarded (reported written already).
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Notes that `block` is to be armed again in `submission` (with
/// [`hot_blocks`]); whether it was in [`HOT_TIMES`] submissions of the
/// last [`HOT_WINDOW`] (counting this one).
fn rearmed_often(g: &mut Guards, block: usize, submission: u64) -> bool {
    let e = g.rearmed.entry(block).or_insert((0, submission, 0));
    if submission >= e.1 + HOT_WINDOW {
        *e = (0, submission, 0);
    }
    if e.0 == 0 || e.2 != submission {
        e.0 += 1;
        e.2 = submission;
    }
    if g.rearmed.len() > 1 << 16 {
        // Blocks armed once and never again: forgotten.
        g.rearmed.retain(|_, e| submission < e.1 + HOT_WINDOW);
    }
    g.rearmed.get(&block).is_some_and(|e| e.0 >= HOT_TIMES)
}

/// Leaves `block` open from `submission` on, with every other view of its
/// pages: a write through any of them must be seen at all of them, and an
/// open view sees none.
fn make_hot(g: &mut Guards, block: usize, submission: u64) {
    count(7);
    let mut blocks = vec![block];
    if view_guard() {
        let shown = crate::inplace::pages_of(block * BLOCK);
        if (2..=MANY_VIEWS).contains(&shown.len()) {
            blocks.extend(shown.into_iter().map(|at| at / BLOCK).filter(|&b| b != block));
        }
    }
    for b in blocks {
        // An armed view: open (it is reported written at its next check,
        // as every hot block is).
        if g.protected.contains(&b) {
            open_here(g, b * BLOCK, (b + 1) * BLOCK);
        }
        g.hot.insert(b, (submission, 0));
    }
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        count(0);
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before. Free memory is no one's yet: taken
        // for a view, its blocks were noted absent and guarded once
        // something was allocated there, the process heap too (its own
        // code then faulted while holding its lock, the fault waited for
        // the guard lock, and the guard lock's holder for the heap: the
        // world froze seconds after the structure builder started,
        // tippythetaidum/VIppy 425228755; Marvisak/VIctory#1849 found it
        // under Wine).
        if info.State == MEM_COMMIT || info.State == MEM_FREE || !holes {
            return false;
        }
        // A hole (a placeholder): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize, epoch: Option<Epoch>) {
    protect_here(g, from, to, epoch);
    if !view_guard() {
        return;
    }
    let runs = alias_runs();
    let hot_epoch = epoch.filter(|_| hot_blocks()).map(|e| e.submission);
    let mut others = Vec::new();
    // Blocks one of whose other views is left open ([`hot_blocks`]): a
    // write there faults nowhere, so they are left open as well.
    let mut rehot = Vec::new();
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            if let Some(now) = hot_epoch
                && shown.iter().any(|&at| at / BLOCK != block && hot_now(g, at / BLOCK, now))
            {
                rehot.push(block);
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    if runs {
                        others.push(alias);
                    } else {
                        protect_alias(g, alias);
                    }
                }
            }
        }
    }
    for block in rehot {
        make_hot(g, block, hot_epoch.unwrap_or(0));
    }
    if !others.is_empty() {
        others.sort_unstable();
        others.dedup();
        others.retain(|b| !g.protected.contains(b));
        for (first, last) in block_runs(&others) {
            protect_aliases(g, first, last);
        }
    }
}

/// Whether `block` is left open ([`hot_blocks`]) in submission `now`; one
/// whose time is up is forgotten.
fn hot_now(g: &mut Guards, block: usize, now: u64) -> bool {
    match g.hot.get(&block) {
        Some(&(since, _)) if now < since + HOT_FOR => true,
        Some(_) => {
            g.hot.remove(&block);
            g.rearmed.remove(&block);
            false
        }
        None => false,
    }
}

/// Whether the other views of the blocks one guarding takes are made
/// read-only a run of blocks at a time (`VICTORY_ALIAS_RUNS=0`: a block at a
/// time, as before). Each change of protection is a call into the kernel
/// that flushes the address's translations on every processor running the
/// process; the other views of a texture or buffer mapped twice lie side by
/// side as the guarded ones do.
fn alias_runs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_RUNS").is_none_or(|v| v != "0"))
}

/// `blocks` (in order, apart) as runs `[first, last)` of consecutive blocks.
fn block_runs(blocks: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &block in blocks {
        match runs.last_mut() {
            Some((_, last)) if *last == block => *last += 1,
            _ => runs.push((block, block + 1)),
        }
    }
    runs
}

/// [`protect_alias`] for the blocks `[first, last)`, none of them guarded:
/// one call for the run, a block at a time if it fails (the run spans
/// views).
fn protect_aliases(g: &mut Guards, first: usize, last: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if last - first > 1 {
        count(5);
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        if unsafe { VirtualProtect((first * BLOCK) as *const _, (last - first) * BLOCK, PAGE_READONLY, &mut old) } != 0 {
            g.protected.extend(first..last);
            g.views.extend(first..last);
            for block in first..last {
                g.dirty.remove(&block);
                g.partly.remove(&block);
            }
            return;
        }
    }
    for block in first..last {
        protect_alias(g, block);
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(&block) {
        return;
    }
    count(5);
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        g.protected.insert(block);
        g.views.insert(block);
        g.dirty.remove(&block);
        g.partly.remove(&block);
    }
}

/// [`protect`] for one range, without the other views of its pages.
///
/// Only view memory is made read-only. A range can run past the views into
/// other committed memory: with the structure builder on, a 4 MiB window
/// at 0x400000 (it begins in mapped memory, so [`is_view`] says yes) had
/// every committed private block in it made read-only block by block, the
/// process heap's among them. The heap's own code then faulted while
/// holding its lock, the faults waited for the guard lock, and the world
/// froze 3 to 18 s after the builder started (tippythetaidum/VIppy
/// ddd10de13, fxceless/victory-fork#1147). Committed private memory and
/// images are never guarded, region by region ([`view_run`]), as
/// Marvisak/VIctory's 280b150 decides for [`track`].
fn protect_here(g: &mut Guards, from: usize, to: usize, epoch: Option<Epoch>) {
    let mut at = from;
    while at < to {
        let (end, view) = kind_run(g, at, to);
        if view {
            protect_run(g, at, end, epoch);
        } else {
            for block in at..end {
                not_guardable(g, block, epoch);
            }
        }
        at = end;
    }
}

/// [`view_run`], without asking Windows for blocks the guards know as
/// view memory already (`Guards::views`: made read-only or noted absent
/// since the runtime last mapped or unmapped them). The blocks a check
/// arms again every frame are such blocks (written since their last
/// check), so the default path asks no more than before; only blocks not
/// known yet (a range guarded the first time, blocks just remapped) are
/// asked about, once per region.
fn kind_run(g: &Guards, block: usize, to: usize) -> (usize, bool) {
    let mut end = block;
    for &known in g.views.range(block..to) {
        if known != end {
            break;
        }
        end += 1;
    }
    if end > block {
        return (end, true);
    }
    view_run(block, to)
}

/// From `block` on (to `to` at most): the end of the run of blocks whose
/// memory is of one kind, and whether that kind may be guarded (anything
/// but committed private memory and images: views, and placeholders,
/// which [`protect_run`] notes absent when a protection fails).
fn view_run(block: usize, to: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    count(0);
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return (block + 1, false);
    }
    let end = (info.BaseAddress as usize).saturating_add(info.RegionSize).div_ceil(BLOCK).clamp(block + 1, to);
    let other = info.State == MEM_COMMIT && (info.Type == MEM_PRIVATE || info.Type == MEM_IMAGE);
    (end, !other)
}

/// A committed block of a range to guard that is not made read-only (part
/// of it is not committed, or it is not ours to protect): reported written
/// at every asking; with `VICTORY_ABSENT_ONCE` not tried again in this
/// submission.
fn not_guardable(g: &mut Guards, block: usize, epoch: Option<Epoch>) {
    if let Some(e) = epoch.filter(|_| absent_once()) {
        g.unguardable_in(e.submission).insert(block);
    }
}

/// [`protect_here`] for a run of blocks that may be guarded: whole, else
/// block by block, leaving out any that fail and noting those not
/// committed (unmapped) absent.
fn protect_run(g: &mut Guards, from: usize, to: usize, epoch: Option<Epoch>) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        count(if to - from > 1 { 3 } else { 4 });
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
        g.views.extend(from..to);
        for block in from..to {
            g.dirty.remove(&block);
            g.partly.remove(&block);
        }
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut block = from;
    while block < to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            block += 1;
            continue;
        }
        count(2);
        let until = uncommitted_end(block);
        if block < until {
            // With `VICTORY_ABSENT_ONCE`: the query answers for the whole
            // uncommitted run, which a block at a time asked again for each
            // (and tried to protect first).
            let run_end = if absent_once() { until.min(to) } else { block + 1 };
            g.absent.extend(block..run_end);
            g.views.extend(block..run_end);
            block = run_end;
        } else {
            not_guardable(g, block, epoch);
            block += 1;
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    // Placeholders only: free memory is not noted absent (see `is_view`).
    if n != 0 && info.State == MEM_RESERVE {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    let mark = mark_only();
    for block in aliased {
        if !mark {
            g.aliased.remove(&block);
        }
        for alias in crate::inplace::pages_of(block * BLOCK) {
            let other = alias / BLOCK != block;
            let done = if mark {
                // The other views stay read-only (a write through one of
                // them faults in its turn, and opens it): they only count
                // as written, which their next check takes without a call.
                other && mark_block(g, alias / BLOCK, range_mask(block, start, end))
            } else {
                other && open_here(g, alias, alias.saturating_add(BLOCK))
            };
            if done {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if !mark {
                g.aliased.remove(&(alias / BLOCK));
            }
        }
    }
    any
}

/// Whether a write the guards learn of other than by its own fault (one
/// through another view of the pages, one into the pages through the
/// runtime's view, a view of them newly mapped) leaves the guarded blocks
/// read-only and only counts them written (`VICTORY_MARK_ONLY=0`: they are
/// opened, as before).
///
/// Opened, a block is made read-only again at its next check: two calls
/// that change protection (each flushing the address on every processor
/// running the process), for a block whose bytes need only be copied again.
/// In the intro a write opened its block's three or four other views too,
/// and the re-arming was most of the submitting thread's 12 ms a frame of
/// `VirtualProtect` (round 2: 172 other views armed a submission against 37
/// blocks). Counted written while read-only, a block's next check reports
/// it written and notes it clean (as one written through the runtime always
/// was), and a later write through it faults as it would have.
fn mark_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MARK_ONLY").is_none_or(|v| v != "0"))
}

/// Counts the guarded blocks of `[start, end)` written, leaving them
/// read-only; true if any was guarded.
fn mark_written(g: &mut Guards, start: usize, end: usize) -> bool {
    let blocks: Vec<usize> = g.protected.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for &block in &blocks {
        mark_block(g, block, range_mask(block, start, end));
    }
    !blocks.is_empty()
}

/// The kernel pages of `block` that `[start, end)` touches, a bit each
/// (all of them where [`guard_pages`] is off).
fn range_mask(block: usize, start: usize, end: usize) -> u16 {
    if !guard_pages() {
        return FULL;
    }
    let (lo, hi) = (start.max(block * BLOCK), end.min((block + 1) * BLOCK));
    if lo >= hi {
        return 0;
    }
    let (a, b) = ((lo % BLOCK) / KERNEL_PAGE, ((hi - 1) % BLOCK) / KERNEL_PAGE);
    (FULL >> (15 - b)) & (FULL << a)
}

/// Counts guarded `block` written in the kernel pages `mask`, leaving it
/// read-only; true if it is guarded. A block already counted written
/// anywhere stays so.
fn mark_block(g: &mut Guards, block: usize, mask: u16) -> bool {
    if !g.protected.contains(&block) {
        return false;
    }
    note_dirty(g, block, mask);
    true
}

/// Notes that guarded `block` may have been written in the kernel pages
/// `mask` since its last report.
fn note_dirty(g: &mut Guards, block: usize, mask: u16) {
    if g.clean.contains(&block) {
        g.clean.remove(&block);
        if mask != FULL {
            g.dirty.insert(block, mask);
        } else {
            g.dirty.remove(&block);
        }
    } else if let Some(m) = g.dirty.get_mut(&block) {
        // Not clean with an entry: written there before; without one:
        // written anywhere already.
        *m |= mask;
    }
}

/// [`mark_written`] for `[start, end)` and every other view of its
/// guarded pages, as [`open`] goes to them.
fn mark_with_views(g: &mut Guards, start: usize, end: usize) {
    mark_written(g, start, end);
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && mark_block(g, alias / BLOCK, range_mask(block, start, end)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// Whether a write that faults on a guarded block opens only the block's
/// kernel pages from the faulting one to its end, and notes those as the
/// ones written (`VICTORY_GUARD_4K=0`: the whole block opens and counts
/// written, as before).
///
/// Guarded view memory has no write watch, so a written block was copied
/// whole: 64 KiB into every copy over it, for a write that may have been a
/// few bytes (in the intro most of the 110 MB a frame the refreshes copy,
/// and what the journal told the descriptor pass of heap pages). A page
/// below the faulting one stays read-only, so a write there faults in its
/// turn and adds its pages; one above it is noted written whether or not
/// it is. The title writes its rings upwards: one fault a block, as before.
/// The block is armed again whole at its next check (one call, as before).
fn guard_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_4K").is_none_or(|v| v != "0"))
}

/// For a write that faulted at `at` in a guarded block ([`guard_pages`]):
/// makes its kernel pages from `at`'s on writable, notes them written (and
/// the same pages at the block's other views, which stay read-only); true
/// if the block is guarded and that was done.
fn open_tail(g: &mut Guards, at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let block = at / BLOCK;
    if !g.protected.contains(&block) {
        return false;
    }
    let k = (at % BLOCK) / KERNEL_PAGE;
    let from = block * BLOCK + k * KERNEL_PAGE;
    let mut old = 0;
    count(14);
    // SAFETY: a block we made read-only; only its protection changes.
    if unsafe { VirtualProtect(from as *const _, (block + 1) * BLOCK - from, PAGE_READWRITE, &mut old) } == 0 {
        return false;
    }
    let mask = FULL << k;
    note_dirty(g, block, mask);
    g.partly.insert(block);
    if g.aliased.contains(&block) {
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && mark_block(g, alias / BLOCK, mask) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    true
}

/// Arms `block` again whole if a fault opened part of it
/// ([`open_tail`]); false if that failed, and the block is then no longer
/// guarded.
fn rearm_tail(g: &mut Guards, block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if !g.partly.remove(&block) {
        return true;
    }
    count(15);
    let mut old = 0;
    // SAFETY: title view memory we guard; only its protection changes.
    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
        return true;
    }
    g.protected.remove(&block);
    false
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        g.dirty.remove(&block);
        g.partly.remove(&block);
        let mut old = 0;
        count(6);
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// After the layer wrote `[at, at + len)` of title memory through the
/// kernel: the dirty bits the kernel holds for those pages are the layer's
/// own writes, and a refresh that took them for the title's copies the same
/// bytes up again every frame. Asks the watch to clear them and drops the
/// answer (nothing is written into title memory between the write and this
/// asking by design: both are the layer's).
pub(crate) fn forgot(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const RESET: u32 = 1;
    if len == 0 {
        return;
    }
    let mut buf = [0usize; 512];
    let first = at / KERNEL_PAGE * KERNEL_PAGE;
    let end = at.saturating_add(len);
    let mut page = first;
    while page < end {
        let pages = (end - page).div_ceil(KERNEL_PAGE).min(buf.len());
        let span = pages * KERNEL_PAGE;
        let mut count = pages;
        let mut granularity = 0u32;
        // SAFETY: `count` entries in the buffer, a whole number of kernel
        // pages of our own process's memory.
        let _ = unsafe {
            GetWriteWatch(
                RESET,
                page as *const _,
                span,
                buf.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        page += span;
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            // Written through another view: these addresses need not be
            // writable, only copied again (`mark_only`).
            if mark_only() {
                mark_with_views(&mut g, at, end);
            } else {
                open(&mut g, at, end);
            }
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        let end = at.saturating_add(n);
        if if mark_only() { mark_written(&mut g, at, end) } else { open_here(&mut g, at, end) } {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    Remapping(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // Whether its blocks are there is asked anew (`texture::block_there`).
    crate::texture::blocks_remapped(at, len);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
    let aliased: Vec<usize> = g.aliased.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    let views: Vec<usize> = g.views.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
    for block in views {
        g.views.remove(&block);
    }
    // What was found unmapped or not to be protected there is no longer
    // known, nor how often its blocks were written. Runs elsewhere stay:
    // the runtime maps and unmaps views all the time as the title streams,
    // and clearing them all made the next refresh ask about every run
    // again (round 2: 422 questions a submission).
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let runs: Vec<usize> = g.absent_runs.range(..last).filter(|&(_, &until)| until > first).map(|(&b, _)| b).collect();
    for block in runs {
        count(13);
        // What lies outside the range is as it was: the parts of the run
        // there stay.
        if let Some(until) = g.absent_runs.remove(&block) {
            if block < first {
                g.absent_runs.insert(block, first);
            }
            if until > last {
                g.absent_runs.insert(last, until);
            }
        }
    }
    g.unguardable.clear();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let hot: Vec<usize> = g.hot.range(range.clone()).map(|(&b, _)| b).collect();
    for block in hot {
        g.hot.remove(&block);
    }
    let rearmed: Vec<usize> = g.rearmed.range(range.clone()).map(|(&b, _)| b).collect();
    for block in rearmed {
        g.rearmed.remove(&block);
    }
    let dirty: Vec<usize> = g.dirty.range(range.clone()).map(|(&b, _)| b).collect();
    for block in dirty {
        g.dirty.remove(&block);
    }
    let partly: Vec<usize> = g.partly.range(range).copied().collect();
    for block in partly {
        g.partly.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks and faults in
/// [`copy_title`]'s routine, once, ahead of the crash reporter's. One
/// handler for both, so that a write of the routine into a guarded block
/// opens the block (and the copy goes on) before the routine is told it
/// failed.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            // SAFETY: as above.
            return unsafe { copy_fault(info) };
        }
        let at = r.ExceptionInformation[1];
        {
            // Held across both checks: another thread may have opened the
            // block since this one faulted, and a copy may have guarded it
            // again since (then it is ours again, not read-only for some
            // other reason).
            let mut g = guards();
            if guard_pages() && open_tail(&mut g, at) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if open(&mut g, at, at + 1) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if writable(at) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            // A block of view memory we guard that is read-only though the
            // guards do not hold it: one a check made read-only after
            // something opened it ([`unlocked_guards`]: the call is made
            // without the lock). Opened and counted written, as a fault on a
            // guarded block is.
            if unlocked_guards() && reopen_stray(&mut g, at) {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        // SAFETY: as above.
        unsafe { copy_fault(info) }
    }
    /// A fault in [`copy_title`]'s routine: it returns how far it got.
    /// Only faults on reading or writing data (not an execute fault), of
    /// the kinds a page that is not there or not readable raises.
    ///
    /// # Safety
    /// `info` is the exception being dispatched.
    unsafe fn copy_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        use windows_sys::Win32::Foundation::{STATUS_GUARD_PAGE_VIOLATION, STATUS_IN_PAGE_ERROR};
        // SAFETY: the system passes the exception being dispatched and the
        // faulting thread's context.
        let (r, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        const EXECUTE: usize = 8;
        let kind = r.ExceptionCode;
        if !in_copy_routine(context.Rip as usize)
            || !(kind == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                || kind == STATUS_GUARD_PAGE_VIOLATION
                || kind == STATUS_IN_PAGE_ERROR)
            || (r.NumberParameters >= 1 && r.ExceptionInformation[0] == EXECUTE)
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        context.Rip = copy_resume(context.Rip as usize) as u64;
        EXCEPTION_CONTINUE_EXECUTION
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections,
    // and the context of a fault in our own copy routine.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

/// Whether title memory is read (and, for the title's own buffer copies,
/// written) by [`copy_title`], a copy of the processor's own that survives
/// a page that is not there, instead of through the kernel
/// (`ReadProcessMemory`) (`VICTORY_USER_COPY=0`: through the kernel, as
/// before).
///
/// The kernel's copy reports an unmapped page instead of faulting, which is
/// why it was used: the title unmaps memory under us as it streams. But each
/// call is a system call that locks and maps the source's pages: 80,000
/// calls a frame of 12 KB on average, 39 ms a frame of the submitting
/// thread's (and the title buffer copies, 14,000 calls of 0.5 KB, 10 ms of
/// it). `VICTORY_COPY_FAST`'s plain copy after asking the kernel whether the
/// pages are there could not survive a page unmapped between the asking and
/// the copy; this copy does: its routine's faults are taken by our handler
/// ([`install`]), which makes it return how far it got, and the rest is read
/// through the kernel as before (zeros where nothing is there).
///
/// What it reads is what the kernel read: reading neither trips a guard of
/// ours (they are read-only) nor sets a write-watch bit. A page the title
/// guards itself (`PAGE_GUARD`) loses its guard on the first read, as it
/// did when the kernel read it.
pub(crate) fn user_copies() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_USER_COPY").is_none_or(|v| v != "0"))
}

// The copy routine. `rcx` destination, `rdx` source, `r8` bytes; returns in
// `rax` the bytes copied. A leaf function (no stack, only volatile
// registers), so a fault in it needs no unwinding: the handler moves the
// instruction pointer to its `ret` and `rax` says how far it got. Every
// load of a step comes before its stores, so a fault on reading leaves
// exactly the bytes before `rax` copied and none after; a fault on writing
// (a title destination) may leave up to 64 bytes after `rax` written.
//
// From [`MOVS_MIN`] bytes on (`victory_gpu_copy_title`, with
// [`movs_copies`]) the copy is one `rep movsb`, which this processor runs
// at its full width (ERMS/FSRM) where the loop moved 16 bytes an
// instruction (4.6 GB/s measured). `rsi` and `rdi` are kept in `r10` and
// `r11` (no stack: still a leaf). A string instruction is interruptible:
// at a fault `rcx` holds the bytes not yet copied and every byte before
// the faulting one is copied, so the handler resumes at
// `victory_gpu_copy_title_movs_resume`, which returns `r8 - rcx` and puts
// `rsi` and `rdi` back. (Bytes after it in the destination may have been
// written as well, as with the loop.) Below that size, or without the
// switch, the loop (`victory_gpu_copy_title_loop`).
#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(
    ".text",
    ".p2align 4",
    ".globl victory_gpu_copy_title",
    "victory_gpu_copy_title:",
    "    cmp r8, 256",
    "    jb victory_gpu_copy_title_loop",
    "    mov r10, rsi",
    "    mov r11, rdi",
    "    mov rdi, rcx",
    "    mov rsi, rdx",
    "    mov rcx, r8",
    ".globl victory_gpu_copy_title_movs",
    "victory_gpu_copy_title_movs:",
    "    rep movsb",
    ".globl victory_gpu_copy_title_movs_resume",
    "victory_gpu_copy_title_movs_resume:",
    "    mov rax, r8",
    "    sub rax, rcx",
    "    mov rsi, r10",
    "    mov rdi, r11",
    "    ret",
    ".p2align 4",
    ".globl victory_gpu_copy_title_loop",
    "victory_gpu_copy_title_loop:",
    "    xor eax, eax",
    "    cmp r8, 64",
    "    jb 3f",
    "    lea r9, [r8 - 64]",
    "2:",
    "    movdqu xmm0, xmmword ptr [rdx + rax]",
    "    movdqu xmm1, xmmword ptr [rdx + rax + 16]",
    "    movdqu xmm2, xmmword ptr [rdx + rax + 32]",
    "    movdqu xmm3, xmmword ptr [rdx + rax + 48]",
    "    movdqu xmmword ptr [rcx + rax], xmm0",
    "    movdqu xmmword ptr [rcx + rax + 16], xmm1",
    "    movdqu xmmword ptr [rcx + rax + 32], xmm2",
    "    movdqu xmmword ptr [rcx + rax + 48], xmm3",
    "    add rax, 64",
    "    cmp rax, r9",
    "    jbe 2b",
    "3:",
    "    lea r9, [rax + 8]",
    "    cmp r9, r8",
    "    ja 4f",
    "    mov r10, qword ptr [rdx + rax]",
    "    mov qword ptr [rcx + rax], r10",
    "    mov rax, r9",
    "    jmp 3b",
    "4:",
    "    cmp rax, r8",
    "    jae victory_gpu_copy_title_resume",
    "    movzx r10d, byte ptr [rdx + rax]",
    "    mov byte ptr [rcx + rax], r10b",
    "    inc rax",
    "    jmp 4b",
    ".globl victory_gpu_copy_title_resume",
    "victory_gpu_copy_title_resume:",
    "    ret",
    ".globl victory_gpu_copy_title_end",
    "victory_gpu_copy_title_end:",
    "    int3",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_gpu_copy_title(dst: *mut u8, src: *const u8, len: usize) -> usize;
    fn victory_gpu_copy_title_loop(dst: *mut u8, src: *const u8, len: usize) -> usize;
    fn victory_gpu_copy_title_resume();
    fn victory_gpu_copy_title_movs();
    fn victory_gpu_copy_title_movs_resume();
    fn victory_gpu_copy_title_end();
}

/// The length from which the routine copies with `rep movsb` (as in its
/// code).
#[cfg_attr(not(test), allow(dead_code))]
const MOVS_MIN: usize = 256;

/// Whether long copies are one `rep movsb` (`VICTORY_COPY_MOVS=0`: the
/// 64-byte loop for every length, as before).
fn movs_copies() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_MOVS").is_none_or(|v| v != "0"))
}

/// Whether `at` is an instruction of the copy routine.
fn in_copy_routine(at: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        let start = victory_gpu_copy_title as *const () as usize;
        let end = victory_gpu_copy_title_end as *const () as usize;
        (start..end).contains(&at)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = at;
        false
    }
}

/// Where a fault in the copy routine at `at` resumes: after the string
/// copy if it was that, else its return.
fn copy_resume(at: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if at == victory_gpu_copy_title_movs as *const () as usize {
            victory_gpu_copy_title_movs_resume as *const () as usize
        } else {
            victory_gpu_copy_title_resume as *const () as usize
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = at;
        0
    }
}

/// Copies `len` bytes of memory at `src` to `dst`, either of which may be
/// title memory that is not there (or not readable, or for `dst` not
/// writable): returns the bytes copied, `len` unless a page of either
/// faulted. Bytes before the answer are copied; of those after it, none of
/// the source's were read, and up to 64 of the destination may have been
/// written. Only memory that may fault is to be given: a fault elsewhere
/// (our own memory) would be taken the same way.
///
/// # Safety
/// `dst..dst + len` must not overlap `src..src + len`, and must be memory
/// nothing else of ours reads or writes meanwhile (a host copy, or title
/// memory).
pub(crate) unsafe fn copy_title(dst: *mut u8, src: *const u8, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    install();
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the caller's; faults in the routine return to its `ret`
        // with the bytes copied so far ([`install`]).
        unsafe {
            if movs_copies() {
                victory_gpu_copy_title(dst, src, len)
            } else {
                victory_gpu_copy_title_loop(dst, src, len)
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, src);
        0
    }
}

#[cfg(test)]
mod copy_tests {
    use super::{KERNEL_PAGE, MOVS_MIN, install, victory_gpu_copy_title, victory_gpu_copy_title_loop};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    /// The routine's two entries: with the string copy for long copies, and
    /// the loop for every length.
    type Copy = unsafe extern "C" fn(*mut u8, *const u8, usize) -> usize;
    const ENTRIES: [(&str, Copy); 2] = [("movs", victory_gpu_copy_title), ("loop", victory_gpu_copy_title_loop)];

    /// Whether `entry` copies `len` bytes with the string copy (whose fault
    /// answer is exact to the byte).
    fn exact(entry: &str, len: usize) -> bool {
        entry == "movs" && len >= MOVS_MIN
    }

    /// Kernel pages committed, filled with a pattern.
    struct Pages(*mut u8, usize);
    impl Pages {
        fn new(n: usize) -> Pages {
            // SAFETY: a fresh allocation of ours, freed on drop.
            let p = unsafe { VirtualAlloc(std::ptr::null(), n * KERNEL_PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }
                .cast::<u8>();
            assert!(!p.is_null());
            for i in 0..n * KERNEL_PAGE {
                // SAFETY: inside the allocation.
                unsafe { p.add(i).write(pattern(i)) };
            }
            Pages(p, n)
        }
        fn at(&self, offset: usize) -> *mut u8 {
            self.0.wrapping_add(offset)
        }
        /// Makes kernel page `k` unreadable, one way or the other.
        fn take_away(&self, k: usize, decommit: bool) {
            assert!(k < self.1);
            // SAFETY: a kernel page of our allocation.
            unsafe {
                if decommit {
                    assert_ne!(VirtualFree(self.at(k * KERNEL_PAGE).cast(), KERNEL_PAGE, MEM_DECOMMIT), 0);
                } else {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(self.at(k * KERNEL_PAGE).cast(), KERNEL_PAGE, PAGE_NOACCESS, &mut old), 0);
                }
            }
        }
    }
    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    fn pattern(i: usize) -> u8 {
        (i * 7 + 3 + i / 251) as u8
    }

    /// Every length class of both entries (empty, the byte tail, the
    /// 8-byte steps, the 64-byte steps, either side of the string copy's
    /// threshold, several pages), at every alignment of source and
    /// destination within 64 bytes.
    #[test]
    fn copies_every_length_exactly() {
        install();
        let pages = Pages::new(8);
        let lens: Vec<usize> = (0..300)
            .chain([MOVS_MIN - 1, MOVS_MIN, MOVS_MIN + 1, 511, 512, 4095, 4096, 4097, 3 * KERNEL_PAGE + 13, 7 * KERNEL_PAGE])
            .collect();
        for (entry, copy) in ENTRIES {
            for &len in &lens {
                for skew in [0usize, 1, 7, 8, 15, 31, 33, 63] {
                    for dst_skew in [1usize, 0, 9, 32] {
                        let mut out = vec![0xeeu8; len + 2 + dst_skew];
                        // SAFETY: readable source, our own destination, apart.
                        let done = unsafe { copy(out.as_mut_ptr().add(1 + dst_skew), pages.at(skew), len) };
                        assert_eq!(done, len, "{entry} {len} {skew} {dst_skew}");
                        let got = &out[1 + dst_skew..1 + dst_skew + len];
                        assert!(got.iter().enumerate().all(|(i, &b)| b == pattern(skew + i)), "{entry} {len} {skew}");
                        assert_eq!((out[dst_skew], out[len + 1 + dst_skew]), (0xee, 0xee), "nothing written around it");
                    }
                }
            }
        }
    }

    /// A page in the middle that is not there (decommitted) or not readable
    /// (`PAGE_NOACCESS`): the copy says how far it got, without a crash, and
    /// the bytes before that are right; with the string copy to the byte.
    /// The fault falls at every position inside a 64-byte step, and the copy
    /// starts at every offset before the page down to two pages before it.
    #[test]
    fn a_missing_page_in_the_middle_ends_the_copy_there() {
        install();
        for decommit in [true, false] {
            let pages = Pages::new(6);
            pages.take_away(3, decommit);
            let boundary_at = 3 * KERNEL_PAGE;
            let skews: Vec<usize> = (0..130).chain([KERNEL_PAGE, 2 * KERNEL_PAGE - 1, 2 * KERNEL_PAGE + 64, 3 * KERNEL_PAGE - 255, 3 * KERNEL_PAGE - 256, 3 * KERNEL_PAGE - 257]).map(|d| boundary_at - 1 - d).collect();
            for (entry, copy) in ENTRIES {
                for &start in &skews {
                    for len in [boundary_at - start + 1, boundary_at - start + 100, 6 * KERNEL_PAGE - start] {
                        let mut out = vec![0xeeu8; len];
                        // SAFETY: as above; the source faults at its page 3.
                        let done = unsafe { copy(out.as_mut_ptr(), pages.at(start), len) };
                        let boundary = boundary_at - start;
                        if exact(entry, len) {
                            assert_eq!(done, boundary, "{entry} {decommit} {start} {len}");
                        } else {
                            assert!(done <= boundary && done + 64 > boundary, "{entry} {decommit} {start} {len}: {done}");
                        }
                        assert!((0..done).all(|i| out[i] == pattern(start + i)), "{entry} {decommit} {start} {len}");
                        assert!(out[boundary..].iter().all(|&b| b == 0xee), "{entry} {start} {len}: nothing read past the fault");
                    }
                }
                // From inside the missing page: nothing.
                for len in [16, MOVS_MIN + 5] {
                    let mut out = vec![0u8; len];
                    // SAFETY: as above.
                    assert_eq!(unsafe { copy(out.as_mut_ptr(), pages.at(boundary_at + 5), len) }, 0, "{entry} {decommit}");
                }
            }
        }
    }

    /// A destination that is not writable ends the copy there too: with
    /// the string copy at exactly its first byte.
    #[test]
    fn a_read_only_destination_ends_the_copy() {
        install();
        let from = Pages::new(4);
        for (entry, copy) in ENTRIES {
            for skew in [0usize, 3, 40] {
                let to = Pages::new(4);
                // SAFETY: the third kernel page of our allocation.
                unsafe {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(to.at(2 * KERNEL_PAGE).cast(), KERNEL_PAGE, PAGE_READONLY, &mut old), 0);
                }
                // SAFETY: two allocations of ours; the destination faults at
                // its third page.
                let done = unsafe { copy(to.at(skew), from.at(8), 3 * KERNEL_PAGE - skew) };
                let boundary = 2 * KERNEL_PAGE - skew;
                if exact(entry, 3 * KERNEL_PAGE) {
                    assert_eq!(done, boundary, "{entry} {skew}");
                } else {
                    assert!(done <= boundary && done + 64 > boundary, "{entry} {skew}: {done}");
                }
                // SAFETY: the bytes before the fault, in our allocation.
                let got = unsafe { std::slice::from_raw_parts(to.at(skew), done) };
                assert!(got.iter().enumerate().all(|(i, &b)| b == pattern(8 + i)), "{entry} {skew}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// The journal never forgets a page: a test's fresh allocation may lie
    /// where another test's was, whose writes it still holds.
    fn forget(start: usize, end: usize) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let pages: Vec<usize> = j.written.range(start / PAGE..end.div_ceil(PAGE)).map(|(&p, _)| p).collect();
        for page in pages {
            j.written.remove(&page);
        }
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        forget(start, end);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The journal says which 4 KiB pages of a page were written, and
    /// since which asking that holds.
    #[test]
    fn the_journal_tells_the_kernel_pages_written() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 2 * PAGE);
        forget(start, end);
        let page = p / PAGE + 1;
        writes_fine(start, end, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 3 * KERNEL_PAGE + 8) as *mut u8).write_volatile(1) };
        let (first, found) = writes_fine(start, end, false).expect("watched");
        assert_eq!(found, [(page, Written { at: first, before: 0, mask: 1 << 3 })]);
        // SAFETY: as above.
        unsafe {
            ((p + PAGE + 5 * KERNEL_PAGE) as *mut u8).write_volatile(1);
            ((p + PAGE + 6 * KERNEL_PAGE + 1) as *mut u8).write_volatile(1);
        }
        let (second, found) = writes_fine(start, end, false).expect("watched");
        assert_eq!(found, [(page, Written { at: second, before: first, mask: 1 << 5 | 1 << 6 })]);
        // Nothing written since: the entry stays.
        let (third, again) = writes_fine(start, end, false).expect("watched");
        assert!(third > second);
        assert_eq!(again, found);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Asked again at the same number, the journal joins the pages; an
    /// answer numbered before the last noted keeps the later number and no
    /// longer says where.
    #[test]
    fn notes_join_and_never_go_back() {
        let mut written = BTreeMap::new();
        note(&mut written, 9, 4, 1);
        note(&mut written, 9, 4, 2);
        assert_eq!(written[&9], Written { at: 4, before: 0, mask: 3 });
        note(&mut written, 9, 6, 8);
        assert_eq!(written[&9], Written { at: 6, before: 4, mask: 8 });
        note(&mut written, 9, 5, 1);
        assert_eq!(written[&9], Written { at: 6, before: 4, mask: FULL });
    }

    /// A remapping forgets the unmapped runs it touches and no others.
    #[test]
    fn a_remapping_forgets_only_the_absent_runs_it_touches() {
        // Far above anything mapped in a test: only the notes change.
        let b = (1usize << 46) / BLOCK;
        {
            let mut g = guards();
            g.absent_for = 7;
            g.absent_runs.insert(b, b + 4);
            g.absent_runs.insert(b + 10, b + 11);
            g.absent_runs.insert(b + 20, b + 30);
        }
        title_memory_remapped((b + 2) * BLOCK, BLOCK);
        title_memory_remapped((b + 29) * BLOCK, 2 * BLOCK);
        let mut g = guards();
        assert_eq!(g.absent_until(b + 10, 7), Some(b + 11));
        // The runs are cut around the remapped blocks, not dropped.
        assert_eq!(g.absent_until(b + 1, 7), Some(b + 2));
        assert_eq!(g.absent_until(b + 2, 7), None);
        assert_eq!(g.absent_until(b + 3, 7), Some(b + 4));
        assert_eq!(g.absent_until(b + 25, 7), Some(b + 29));
        assert_eq!(g.absent_until(b + 29, 7), None);
        for k in [b, b + 3, b + 10, b + 20] {
            g.absent_runs.remove(&k);
        }
    }

    /// A write to a guarded block opens it from the written kernel page on
    /// and is reported as those pages; a write below faults in its turn.
    #[test]
    fn a_fault_opens_and_reports_from_its_kernel_page_on() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        if !guard_pages() {
            return;
        }
        assert_eq!(range_mask(0, 5 * KERNEL_PAGE + 3, 6 * KERNEL_PAGE + 1), 0b110_0000);
        assert_eq!(range_mask(1, 0, 4 * BLOCK), FULL);
        // SAFETY: a pagefile section of one block, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let block = view / BLOCK;
        assert_eq!(guard_fine(view, view + BLOCK, None), Some(vec![(block, FULL)]));
        // SAFETY: inside our view; the write faults and opens the tail.
        unsafe { ((view + 5 * KERNEL_PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(guards().protected.contains(&block), "the pages below stay guarded");
        // SAFETY: as above, above the fault: open, no fault.
        unsafe { ((view + 9 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        // SAFETY: as above, below it: faults, opens from there on.
        unsafe { ((view + 2 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(view, view + BLOCK, None), Some(vec![(block, FULL << 2)]));
        assert_eq!(guard_fine(view, view + BLOCK, None), Some(Vec::new()), "armed again whole");
        // SAFETY: as above; faults again: armed whole.
        unsafe { ((view + 15 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(view, view + BLOCK, None), Some(vec![(block, 1 << 15)]));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, BLOCK);
    }

    /// A view block made read-only that the guards do not hold (a check's
    /// late call after something opened it) is opened by a write's fault,
    /// not left to crash the writer, and reported written.
    #[test]
    fn a_block_armed_after_it_was_opened_is_opened_by_its_fault() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        if !unlocked_guards() {
            return;
        }
        // SAFETY: a pagefile section of one block, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let block = view / BLOCK;
        assert!(guard_fine(view, view + BLOCK, None).is_some());
        {
            // As if opened by another thread and armed again by a check
            // that made its call afterwards.
            let mut g = guards();
            g.protected.remove(&block);
            g.clean.remove(&block);
        }
        // SAFETY: inside our view, read-only now: the write faults.
        unsafe { ((view + 100) as *mut u8).write_volatile(9) };
        // SAFETY: as above.
        assert_eq!(unsafe { ((view + 100) as *const u8).read_volatile() }, 9);
        assert_eq!(guard_fine(view, view + BLOCK, None), Some(vec![(block, FULL)]));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, BLOCK);
    }

    #[test]
    fn block_sets_answer_as_a_set_of_blocks_does() {
        let mut set = BlockSet::new();
        let mut naive = BTreeSet::new();
        for b in [0usize, 1, 2, 63, 64, 65, 127, 128, 300, 301, 302, 303] {
            set.insert(b);
            naive.insert(b);
        }
        set.extend(500..700);
        naive.extend(500..700);
        set.remove(&65);
        naive.remove(&65);
        set.remove(&999);
        for first in (0..720).step_by(7) {
            for last in [first, first + 1, first + 3, first + 64, first + 130] {
                let want: Vec<usize> = naive.range(first..last).copied().collect();
                assert_eq!(set.range(first, last), want, "{first}..{last}");
                assert_eq!(set.all(first, last), want.len() == last - first, "{first}..{last}");
            }
        }
        assert!(set.contains(&64) && !set.contains(&65));
    }

    /// A block re-armed in three submissions is left open with the other
    /// views of its pages: no fault, reported written once a submission while
    /// its asking stands, at every check otherwise; its time up, it is
    /// guarded again.
    #[test]
    fn a_block_written_again_and_again_is_left_open() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        if !hot_blocks() {
            return;
        }
        // SAFETY: a pagefile section of one block, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let block = view / BLOCK;
        let page = vec![view / PAGE];
        // Far from the numbers other tests' checks use.
        let base = 1 << 40;
        let at = |submission: u64, standing: bool| Some(Epoch { submission: base + submission, standing });
        // SAFETY: inside our view; a write faults while it is guarded.
        let write = || unsafe { ((view + 8) as *mut u8).write_volatile(7) };
        assert_eq!(guard(view, view + BLOCK, at(0, true)), Some(page.clone()));
        // The first arming counts as one: armed in `HOT_TIMES - 1`
        // submissions, then left open.
        let hot = HOT_TIMES as u64 - 1;
        for submission in 1..hot {
            assert!(guards().protected.contains(&block), "armed after submission {submission}");
            write();
            assert_eq!(guard(view, view + BLOCK, at(submission, true)), Some(page.clone()));
        }
        assert!(guards().protected.contains(&block));
        write();
        assert_eq!(guard(view, view + BLOCK, at(hot, true)), Some(page.clone()));
        assert!(!guards().protected.contains(&block) && guards().hot.contains_key(&block));
        write();
        assert_eq!(guard(view, view + BLOCK, at(hot, true)), Some(Vec::new()), "once while the asking stands");
        assert_eq!(guard(view, view + BLOCK, at(hot + 1, true)), Some(page.clone()), "again in the next submission");
        assert_eq!(guard(view, view + BLOCK, at(hot + 1, false)), Some(page.clone()), "at every check without one");
        assert_eq!(guard(view, view + BLOCK, at(hot + 1, false)), Some(page.clone()));
        // Its time up, it is guarded again.
        assert_eq!(guard(view, view + BLOCK, at(hot + 1 + HOT_FOR, true)), Some(page.clone()));
        assert!(guards().protected.contains(&block) && !guards().hot.contains_key(&block));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, BLOCK);
    }

    /// A guarded block a write opened stays known as view memory (no
    /// question to Windows, no write-watch asking) until it is remapped.
    #[test]
    fn an_opened_view_block_stays_known_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of one block, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(!known_view(view));
        assert!(guard(view, view + BLOCK, None).is_some());
        assert!(known_view(view));
        // SAFETY: our view; the write faults and opens the block.
        unsafe { ((view + 8) as *mut u8).write_volatile(1) };
        // Written: open (whole, or from the written kernel page on).
        assert!(!guards().clean.contains(&(view / BLOCK)));
        assert_eq!(known_view(view), views_known());
        assert_eq!(guard(view, view + BLOCK, None), Some(vec![view / PAGE]));
        title_memory_remapped(view, BLOCK);
        assert!(!known_view(view));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, BLOCK);
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK, None).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK, None),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE, None).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Free memory is no view: taken for one, it would be guarded once
    /// something is allocated there (the process heap: its writes then
    /// fault into the handler while a thread holds the guard lock).
    #[test]
    fn free_memory_is_no_view() {
        use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: a reservation of ours, released at once: free memory.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
        let view = is_view(r, r + 2 * BLOCK);
        // Other tests allocate meanwhile: only memory still free counts.
        // SAFETY: plain data filled in by the query, of the right size.
        let still_free = unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(r as *const _, &mut info, size_of_val(&info)) != 0
                && info.State == MEM_FREE
                && info.BaseAddress as usize + info.RegionSize >= r + 2 * BLOCK
        };
        assert!(!(view && still_free), "free memory taken for a view");
    }

    /// A range that begins in a view and runs into committed private
    /// memory guards the view only: the private block (the process heap's,
    /// say) stays writable.
    #[test]
    fn committed_private_memory_beside_a_view_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MapViewOfFile3, PAGE_NOACCESS,
            PAGE_READONLY, UnmapViewOfFile2, VirtualAlloc2, VirtualQuery,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of two blocks, split; a pagefile
        // section's block mapped over the first, private memory committed
        // over the second.
        let (p, section, view) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), 0);
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, p);
            let private = VirtualAlloc2(
                GetCurrentProcess(),
                (p + BLOCK) as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_eq!(private, p + BLOCK);
            (p, section, view)
        };
        title_memory_remapped(p, BLOCK);
        assert!(is_view(p, p + 2 * BLOCK), "it begins in a view");
        assert!(guard(p, p + 2 * BLOCK, None).is_some());
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query, of the right size.
            unsafe {
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(at as *const _, &mut info, size_of_val(&info)), 0);
                info.Protect
            }
        };
        assert_eq!(protection(p), PAGE_READONLY, "the view is guarded");
        assert_eq!(protection(p + BLOCK), PAGE_READWRITE, "the private block is not");
        // SAFETY: committed private memory of ours, writable.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our view, section and private block.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, 0);
            VirtualFree((p + BLOCK) as *mut _, 0, MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK, None),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK, None), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE, None), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE, None), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK, None), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK, None), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
