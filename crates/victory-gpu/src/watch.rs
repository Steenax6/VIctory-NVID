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
//! system call writing into a guarded block fails; the runtime opens the
//! blocks before the title's file reads ([`writing`]) and says when a read
//! is done ([`kernel_wrote`]), for the blocks guarded again meanwhile.

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

/// A page nothing was noted of: askings are numbered from 1.
const UNNOTED: Written = Written { at: 0, before: 0, mask: 0 };
/// Pages in a chunk of [`Pages`] (64 MiB of address space), and the pages
/// it holds in chunks: the title's addresses (40 bits).
const CHUNK_PAGES: usize = 1 << 10;
const NEAR_PAGES: usize = (1 << 40) / PAGE;

/// What the journal knows of each 64 KiB page ([`Written`]), by page
/// number: a table in chunks, made where a page is first noted, for the
/// title's addresses, and a map for whatever lies above them.
///
/// It was a map for all of it. A refresh asks for the entries of its whole
/// range (2,784 pages for the title's 174 MB buffer, fifteen times a
/// frame), and walking the map's nodes for them was 4 ms a frame of the
/// submitting thread (profile of 2026-11-26: `entries` 3.97 ms); a chunk is
/// walked as an array.
struct Pages {
    chunks: Vec<Option<Box<[Written]>>>,
    far: BTreeMap<usize, Written>,
}

impl Pages {
    const fn new() -> Pages {
        Pages { chunks: Vec::new(), far: BTreeMap::new() }
    }

    /// What is noted of `page`, if anything is.
    fn get_mut(&mut self, page: usize) -> Option<&mut Written> {
        if page >= NEAR_PAGES {
            return self.far.get_mut(&page);
        }
        self.chunks
            .get_mut(page / CHUNK_PAGES)?
            .as_mut()?
            .get_mut(page % CHUNK_PAGES)
            .filter(|w| w.at != 0)
    }

    /// Notes `w` (an asking's number is never 0) of `page`.
    fn insert(&mut self, page: usize, w: Written) {
        if page >= NEAR_PAGES {
            self.far.insert(page, w);
            return;
        }
        let chunk = page / CHUNK_PAGES;
        if self.chunks.len() <= chunk {
            self.chunks.resize_with(chunk + 1, || None);
        }
        let entries = self.chunks[chunk].get_or_insert_with(|| vec![UNNOTED; CHUNK_PAGES].into_boxed_slice());
        if let Some(entry) = entries.get_mut(page % CHUNK_PAGES) {
            *entry = w;
        }
    }

    /// Forgets what is noted of `page` (tests).
    #[cfg(test)]
    fn remove(&mut self, page: usize) {
        if let Some(w) = self.get_mut(page) {
            *w = UNNOTED;
        }
        self.far.remove(&page);
    }

    /// Calls `f` with every noted page of `[first, last)` and what is
    /// noted of it, in page order.
    fn each(&self, first: usize, last: usize, mut f: impl FnMut(usize, &Written)) {
        if first >= last {
            return;
        }
        let near = last.min(NEAR_PAGES);
        let mut page = first;
        while page < near {
            let chunk = page / CHUNK_PAGES;
            if chunk >= self.chunks.len() {
                break;
            }
            let end = ((chunk + 1) * CHUNK_PAGES).min(near);
            if let Some(entries) = &self.chunks[chunk] {
                let from = page % CHUNK_PAGES;
                for (i, w) in entries.iter().enumerate().take(from + (end - page)).skip(from) {
                    if w.at != 0 {
                        f(chunk * CHUNK_PAGES + i, w);
                    }
                }
            }
            page = end;
        }
        if last > NEAR_PAGES {
            for (&page, w) in self.far.range(first.max(NEAR_PAGES)..last) {
                f(page, w);
            }
        }
    }
}

/// How a copy in view memory is brought up to date: only when something
/// happened in its memory since it last was (1, the way; unset or any value
/// but 2), or, `VICTORY_REFRESH_EVENTS=2`, the check: it asks the guards
/// every time, and every refresh that 1 would have left out and that found
/// bytes to copy is counted and named (0 of them is what 1 needs). Asking
/// every time with no check (the old way) is gone: the check counted 0 in
/// the prologue and at the open world's entry (runs s_y2_p_check and
/// s_we_y2_check of 2026-11-26).
///
/// A submission brings some 6,350 ranges of copies up to date a frame in
/// the first room and about 5,900 of them find nothing written (the census
/// of 2026-11-26). For a range in view memory each of those is a look at
/// every guarded 64 KiB block of it under the guards' lock. What can make
/// a block count as written is known where it happens: a write that
/// faults, a write of ours that opens or marks it, the runtime mapping
/// its memory anew, a page the journal notes. Each raises a count for its
/// 4 MiB of title memory ([`event`]), and a copy remembers the counts of
/// the range it last brought up to date ([`Seen`]): the same counts,
/// nothing to ask. Private memory is not left out: only the kernel knows
/// what was written there, and it must be asked.
///
/// Every count is raised with the lock held under which the thing counted
/// becomes true for an asking (the guards' lock, the journal's), so a look
/// at the counts without a lock errs to one side only: a change the look
/// did not see is counted after it, and the next refresh asks.
///
/// One thing has no event: a block noted absent (not committed) that is
/// committed with nothing told (the title committing part of a reservation
/// of its own; whatever the runtime maps is told). An asking finds that
/// when the guards look such blocks up again, once in [`absent_frames`]
/// frames, so a copy whose range held absent blocks asks once in each such
/// stretch ([`Seen::quiet`]) and learns of it as late as it did before.
///
/// The check (2) decides with the locks held ([`Look::quiet`]). Its first
/// form decided before the asking took them, and whatever the asking then
/// found counted as a find: 7 in the prologue and 2 at the open world's
/// entry (runs s_g9_p_check and s_we_g9_check of 2026-11-26), every one a
/// copy shaders or the title write. A write whose fault or mark was
/// counted in that moment is such a find, and with 1 the next refresh
/// would have asked for it; those are counted apart now.
pub(crate) fn events_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_REFRESH_EVENTS").as_deref().map(str::trim) {
        Ok("2") => 2,
        _ => 1,
    })
}

/// The title memory one count of [`EVENTS`] stands for.
const EVENT_SPAN: usize = 4 << 20;
/// What happened in each 4 MiB of title memory (the first 1 TiB each its
/// own count, memory above shares them): raised, never lowered.
static EVENTS: [std::sync::atomic::AtomicU32; 1 << 18] = [const { std::sync::atomic::AtomicU32::new(0) }; 1 << 18];

/// Something at `at` came to count as written, or the guards over it
/// changed.
fn event(at: usize) {
    EVENTS[(at / EVENT_SPAN) & (EVENTS.len() - 1)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// [`event`] for all of `[at, at + len)`.
fn event_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / EVENT_SPAN, at.saturating_add(len - 1) / EVENT_SPAN);
    // (No more counts than there are.)
    for span in (first..=last).take(EVENTS.len()) {
        EVENTS[span & (EVENTS.len() - 1)].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The counts of `[start, end)` together: the same number as at an earlier
/// call means nothing happened there between the two.
pub(crate) fn events(start: usize, end: usize) -> u64 {
    if start >= end {
        return 0;
    }
    let (first, last) = (start / EVENT_SPAN, (end - 1) / EVENT_SPAN);
    (first..=last)
        .take(EVENTS.len())
        .map(|span| u64::from(EVENTS[span & (EVENTS.len() - 1)].load(std::sync::atomic::Ordering::Relaxed)))
        .sum()
}

/// The frames an answer about absent blocks holds for, numbered: what the
/// guards' cache of absent runs goes by ([`absent_once`]).
fn absent_stretch() -> u64 {
    crate::frame::current() / absent_frames()
}

/// [`Seen::absent`] of a range that must be asked every time: its absent
/// blocks are looked up at every check (no cache), or the kernel's write
/// watch answered for it and not the guards.
const EVERY_TIME: u64 = u64::MAX;

/// What a copy in view memory remembers of the range it last brought up to
/// date by asking ([`events_mode`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Seen {
    /// The range.
    from: usize,
    to: usize,
    /// [`events`] of it, taken just before that asking.
    events: u64,
    /// If blocks of it were noted absent after that asking: the stretch of
    /// frames ([`absent_stretch`]) in which that was looked up, or
    /// [`EVERY_TIME`].
    absent: Option<u64>,
}

impl Seen {
    /// What an asking of `[from, to)` that begins now leaves a copy to
    /// remember, once `look` has been through it: call before the asking.
    pub(crate) fn before(from: usize, to: usize) -> Seen {
        Seen { from, to, events: events(from, to), absent: None }
    }

    /// [`Seen::before`] with what the asking learned of absent blocks.
    pub(crate) fn after(self, look: &Look) -> Seen {
        Seen { absent: look.absent, ..self }
    }

    /// Whether `[from, to)` lies in the range.
    pub(crate) fn holds(&self, from: usize, to: usize) -> bool {
        self.from <= from && to <= self.to
    }

    /// Whether nothing happened in the range since it was brought up to
    /// date, as far as the counts say now: a refresh inside it then has
    /// nothing to ask. With absent blocks in it, only within the frames
    /// their lookup holds for.
    pub(crate) fn quiet(&self) -> bool {
        self.uncounted() && self.absent.is_none_or(|stretch| stretch == absent_stretch())
    }

    /// Whether nothing was counted in the range since it was brought up to
    /// date, whatever the frames of its absent blocks' lookup say. (Not
    /// for a range that is asked every time: counts say nothing of it.)
    pub(crate) fn uncounted(&self) -> bool {
        self.from < self.to && events(self.from, self.to) == self.events && self.absent != Some(EVERY_TIME)
    }
}

/// What an asking is handed by a copy in view memory and tells it besides
/// the pages written ([`writes_fine_look`]).
#[derive(Debug, Default)]
pub(crate) struct Look {
    /// In, for the check: what the copy remembers.
    pub(crate) seen: Option<Seen>,
    /// Out: [`Seen::quiet`] of `seen`, taken with the journal's lock held
    /// and (for view memory) the guards', before the asking took anything:
    /// nothing can be counted between this look and what the asking finds.
    pub(crate) quiet: bool,
    /// Out: [`Seen::uncounted`] of `seen`, taken at the same moment.
    pub(crate) uncounted: bool,
    /// Out: [`Seen::absent`] for the range asked.
    pub(crate) absent: Option<u64>,
    /// Out, for the check's warning: the first block the guards reported
    /// written, the kernel pages of it, and what it was to them.
    pub(crate) found: Option<(usize, u16, &'static str)>,
    /// Out: whether the guards reported a block that was not one noted
    /// absent and found there now (what has no event, and is looked for
    /// once in [`absent_frames`] frames).
    pub(crate) other: bool,
}

impl Look {
    fn found(&mut self, block: usize, mask: u16, what: &'static str, arrived: bool) {
        self.found.get_or_insert((block, mask, what));
        self.other |= !arrived;
    }

    /// Looks at what the copy remembers, with the asking's locks held.
    fn look(&mut self) {
        self.quiet = self.seen.is_some_and(|seen| seen.quiet());
        self.uncounted = self.seen.is_some_and(|seen| seen.uncounted());
    }
}

/// Since the last frame line, of the refreshes of ranges in view memory
/// with [`events_mode`] on: those with nothing happened since the range
/// was last brought up to date (left out, or with the check asked all the
/// same) and nothing to copy; those of them that had bytes to copy all the
/// same (the check's finds: an event that was not counted); those asked
/// because something had happened; those asked because the range was not
/// inside the one last brought up to date; (the check) those that had
/// nothing happened when the refresh looked, had something counted before
/// the asking held its locks, and found bytes to copy; and those asked
/// because the frames their absent blocks' lookup holds for had passed,
/// with nothing counted, that found bytes to copy: in blocks that had been
/// absent and nowhere else (what the turn is for), and elsewhere (bytes no
/// event announced, which 1 takes that many frames late).
static EVENT_CHECKS: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// What a refresh of a range in view memory came to ([`event_checked`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Checked {
    /// Whether the range lay in the one last brought up to date.
    pub(crate) inside: bool,
    /// Whether nothing had happened there when the refresh looked, without
    /// a lock (what 1 goes by).
    pub(crate) quiet: bool,
    /// The same with the asking's locks held ([`Look::quiet`]); as `quiet`
    /// where nothing was asked.
    pub(crate) quiet_locked: bool,
    /// Whether nothing was counted there, with the asking's locks held and
    /// whatever the frames of its absent blocks say ([`Look::uncounted`]).
    pub(crate) uncounted: bool,
    /// The bytes the asking found to copy (`None`: it was left out).
    pub(crate) copied: Option<usize>,
}

/// A refresh of `[from, to)` of the copy `[start, start + len)` (`copy`)
/// in view memory was looked at; `look`: its asking's, if it asked.
pub(crate) fn event_checked(c: Checked, copy: (u64, u64), from: usize, to: usize, look: Option<&Look>) {
    use std::sync::atomic::Ordering::Relaxed;
    let bytes = c.copied.is_some_and(|n| n > 0);
    let found = look.and_then(|look| look.found);
    // Whether all the guards reported were blocks that had been absent
    // (and they reported some: bytes with nothing reported are a note in
    // the journal).
    let arrivals = found.is_some() && look.is_some_and(|look| !look.other);
    let place = match (c.inside, c.quiet, c.quiet_locked) {
        (false, _, _) => 3,
        (true, true, true) if bytes => 1,
        (true, true, true) => 0,
        // Counted between the look and the asking: 1 leaves this refresh
        // out and asks at the next.
        (true, true, false) if bytes => 4,
        // Asked for its absent blocks' sake alone, and bytes were there.
        (true, false, false) if bytes && c.uncounted => {
            if arrivals {
                5
            } else {
                6
            }
        }
        (true, _, _) => 2,
    };
    EVENT_CHECKS[place].fetch_add(1, Relaxed);
    if place == 1 || place == 6 {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, Relaxed) < 32 {
            let found = match found {
                Some((block, mask, what)) => format!("the guards reported block {:#x} (kernel pages {mask:#06x}): {what}", block.wrapping_mul(BLOCK)),
                None => "the guards reported nothing: a note in the journal later than the copy's own".to_string(),
            };
            tracing::warn!(
                copy = format_args!("{:#x}+{:#x}", copy.0, copy.1),
                range = format_args!("{from:#x}..{to:#x}"),
                copied = c.copied.unwrap_or(0),
                found,
                "a refresh found bytes to copy where no event was counted since the range was last brought up to date (looked at with the asking's locks held): with VICTORY_REFRESH_EVENTS=1 the copy would have kept the bytes from before{}",
                if place == 6 {
                    " until its absent blocks were looked up again (it was asked for their sake, and these are not those blocks)"
                } else {
                    ""
                }
            );
        }
    }
}

/// With the frame line.
pub(crate) fn events_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let mode = events_mode();
    let [quiet, wrong, happened, outside, between, arrived, late] = [0, 1, 2, 3, 4, 5, 6].map(|i| EVENT_CHECKS[i].swap(0, Relaxed));
    tracing::info!(
        "refreshes by event, of ranges in view memory, per frame (VICTORY_REFRESH_EVENTS={mode}): {:.0} with nothing happened since \
         the range was last brought up to date ({}); {:.0} asked because something had; {:.0} asked because the range was not \
         inside that one; since the last line {wrong} that the check found with bytes to copy and no event, looked at with the \
         asking's locks held (0 is what =1 needs), and {between} with bytes to copy whose event was counted between the \
         refresh's look and its asking (=1 asks for those at the next refresh; the check's first form took them for finds); \
         of the ranges asked only because the frames their absent blocks' lookup holds for had passed, {arrived} found blocks \
         that had been absent there with nothing told (what that asking is for) and {late} found other bytes no event had \
         announced (0 is what =1 needs: it takes those up to VICTORY_ABSENT_FRAMES frames late, and never in a range with no \
         absent block)",
        quiet as f64 / frames,
        if mode == 1 { "left out" } else { "asked all the same: the check" },
        (happened + arrived + late) as f64 / frames,
        outside as f64 / frames,
    );
}

/// Notes in `written` that asking `now` found `page` written in the kernel
/// pages `mask`.
fn note(written: &mut Pages, page: usize, now: u64, mask: u16) {
    event(page.wrapping_mul(PAGE));
    match written.get_mut(page) {
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
    written: Pages,
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
    written: Pages::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
    once_id: 0,
});

/// The journal, locked (a wait of 0.2 ms or more for it is an event of
/// `crate::timeline`).
fn journal() -> std::sync::MutexGuard<'static, Journal> {
    crate::timeline::waited("L:journal", || JOURNAL.lock().unwrap_or_else(|e| e.into_inner()))
}

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
    let mut out = Vec::new();
    j.written.each(first, last, |page, w| out.push((page, *w)));
    out
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
    let g = guards(Holder::Look);
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
    let mut j = journal();
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
    let mut j = journal();
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
    let mut j = journal();
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
    let mut j = journal();
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` (the title's buffer copies, done
/// at submission): its pages count as written from now on, so the
/// refreshes that follow copy them.
///
/// Whether or not a submission's one asking stands. The copy clears the
/// kernel's written bits of its destination ([`forgot`]), so this note is
/// all that tells of it: it was made only while an asking stood, and a
/// copy made after a step that ended the asking, or in a submission that
/// asked the kernel about nothing, was told to nobody. A host copy of its
/// destination that was brought up to date later kept the bytes from
/// before it, and with them whatever the title had written into the same
/// 4 KiB pages since the last asking.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = journal();
    if len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    if j.kept {
        j.ahead_asked = now;
    }
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        note(&mut j.written, page, now, FULL);
    }
}

/// For a landing that wrote private title memory (`crate::shadow::landed`):
/// takes the kernel's written bits of `ranges` now, and notes the pages
/// they name as written at an asking of its own, whose number is returned
/// with the pages noted (each with what the journal now holds of it, in
/// page order). Ranges that lie close together are one kernel asking; one
/// that is not write-watched memory of one reservation is left as it is.
///
/// What it is for: the landing's write sets those bits, and the next
/// refresh's asking took them for the title's, so the landing's own copy
/// read the pages it had just been given (22.8 MB a frame in the first
/// room). Taken here, right behind the write and with the journal's lock
/// held as every asking holds it, the bits stand for "written up to now",
/// by the landing or by the title; the note tells every other copy over
/// the pages, exactly as the next asking would have; and the landing's
/// copy can compare each page with its own bytes and count it taken if
/// they are the same. A write of the title's after this sets the bit
/// again and is found by the next asking, as ever.
pub(crate) fn take_landed(ranges: &[(usize, usize)]) -> (u64, Vec<(usize, Written)>) {
    let mut pieces: Vec<(usize, usize)> =
        ranges.iter().filter(|r| r.0 < r.1).map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE))).collect();
    pieces.sort_unstable();
    let mut j = journal();
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in `wrote`: an asking that stands moves with the note.
    if j.kept {
        j.ahead_asked = now;
    }
    let mut noted = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len() && pieces[k].0 <= last + LANDED_GAP && last.max(pieces[k].1) - first <= AHEAD_MAX {
            last = last.max(pieces[k].1);
            k += 1;
        }
        take_run(j, &pieces[i..k], now, &mut noted);
        i = k;
    }
    (now, noted)
}

/// Pieces of one landing closer than this are one kernel asking
/// ([`take_landed`]): 1 MiB. The pages between are asked about too, and
/// what the title wrote there is noted a little earlier than it would
/// have been.
const LANDED_GAP: usize = (1 << 20) / PAGE;

/// [`take_landed`] for `pieces` (page ranges in order): one kernel asking
/// for all of them, by halves if it fails.
fn take_run(j: &mut Journal, pieces: &[(usize, usize)], now: u64, noted: &mut Vec<(usize, Written)>) {
    let Some(&(first, _)) = pieces.first() else {
        return;
    };
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel_as(j, first, last, now) {
        j.written.each(first, last, |page, w| {
            if w.at == now {
                noted.push((page, *w));
            }
        });
        return;
    }
    if pieces.len() < 2 {
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    take_run(j, a, now, noted);
    take_run(j, b, now, noted);
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
    let now = j.ahead_asked;
    ask_kernel_as(j, first, last, now)
}

/// [`ask_kernel`], with what is found noted under asking `now`.
fn ask_kernel_as(j: &mut Journal, first: usize, last: usize, now: u64) -> bool {
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
    let j = journal();
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| {
            let mut out = Vec::new();
            j.written.each(first, last, |page, w| out.push((page, w.at)));
            (j.ahead_asked, out)
        })
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
    writes_fine_look(start, end, guarded, &mut Look::default())
}

/// [`writes_fine`] for a copy that goes by events ([`events_mode`]):
/// `look` takes what the copy remembers and brings back what the asking
/// learned beside the pages.
pub(crate) fn writes_fine_look(start: usize, end: usize, guarded: bool, look: &mut Look) -> Option<(u64, Vec<(usize, Written)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = journal();
    let j = &mut *j;
    // Where the kernel's write watch answers (here and below) no event
    // tells of a write: such a range is asked every time. The guards' own
    // look, with their lock too, replaces this one.
    look.look();
    look.absent = Some(EVERY_TIME);
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
        let epoch = once().then_some(Epoch { submission: j.once_id });
        let written = guard_fine(first * PAGE, last * PAGE, epoch, look);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            if written.is_some() {
                crate::copystat::buffers::view(first * PAGE, last.saturating_sub(first).saturating_mul(PAGE));
            }
        }
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
    !view_ask() && held(&guards(Holder::Look), start / BLOCK)
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

/// What a check of the guards is told of the submission it is made for.
#[derive(Clone, Copy, Debug)]
struct Epoch {
    /// The submission's number ([`Journal::once_id`]).
    submission: u64,
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
    dirty: BTreeMap::new(),
    partly: BTreeSet::new(),
});

/// Diagnostics, since the last [`guard_line`]: kernel calls the guards
/// made, by what for: `VirtualQuery` for whether a range is a view, for
/// whether absent blocks are still unmapped, after a failed protection;
/// `VirtualProtect` arming runs of blocks, arming blocks one by one (a run
/// that failed), arming other views, opening blocks; (two no longer
/// counted); checks that found their range all clean, all
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
    let (runs, absent, partly) = {
        let g = guards(Holder::Look);
        (g.absent_runs.len(), g.absent.len(), g.dirty.len())
    };
    tracing::info!(
        "guards per submission: {:.1} checks, {:.1} all clean; VirtualQuery {:.1} is-view, {:.1} absent, {:.1} after a failed \
         protection; VirtualProtect {:.1} runs armed, {:.1} blocks armed one by one, {:.1} other views armed, {:.1} blocks opened; \
         absent runs: {:.1} found in the cache, cache emptied {:.2} times, {:.1} runs dropped by remappings, {runs} cached, {absent} \
         absent blocks; faults that opened part of a block {:.1}, such blocks armed again {:.1}, {partly} blocks with 4 KiB pages \
         noted written now, {:.2} faults on blocks armed after they were opened; refreshes of copies in the card's memory: {:.1} 4 KiB blocks ({:.2} MB) equal to the CPU side, not sent \
         up (VICTORY_SEND_CHANGED=0 sends them)",
        c[10], c[9], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[11], c[12], c[13], c[14], c[15], c[16], same[0], same[1] / 1e6
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

/// Who takes the guards' lock: a hold is an event of `crate::timeline`
/// under the holder's name ([`Held`]).
#[derive(Clone, Copy, Debug)]
enum Holder {
    /// A check of a range ([`guard_fine`]): its walk over the blocks.
    Check,
    /// What a check undoes for the protections that failed ([`settle`]).
    Settle,
    /// A write of the title's that faulted ([`install`]'s handler).
    Fault,
    /// Blocks opened before a write through the kernel ([`writing_as`]).
    Opening,
    /// Blocks counted written after a write the guards do not see
    /// ([`written_beside`], [`kernel_wrote`], [`title_memory_written`]).
    Marking,
    /// A view of the runtime's mapped ([`views_mapped`]).
    Mapped,
    /// Memory the runtime mapped or unmapped ([`title_memory_remapped`]).
    Remapped,
    /// A look that changes nothing ([`known_view`], [`watchable`],
    /// [`seen_since`], [`guard_state`], the guards' line).
    Look,
}

impl Holder {
    /// The events of a hold of 0.2 ms and more, and of a thread's holds
    /// summed.
    fn events(self) -> (&'static str, &'static str) {
        match self {
            Holder::Check => ("H:check", "HS:check"),
            Holder::Settle => ("H:settle", "HS:settle"),
            Holder::Fault => ("H:fault", "HS:fault"),
            Holder::Opening => ("H:opening", "HS:opening"),
            Holder::Marking => ("H:marking", "HS:marking"),
            Holder::Mapped => ("H:mapped", "HS:mapped"),
            Holder::Remapped => ("H:remapped", "HS:remapped"),
            Holder::Look => ("H:look", "HS:look"),
        }
    }
}

const _: () = assert!((Holder::Look as usize) < crate::timeline::HOLDERS);

/// The guards, locked. While `crate::timeline` keeps events, letting go
/// says how long they were held and by whom: the lock is the fault
/// handler's on the title's threads, and a wait for it (`L:guards`) names
/// no holder.
struct Held {
    guards: std::sync::MutexGuard<'static, Guards>,
    by: Holder,
    since: Option<std::time::Instant>,
    /// A number of the holder's own for the event (the blocks a check
    /// walked).
    n: u64,
}

impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        &self.guards
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        &mut self.guards
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        let (long, sum) = self.by.events();
        crate::timeline::held(self.by as usize, long, sum, self.since, self.n);
    }
}

/// The guards' lock, taken by `by` (a wait of 0.2 ms or more for it is an
/// event of `crate::timeline`, and so is the hold: [`Held`]).
fn guards(by: Holder) -> Held {
    let guards = crate::timeline::waited("L:guards", || GUARDS.lock().unwrap_or_else(|e| e.into_inner()));
    Held { guards, by, since: crate::timeline::clock(), n: 0 }
}

/// [`guard_fine`] without the kernel pages: the pages written.
#[cfg(test)]
fn guard(start: usize, end: usize, epoch: Option<Epoch>) -> Option<Vec<usize>> {
    guard_fine(start, end, epoch, &mut Look::default()).map(|w| w.into_iter().map(|(page, _)| page).collect())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
/// `epoch`: the submission the check is made for, with [`once`]
/// ([`absent_once`]). `look`: see [`Look`].
fn guard_fine(start: usize, end: usize, epoch: Option<Epoch>, look: &mut Look) -> Option<Vec<(usize, u16)>> {
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
    let mut g = guards(Holder::Check);
    if ask || !held(&g, start / BLOCK) {
        drop(g);
        if !is_view(start, end) {
            return None;
        }
        g = guards(Holder::Check);
    }
    g.n = (end.div_ceil(BLOCK) - start / BLOCK) as u64;
    // With this lock and the journal's (the caller's) nothing is counted
    // until the check is through: what it finds below is what these counts
    // stand for.
    look.look();
    look.absent = None;
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
    let absent_epoch = epoch.filter(|_| absent_once()).map(|_| absent_stretch());
    let unguardable_epoch = epoch.filter(|_| absent_once()).map(|e| e.submission);
    // The changes of protection this check decides are made after the
    // guards' lock is let go ([`Deferred`]).
    let mut later = Deferred::default();
    for block in first..=last {
        if block / 64 != chunk {
            chunk = block / 64;
            word = g.clean.word(chunk);
        }
        let is_clean = word & (1 << (block % 64)) != 0;
        let mut open = block < last && !is_clean;
        let mut arrived = false;
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
                // There now, and nothing told of it (a mapping the runtime
                // tells takes the block out of the absent ones): no event
                // was counted for this.
                g.absent.remove(&block);
                arrived = true;
            } else {
                // Still absent, by a lookup that holds for some frames
                // (or is made at every check): see [`Seen::quiet`].
                look.absent = Some(absent_epoch.unwrap_or(EVERY_TIME));
            }
        }
        if open {
            let rearmed = g.protected.contains(&block) && {
                // Written through the runtime or another view, still
                // read-only; or opened by a fault from its kernel page on
                // ([`guard_pages`]), armed again whole now. Reported with
                // the kernel pages that may have been written.
                let mask = g.dirty.remove(&block).unwrap_or(FULL);
                written.push((block, mask));
                look.found(
                    block,
                    mask,
                    if arrived {
                        "noted absent, there now with nothing told of it, and guarded meanwhile as another view of its pages"
                    } else {
                        "guarded and counted written (a fault, a write of ours, or guarded as another view of its pages)"
                    },
                    arrived,
                );
                if g.partly.remove(&block) {
                    later.rearm.push(block);
                }
                true
            };
            if !g.protected.contains(&block) && !written.last().is_some_and(|&(b, _)| b == block) {
                written.push((block, FULL));
                look.found(
                    block,
                    FULL,
                    if arrived {
                        "noted absent, and there now with nothing told of it"
                    } else {
                        "not guarded (opened, mapped anew, never taken, or not to be guarded)"
                    },
                    arrived,
                );
            }
            if rearmed {
                g.clean.insert(block);
            } else if unguardable_epoch.is_some_and(|e| g.unguardable_in(e).contains(&block)) {
                // Could not be made read-only in this submission: reported
                // written at every check, as it would be after trying.
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            plan_protect(&mut g, from, block, &mut later, epoch);
        }
    }
    drop(g);
    if !later.is_empty() {
        let failed = later.apply();
        if !failed.is_empty() {
            settle(&mut guards(Holder::Settle), &failed, epoch);
        }
    }
    Some(written)
}

/// The changes of protection a check decided, made with the guards' lock
/// let go: runs of blocks to make read-only (noted guarded and clean
/// already), other views of their pages (noted guarded), and blocks a
/// fault opened in part to make read-only whole again (noted clean).
///
/// A check held the lock across every `VirtualProtect` it made (and the
/// lookups of a block's other views), and the lock is the fault handler's:
/// the title's threads that wrote a guarded block waited for it, 15 ms a
/// frame on the recording threads (round 5's profile). A check notes the
/// blocks guarded first (so that a fault on one once it is read-only
/// finds it the guards'), lets the lock go, makes the calls, and takes the
/// lock again only for those that failed. A write landing on a block
/// between the note and the call is one before the check's copy reads the
/// block, and the check reports the block written into the journal, whose
/// lock it holds throughout (every other copy learns of it there). (The
/// calls under the lock were `VICTORY_GUARD_UNLOCKED=0`, gone with the
/// blocks left open, `VICTORY_HOT_BLOCKS`, which were never run.)
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

/// For a check: notes the blocks of `[from, to)` that may be guarded
/// guarded and clean and their pages' other views guarded, and leaves the
/// calls to `later`.
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

/// [`plan_protect`] for a run of blocks that may be guarded. The same
/// physical pages at every other view of them are made read-only too: a
/// write through one of those is this memory written, and only a fault
/// can say so (`VICTORY_VIEW_GUARD=0` leaves the other views as they are,
/// as before). One view (nothing else to guard), or the page the title
/// shows wherever it has nothing (thousands of views): each view of that
/// one is guarded by itself.
///
/// Another view is not noted clean. The copies of title memory at that
/// view have not taken the pages: noted clean, the block would not be
/// reported written at their next check and they would keep the bytes from
/// before the write (a check reports a block that is read-only and not
/// clean as written, then notes it clean). A block that is clean stays so.
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
/// know but do not hold guarded: if the block is
/// read-only (a check made it so after something opened it), makes it
/// writable and notes it written; true if that was done.
///
/// Known to the guards, or lying in a view the runtime told of
/// (`crate::inplace::view_at`). A check notes its blocks guarded, lets
/// the lock go and then changes their protection; a mapping of the
/// runtime's that takes the lock in between drops all the guards knew of
/// the blocks, and the check's call then makes the new view read-only
/// with nothing left to say it was ours. The title's first write there
/// faulted, found no guard, and went on as an access violation of the
/// title's. The runtime maps the title's views read-write and nothing but
/// a guard makes one read-only, so a read-only block of such a view is a
/// guard's.
fn reopen_stray(g: &mut Guards, at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
    };
    let block = at / BLOCK;
    if g.protected.contains(&block) || !(g.views.contains(&block) || crate::inplace::view_at(at).is_some()) {
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
    event(block.wrapping_mul(BLOCK));
    g.dirty.remove(&block);
    g.partly.remove(&block);
    true
}

/// Undoes the notes of the blocks whose protection could not be changed
/// ([`Deferred::apply`]): one that is not committed (unmapped) is noted
/// absent, any other is not tried again in this submission.
fn settle(g: &mut Guards, failed: &Failed, epoch: Option<Epoch>) {
    for &block in &failed.runs {
        g.protected.remove(&block);
        event(block.wrapping_mul(BLOCK));
        g.clean.remove(&block);
        event(block.wrapping_mul(BLOCK));
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
        event(block.wrapping_mul(BLOCK));
    }
    for &block in &failed.rearm {
        // No longer guarded (reported written already).
        g.protected.remove(&block);
        event(block.wrapping_mul(BLOCK));
        g.clean.remove(&block);
        event(block.wrapping_mul(BLOCK));
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
/// which [`settle`] notes absent when a protection fails).
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
        event(block.wrapping_mul(BLOCK));
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
        event(block.wrapping_mul(BLOCK));
        g.clean.remove(&block);
        event(block.wrapping_mul(BLOCK));
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
    writing_as(at, len, crate::writes::Cause::Kernel);
}

/// [`writing`] for the layer's own writes, which say what they are
/// (`cause`).
pub(crate) fn writing_as(at: usize, len: usize, cause: crate::writes::Cause) {
    // (A diagnostic's inaccessible pages there: the kernel would not write them.)
    crate::read_census::disarm(at, len);
    // The length is the title's (`fread`'s size times count, saturated).
    {
        let mut g = guards(Holder::Opening);
        // Every block is opened under the guards' lock, so what the count
        // of opened blocks grows by here is this write's.
        #[cfg(feature = "diagnostics")]
        let before = GUARD_CALLS[6].load(std::sync::atomic::Ordering::Relaxed);
        open(&mut g, at, at.saturating_add(len));
        #[cfg(feature = "diagnostics")]
        crate::copystat::buffers::opened(
            cause as usize,
            GUARD_CALLS[6].load(std::sync::atomic::Ordering::Relaxed).saturating_sub(before),
        );
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len, cause);
}

/// Since the start: reads of the kernel's into title memory that finished
/// ([`kernel_wrote`]), their bytes, and the blocks under them that a check
/// had guarded again while the kernel was still writing.
pub(crate) static KERNEL_WROTE: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The kernel has written title memory `[at, at + len)`: a read of the
/// title's into its own memory, of which [`writing`] was told before it
/// began, is done. The runtime calls it (as `kernel_wrote`).
///
/// [`writing`] opens the guarded blocks, and nothing holds them open: a
/// check that comes while the kernel is still writing reports them
/// written, guards them again and has its copy read what is there so far.
/// A read through the system's cache fails on such a block (and the title
/// reads again); one that had the pages locked when it began (a file
/// opened without buffering) goes on into them, with no fault and nothing
/// counted, and every host copy over them kept the bytes of the moment it
/// looked. So the blocks guarded again count as written once more now
/// that the bytes are all there, at every view of their pages
/// ([`mark_with_views`], or opened where [`mark_only`] is off), and the
/// write is numbered for the texture pass as DirectStorage's are
/// ([`title_memory_written`]). Upstream holds such blocks open over the
/// read and marks them at its end (its c75fdba; pull request 3231 there
/// carries that to overlapped reads, which this title does not import).
pub fn kernel_wrote(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    {
        let mut g = guards(Holder::Marking);
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        KERNEL_WROTE[0].fetch_add(1, Relaxed);
        KERNEL_WROTE[1].fetch_add(len as u64, Relaxed);
        KERNEL_WROTE[2].fetch_add(guarded, Relaxed);
        if guarded != 0 {
            if mark_only() {
                mark_with_views(&mut g, at, end);
            } else {
                open(&mut g, at, end);
            }
        }
    }
    title_memory_written(at, len, false);
}

/// With the frame line, once the kernel has read into title memory.
pub(crate) fn kernel_line() {
    use std::sync::atomic::Ordering::Relaxed;
    let [reads, bytes, guarded] = [0, 1, 2].map(|i| KERNEL_WROTE[i].load(Relaxed));
    if reads == 0 {
        return;
    }
    tracing::info!(
        "reads of the kernel's into title memory that finished, since the start: {reads} ({} MB); {guarded} blocks under them \
         had been guarded again while the kernel was writing and count as written once more (without that, {guarded} blocks \
         of host copies with the bytes of the moment they were looked at)",
        bytes >> 20
    );
}

/// After a write of the layer's own into title memory `[at, at + len)` that
/// went through a view of its own of the same pages (`crate::beside`): the
/// guarded blocks there, and at every other view of their pages, count as
/// written and stay read-only ([`mark_only`]), so every host copy over
/// them takes them again at its next check, as after [`writing_as`] opened
/// them, without the two changes of protection. Called once the bytes are
/// there: a check between the write and this takes them a second time, one
/// before the write would have noted the block clean with the old bytes.
pub(crate) fn written_beside(at: usize, len: usize, cause: crate::writes::Cause) {
    {
        let mut g = guards(Holder::Marking);
        mark_with_views(&mut g, at, at.saturating_add(len));
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len, cause);
}

/// What the journal and the guards hold of title memory `[start, end)`
/// against the asking numbered `asked` (the one a texture's upload read
/// after, `crate::texture` `Staging::asked`), asked without changing
/// either (no kernel asking, no block guarded again): whether a page of it
/// is noted written after that asking; how many of its 64 KiB blocks are
/// guarded and counted written since they were last taken; and how many
/// are not guarded at all, so that a store there is seen by nobody. For
/// the count of what was known of a texture a look found changed
/// (`crate::census`): two locks, one after the other.
pub(crate) fn seen_since(start: usize, end: usize, asked: u64) -> (bool, usize, usize) {
    if start >= end {
        return (false, 0, 0);
    }
    // (No more than 4 GiB of it: no texture is longer, and a range that is
    // nonsense must not keep a recording thread walking.)
    let first = start / BLOCK;
    let last = end.div_ceil(BLOCK).min(first.saturating_add(1 << 16));
    let noted = asked != 0 && {
        let j = journal();
        let mut after = false;
        j.written.each(first, last, |_, w| after |= w.at > asked);
        after
    };
    let (mut written, mut open) = (0, 0);
    {
        let g = guards(Holder::Look);
        for block in first..last {
            if !g.protected.contains(&block) {
                open += usize::from(!g.absent.contains(&block));
            } else if !g.clean.contains(&block) {
                written += 1;
            }
        }
    }
    // Blocks no guard holds may be write-watched memory: the kernel says
    // whether a page of them was written since its bits were last taken,
    // and leaves the bits (no reset). What another asking took meanwhile
    // is in the journal, above.
    if open > 0
        && let Some(pages) = watched_dirty(first * BLOCK, (last - first) * BLOCK)
    {
        (written, open) = (written + pages, 0);
    }
    (noted, written, open)
}

/// How many pages of `[at, at + len)` the kernel's write watch holds as
/// written (up to a few), without taking the bits; `None` if the range is
/// not write-watched memory of one reservation.
pub(crate) fn watched_dirty(at: usize, len: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let mut found = [0usize; 8];
    let mut count = found.len();
    let mut granularity = 0u32;
    // SAFETY: the answer buffer holds `count` entries; a range that is not
    // watched makes the call fail.
    let failed =
        unsafe { GetWriteWatch(0, at as *const _, len, found.as_mut_ptr().cast(), &mut count, &mut granularity) } != 0;
    (!failed).then_some(count)
}

/// The number of the journal's last asking: what a read of title memory
/// that begins now comes after.
pub(crate) fn asked_now() -> u64 {
    journal().asked
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write into physical
/// pages that the guards do not see, once its bytes are there
/// ([`title_memory_written`]), as bits: 1, the guarded blocks under it are
/// opened and count as written (host copies of buffers and textures read
/// through tables take them again); 2, the placed textures picked by index
/// whose last upload read before the write (by the writes' numbers,
/// `crate::writes`) are due again at once, at every pass (without it they
/// are counted and left as they are). Unset, `1` or `all`: bits 1 and 2.
/// `guards`: bit 1 alone. `slots`: bit 2 alone. `0`: nothing. Every such write and
/// every read to a plain address that finished is numbered and noted for
/// the texture pass whatever this says (a read to a plain address writes
/// from user mode, which the guards see).
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 uploaded some 120 textures a second again in the opening
/// cutscene (12,011 in 100 s), each of them whole, and was off for it; in
/// its earlier form, two frames after the write, it cost 8 ms a frame. It
/// is on now that an upload brings only the levels whose bytes changed and
/// a write is taken for the slots whose view holds its bytes
/// (`crate::descriptor::host` `told_slots`): the game maps a texture's
/// pages, writes its view and has DirectStorage fill the pages afterwards,
/// and a slot uploaded in between showed what the pages held before until
/// a look found it changed, 4 to 400 frames later, or never.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("guards") => 1,
        Ok("slots") => 2,
        _ => 3,
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
/// each address showing them). Here it is counted always, the guards are
/// acted on as [`page_writes`] says, and the range is kept with a number
/// for the texture pass (`crate::writes`). Called without the runtime's
/// page-table lock; takes the guards' lock, then, alone, the lock of the
/// ranges kept for the texture pass.
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
    }
    if pages {
        let mut g = guards(Holder::Marking);
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        WRITTEN[2].fetch_add(guarded, Relaxed);
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
    crate::descriptor::host::page_written(at, len, pages);
    // The refresh check: what it holds as differing there was under this
    // write (the guards' lock is let go by now).
    crate::shadow::refresh_check_told(at, len);
}

/// What the guards hold of the block of `at`, in words (for the refresh
/// check's warnings).
pub(crate) fn guard_state(at: usize) -> &'static str {
    let g = guards(Holder::Look);
    let block = at / BLOCK;
    match (g.protected.contains(&block), g.clean.contains(&block)) {
        (true, true) => "guarded, not written since it was taken",
        (true, false) => "guarded, counted written",
        _ if g.absent.contains(&block) => "noted absent",
        _ if g.views.contains(&block) => "view memory, not guarded now",
        _ => "not known to the guards (private memory, or never taken)",
    }
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written.
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards(Holder::Mapped);
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
    Remapping(crate::timeline::waited("L:mapping.w", || MAPPING.write().unwrap_or_else(|e| e.into_inner())))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(crate::timeline::waited("L:mapping.r", || MAPPING.read().unwrap_or_else(|e| e.into_inner())))
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
    // Copies of shader-written textures back into it are dropped.
    crate::texback::remapped(at, len);
    // Whether its blocks are there is asked anew (`texture::block_there`),
    // and the textures staged there are read anew by the next list.
    crate::texture::blocks_remapped(at, len);
    crate::texture::remapped(at, len);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len, crate::writes::Cause::Mapped);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards(Holder::Remapped);
    // What a copy knew of this memory is no longer so: counted with the
    // guards' lock held, as everything is that changes what a check finds.
    // Counted before the lock was taken, a refresh could look at the
    // counts after this one and be checked before the blocks noted absent
    // here were dropped below (a drop that counts nothing by itself): its
    // copy then remembered counts that held the event and not the change,
    // and was never asked about the memory mapped here.
    event_range(at, len);
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).copied().collect();
    for block in absent {
        g.absent.remove(&block);
    }
    let blocks: Vec<usize> = g.protected.range(range).copied().collect();
    for block in blocks {
        g.protected.remove(&block);
        event(block.wrapping_mul(BLOCK));
        g.clean.remove(&block);
        event(block.wrapping_mul(BLOCK));
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
    // known. Runs elsewhere stay:
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
        // A page the read census made inaccessible (`crate::read_census`,
        // a diagnostic): given back and counted, whoever touched it.
        if crate::read_census::every() != 0
            && r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.NumberParameters >= 2
            && r.ExceptionInformation[0] <= WRITE
            // SAFETY: the faulting thread's context, as dispatched.
            && crate::read_census::fault(r.ExceptionInformation[1], r.ExceptionInformation[0] == WRITE, unsafe { (*(*info).ContextRecord).Rip } as usize)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
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
            let mut g = guards(Holder::Fault);
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
            // something opened it (the call is made without the lock).
            // Opened and counted written, as a fault on a guarded block is.
            if reopen_stray(&mut g, at) {
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

    /// Write-watched memory is asked without its bits being taken: what
    /// was written stays written for the asking that copies it, and the
    /// memory counts as watched, not as seen by nobody.
    #[test]
    fn the_write_watch_is_asked_without_taking_its_bits() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        /// The allocation type that has the kernel keep a dirty bit a page.
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let len = 4 * BLOCK;
        // SAFETY: a fresh allocation of ours, written inside its bounds and
        // released at the end.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            let at = p as usize;
            assert_eq!(watched_dirty(at, len), Some(0));
            assert_eq!(seen_since(at, at + len, 0), (false, 0, 0));
            p.add(BLOCK + 5).write(1);
            assert_eq!(watched_dirty(at, len), Some(1));
            assert_eq!(watched_dirty(at, len), Some(1), "the bits are left");
            assert_eq!(seen_since(at, at + len, 0), (false, 1, 0));
            assert_ne!(VirtualFree(p.cast(), 0, MEM_RELEASE), 0);
            // Memory that is not watched: nothing to ask.
            let plain = VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!plain.is_null());
            assert_eq!(watched_dirty(plain as usize, len), None);
            assert_ne!(VirtualFree(plain.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// Memory the guards do not hold is seen by nobody: every block of it
    /// counts as not guarded, and an empty range has none.
    #[test]
    fn memory_no_guard_holds_is_seen_by_nobody() {
        // An address no test maps and no guard holds.
        let at = 0x7fe9_0000_0000usize;
        assert_eq!(seen_since(at, at + 3 * BLOCK, 0), (false, 0, 3));
        assert_eq!(seen_since(at + 1, at + BLOCK + 1, 5), (false, 0, 2));
        assert_eq!(seen_since(at, at, 0), (false, 0, 0));
        // A range that is none, address 0 and the top of the address space.
        assert_eq!(seen_since(at + BLOCK, at, 7), (false, 0, 0));
        assert_eq!(seen_since(0, 0, 0), (false, 0, 0));
        assert_eq!(seen_since(0, 1, 0).1, 0);
        assert_eq!(seen_since(usize::MAX, usize::MAX, 0), (false, 0, 0));
        assert_eq!(seen_since(usize::MAX - 1, usize::MAX, u64::MAX).1, 0);
        // All of the address space: looked at for 4 GiB of it, no longer.
        assert_eq!(seen_since(0, usize::MAX, 0), (false, 0, 1 << 16));
        assert!(asked_now() <= asked_now());
    }
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// The journal never forgets a page: a test's fresh allocation may lie
    /// where another test's was, whose writes it still holds.
    fn forget(start: usize, end: usize) {
        let mut j = journal();
        for page in start / PAGE..end.div_ceil(PAGE) {
            j.written.remove(page);
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

    /// A write of ours whose kernel bits were dropped ([`forgot`], as a
    /// title buffer copy made at submission drops them) is told to the
    /// journal whether or not a submission's asking stands: the next
    /// asking names its page, written after the asking before.
    #[test]
    fn a_write_of_ours_is_noted_with_no_asking_standing() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 2 * PAGE);
        forget(start, end);
        let (first, found) = writes_fine(start, end, false).expect("watched");
        assert_eq!(found, []);
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        forgot(p + PAGE + 8, 1);
        wrote(p + PAGE + 8, 1);
        let (_, found) = writes_fine(start, end, false).expect("watched");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, p / PAGE + 1);
        assert!(found[0].1.at > first && found[0].1.mask == FULL);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Asked again at the same number, the journal joins the pages; an
    /// answer numbered before the last noted keeps the later number and no
    /// longer says where.
    #[test]
    fn notes_join_and_never_go_back() {
        let mut written = Pages::new();
        note(&mut written, 9, 4, 1);
        note(&mut written, 9, 4, 2);
        assert_eq!(written.get_mut(9).copied(), Some(Written { at: 4, before: 0, mask: 3 }));
        note(&mut written, 9, 6, 8);
        assert_eq!(written.get_mut(9).copied(), Some(Written { at: 6, before: 4, mask: 8 }));
        note(&mut written, 9, 5, 1);
        assert_eq!(written.get_mut(9).copied(), Some(Written { at: 6, before: 4, mask: FULL }));
    }

    /// The table answers as the map did: the noted pages of a range, in
    /// order, across its chunks and above the addresses it has chunks for.
    #[test]
    fn the_pages_noted_in_a_range_come_in_order_wherever_they_lie() {
        let mut written = Pages::new();
        let mut map = BTreeMap::new();
        let w = |at| Written { at, before: 0, mask: 1 };
        let pages = [
            0,
            5,
            CHUNK_PAGES - 1,
            CHUNK_PAGES,
            7 * CHUNK_PAGES + 3,
            NEAR_PAGES - 1,
            NEAR_PAGES,
            NEAR_PAGES + 12,
            usize::MAX / PAGE,
        ];
        for (i, &page) in pages.iter().enumerate() {
            written.insert(page, w(i as u64 + 1));
            map.insert(page, w(i as u64 + 1));
        }
        assert!(written.get_mut(1).is_none() && written.get_mut(NEAR_PAGES + 1).is_none());
        let ranges = [
            (0, 1),
            (0, usize::MAX),
            (1, 5),
            (5, CHUNK_PAGES),
            (CHUNK_PAGES - 1, CHUNK_PAGES + 1),
            (CHUNK_PAGES + 1, 7 * CHUNK_PAGES + 3),
            (6 * CHUNK_PAGES, NEAR_PAGES),
            (NEAR_PAGES - 1, NEAR_PAGES + 13),
            (NEAR_PAGES + 1, usize::MAX),
            (9, 9),
            (9, 3),
        ];
        for (first, last) in ranges {
            let mut got = Vec::new();
            written.each(first, last, |page, w| got.push((page, *w)));
            let want: Vec<(usize, Written)> = if first < last { map.range(first..last).map(|(&p, &w)| (p, w)).collect() } else { Vec::new() };
            assert_eq!(got, want, "{first:#x}..{last:#x}");
        }
        // A page forgotten is not noted; noting it again is.
        written.remove(5);
        written.remove(NEAR_PAGES);
        assert!(written.get_mut(5).is_none() && written.get_mut(NEAR_PAGES).is_none());
        note(&mut written, 5, 9, 2);
        assert_eq!(written.get_mut(5).copied(), Some(Written { at: 9, before: 0, mask: 2 }));
    }

    /// A remapping forgets the unmapped runs it touches and no others.
    #[test]
    fn a_remapping_forgets_only_the_absent_runs_it_touches() {
        // Far above anything mapped in a test: only the notes change.
        let b = (1usize << 46) / BLOCK;
        {
            let mut g = guards(Holder::Look);
            g.absent_for = 7;
            g.absent_runs.insert(b, b + 4);
            g.absent_runs.insert(b + 10, b + 11);
            g.absent_runs.insert(b + 20, b + 30);
        }
        title_memory_remapped((b + 2) * BLOCK, BLOCK);
        title_memory_remapped((b + 29) * BLOCK, 2 * BLOCK);
        let mut g = guards(Holder::Look);
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
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, FULL)]));
        // SAFETY: inside our view; the write faults and opens the tail.
        unsafe { ((view + 5 * KERNEL_PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(guards(Holder::Look).protected.contains(&block), "the pages below stay guarded");
        // SAFETY: as above, above the fault: open, no fault.
        unsafe { ((view + 9 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        // SAFETY: as above, below it: faults, opens from there on.
        unsafe { ((view + 2 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, FULL << 2)]));
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(Vec::new()), "armed again whole");
        // SAFETY: as above; faults again: armed whole.
        unsafe { ((view + 15 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, 1 << 15)]));
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
        // SAFETY: a pagefile section of one block, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let block = view / BLOCK;
        assert!(guard_fine(view, view + BLOCK, None, &mut Look::default()).is_some());
        {
            // As if opened by another thread and armed again by a check
            // that made its call afterwards.
            let mut g = guards(Holder::Look);
            g.protected.remove(&block);
            g.clean.remove(&block);
        }
        // SAFETY: inside our view, read-only now: the write faults.
        unsafe { ((view + 100) as *mut u8).write_volatile(9) };
        // SAFETY: as above.
        assert_eq!(unsafe { ((view + 100) as *const u8).read_volatile() }, 9);
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, FULL)]));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, BLOCK);
    }

    /// The same when a mapping dropped all the guards knew of the block
    /// between a check's note and its late call: the block lies in a view
    /// the runtime told of, read-only by that call alone, and the title's
    /// write there is opened and reported, not left to fault unanswered.
    #[test]
    fn a_block_armed_after_it_was_mapped_anew_is_opened_by_its_fault() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let block = (one + BLOCK) / BLOCK;
        // (A check of the other view: the handler is there.)
        assert!(guard(two, two + SIZE, None).is_some());
        // Mapped anew, as the runtime tells it: nothing is known of it.
        title_memory_remapped(one, SIZE);
        crate::inplace::title_memory_view(one, SIZE, section, 0);
        assert!(!guards(Holder::Look).views.contains(&block));
        // The check's late call.
        let mut old = 0;
        // SAFETY: a block of our own view; only its protection changes.
        assert_ne!(unsafe { VirtualProtect((one + BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) }, 0);
        // SAFETY: inside the view, read-only now: the write faults.
        unsafe { ((one + BLOCK + 100) as *mut u8).write_volatile(9) };
        // SAFETY: as above.
        assert_eq!(unsafe { ((one + BLOCK + 100) as *const u8).read_volatile() }, 9);
        let written = guard(one, one + SIZE, None).expect("view memory");
        assert!(written.contains(&block), "reported written");
        drop_views(section, &[one, two], SIZE);
    }

    #[test]
    fn block_sets_answer_as_a_set_of_blocks_does() {
        let mut set = BlockSet::new();
        let mut naive = BTreeSet::new();
        for b in [0usize, 1, 2, 63, 64, 65, 127, 128, 300, 301, 302, 303] {
            set.insert(b);
            naive.insert(b);
        }
        for block in 500..700 {
            set.insert(block);
        }
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
        assert!(!guards(Holder::Look).clean.contains(&(view / BLOCK)));
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

    /// A write of the layer's own that went beside the guards, through a
    /// view of its own of the pages (`crate::beside`, [`written_beside`]):
    /// the next check of every guarded view of the pages reports them
    /// written, the bytes are there, and no view was opened for it.
    #[test]
    fn a_write_beside_the_guards_is_reported_at_every_view_and_opens_none() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        crate::inplace::title_memory_section(section, SIZE as u64, true);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 0), pages(two, 1)]));
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE, None), Some(Vec::new()), "nothing written yet");
        // The bytes first, then the guards are told.
        let place = crate::beside::find(one + BLOCK + 16, 32).expect("a place in a view of our own");
        assert_eq!(place.write(one + BLOCK + 16, &[7; 32]), 32);
        written_beside(one + BLOCK + 16, 32, crate::writes::Cause::Back);
        assert_eq!(protection(one + BLOCK), PAGE_READONLY, "the view written at stays guarded");
        assert_eq!(protection(two + BLOCK), PAGE_READONLY, "and so does the other view of its pages");
        // SAFETY: both views are readable.
        unsafe {
            assert_eq!(std::slice::from_raw_parts((one + BLOCK + 16) as *const u8, 32), [7; 32]);
            assert_eq!(std::slice::from_raw_parts((two + BLOCK + 16) as *const u8, 32), [7; 32]);
        }
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 1)]), "the other view's copy takes them too");
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE, None), Some(Vec::new()));
        // A write of the title's afterwards faults and is seen, as ever.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 0)]));
        crate::beside::section_made(section);
        drop_views(section, &[one, two], SIZE);
    }

    /// Whatever makes a guarded block count as written raises the count of
    /// its memory ([`events`], what `VICTORY_REFRESH_EVENTS` goes by), at
    /// the view written through and at every other view of the pages: a
    /// write of the title's that faults, a write of ours beside the
    /// guards, the memory mapped anew. (Counts are by 4 MiB and other
    /// tests' memory may share them: only that they rise is asked.)
    #[test]
    fn what_makes_a_guarded_block_count_as_written_is_an_event() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        crate::inplace::title_memory_section(section, SIZE as u64, true);
        let seen = |view: usize| events(view, view + SIZE);
        let clean = || {
            guard(one, one + SIZE, None);
            guard(two, two + SIZE, None);
            assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()));
            assert_eq!(guard(two, two + SIZE, None), Some(Vec::new()));
            (seen(one), seen(two))
        };
        // A write of the title's through one view.
        let (a, b) = clean();
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert!(seen(one) > a, "the view written through");
        assert!(seen(two) > b, "the other view of its pages");
        // A write of ours beside the guards.
        let (a, b) = clean();
        let place = crate::beside::find(two + BLOCK + 16, 32).expect("a place in a view of our own");
        assert_eq!(place.write(two + BLOCK + 16, &[9; 32]), 32);
        written_beside(two + BLOCK + 16, 32, crate::writes::Cause::Back);
        assert!(seen(two) > b);
        assert!(seen(one) > a);
        // The memory mapped anew (here: told so).
        let (a, _) = clean();
        title_memory_remapped(one, SIZE);
        crate::inplace::title_memory_view(one, SIZE, section, 0);
        assert!(seen(one) > a);
        // The counts of a range are those of every 4 MiB it touches.
        let far = 0x7_e000_0000usize;
        let before = (events(far, far + 1), events(far, far + 3 * EVENT_SPAN), events(far + EVENT_SPAN, far + 2 * EVENT_SPAN));
        event_range(far + EVENT_SPAN - 1, 2);
        assert_eq!(events(far, far + 1), before.0 + 1);
        assert_eq!(events(far, far + 3 * EVENT_SPAN), before.1 + 2);
        assert_eq!(events(far + EVENT_SPAN, far + 2 * EVENT_SPAN), before.2 + 1);
        assert_eq!(events(far, far), 0);
        crate::beside::section_made(section);
        drop_views(section, &[one, two], SIZE);
    }

    /// An asking says, with its locks held, whether anything was counted
    /// in the range a copy remembers since it was brought up to date, and
    /// what the guards held of the first block they report: what the check
    /// of `VICTORY_REFRESH_EVENTS=2` goes by. (Other tests' memory may
    /// share a count: "nothing counted" is asked until it is seen once.)
    #[test]
    fn an_asking_says_whether_anything_was_counted_and_what_it_found() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let ask = |seen: Seen| {
            let mut look = Look { seen: Some(seen), ..Look::default() };
            assert!(guard_fine(one, one + SIZE, None, &mut look).is_some());
            look
        };
        ask(Seen::default());
        // Nothing happens: the asking finds nothing, and says nothing was
        // counted.
        let still = (0..1000).any(|_| {
            let look = ask(Seen::before(one, one + SIZE));
            assert_eq!(look.found, None);
            assert_eq!(look.absent, None);
            look.quiet
        });
        assert!(still, "an asking with nothing counted before it");
        // A write of the title's through the other view of the pages is
        // counted where the copy looks: no longer quiet, and the asking
        // names the block.
        let seen = Seen::before(one, one + SIZE);
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert!(!seen.quiet());
        let look = ask(seen);
        assert!(!look.quiet && !look.uncounted);
        assert_eq!(look.found.map(|f| f.0), Some((one + BLOCK) / BLOCK));
        assert!(look.other, "a block that was there all along");
        // A range a copy never brought up to date holds nothing.
        assert!(!Seen::default().holds(one, one + SIZE) && !Seen::default().quiet());
        assert!(seen.holds(one + 8, one + SIZE) && !seen.holds(one, one + SIZE + 1));
        drop_views(section, &[one, two], SIZE);
    }

    /// A range with a block noted absent is asked again when the guards
    /// look such blocks up again (here, with no cache: every time): a
    /// block committed with nothing told is found by that asking alone.
    #[test]
    fn a_range_with_absent_blocks_is_asked_again_when_they_are_looked_up() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3,
            PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of two blocks and a pagefile section
        // of two.
        let (p, section) = unsafe {
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
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (2 * BLOCK) as u32, std::ptr::null());
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
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
        };
        map(0);
        let ask = || {
            let mut look = Look::default();
            let written = guard_fine(p, p + 2 * BLOCK, None, &mut look).expect("view memory");
            (written, look)
        };
        ask();
        // Block 1 is not there: the copy must ask again.
        let (written, look) = ask();
        assert_eq!(written, Vec::new());
        assert_eq!(look.absent, Some(EVERY_TIME));
        let seen = Seen::before(p, p + 2 * BLOCK).after(&look);
        assert!(!seen.quiet(), "looked up at every check: asked every time");
        // Within the frames a lookup holds for, it need not; past them it
        // must.
        let far = 0x7_d000_0000usize;
        let held = |stretch: u64| Seen { absent: Some(stretch), ..Seen::before(far, far + BLOCK) };
        assert!((0..1000).any(|_| held(absent_stretch()).quiet()));
        assert!(!held(absent_stretch() + (1 << 40)).quiet());
        // Past them nothing need have been counted: such a range is asked
        // for its absent blocks' sake alone.
        assert!((0..1000).any(|_| held(absent_stretch() + (1 << 40)).uncounted()));
        assert!(!seen.uncounted(), "a range asked every time has no counts to go by");
        // Mapped with nothing told: the asking finds it, and says how.
        map(1);
        let (written, look) = ask();
        assert_eq!(written, vec![((p + BLOCK) / BLOCK, FULL)]);
        assert_eq!(look.found.map(|f| f.2), Some("noted absent, and there now with nothing told of it"));
        assert!(!look.other, "nothing but the block that had been absent");
        assert_eq!(look.absent, None);
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: the two views mapped above, and the section's handle.
        unsafe {
            use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: p as *mut _ });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: (p + BLOCK) as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A read of the kernel's into guarded memory: the blocks are opened
    /// for it; one a check guards again while the kernel is still writing
    /// counts as written once more when the read says it is done, at every
    /// view of its pages.
    #[test]
    fn a_block_guarded_again_under_a_kernel_read_counts_as_written_at_its_end() {
        use std::sync::atomic::Ordering::Relaxed;
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        guard(one, one + SIZE, None);
        guard(two, two + SIZE, None);
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE, None), Some(Vec::new()));
        // The read begins; a check comes before it is done and takes the
        // block as it is.
        writing(one + BLOCK, 64);
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 1)]));
        let before = KERNEL_WROTE[2].load(Relaxed);
        // The read is done: both views' copies take the block again.
        kernel_wrote(one + BLOCK, 64);
        assert!(KERNEL_WROTE[2].load(Relaxed) > before, "counted as a block guarded under the read");
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![pages(two, 1)]));
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()));
        // A read into a block no check touched meanwhile marks nothing new:
        // the block is open, and its next check reports it as ever.
        writing(one, 64);
        kernel_wrote(one, 64);
        assert_eq!(guard(one, one + SIZE, None), Some(vec![pages(one, 0)]));
        drop_views(section, &[one, two], SIZE);
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
