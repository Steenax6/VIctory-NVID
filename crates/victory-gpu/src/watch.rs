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
    // (With it: what the landings took of the guards' marks.)
    landed_views_line(frames);
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
    // (Read without a lock: a range left in wrongly fails its asking and is
    // asked by halves, one left out wrongly is asked by `writes`.)
    std::borrow::Cow::Owned(ranges.iter().copied().filter(|r| !held(r.0 / BLOCK)).collect())
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

/// [`take_landed`]'s twin for view memory, which the kernel does not
/// write-watch: for a landing that wrote `ranges` of title memory beside
/// the guards and has told them (`crate::beside`, [`written_beside`]).
/// Takes the guards' marks of the blocks there now, and notes the pages
/// they name as written at an asking of its own, whose number is returned
/// with the pages noted (each with what the journal now holds of it, in
/// page order).
///
/// What it is for: the landing's mark counts its blocks written, the next
/// refresh's check reported them, and the landing's own copy read the
/// kernel pages it had just been given and found them equal: the half of
/// the echo that lies in view memory ([`take_landed`] has dealt with the
/// half in private memory). Taken here, right behind the telling and with
/// the journal's lock held as every asking holds it, the marks stand for
/// "written up to now": the note tells every other copy over the pages,
/// exactly as the next check would have, and the landing's copy can
/// compare each page with its own bytes and count it taken if they are the
/// same.
///
/// A block is taken only if a check could note it clean without a call:
/// guarded, counted written in kernel pages that are known (a mark's; a
/// block counted written anywhere is its next check's), not open from a
/// fault's page on, no opening of it under way, and no view of its pages
/// become writable since it was last clean ([`Region::apart`]). So the
/// rule of [`guard_fine`] holds for the block noted clean here: it is
/// read-only in full at every view. A store of the title's since the
/// landing's write has faulted and opened the block in part, or another
/// view of its pages: either leaves it to the next check. A write told
/// since (another landing's, a read's) is in the marks taken, noted for
/// every copy, and in the bytes the landing's copy compares. A store after
/// this faults, as after a check. No check is between its notes and its
/// changes of protection while this runs: each is made with the
/// journal's lock held throughout.
pub(crate) fn take_landed_views(ranges: &[(usize, usize)]) -> (u64, Vec<(usize, Written)>) {
    let mut j = journal();
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in `wrote`: an asking that stands moves with the note.
    if j.kept {
        j.ahead_asked = now;
    }
    // The blocks that may be taken, read without a lock; decided again
    // with their regions locked.
    let takable = |r: &Region| r.protected.get() & !r.clean.get() & !r.partly.get() & !r.apart.get() & !r.absent.get();
    let mut blocks: Vec<usize> = Vec::new();
    for &(start, end) in ranges {
        each_region(start / BLOCK, end.div_ceil(BLOCK), |index, r, want| {
            blocks.extend(each_bit(want & takable(r)).map(|i| index * REGION + i));
        });
    }
    blocks.sort_unstable();
    blocks.dedup();
    let mut noted = Vec::new();
    if blocks.is_empty() {
        return (now, noted);
    }
    let mut l = Locked::blocks(Holder::Landing, blocks.iter().copied());
    for &block in &blocks {
        let Some((r, m, i)) = l.at(block) else {
            continue;
        };
        let b = 1u64 << i;
        if takable(r) & b != 0 && m.opening[i] == 0 && m.dirty[i] != 0 {
            let mask = std::mem::take(&mut m.dirty[i]);
            r.clean.set(b);
            note(&mut j.written, block, now, mask);
            if let Some(w) = j.written.get_mut(block) {
                noted.push((block, *w));
            }
        }
    }
    (now, noted)
}

/// Since the last frame line ([`landed_views`]): landings that took marks
/// in view memory, the 64 KiB blocks noted by them, and their kernel
/// pages.
static LANDED_VIEWS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// For `crate::shadow::landed`, the one call it makes for view memory.
/// `pieces`: what a landing merged into title memory, as `(copy, start,
/// end)` in title addresses, `copy` a number that is the same for the
/// pieces of one host copy and for no other. Of the pieces that lie in
/// views of the title's physical pages, the guards' marks are taken
/// ([`take_landed_views`]), and `taken` is called once for each copy with
/// pages noted: the copy's number, the pages noted (all of them, in page
/// order), the title memory the landing wrote of that copy, and the
/// asking's number. The copy then compares each noted page with its own
/// bytes and counts it taken, as it does for the kernel's bits in private
/// memory. Pieces in private memory are passed by: `landed` takes the
/// kernel's bits of those itself.
pub(crate) fn landed_views(pieces: impl Iterator<Item = (usize, usize, usize)>, mut taken: impl FnMut(usize, &[(usize, Written)], &[(usize, usize)], u64)) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut views: Vec<(usize, usize, usize)> =
        pieces.filter(|&(_, start, end)| start < end && crate::inplace::view_at(start).is_some()).collect();
    if views.is_empty() {
        return;
    }
    let ranges: Vec<(usize, usize)> = views.iter().map(|&(_, start, end)| (start, end)).collect();
    let (now, noted) = take_landed_views(&ranges);
    if noted.is_empty() {
        return;
    }
    LANDED_VIEWS[0].fetch_add(1, Relaxed);
    LANDED_VIEWS[1].fetch_add(noted.len() as u64, Relaxed);
    LANDED_VIEWS[2].fetch_add(noted.iter().map(|(_, w)| u64::from(w.mask.count_ones())).sum(), Relaxed);
    // Copy by copy, each with what the landing wrote of it.
    views.sort_unstable();
    for of_one in views.chunk_by(|a, b| a.0 == b.0) {
        let runs: Vec<(usize, usize)> = of_one.iter().map(|&(_, start, end)| (start, end)).collect();
        // (Only if a page of it was noted.)
        if runs.iter().any(|&(start, end)| noted.iter().any(|&(page, _)| page * PAGE < end && (page + 1) * PAGE > start)) {
            taken(of_one[0].0, &noted, &runs, now);
        }
    }
}

/// With the frame line ([`events_line`]).
fn landed_views_line(frames: f64) {
    let [takings, blocks, kernel_pages] = [0, 1, 2].map(|i| LANDED_VIEWS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    tracing::info!(
        "landings' marks taken in view memory, per frame: {takings:.1} landings took them, {blocks:.0} blocks of 64 KiB with \
         {kernel_pages:.0} 4 KiB pages ({:.2} MB) noted; what the landings' copies count taken of them and no longer read is in \
         the line of the landings' written bits, with the pages of private memory, and so are the counts of its check \
         (VICTORY_LAND_TAKEN=2)",
        kernel_pages * KERNEL_PAGE as f64 / 1e6,
    );
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
    // While the submission's one asking stands, what the guards found
    // unmapped stays so for it (see [`absent_once`]).
    let epoch = once().then_some(Epoch { submission: j.once_id });
    let check = |look: &mut Look| {
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard_fine(first * PAGE, last * PAGE, epoch, look);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            if written.is_some() {
                crate::copystat::buffers::view(first * PAGE, last.saturating_sub(first).saturating_mul(PAGE));
            }
        }
        written
    };
    // (The guards said so without a lock: if the memory is no view by now,
    // mapped anew in between, the check says that and the kernel is asked
    // as if they had not.)
    let by_guards = if known_view { check(look) } else { None };
    if known_view && by_guards.is_none() {
        look.absent = Some(EVERY_TIME);
    }
    // SAFETY: the answer buffer holds `count` entries.
    let failed = by_guards.is_some()
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
        if by_guards.is_some() {
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
        match by_guards {
            Some(written) => written,
            // Neither watched by the kernel nor a view.
            None if known_view => return None,
            None => check(look)?,
        }
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
///
/// Answered without a lock ([`held`]). A yes that is no longer so when the
/// check comes (the memory was mapped anew in between and is no view now)
/// is found there: the check says the memory is no view, and the kernel is
/// asked after all ([`writes_fine_look`]).
fn known_view(start: usize) -> bool {
    !view_ask() && held(start / BLOCK)
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
/// last mapped or unmapped it. Read without a lock: a hint, for what need
/// not be asked of the kernel ([`known_view`], [`watchable`]) and for
/// where a check need not ask Windows first ([`guard_fine`]); what a check
/// does goes by what it finds with its regions locked.
fn held(block: usize) -> bool {
    region(block / REGION).is_some_and(|r| {
        let known = r.protected.get() | r.absent.get() | if views_known() { r.views.get() } else { 0 };
        known & bit(block) != 0
    })
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

/// The blocks of one region of the guards: 4 MiB of title memory, the span
/// one count of events stands for ([`EVENT_SPAN`]); a bit a block in each
/// of a region's sets.
const REGION: usize = EVENT_SPAN / BLOCK;
const _: () = assert!(REGION == 64);

/// The bit of `block` in its region's sets.
fn bit(block: usize) -> u64 {
    1 << (block % REGION)
}

/// The bits of the blocks `[first, last)` that lie in region `index`.
fn span(index: usize, first: usize, last: usize) -> u64 {
    let lo = first.saturating_sub(index.saturating_mul(REGION)).min(REGION);
    let hi = last.saturating_sub(index.saturating_mul(REGION)).min(REGION);
    if lo >= hi {
        return 0;
    }
    (if hi == REGION { u64::MAX } else { (1u64 << hi) - 1 }) & !((1u64 << lo) - 1)
}

/// The places of the bits set in `bits`, lowest first.
fn each_bit(mut bits: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (bits != 0).then(|| {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            i
        })
    })
}

/// A set of a region's blocks. Written only with the region's lock held;
/// read by anyone, without it.
struct Bits(std::sync::atomic::AtomicU64);

impl Bits {
    const fn new() -> Bits {
        Bits(std::sync::atomic::AtomicU64::new(0))
    }

    fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn has(&self, bit: u64) -> bool {
        self.get() & bit != 0
    }

    fn set(&self, bits: u64) {
        self.0.fetch_or(bits, std::sync::atomic::Ordering::SeqCst);
    }

    fn clear(&self, bits: u64) {
        self.0.fetch_and(!bits, std::sync::atomic::Ordering::SeqCst);
    }
}

/// What the guards hold of one region of title memory ([`REGION`] blocks).
///
/// It was one table for all title memory under one lock, which was also
/// the fault handler's on every thread of the title's. Measured before
/// this (2026-11-26, first room, twelve frames each, runs k_hold_plain and
/// k_hold_thin; `H:*` and `HS:*` of `crate::timeline`): the lock was held
/// 11.5 and 12.3 ms a frame in some 4,600 holds of one to five
/// microseconds (faults 46 to 50 %, each across its change of protection;
/// checks 17 to 19 %; the landing's openings 15 to 17 %; looks 11 to
/// 12 %; markings 7 %), and waited for 15.2 ms a frame without thin
/// recording and 58.2 with it, 44.6 of that on the title's six recording
/// threads: four to five times as long as it was held. No holder kept it
/// long; a waiter that sleeps is woken late on a machine whose processors
/// are all busy, and with one lock every thread that writes guarded
/// memory meets every other.
///
/// So: a lock a region, taken for the blocks an operation changes, in the
/// order of addresses where it needs several ([`Locked`]); no call into
/// the kernel and no walk over other regions under it; and what only asks
/// (is this a view, is this range all clean, how many blocks are counted
/// written) reads the sets without it.
///
/// The sets are read without the lock by design, and what makes that
/// exact is the order in which a writer changes them: a block is noted
/// not clean before the write that makes it so can land (a fault's
/// handler notes it before it gives write access back; a write of ours
/// that goes beside the guards is told after its bytes are there, as
/// before, and is found by the check after the telling). A look that
/// finds a block clean therefore comes before that write, and the next
/// one finds it; a look that finds it not clean takes the region's lock
/// and decides there.
struct Region {
    /// Blocks we made read-only.
    protected: Bits,
    /// Of those, the ones not written since a copy took them.
    clean: Bits,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Bits,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one counts them all
    /// written. A write range can be huge (`fread`), so only these are
    /// looked up.
    aliased: Bits,
    /// Blocks made read-only or noted absent since the runtime last mapped
    /// or unmapped them: view memory ([`views_known`]).
    views: Bits,
    /// Of the guarded ones, those a fault made writable in part (from the
    /// faulting kernel page to the block's end).
    partly: Bits,
    /// Guarded blocks that were counted written because another view of
    /// their pages became writable while they stayed read-only: a view
    /// mapped ([`views_mapped`]), one opened in part by a fault
    /// ([`open_tail`]) or whole for a write of ours ([`open`]), one that
    /// would not be made read-only ([`settle`]). The check that reports
    /// such a block and notes it clean makes the other views read-only
    /// first: see [`guard_fine`].
    apart: Bits,
    /// The openings under way on blocks of the region ([`Marks::opening`],
    /// summed), for a look without its lock ([`openings_done`]). Raised
    /// before the block's guard is taken away and lowered after the block
    /// is writable: a look that finds a block not guarded and this 0 finds
    /// it writable as far as the guards go.
    openings: std::sync::atomic::AtomicU32,
    marks: Mutex<Marks>,
}

/// What a region holds that only the holder of its lock reads.
struct Marks {
    /// For each guarded block counted written since it was last reported:
    /// its kernel pages that may have been, a bit each; 0: anywhere.
    dirty: [u16; REGION],
    /// For each block: the changes of protection that make it writable
    /// which were decided under this lock and are not yet said done (a
    /// fault's, an opening's: the call into the kernel is made with the
    /// lock let go). While there are any, a check reports the block
    /// written and neither notes it clean nor makes it read-only: its own
    /// call could come before the one that opens.
    opening: [u8; REGION],
}

impl Region {
    fn new() -> Region {
        Region {
            protected: Bits::new(),
            clean: Bits::new(),
            absent: Bits::new(),
            aliased: Bits::new(),
            views: Bits::new(),
            partly: Bits::new(),
            apart: Bits::new(),
            openings: std::sync::atomic::AtomicU32::new(0),
            marks: Mutex::new(Marks { dirty: [0; REGION], opening: [0; REGION] }),
        }
    }
}

/// Regions of the title's addresses (the first 1 TiB), by number.
const NEAR: usize = (1 << 40) / EVENT_SPAN;

/// The regions the guards hold of the title's addresses, each made where
/// something is first noted in it and never freed; null: none yet.
static REGIONS: [std::sync::atomic::AtomicPtr<Region>; NEAR] = [const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; NEAR];
/// The first of them made and one past the last: a walk over a long range
/// looks no further.
static MADE: [std::sync::atomic::AtomicUsize; 2] =
    [std::sync::atomic::AtomicUsize::new(usize::MAX), std::sync::atomic::AtomicUsize::new(0)];
/// The regions above them (views the system places, in tests).
static FAR: std::sync::RwLock<BTreeMap<usize, &'static Region>> = std::sync::RwLock::new(BTreeMap::new());
/// Blocks noted absent now, for the guards' line.
static ABSENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The region numbered `index` (blocks `index * REGION..`), if the guards
/// ever noted anything in it.
fn region(index: usize) -> Option<&'static Region> {
    if index < NEAR {
        let made = REGIONS[index].load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: a region is leaked where it is made ([`region_made`]) and
        // never freed.
        unsafe { made.as_ref() }
    } else {
        FAR.read().unwrap_or_else(|e| e.into_inner()).get(&index).copied()
    }
}

/// [`region`], made if it is not there.
fn region_made(index: usize) -> &'static Region {
    use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
    if let Some(r) = region(index) {
        return r;
    }
    if index >= NEAR {
        return FAR.write().unwrap_or_else(|e| e.into_inner()).entry(index).or_insert_with(|| Box::leak(Box::new(Region::new())));
    }
    // (Before it is there: a walk that finds the region finds it inside.)
    MADE[0].fetch_min(index, Relaxed);
    MADE[1].fetch_max(index + 1, Relaxed);
    let new = Box::into_raw(Box::new(Region::new()));
    match REGIONS[index].compare_exchange(std::ptr::null_mut(), new, AcqRel, Acquire) {
        // SAFETY: made just now, and leaked from here on.
        Ok(_) => unsafe { &*new },
        Err(there) => {
            // SAFETY: `new` was made here and given to nobody; `there` is
            // another thread's, leaked as above.
            unsafe {
                drop(Box::from_raw(new));
                &*there
            }
        }
    }
}

/// Calls `f` with every region the guards hold that has blocks in
/// `[first, last)`, in the order of addresses: its number, the region, the
/// bits of those blocks.
fn each_region(first: usize, last: usize, mut f: impl FnMut(usize, &'static Region, u64)) {
    use std::sync::atomic::Ordering::Relaxed;
    if first >= last {
        return;
    }
    let (lo, hi) = (first / REGION, (last - 1) / REGION);
    for index in lo.max(MADE[0].load(Relaxed))..hi.saturating_add(1).min(MADE[1].load(Relaxed)).min(NEAR) {
        if let Some(r) = region(index) {
            f(index, r, span(index, first, last));
        }
    }
    if hi >= NEAR {
        let far: Vec<(usize, &'static Region)> =
            FAR.read().unwrap_or_else(|e| e.into_inner()).range(lo.max(NEAR)..=hi).map(|(&index, &r)| (index, r)).collect();
        for (index, r) in far {
            f(index, r, span(index, first, last));
        }
    }
}

/// Whether every block of `[first, last)` is noted clean, read without a
/// lock (as a check looks first, [`guard_fine`]).
#[cfg(test)]
fn all_clean(first: usize, last: usize) -> bool {
    if first >= last {
        return true;
    }
    (first / REGION..=(last - 1) / REGION).all(|index| {
        let want = span(index, first, last);
        region(index).is_some_and(|r| r.clean.get() & want == want)
    })
}

/// The guarded blocks of `[first, last)`, and those known as views of
/// shared pages, read without a lock (what a caller then does to them it
/// decides again with their regions locked).
fn guarded_and_shared(first: usize, last: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut guarded, mut shared) = (Vec::new(), Vec::new());
    each_region(first, last, |index, r, want| {
        guarded.extend(each_bit(r.protected.get() & want).map(|i| index * REGION + i));
        shared.extend(each_bit(r.aliased.get() & want).map(|i| index * REGION + i));
    });
    (guarded, shared)
}

/// With [`absent_once`]: runs of blocks found wholly uncommitted, first
/// block → end, for the stretch of frames numbered `stretch`
/// ([`absent_stretch`]). A lock of its own, taken last and alone.
struct AbsentRuns {
    runs: BTreeMap<usize, usize>,
    stretch: u64,
    /// The remappings told so far: a run asked of Windows before one of
    /// them and noted after it could hold memory mapped in between.
    remaps: u64,
}

static ABSENT_RUNS: Mutex<AbsentRuns> = Mutex::new(AbsentRuns { runs: BTreeMap::new(), stretch: 0, remaps: 0 });

fn absent_runs() -> std::sync::MutexGuard<'static, AbsentRuns> {
    ABSENT_RUNS.lock().unwrap_or_else(|e| e.into_inner())
}

impl AbsentRuns {
    /// The end of the run of uncommitted blocks from `block` on, as found
    /// in the stretch of frames `stretch`, if it was.
    fn until(&mut self, block: usize, stretch: u64) -> Option<usize> {
        if self.stretch != stretch {
            count(12);
            self.stretch = stretch;
            self.runs.clear();
            return None;
        }
        let found = self.runs.range(..=block).next_back().and_then(|(_, &until)| (until > block).then_some(until));
        if found.is_some() {
            count(11);
        }
        found
    }

    /// The runs that touch blocks `[first, last)` are no longer known
    /// (the runtime mapped or unmapped there); what of them lies outside
    /// stays.
    fn remapped(&mut self, first: usize, last: usize) {
        self.remaps += 1;
        // Runs lie apart: the one that begins at or before `first`, and
        // those that begin inside.
        let before = self.runs.range(..=first).next_back().filter(|&(_, &until)| until > first).map(|(&block, _)| block);
        let inside: Vec<usize> = self.runs.range(first.saturating_add(1)..last).map(|(&block, _)| block).collect();
        for block in before.into_iter().chain(inside) {
            count(13);
            if let Some(until) = self.runs.remove(&block) {
                if block < first {
                    self.runs.insert(block, first);
                }
                if until > last {
                    self.runs.insert(last, until);
                }
            }
        }
    }
}

/// With [`absent_once`]: committed blocks that could not be made read-only
/// in the submission numbered by the first (they are reported written at
/// every check, as before, without trying again). A lock of its own, taken
/// last and alone.
static UNGUARDABLE: Mutex<(u64, BTreeSet<usize>)> = Mutex::new((0, BTreeSet::new()));

/// The blocks that could not be made read-only in `submission`.
fn unguardable_in(submission: u64) -> std::sync::MutexGuard<'static, (u64, BTreeSet<usize>)> {
    let mut held = UNGUARDABLE.lock().unwrap_or_else(|e| e.into_inner());
    if held.0 != submission {
        held.0 = submission;
        held.1.clear();
    }
    held
}

/// Diagnostics, since the last [`guard_line`]: kernel calls the guards
/// made, by what for: `VirtualQuery` for whether a range is a view, for
/// whether absent blocks are still unmapped, after a failed protection;
/// `VirtualProtect` arming runs of blocks, arming blocks one by one (a run
/// that failed), arming other views, opening blocks; guarded blocks whose
/// other views a check looked at again because one had become writable
/// ([`Region::apart`]), other views it then armed again whole; checks that
/// found their range all clean, all
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
    let (runs, absent) = (absent_runs().runs.len(), ABSENT.load(Relaxed));
    tracing::info!(
        "guards per submission: {:.1} checks, {:.1} all clean; VirtualQuery {:.1} is-view, {:.1} absent, {:.1} after a failed \
         protection; VirtualProtect {:.1} runs armed, {:.1} blocks armed one by one, {:.1} other views armed, {:.1} blocks opened; \
         {:.1} guarded blocks reported with a view of their pages become writable, {:.1} such views armed again whole; \
         absent runs: {:.1} found in the cache, cache emptied {:.2} times, {:.1} runs dropped by remappings, {runs} cached, {absent} \
         absent blocks; faults that opened part of a block {:.1}, such blocks armed again {:.1}, {:.2} faults on blocks armed \
         after they were opened; refreshes of copies in the card's memory: {:.1} 4 KiB blocks ({:.2} MB) equal to the CPU side, \
         not sent up (VICTORY_SEND_CHANGED=0 sends them)",
        c[10], c[9], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[11], c[12], c[13], c[14], c[15], c[16], same[0], same[1] / 1e6
    );
    let [waited, found, vain] = [0, 1, 2].map(|i| OPENED_BESIDE[i].load(Relaxed));
    let [stood, stood_vain] = [0, 1].map(|i| OPENINGS_AWAITED[i].load(Relaxed));
    if waited + found + vain + stood > 0 {
        tracing::info!(
            "guards, since the start: {waited} faults on a block another thread was opening waited for it, {found} found it \
             opened by the time they asked, {vain} waited in vain; {stood} openings for a write of ours or the kernel's \
             waited for another thread's opening of a block of theirs, {stood_vain} of them in vain"
        );
    }
}

/// Who takes locks of the guards: a hold is an event of `crate::timeline`
/// under the holder's name ([`Locked`]).
#[derive(Clone, Copy, Debug)]
enum Holder {
    /// A check of a range ([`guard_fine`]): its walk over the blocks, and
    /// what it notes of other views.
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
    /// A landing taking its own marks ([`take_landed_views`]).
    Landing,
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
            Holder::Landing => ("H:landing", "HS:landing"),
        }
    }
}

const _: () = assert!((Holder::Landing as usize) < crate::timeline::HOLDERS);

/// Regions of the guards locked together, always in the order of their
/// addresses: two takers that share regions cannot wait for each other.
/// A wait of 0.2 ms or more for them is an event of `crate::timeline`
/// (`L:guards`), and while it keeps events letting go says how long they
/// were held and by whom.
///
/// Nothing that calls into the kernel, and no lookup in the table of the
/// title's views, is done while one of these is held: the lock is the
/// fault handler's on the title's threads, and what waits for it waits
/// with a store half done.
struct Locked {
    regions: Vec<(usize, &'static Region, std::sync::MutexGuard<'static, Marks>)>,
    by: Holder,
    since: Option<std::time::Instant>,
    /// A number of the holder's own for the event (the blocks a check
    /// walked).
    n: u64,
}

impl Locked {
    /// Locks the regions numbered `indices` (in any order, with repeats),
    /// making those the guards do not hold yet.
    fn take(by: Holder, mut indices: Vec<usize>) -> Locked {
        indices.sort_unstable();
        indices.dedup();
        let regions = crate::timeline::waited("L:guards", || {
            indices
                .iter()
                .map(|&index| {
                    let r = region_made(index);
                    (index, r, r.marks.lock().unwrap_or_else(|e| e.into_inner()))
                })
                .collect()
        });
        Locked { regions, by, since: crate::timeline::clock(), n: 0 }
    }

    /// [`Locked::take`] for the regions of `blocks`.
    fn blocks(by: Holder, blocks: impl IntoIterator<Item = usize>) -> Locked {
        Locked::take(by, blocks.into_iter().map(|block| block / REGION).collect())
    }

    /// The region of `block`, what its lock keeps and the block's place in
    /// it, if the region is among those locked.
    fn at(&mut self, block: usize) -> Option<(&'static Region, &mut Marks, usize)> {
        let k = self.regions.binary_search_by_key(&(block / REGION), |held| held.0).ok()?;
        let (_, r, marks) = &mut self.regions[k];
        Some((*r, &mut **marks, block % REGION))
    }
}

impl Drop for Locked {
    fn drop(&mut self) {
        self.regions.clear();
        let (long, sum) = self.by.events();
        crate::timeline::held(self.by as usize, long, sum, self.since, self.n);
    }
}

/// [`guard_fine`] without the kernel pages: the pages written.
#[cfg(test)]
fn guard(start: usize, end: usize, epoch: Option<Epoch>) -> Option<Vec<usize>> {
    guard_fine(start, end, epoch, &mut Look::default()).map(|w| w.into_iter().map(|(page, _)| page).collect())
}

/// What a guarded block that a check reports was to the guards.
const WAS_WRITTEN: &str = "guarded and counted written (a fault, a write of ours, or guarded as another view of its pages)";
/// What a block that a check reports and no guard holds was to them.
const WAS_OPEN: &str = "not guarded (opened, mapped anew, never taken, or not to be guarded)";

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
/// `epoch`: the submission the check is made for, with [`once`]
/// ([`absent_once`]). `look`: see [`Look`].
///
/// In four steps, of which only the first and the third hold locks, and
/// those only of the regions they change:
/// 1. With the range's regions locked: every block not clean is reported.
///    One that is guarded is noted clean (its kernel pages counted written
///    taken); one known as view memory and not guarded is noted guarded
///    and clean; of the rest it is noted down what must be asked.
/// 2. With no lock: Windows says whether blocks noted absent are still
///    not committed and what kind of memory the blocks not known yet are;
///    the table of the title's views says which other views show the
///    pages of the blocks noted clean.
/// 3. With those blocks' regions and their other views' locked: what
///    step 2 learned is noted, and the other views are noted guarded, or
///    to be made read-only whole again.
/// 4. With no lock: the changes of protection ([`Deferred`]), and what
///    failed of them undone ([`settle`]).
///
/// The caller's copy reads after step 4.
///
/// Why the steps may be apart. A block is noted clean in step 1 while it,
/// or another view of its pages, is still writable; a write that lands
/// before step 4 has made them read-only is one before the copy's read,
/// and one after it faults (the block's own handler finds it guarded; a
/// view's finds it guarded and known as a view of shared pages since
/// step 3, and counts this block written). A fault's handler that noted a
/// block before step 1 and has not yet made its call into the kernel is
/// seen there ([`Marks::opening`]): such a block is reported and left as
/// it is, and one whose other view is being opened so is counted written
/// again in step 3. What the runtime maps or unmaps between the steps is
/// told, and takes out what the steps noted; the call of step 4 may then
/// make a new view read-only, which its first write's fault gives back
/// ([`reopen_stray`]). What Windows said in step 2 of blocks noted absent
/// is kept only if no mapping was told since step 1 ([`AbsentRuns`]).
///
/// The rule kept at every check: **while a block is noted clean, no view
/// of its pages is writable** (not one of up to [`MANY_VIEWS`]), and each
/// is known as a view of shared pages, so that a fault on one counts the
/// others written. A write through a writable view faults nowhere, and
/// the journal goes by address: a clean block whose pages can be written
/// elsewhere unseen is a copy that keeps old bytes for good (the title
/// maps streamed pages into its pool, has them shown in the GPU's address
/// space as well, and writes through the pool's view). A block guarded
/// for the first time has its other views guarded with it; one reported
/// while still guarded has them looked at again if one of them became
/// writable since ([`Region::apart`]): one not guarded is guarded (not
/// clean: its own copies have not taken the pages), one a fault opened in
/// part is made read-only whole again and stays counted written.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
/// One view (nothing else to guard), or the page the title shows wherever
/// it has nothing (thousands of views): each view of that one is guarded
/// by itself.
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
    if (ask || !held(start / BLOCK)) && !is_view(start, end) {
        return None;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    look.absent = None;
    if first >= last {
        return Some(Vec::new());
    }
    // A frame asks about some 10 GB of view memory, 150,000 blocks, nearly
    // all clean: a word a region, read without a lock. The regions with a
    // block that is not clean are locked below, and no others: a block
    // found clean here is one whose next write the next look finds (see
    // [`Region`]). The regions the guards hold nothing of are all to look
    // at.
    let (mut marked, mut missing): (Vec<usize>, Vec<usize>) = Default::default();
    for index in first / REGION..=(last - 1) / REGION {
        let want = span(index, first, last);
        match region(index) {
            None => missing.push(index),
            Some(r) if r.clean.get() & want != want => marked.push(index),
            Some(_) => {}
        }
    }
    if marked.is_empty() && missing.is_empty() {
        // Nothing written, mapped or new since the range was last taken.
        // (What the guards report is what the look below stands for, and
        // here they report nothing; the caller holds the journal's lock,
        // under which every note of the journal's is counted.)
        look.look();
        count(9);
        return Some(Vec::new());
    }
    // Unmapped blocks found so stay so for some frames ([`absent_once`]);
    // blocks that would not be protected, for the submission.
    let absent_epoch = epoch.filter(|_| absent_once()).map(|_| absent_stretch());
    let unguardable_epoch = epoch.filter(|_| absent_once()).map(|e| e.submission);
    let mut written: Vec<(usize, u16)> = Vec::new();
    let mut later = Deferred::default();
    // Blocks noted clean whose pages' other views are to be looked at; the
    // blocks noted absent that Windows must be asked about; the blocks the
    // guards know nothing of.
    let (mut together, mut asked, mut unknown): (Vec<usize>, Vec<usize>, Vec<usize>) = Default::default();
    let remaps;
    {
        let mut l = Locked::take(Holder::Check, marked.clone());
        l.n = (last - first) as u64;
        // With these locks and the journal's (the caller's) nothing is
        // counted for these regions until the step is through: what it
        // finds is what the counts stand for.
        look.look();
        let unguardable = unguardable_epoch.map(unguardable_in);
        let mut runs = absent_runs();
        remaps = runs.remaps;
        for &index in &missing {
            for block in each_bit(span(index, first, last)).map(|i| index * REGION + i) {
                written.push((block, FULL));
                look.found(block, FULL, WAS_OPEN, false);
                if !unguardable.as_ref().is_some_and(|u| u.1.contains(&block)) {
                    unknown.push(block);
                }
            }
        }
        let mut uncommitted_until = first;
        for &index in &marked {
            let (want, base) = (span(index, first, last), index * REGION);
            let Some((r, m, _)) = l.at(base) else {
                continue;
            };
            for i in each_bit(want & !r.clean.get()) {
                let (block, b) = (base + i, 1u64 << i);
                if r.absent.has(b) {
                    // Ordinary commits need no runtime mapping notification:
                    // whether it is still absent is in the cache of runs
                    // found so, or asked of Windows in the next step.
                    if block >= uncommitted_until {
                        match absent_epoch.and_then(|e| runs.until(block, e)) {
                            Some(until) => uncommitted_until = until,
                            None => {
                                asked.push(block);
                                continue;
                            }
                        }
                    }
                    // Still absent, by a lookup that holds for some frames:
                    // see [`Seen::quiet`].
                    look.absent = Some(absent_epoch.unwrap_or(EVERY_TIME));
                    continue;
                }
                if r.protected.has(b) {
                    if m.opening[i] != 0 {
                        written.push((block, FULL));
                        look.found(block, FULL, WAS_WRITTEN, false);
                        continue;
                    }
                    // Written through the runtime or another view, still
                    // read-only; or opened by a fault from its kernel page
                    // on ([`guard_pages`]), armed again whole in step 4.
                    // Reported with the kernel pages that may have been
                    // written.
                    let mask = if m.dirty[i] == 0 { FULL } else { m.dirty[i] };
                    m.dirty[i] = 0;
                    written.push((block, mask));
                    look.found(block, mask, WAS_WRITTEN, false);
                    if r.partly.has(b) {
                        r.partly.clear(b);
                        later.rearm.push(block);
                    }
                    if r.apart.has(b) {
                        r.apart.clear(b);
                        count(7);
                        together.push(block);
                    }
                    r.clean.set(b);
                    continue;
                }
                written.push((block, FULL));
                look.found(block, FULL, WAS_OPEN, false);
                if m.opening[i] != 0 || unguardable.as_ref().is_some_and(|u| u.1.contains(&block)) {
                    // Being opened for a write, or could not be made
                    // read-only in this submission: reported written at
                    // every check.
                } else if r.views.has(b) {
                    // Known as view memory (made read-only or noted absent
                    // since the runtime last mapped or unmapped it): no
                    // question to Windows.
                    note_guarded(r, m, i, block);
                    later.runs.push(block);
                    together.push(block);
                } else {
                    unknown.push(block);
                }
            }
        }
    }
    // Step 2. Blocks noted absent: one question for each run of them.
    let (mut arrived, mut still): (Vec<usize>, Vec<(usize, usize)>) = Default::default();
    let mut until = 0;
    for &block in &asked {
        if block < until {
            continue;
        }
        count(1);
        let end = uncommitted_end(block);
        if end > block {
            still.push((block, end));
            until = end;
        } else {
            arrived.push(block);
        }
    }
    // Blocks not known: one question for each run of one kind of memory.
    let mut guardable: Vec<usize> = Vec::new();
    unknown.sort_unstable();
    for (from, to) in block_runs(&unknown) {
        let mut at = from;
        while at < to {
            let (end, view) = view_run(at, to);
            if view {
                guardable.extend(at..end);
            } else if let Some(submission) = unguardable_epoch {
                // Committed and not ours to protect: reported written at
                // every asking, and not tried again in this submission.
                unguardable_in(submission).1.extend(at..end);
            }
            at = end;
        }
    }
    let others = if view_guard() { crate::inplace::others_of(together.iter().chain(&guardable).copied()) } else { Vec::new() };
    // Step 3.
    if !(arrived.is_empty() && still.is_empty() && guardable.is_empty() && others.is_empty()) {
        let mut l = Locked::blocks(
            Holder::Check,
            arrived.iter().chain(&guardable).copied().chain(others.iter().flat_map(|(block, views)| std::iter::once(*block).chain(views.iter().copied()))),
        );
        for &block in &arrived {
            let Some((r, _, i)) = l.at(block) else {
                continue;
            };
            if r.absent.has(1 << i) {
                // There now, and nothing told of it (a mapping the runtime
                // tells takes the block out of the absent ones): no event
                // was counted for this. Reported; guarded by the next
                // check, which finds it known as view memory.
                r.absent.clear(1 << i);
                ABSENT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                written.push((block, FULL));
                look.found(block, FULL, "noted absent, and there now with nothing told of it", true);
            }
        }
        if !still.is_empty() {
            look.absent = Some(absent_epoch.unwrap_or(EVERY_TIME));
            if let Some(stretch) = absent_epoch {
                let mut runs = absent_runs();
                if runs.remaps == remaps && runs.stretch == stretch {
                    runs.runs.extend(still.iter().copied());
                }
            }
        }
        for &block in &guardable {
            let Some((r, m, i)) = l.at(block) else {
                continue;
            };
            let b = 1u64 << i;
            if !(r.protected.has(b) || r.absent.has(b) || m.opening[i] != 0) {
                r.views.set(b);
                note_guarded(r, m, i, block);
                later.runs.push(block);
            }
        }
        for (block, views) in &others {
            // (Not if it was opened or mapped anew since step 1: nothing
            // is noted clean there now.)
            if !l.at(*block).is_some_and(|(r, _, i)| {
                let guarded = r.protected.has(1 << i);
                if guarded {
                    r.aliased.set(1 << i);
                }
                guarded
            }) {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut all = true;
            for &view in views {
                let Some((r, m, i)) = l.at(view) else {
                    continue;
                };
                let b = 1u64 << i;
                r.aliased.set(b);
                if m.opening[i] != 0 {
                    // A fault or a write of ours is opening it just now.
                    all = false;
                } else if !r.protected.has(b) {
                    // Not clean: its copies have not taken these pages.
                    r.protected.set(b);
                    r.views.set(b);
                    m.dirty[i] = 0;
                    later.aliases.push(view);
                } else if r.partly.has(b) {
                    // Open from a fault's page on: whole again, and still
                    // counted written where it was.
                    r.partly.clear(b);
                    count(8);
                    later.rearm.push(view);
                }
            }
            if !all && let Some((r, m, i)) = l.at(*block) {
                // A view of its pages stays writable: not clean after all,
                // and looked at again by the next check.
                note_dirty(r, m, i, *block, FULL);
                r.apart.set(1 << i);
            }
        }
    }
    // Step 4.
    if !later.is_empty() {
        let failed = later.apply();
        if !failed.is_empty() {
            settle(&failed, epoch);
        }
    }
    written.sort_unstable();
    #[cfg(feature = "diagnostics")]
    for &(block, _) in &written {
        history(block, "reported");
    }
    Some(written)
}

/// Notes the block at `i` of `r` guarded and clean (a check's, which makes
/// it read-only before its copy reads it).
fn note_guarded(r: &Region, m: &mut Marks, i: usize, block: usize) {
    let b = 1u64 << i;
    r.protected.set(b);
    r.views.set(b);
    r.partly.clear(b);
    r.apart.clear(b);
    m.dirty[i] = 0;
    r.clean.set(b);
    history(block, "guarded");
}

/// The changes of protection a check decided, made with no lock of the
/// guards held: blocks to make read-only (noted guarded and clean
/// already), other views of their pages (noted guarded), and blocks a
/// fault opened in part to make read-only whole again.
///
/// A check held the lock across every `VirtualProtect` it made (and the
/// lookups of a block's other views), and the lock is the fault handler's:
/// the title's threads that wrote a guarded block waited for it, 15 ms a
/// frame on the recording threads (round 5's profile). A check notes the
/// blocks guarded first (so that a fault on one once it is read-only
/// finds it the guards'), lets the locks go, makes the calls, and takes
/// locks again only for those that failed. A write landing on a block
/// between the note and the call is one before the check's copy reads the
/// block, and the check reports the block written into the journal, whose
/// lock it holds throughout (every other copy learns of it there). (The
/// calls under the lock were `VICTORY_GUARD_UNLOCKED=0`, gone with the
/// blocks left open, `VICTORY_HOT_BLOCKS`, which were never run.)
#[derive(Default)]
struct Deferred {
    runs: Vec<usize>,
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

    /// Makes the calls: a run of blocks at a time (each change of
    /// protection is a call into the kernel that flushes the address's
    /// translations on every processor running the process), a block at a
    /// time where a run fails (it spans views).
    fn apply(mut self) -> Failed {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        let set = |from: usize, to: usize| {
            let mut old = 0;
            // SAFETY: title view memory the guards hold; only its
            // protection changes.
            unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) != 0 }
        };
        let mut failed = Failed::default();
        self.runs.sort_unstable();
        self.runs.dedup();
        for (from, to) in block_runs(&self.runs) {
            count(if to - from > 1 { 3 } else { 4 });
            if set(from, to) {
                continue;
            }
            for block in from..to {
                if to - from > 1 {
                    count(4);
                }
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

/// Undoes the notes of the blocks whose protection could not be changed
/// ([`Deferred::apply`]): one that is not committed (unmapped) is noted
/// absent, any other is not tried again in this submission. A view that
/// would not be made read-only, or read-only whole again, is no longer
/// guarded, and the guarded views of its pages count as written and apart
/// ([`Region::apart`]): the check that asked for it noted its own block
/// clean.
fn settle(failed: &Failed, epoch: Option<Epoch>) {
    let runs: Vec<(usize, bool)> = failed
        .runs
        .iter()
        .map(|&block| {
            count(2);
            (block, block < uncommitted_end(block))
        })
        .collect();
    let others = if view_guard() { crate::inplace::others_of(failed.aliases.iter().chain(&failed.rearm).copied()) } else { Vec::new() };
    let mut l = Locked::blocks(
        Holder::Settle,
        failed.runs.iter().chain(&failed.aliases).chain(&failed.rearm).copied().chain(others.iter().flat_map(|(_, views)| views.iter().copied())),
    );
    for &(block, absent) in &runs {
        let Some((r, m, i)) = l.at(block) else {
            continue;
        };
        unguard(r, m, i, block);
        if absent {
            if !r.absent.has(1 << i) {
                r.absent.set(1 << i);
                ABSENT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            history(block, "noted absent");
        } else {
            r.views.clear(1 << i);
            if let Some(e) = epoch.filter(|_| absent_once()) {
                unguardable_in(e.submission).1.insert(block);
            }
        }
    }
    for &block in failed.aliases.iter().chain(&failed.rearm) {
        if let Some((r, m, i)) = l.at(block) {
            unguard(r, m, i, block);
        }
    }
    for (_, views) in &others {
        for &view in views {
            if mark_block(&mut l, view, FULL)
                && let Some((r, _, i)) = l.at(view)
            {
                r.apart.set(1 << i);
            }
        }
    }
}

/// An opening of the block at `i` of `r` begins: noted with the region's
/// lock held, before anything else of the block changes
/// ([`Region::openings`]).
fn opening_begins(r: &Region, m: &mut Marks, i: usize) {
    if m.opening[i] < u8::MAX {
        r.openings.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        m.opening[i] += 1;
    }
}

/// The opening is said done (the region's lock held): its change of
/// protection has been made.
fn opening_ends(r: &Region, m: &mut Marks, i: usize) {
    if m.opening[i] > 0 {
        m.opening[i] -= 1;
        r.openings.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The block at `i` of `r` is no longer guarded (opened, or its protection
/// failed): counted, as everything is that changes what a check finds.
fn unguard(r: &Region, m: &mut Marks, i: usize, block: usize) {
    let b = 1u64 << i;
    r.protected.clear(b);
    r.clean.clear(b);
    r.partly.clear(b);
    r.apart.clear(b);
    m.dirty[i] = 0;
    event(block.wrapping_mul(BLOCK));
    history(block, "unguarded");
}

/// What became of a write's fault ([`open_tail`], [`reopen_stray`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    /// The block is writable where the write goes: once more.
    Opened,
    /// No block of the guards' to open.
    No,
    /// The block was guarded between the looks: [`open_tail`] opens it.
    Again,
}

/// For a write that faulted at `at` on a block of view memory the guards
/// know but do not hold guarded: if the block is read-only (a check made
/// it so after something opened it), makes it writable and notes it
/// written.
///
/// Known to the guards, or lying in a view the runtime told of
/// (`crate::inplace::view_at`). A check notes its blocks guarded, lets
/// its locks go and then changes their protection; a mapping of the
/// runtime's told in between drops all the guards knew of the blocks, and
/// the check's call then makes the new view read-only with nothing left
/// to say it was ours. The title's first write there faulted, found no
/// guard, and went on as an access violation of the title's. The runtime
/// maps the title's views read-write and nothing but a guard makes one
/// read-only, so a read-only block of such a view is a guard's.
///
/// The guarded views of its pages count as written and apart from then on
/// ([`Region::apart`]): it is writable and no guard's.
fn reopen_stray(at: usize) -> Fault {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
    };
    let block = at / BLOCK;
    if !(region(block / REGION).is_some_and(|r| r.views.has(bit(block))) || crate::inplace::view_at(at).is_some()) {
        return Fault::No;
    }
    let others = other_views(block);
    {
        let mut l = Locked::blocks(Holder::Fault, std::iter::once(block).chain(others.iter().copied()));
        let Some((r, m, i)) = l.at(block) else {
            return Fault::No;
        };
        if r.protected.has(1 << i) {
            return Fault::Again;
        }
        // (No check guards it while this stands.)
        opening_begins(r, m, i);
        for &view in &others {
            if mark_block(&mut l, view, FULL)
                && let Some((r, _, i)) = l.at(view)
            {
                r.apart.set(1 << i);
            }
        }
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    let mut old = 0;
    let opened = n != 0
        && info.State == MEM_COMMIT
        && info.Protect == PAGE_READONLY
        // SAFETY: a block of title view memory we made read-only.
        && unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) } != 0;
    let mut l = Locked::blocks(Holder::Fault, [block]);
    if let Some((r, m, i)) = l.at(block) {
        opening_ends(r, m, i);
    }
    if !opened {
        return Fault::No;
    }
    count(16);
    event(block.wrapping_mul(BLOCK));
    Fault::Opened
}

/// The blocks at which the other views of `block`'s pages show them (none
/// for a page of one view, or of more than [`MANY_VIEWS`]).
fn other_views(block: usize) -> Vec<usize> {
    if !view_guard() {
        return Vec::new();
    }
    crate::inplace::others_of([block]).pop().map_or_else(Vec::new, |(_, views)| views)
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
/// notes them written, for a write of ours or the kernel's that is about
/// to go there; how many were guarded. The same physical pages at every
/// other view of them count as written there: the write goes through one
/// view, and every guard over the pages must copy them again.
///
/// The other views stay read-only (a write through one of them faults in
/// its turn, and opens it): they only count as written, which their next
/// check takes without a call, and that check guards the opened view
/// again ([`Region::apart`]). Opened, a view was made read-only again at
/// its next check: two calls that change protection (each flushing the
/// address on every processor running the process), for a block whose
/// bytes need only be copied again. In the intro a write opened its
/// block's three or four other views too, and the re-arming was most of
/// the submitting thread's 12 ms a frame of `VirtualProtect` (round 2:
/// 172 other views armed a submission against 37 blocks). (Opening them
/// all was `VICTORY_MARK_ONLY=0`; gone: what keeps a clean block's other
/// views read-only goes by the marks.)
///
/// A block opened here is writable anywhere from now on, whatever part of
/// it the write is for: its pages count as written anywhere at their
/// other views. A block already open counts for the kernel pages of this
/// write.
///
/// The blocks are noted open with their regions locked, and made writable
/// after the locks are let go; until that is said done no check guards
/// them again ([`Marks::opening`]).
fn open(by: Holder, start: usize, end: usize) -> usize {
    if start >= end {
        return 0;
    }
    let opened = open_own(by, start, end);
    // What another thread is opening there is writable when this returns
    // as well: the write this is for may be the kernel's, which no fault
    // of ours answers.
    openings_done(by, start / BLOCK, end.div_ceil(BLOCK));
    opened
}

/// [`open`] for the blocks this call finds guarded.
fn open_own(by: Holder, start: usize, end: usize) -> usize {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let (guarded, shared) = guarded_and_shared(start / BLOCK, end.div_ceil(BLOCK));
    if guarded.is_empty() && shared.is_empty() {
        return 0;
    }
    let others = if view_guard() { crate::inplace::others_of(shared.iter().copied()) } else { Vec::new() };
    let mut opened: Vec<usize> = Vec::new();
    {
        let mut l = Locked::blocks(
            by,
            guarded.iter().copied().chain(others.iter().flat_map(|(block, views)| std::iter::once(*block).chain(views.iter().copied()))),
        );
        for &block in &guarded {
            let Some((r, m, i)) = l.at(block) else {
                continue;
            };
            if r.protected.has(1 << i) {
                count(6);
                opening_begins(r, m, i);
                unguard(r, m, i, block);
                opened.push(block);
            }
        }
        for (block, views) in &others {
            let mask = if opened.binary_search(block).is_ok() { FULL } else { range_mask(*block, start, end) };
            for &view in views {
                if mark_block(&mut l, view, mask)
                    && let Some((r, _, i)) = l.at(view)
                {
                    // (This view is writable from now on.)
                    r.apart.set(1 << i);
                    #[cfg(feature = "diagnostics")]
                    ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
    if opened.is_empty() {
        return 0;
    }
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: blocks we made read-only.
        unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READWRITE, &mut old) != 0 }
    };
    for (from, to) in block_runs(&opened) {
        if !set(from, to) && to - from > 1 {
            // (A run across views: a block at a time.)
            for block in from..to {
                set(block, block + 1);
            }
        }
    }
    let mut l = Locked::blocks(by, opened.iter().copied());
    for &block in &opened {
        if let Some((r, m, i)) = l.at(block) {
            opening_ends(r, m, i);
        }
    }
    opened.len()
}

/// Since the start: calls of [`open`] that waited for another thread's
/// opening of a block of theirs; those that waited [`BESIDE_WAIT`] in vain.
pub(crate) static OPENINGS_AWAITED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Waits until no opening is under way on a block of `[first, last)`.
///
/// A block another thread has noted open and not yet made writable is no
/// guard's any more: [`open`] finds nothing to do for it. A write of the
/// processor's then faults and is answered ([`opened_beside`]); the
/// kernel's does not fault into a handler, it fails (a read of the title's
/// into its own memory then ends with an error, and the title's loading
/// with it). So the caller stands here until the other thread has said its
/// opening done. Regions with none under way are passed without a lock.
fn openings_done(by: Holder, first: usize, last: usize) {
    use std::sync::atomic::Ordering::{Relaxed, SeqCst};
    let mut began = None;
    loop {
        let mut busy: Vec<(usize, u64)> = Vec::new();
        each_region(first, last, |index, r, want| {
            if r.openings.load(SeqCst) != 0 {
                busy.push((index, want));
            }
        });
        let under_way = busy.into_iter().any(|(index, want)| {
            let mut l = Locked::take(by, vec![index]);
            l.at(index * REGION).is_some_and(|(_, m, _)| each_bit(want).any(|i| m.opening[i] != 0))
        });
        if !under_way {
            return;
        }
        let began = *began.get_or_insert_with(|| {
            OPENINGS_AWAITED[0].fetch_add(1, Relaxed);
            std::time::Instant::now()
        });
        if began.elapsed() > BESIDE_WAIT {
            OPENINGS_AWAITED[1].fetch_add(1, Relaxed);
            return;
        }
        std::thread::yield_now();
    }
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
/// read-only; true if it is guarded (and its region among those locked).
/// A block already counted written anywhere stays so.
fn mark_block(l: &mut Locked, block: usize, mask: u16) -> bool {
    let Some((r, m, i)) = l.at(block) else {
        return false;
    };
    if !r.protected.has(1 << i) {
        return false;
    }
    note_dirty(r, m, i, block, mask);
    true
}

/// Notes that the guarded block at `i` of `r` may have been written in the
/// kernel pages `mask` since its last report.
fn note_dirty(r: &Region, m: &mut Marks, i: usize, block: usize, mask: u16) {
    let b = 1u64 << i;
    if r.clean.has(b) {
        // Not clean first, then the count: an asking that sees the count
        // sees the block.
        r.clean.clear(b);
        event(block.wrapping_mul(BLOCK));
        m.dirty[i] = mask;
        history(block, "counted written");
    } else if m.dirty[i] != 0 {
        // Not clean with pages noted: written there before; with none:
        // written anywhere already.
        m.dirty[i] |= mask;
    }
}

/// Counts the guarded blocks of `[start, end)` written in the kernel pages
/// the range touches, and the same pages at every other view of them,
/// leaving all of them read-only: for a write the guards do not see whose
/// bytes are there (through the runtime's own view, a view of ours, the
/// kernel into a block guarded again). Their next checks report them and
/// note them clean, as one written through the runtime always was, and a
/// later write through them faults as it would have.
fn mark_with_views(by: Holder, start: usize, end: usize) {
    if start >= end {
        return;
    }
    let (guarded, shared) = guarded_and_shared(start / BLOCK, end.div_ceil(BLOCK));
    if guarded.is_empty() && shared.is_empty() {
        return;
    }
    let others = if view_guard() { crate::inplace::others_of(shared.iter().copied()) } else { Vec::new() };
    let mut l = Locked::blocks(by, guarded.iter().copied().chain(others.iter().flat_map(|(_, views)| views.iter().copied())));
    for &block in &guarded {
        mark_block(&mut l, block, range_mask(block, start, end));
    }
    for (block, views) in &others {
        for &view in views {
            if mark_block(&mut l, view, range_mask(*block, start, end)) {
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

/// For a write that faulted at `at`: if its block is guarded, makes its
/// kernel pages from `at`'s on writable ([`guard_pages`]; the whole block
/// without it) and notes them written, and the same pages at the block's
/// other views, which stay read-only.
///
/// In three steps. With the block's region locked, and its other views'
/// if it is known as a view of shared pages (looked up before, with no
/// lock): the pages are noted written, here and at the other views, which
/// are noted apart ([`Region::apart`]: this view is writable from the
/// faulting page on until a check arms it again), and the opening is
/// noted as under way ([`Marks::opening`]). With no lock: the change of
/// protection. With the region locked again: the opening is said done.
/// A check between the first and the last reports the block and leaves it
/// as it is; without that, its own call to make the block read-only again
/// could come before this one's, and the block would stay writable while
/// noted clean.
fn open_tail(at: usize) -> Fault {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let block = at / BLOCK;
    let b = bit(block);
    let Some(r) = region(block / REGION) else {
        return Fault::No;
    };
    let k = if guard_pages() { (at % BLOCK) / KERNEL_PAGE } else { 0 };
    let mask = FULL << k;
    loop {
        if !r.protected.has(b) {
            return Fault::No;
        }
        let shared = r.aliased.has(b);
        let others = if shared { other_views(block) } else { Vec::new() };
        let mut l = Locked::blocks(Holder::Fault, std::iter::once(block).chain(others.iter().copied()));
        let Some((r, m, i)) = l.at(block) else {
            return Fault::No;
        };
        if !r.protected.has(b) {
            return Fault::No;
        }
        if !shared && r.aliased.has(b) {
            // Known as a view of shared pages since the look: once more,
            // with the other views.
            continue;
        }
        note_dirty(r, m, i, block, mask);
        r.partly.set(b);
        opening_begins(r, m, i);
        for &view in &others {
            if mark_block(&mut l, view, mask)
                && let Some((r, _, i)) = l.at(view)
            {
                r.apart.set(1 << i);
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        break;
    }
    let from = block * BLOCK + k * KERNEL_PAGE;
    let mut old = 0;
    count(14);
    // SAFETY: a block we made read-only; only its protection changes.
    let opened = unsafe { VirtualProtect(from as *const _, (block + 1) * BLOCK - from, PAGE_READWRITE, &mut old) } != 0;
    let mut l = Locked::blocks(Holder::Fault, [block]);
    if let Some((r, m, i)) = l.at(block) {
        opening_ends(r, m, i);
        if !opened && r.protected.has(b) {
            // Not to be opened (its memory went meanwhile): no guard's
            // any more. The write faults once more and is the title's.
            unguard(r, m, i, block);
        }
    }
    Fault::Opened
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
    let opened = open(Holder::Opening, at, at.saturating_add(len));
    #[cfg(feature = "diagnostics")]
    crate::copystat::buffers::opened(cause as usize, opened as u64);
    #[cfg(not(feature = "diagnostics"))]
    let _ = opened;
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
/// ([`mark_with_views`]), and the
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
    let guarded = guarded_and_shared(at / BLOCK, end.div_ceil(BLOCK)).0.len() as u64;
    KERNEL_WROTE[0].fetch_add(1, Relaxed);
    KERNEL_WROTE[1].fetch_add(len as u64, Relaxed);
    KERNEL_WROTE[2].fetch_add(guarded, Relaxed);
    if guarded != 0 {
        mark_with_views(Holder::Marking, at, end);
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
/// written and stay read-only ([`mark_with_views`]), so every host copy over
/// them takes them again at its next check, as after [`writing_as`] opened
/// them, without the two changes of protection. Called once the bytes are
/// there: a check between the write and this takes them a second time, one
/// before the write would have noted the block clean with the old bytes.
pub(crate) fn written_beside(at: usize, len: usize, cause: crate::writes::Cause) {
    mark_with_views(Holder::Marking, at, at.saturating_add(len));
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
/// (`crate::census`): the journal's lock, and of the guards a look
/// without one.
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
    for index in first / REGION..=(last - 1) / REGION {
        let want = span(index, first, last);
        let (guarded, clean, absent) = region(index).map_or((0, 0, 0), |r| (r.protected.get(), r.clean.get(), r.absent.get()));
        open += (want & !guarded & !absent).count_ones() as usize;
        written += (want & guarded & !clean).count_ones() as usize;
    }
    // Blocks no guard holds may be write-watched memory: the kernel says
    // whether a page of them was written since its bits were last taken,
    // and leaves the bits (no reset). What another asking took meanwhile
    // is in the journal, above.
    if open > 0
        && let Some(pages) = kernel_dirty(first * BLOCK, (last - first) * BLOCK)
    {
        (written, open) = (written + pages, 0);
    }
    (noted, written, open)
}

/// How many pages of `[at, at + len)` count as written since whatever
/// watches them was last asked, without taking anything: for write-watched
/// memory the pages whose bit the kernel holds (up to a few,
/// [`kernel_dirty`]); for view memory the guards hold, the blocks not
/// noted clean (counted written, open, or not guarded). `None` if the
/// range is neither. For the check of what a landing's taking would have
/// had its copy pass by (`crate::shadow`, `VICTORY_LAND_TAKEN=2`): a page
/// written again since the taking is none of its finds.
pub(crate) fn watched_dirty(at: usize, len: usize) -> Option<usize> {
    if let Some(pages) = kernel_dirty(at, len) {
        return Some(pages);
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    (first < last && held(first)).then(|| {
        (first / REGION..=(last - 1) / REGION)
            .map(|index| {
                let want = span(index, first, last);
                (want & !region(index).map_or(0, |r| r.clean.get())).count_ones() as usize
            })
            .sum()
    })
}

/// How many pages of `[at, at + len)` the kernel's write watch holds as
/// written (up to a few), without taking the bits; `None` if the range is
/// not write-watched memory of one reservation.
fn kernel_dirty(at: usize, len: usize) -> Option<usize> {
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
/// page-table lock; takes the locks of the guards' regions under it, then,
/// alone, the lock of the ranges kept for the texture pass.
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
        let guarded = guarded_and_shared(at / BLOCK, end.div_ceil(BLOCK)).0.len() as u64;
        WRITTEN[2].fetch_add(guarded, Relaxed);
        if mode & 1 != 0 && guarded != 0 {
            // Written through another view: these addresses need not be
            // writable, only copied again.
            mark_with_views(Holder::Marking, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len, pages);
    // The refresh check: what it holds as differing there was under this
    // write (the guards' locks are let go by now).
    crate::shadow::refresh_check_told(at, len);
}

/// What the guards hold of the block of `at`, in words, and of every other
/// view of its pages with what the system says of that view's protection
/// (for the refresh check's warnings: a block "not written since it was
/// taken" that differs from title memory has a view of its pages that is
/// writable, and this names it).
pub(crate) fn guard_state(at: usize) -> String {
    use std::fmt::Write;
    let block = at / BLOCK;
    let state = |block: usize| {
        let Some(r) = region(block / REGION) else {
            return "not known to the guards (private memory, or never taken)";
        };
        let b = bit(block);
        match (r.protected.has(b), r.clean.has(b)) {
            (true, true) => "guarded, not written since it was taken",
            (true, false) if r.partly.has(b) => "guarded, counted written, open from a fault's page on",
            (true, false) => "guarded, counted written",
            _ if r.absent.has(b) => "noted absent",
            _ if r.views.has(b) => "view memory, not guarded now",
            _ => "not known to the guards (private memory, or never taken)",
        }
    };
    let mut out = state(block).to_string();
    let others = crate::inplace::others_of([block]).pop().map_or_else(Vec::new, |(_, views)| views);
    if !others.is_empty() {
        let _ = write!(out, "; {} other views of its pages", others.len());
        for &view in others.iter().take(4) {
            let at = view * BLOCK;
            let _ = write!(out, ", {at:#x}: {}, {}", state(view), if writable(at) { "writable" } else { "not writable" });
        }
    }
    out
}

/// Diagnostics: what last happened to a block, for [`block_report`]: the
/// last few changes of each block (microseconds since the start, and what),
/// newest last. A lock of its own, taken last and alone; at most
/// [`HISTORY_BLOCKS`] blocks are followed, the first ones met.
#[cfg(feature = "diagnostics")]
static HISTORY: Mutex<Option<std::collections::HashMap<usize, [(u64, &'static str); 6]>>> = Mutex::new(None);

#[cfg(feature = "diagnostics")]
const HISTORY_BLOCKS: usize = 1 << 20;

/// Notes in [`HISTORY`] that `what` happened to `block`.
#[cfg(feature = "diagnostics")]
fn history(block: usize, what: &'static str) {
    let at = crate::host::now_us();
    let mut h = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
    let map = h.get_or_insert_with(std::collections::HashMap::new);
    let full = map.len() >= HISTORY_BLOCKS;
    let entry = match map.get_mut(&block) {
        Some(e) => e,
        None if full => return,
        None => map.entry(block).or_insert([(0, ""); 6]),
    };
    entry.rotate_left(1);
    entry[5] = (at, what);
}

#[cfg(not(feature = "diagnostics"))]
fn history(_: usize, _: &'static str) {}

/// Diagnostics, for the report of a removed device: what the guards, the
/// journal and Windows hold of the 64 KiB block at `at` now, for a host
/// copy that differs from title memory there (why the copy can be behind).
/// The region's sets are read without its lock; its marks and the journal
/// are only looked at if their locks can be had at once (the thread that
/// reports may hold them, or another that stands). Windows is asked last,
/// with no lock held.
#[cfg(feature = "diagnostics")]
pub(crate) fn block_report(at: usize) -> String {
    use std::fmt::Write;
    use std::sync::atomic::Ordering::Relaxed;
    let block = at / BLOCK;
    let b = bit(block);
    let mut out = format!("block {:#x}: ", block * BLOCK);
    match region(block / REGION) {
        None => out.push_str("no region of the guards"),
        Some(r) => {
            let sets = [
                ("protected", &r.protected),
                ("clean", &r.clean),
                ("absent", &r.absent),
                ("aliased", &r.aliased),
                ("views", &r.views),
                ("partly", &r.partly),
                ("apart", &r.apart),
            ];
            let on: Vec<&str> = sets.iter().filter(|(_, s)| s.has(b)).map(|(n, _)| *n).collect();
            let _ = write!(out, "sets [{}], region openings {}", on.join(" "), r.openings.load(Relaxed));
            match r.marks.try_lock() {
                Ok(m) => {
                    let i = block % REGION;
                    let _ = write!(out, ", dirty {:#06x} opening {}", m.dirty[i], m.opening[i]);
                }
                Err(_) => out.push_str(", marks locked"),
            }
        }
    }
    let _ = write!(out, "; events of its 4 MiB {}", events(block * BLOCK, block * BLOCK + 1));
    match JOURNAL.try_lock() {
        Ok(j) => {
            let mut entry = None;
            j.written.each(block, block + 1, |_, w| entry = Some(*w));
            let ahead = j.ahead.iter().any(|&(a, e)| a <= block && block < e);
            let _ = write!(
                out,
                "; journal asked {} (ahead {} {}, kept {}, once {}), its entry {}",
                j.asked,
                j.ahead_asked,
                if ahead { "holds it" } else { "not over it" },
                j.kept,
                j.once_id,
                entry.map_or_else(|| "none".to_owned(), |w| format!("at {} before {} mask {:#06x}", w.at, w.before, w.mask))
            );
        }
        Err(_) => out.push_str("; journal locked"),
    }
    match ABSENT_RUNS.try_lock() {
        Ok(runs) => {
            let run = runs.runs.range(..=block).next_back().filter(|r| *r.1 > block);
            let _ = write!(
                out,
                "; absent runs: stretch {} (now {}), remaps {}, {}",
                runs.stretch,
                absent_stretch(),
                runs.remaps,
                run.map_or_else(|| "none over it".to_owned(), |(&f, &u)| format!("run {:#x}..{:#x}", f * BLOCK, u * BLOCK))
            );
        }
        Err(_) => out.push_str("; absent runs locked"),
    }
    if let Ok(u) = UNGUARDABLE.try_lock()
        && u.1.contains(&block)
    {
        let _ = write!(out, "; unguardable in submission {}", u.0);
    }
    if let Ok(h) = HISTORY.try_lock() {
        let now = crate::host::now_us();
        match h.as_ref().and_then(|m| m.get(&block)) {
            Some(e) => {
                let said: Vec<String> =
                    e.iter().filter(|c| !c.1.is_empty()).map(|c| format!("{} {:.1} ms ago", c.1, now.saturating_sub(c.0) as f64 / 1000.0)).collect();
                let _ = write!(out, "; its last changes: {}", said.join(", "));
            }
            None => out.push_str("; no changes followed"),
        }
    }
    out
}

/// Diagnostics: what Windows says of the memory at `at` now (state,
/// protection, type, and the region's extent).
#[cfg(feature = "diagnostics")]
pub(crate) fn query_report(at: usize) -> String {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return "Windows: no answer".to_owned();
    }
    format!(
        "Windows: state {:#x} protect {:#x} type {:#x}, region {:#x}+{:#x}",
        info.State, info.Protect, info.Type, info.BaseAddress as usize, info.RegionSize
    )
}

#[cfg(feature = "diagnostics")]
impl Seen {
    /// Diagnostics: the range remembered, its counts then and now, and the
    /// absent stretch, with what [`Seen::quiet`] says now.
    pub(crate) fn describe(&self) -> String {
        format!(
            "seen {:#x}..{:#x} events then {} now {} absent {:?} (stretch now {}), quiet {}",
            self.from,
            self.to,
            self.events,
            events(self.from, self.to),
            self.absent,
            absent_stretch(),
            self.quiet()
        )
    }
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again: its pages are noted written there, and its next check guards
/// the new view with it ([`guard_fine`]).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    // The guarded blocks of every view of those bytes, with the kernel
    // pages of each the bytes lie in (read without a lock; marked below
    // only if still guarded).
    let mut guarded: Vec<(usize, u16)> = Vec::new();
    for &(at, n) in &ranges {
        let end = at.saturating_add(n);
        let before = guarded.len();
        each_region(at / BLOCK, end.div_ceil(BLOCK), |index, r, want| {
            guarded.extend(each_bit(r.protected.get() & want).map(|i| (index * REGION + i, range_mask(index * REGION + i, at, end))));
        });
        if guarded.len() > before {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if guarded.is_empty() {
        return;
    }
    // Read-only still and counted written; what makes the new view
    // read-only with them is their next check ([`Region::apart`]).
    let mut l = Locked::blocks(Holder::Mapped, guarded.iter().map(|g| g.0));
    for (block, mask) in guarded {
        if mark_block(&mut l, block, mask)
            && let Some((r, _, i)) = l.at(block)
        {
            r.apart.set(1 << i);
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
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    // What a copy knew of this memory is no longer so: counted with the
    // region's lock held, as everything is that changes what a check finds.
    // Counted before the lock was taken, a refresh could look at the
    // counts after this one and be checked before the blocks noted absent
    // here were dropped below (a drop that counts nothing by itself): its
    // copy then remembered counts that held the event and not the change,
    // and was never asked about the memory mapped here. A region at a
    // time, in the order of addresses: a check that comes between two of
    // them finds the one dropped and the other not yet, as it would
    // between two mappings.
    let mut regions = Vec::new();
    each_region(first, last, |index, _, want| regions.push((index, want)));
    for (index, want) in regions {
        let mut l = Locked::take(Holder::Remapped, vec![index]);
        let Some((r, m, _)) = l.at(index * REGION) else {
            continue;
        };
        event(index.wrapping_mul(EVENT_SPAN));
        ABSENT.fetch_sub(u64::from((r.absent.get() & want).count_ones()), std::sync::atomic::Ordering::Relaxed);
        for set in [&r.absent, &r.protected, &r.clean, &r.aliased, &r.views, &r.partly, &r.apart] {
            set.clear(want);
        }
        for i in each_bit(want) {
            m.dirty[i] = 0;
            if len <= 16 << 20 {
                history(index * REGION + i, "remapped");
            }
        }
    }
    // (And the counts of the spans the guards hold nothing of: nothing
    // changes there for a check, whenever it is counted.)
    event_range(at, len);
    // What was found unmapped or not to be protected there is no longer
    // known. Runs elsewhere stay: the runtime maps and unmaps views all the
    // time as the title streams, and clearing them all made the next
    // refresh ask about every run again (round 2: 422 questions a
    // submission).
    absent_runs().remapped(first, last);
    UNGUARDABLE.lock().unwrap_or_else(|e| e.into_inner()).1.clear();
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
        // (More than once only where a check guards the block between two
        // of the looks below: it is the guards' again then, not read-only
        // for some other reason.)
        for _ in 0..8 {
            if open_tail(at) == Fault::Opened {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            // Another thread may have opened the block since this one
            // faulted.
            if writable(at) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            // A block of view memory we guard that is read-only though the
            // guards do not hold it: one a check made read-only after
            // something opened it (the call is made without a lock).
            // Opened and counted written, as a fault on a guarded block is.
            match reopen_stray(at) {
                Fault::Opened => {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
                Fault::Again => {}
                // Not the guards' to open, as far as those looks went: they
                // were made one after the other, and another thread that
                // opens the block makes it writable only after it has noted
                // it open. Waited for, and looked at once more.
                Fault::No => {
                    if !opened_beside(at) {
                        break;
                    }
                }
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

/// Since the start: faults on a block that another thread was opening
/// ([`opened_beside`]) which waited for it; faults that found the block
/// writable by the time they asked with its region locked; faults that
/// waited [`BESIDE_WAIT`] in vain.
pub(crate) static OPENED_BESIDE: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The longest a fault waits for another thread's opening of its block.
const BESIDE_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// For a write that faulted at `at` and found no guard to open
/// ([`open_tail`], [`reopen_stray`]) on memory that was not writable:
/// whether the block is another thread's to open or opened by it, so that
/// the write is to be looked at once more.
///
/// [`open`] and [`open_tail`] note a block open with its region locked,
/// let the lock go and only then make it writable ([`Marks::opening`]). A
/// second thread that writes into the block between the two faults on a
/// block that is no guard's any more and still read-only: its looks found
/// nothing to open, and the write went on as an access violation nobody
/// answered (a thread of the runtime's that fills a read's destination
/// beside another one's, in the loading: the process ended there). Such a
/// fault waits here until the opening is said done.
///
/// With the region locked, a block that is not guarded and has no opening
/// under way has no change of protection of the guards' pending: what the
/// system says of it then stands (asked under the lock for that reason; it
/// is one call, and this is no path a frame takes).
fn opened_beside(at: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let block = at / BLOCK;
    // (Nothing is made for an address the guards never held.)
    if region(block / REGION).is_none() {
        return false;
    }
    let mut began = None;
    loop {
        {
            let mut l = Locked::blocks(Holder::Fault, [block]);
            let Some((r, m, i)) = l.at(block) else {
                return false;
            };
            if r.protected.has(1 << i) {
                // Guarded again meanwhile: a fault of the guards'.
                return true;
            }
            if m.opening[i] == 0 {
                let open = writable(at);
                if open {
                    OPENED_BESIDE[usize::from(began.is_none())].fetch_add(1, Relaxed);
                }
                return open;
            }
        }
        let began = *began.get_or_insert_with(std::time::Instant::now);
        if began.elapsed() > BESIDE_WAIT {
            OPENED_BESIDE[2].fetch_add(1, Relaxed);
            return false;
        }
        std::thread::yield_now();
    }
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

    /// Whether `block` is in the set `of` of its region.
    fn is(of: impl Fn(&Region) -> &Bits, block: usize) -> bool {
        region(block / REGION).is_some_and(|r| of(r).has(bit(block)))
    }

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
        // (The stretch of frames the tests' checks are made in, were one
        // made for a submission: a look for another empties the runs.)
        let stretch = absent_stretch();
        {
            let mut runs = absent_runs();
            runs.until(b, stretch);
            runs.runs.insert(b, b + 4);
            runs.runs.insert(b + 10, b + 11);
            runs.runs.insert(b + 20, b + 30);
        }
        title_memory_remapped((b + 2) * BLOCK, BLOCK);
        title_memory_remapped((b + 29) * BLOCK, 2 * BLOCK);
        let mut runs = absent_runs();
        assert_eq!(runs.until(b + 10, stretch), Some(b + 11));
        // The runs are cut around the remapped blocks, not dropped.
        assert_eq!(runs.until(b + 1, stretch), Some(b + 2));
        assert_eq!(runs.until(b + 2, stretch), None);
        assert_eq!(runs.until(b + 3, stretch), Some(b + 4));
        assert_eq!(runs.until(b + 25, stretch), Some(b + 29));
        assert_eq!(runs.until(b + 29, stretch), None);
        for k in [b, b + 3, b + 10, b + 20] {
            runs.runs.remove(&k);
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
        assert!(is(|r| &r.protected, block), "the pages below stay guarded");
        // SAFETY: as above, above the fault: open, no fault.
        unsafe { ((view + 9 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        // SAFETY: as above, below it: faults, opens from there on.
        unsafe { ((view + 2 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, FULL << 2)]));
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(Vec::new()), "armed again whole");
        // SAFETY: as above; faults again: armed whole.
        unsafe { ((view + 15 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, 1 << 15)]));
        // (Told before the view goes, as the runtime tells it: its address
        // may be another test's the moment it is free.)
        title_memory_remapped(view, BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
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
            let mut l = Locked::blocks(Holder::Opening, [block]);
            let (r, m, i) = l.at(block).expect("its region");
            unguard(r, m, i, block);
        }
        // SAFETY: inside our view, read-only now: the write faults.
        unsafe { ((view + 100) as *mut u8).write_volatile(9) };
        // SAFETY: as above.
        assert_eq!(unsafe { ((view + 100) as *const u8).read_volatile() }, 9);
        assert_eq!(guard_fine(view, view + BLOCK, None, &mut Look::default()), Some(vec![(block, FULL)]));
        // (Told before the view goes, as the runtime tells it: its address
        // may be another test's the moment it is free.)
        title_memory_remapped(view, BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
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
        assert!(!is(|r| &r.views, block));
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

    /// A write into a block that another thread has noted open and not
    /// yet made writable ([`open`] between its two steps) waits for that
    /// thread and goes through; it is no access violation of the writer's.
    #[test]
    fn a_write_into_a_block_being_opened_waits_for_the_opening() {
        use std::sync::atomic::Ordering::Relaxed;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
        };
        install();
        // SAFETY: private memory of our own, of two blocks.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        assert_eq!(p % BLOCK, 0);
        let block = p / BLOCK;
        let protect = |how: u32| {
            let mut old = 0;
            // SAFETY: a block of our allocation; only its protection changes.
            assert_ne!(unsafe { VirtualProtect(p as *const _, BLOCK, how, &mut old) }, 0);
        };
        let opening = |begins: bool| {
            let mut l = Locked::blocks(Holder::Opening, [block]);
            let (r, m, i) = l.at(block).expect("its region");
            if begins { opening_begins(r, m, i) } else { opening_ends(r, m, i) }
        };
        // As `open` leaves a guarded block when its locks are let go: no
        // guard's, an opening under way, read-only still.
        protect(PAGE_READONLY);
        opening(true);
        let before = OPENED_BESIDE[0].load(Relaxed);
        let writer = std::thread::spawn(move || {
            // SAFETY: inside our allocation; the write faults and waits.
            unsafe { ((p + 3 * KERNEL_PAGE) as *mut u8).write_volatile(7) };
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!writer.is_finished(), "the write waits for the opening");
        // The opening's second step.
        protect(PAGE_READWRITE);
        opening(false);
        writer.join().expect("the write went through");
        // SAFETY: inside our allocation.
        assert_eq!(unsafe { ((p + 3 * KERNEL_PAGE) as *const u8).read_volatile() }, 7);
        assert!(OPENED_BESIDE[0].load(Relaxed) > before, "counted as a fault that waited");
        // Asked with nothing under way: what the system says stands.
        assert!(opened_beside(p + 3 * KERNEL_PAGE), "writable, opened by another thread before the asking");
        protect(PAGE_READONLY);
        assert!(!opened_beside(p + 3 * KERNEL_PAGE), "read-only with no opening under way: not the guards'");
        protect(PAGE_READWRITE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// An opening for a write of the kernel's does not return while another
    /// thread is opening a block of its range: the kernel's write meets no
    /// read-only page (it would fail there with no fault of ours).
    #[test]
    fn an_opening_waits_for_another_threads_opening_of_its_blocks() {
        use std::sync::atomic::Ordering::Relaxed;
        use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: private memory of our own, of two blocks.
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let block = p / BLOCK;
        let protect = |how: u32| {
            let mut old = 0;
            // SAFETY: a block of our allocation; only its protection changes.
            assert_ne!(unsafe { VirtualProtect(p as *const _, BLOCK, how, &mut old) }, 0);
        };
        let opening = |begins: bool| {
            let mut l = Locked::blocks(Holder::Opening, [block]);
            let (r, m, i) = l.at(block).expect("its region");
            if begins { opening_begins(r, m, i) } else { opening_ends(r, m, i) }
        };
        // As `open` leaves a guarded block when its locks are let go.
        protect(PAGE_READONLY);
        opening(true);
        let before = OPENINGS_AWAITED[0].load(Relaxed);
        let writer = std::thread::spawn(move || {
            writing(p + 100, 3 * KERNEL_PAGE);
            // The kernel's write, as a read of the title's into its memory
            // makes it: no fault of ours, it fails on a read-only page.
            let bytes = [7u8; 16];
            let mut done = 0;
            // SAFETY: inside our allocation, from a buffer of that length.
            let ok = unsafe { WriteProcessMemory(GetCurrentProcess(), (p + 100) as *const _, bytes.as_ptr().cast(), bytes.len(), &mut done) };
            ok != 0 && done == bytes.len()
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!writer.is_finished(), "the opening waits for the other thread's");
        // The other thread's second step.
        protect(PAGE_READWRITE);
        opening(false);
        assert!(writer.join().expect("no panic"), "the kernel wrote");
        // SAFETY: inside our allocation.
        assert_eq!(unsafe { ((p + 100) as *const u8).read_volatile() }, 7);
        assert!(OPENINGS_AWAITED[0].load(Relaxed) > before, "counted as an opening that waited");
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The bits of a range in a region are those of its blocks there, and
    /// a range is all clean only if every block of it is noted so, in
    /// every region it touches.
    #[test]
    fn a_range_is_all_clean_only_if_every_block_of_it_is() {
        for (index, first, last) in [(0usize, 0usize, 64usize), (0, 3, 9), (1, 3, 70), (1, 64, 128), (2, 3, 70), (5, 330, 331)] {
            let want: Vec<usize> = (first..last).filter(|block| block / REGION == index).map(|block| block % REGION).collect();
            assert_eq!(each_bit(span(index, first, last)).collect::<Vec<_>>(), want, "{index}: {first}..{last}");
        }
        assert_eq!(span(usize::MAX / REGION, usize::MAX - 2, usize::MAX), 0b11 << 61);
        // Far above anything mapped in a test, across three regions.
        let b = (1usize << 45) / BLOCK + 60;
        assert!(!all_clean(b, b + 70), "nothing is known there");
        {
            let mut l = Locked::blocks(Holder::Check, b..b + 70);
            for block in b..b + 70 {
                let (r, m, i) = l.at(block).expect("its region");
                note_guarded(r, m, i, block);
            }
        }
        assert!(all_clean(b, b + 70) && all_clean(b + 5, b + 6) && all_clean(b, b));
        assert!(!all_clean(b - 1, b + 70) && !all_clean(b, b + 71));
        assert_eq!(guarded_and_shared(b + 3, b + 9).0, (b + 3..b + 9).collect::<Vec<_>>());
        {
            let mut l = Locked::blocks(Holder::Marking, [b + 66]);
            assert!(mark_block(&mut l, b + 66, 1 << 3));
            assert!(mark_block(&mut l, b + 66, 1 << 5));
        }
        assert!(!all_clean(b, b + 70) && all_clean(b, b + 66) && all_clean(b + 67, b + 70));
        assert_eq!(seen_since(b * BLOCK, (b + 70) * BLOCK, 0), (false, 1, 0));
        title_memory_remapped(b * BLOCK, 70 * BLOCK);
        assert!(!all_clean(b, b + 1));
        assert_eq!(seen_since(b * BLOCK, (b + 70) * BLOCK, 0).1, 0);
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
        assert!(!is(|r| &r.clean, view / BLOCK));
        assert_eq!(known_view(view), views_known());
        assert_eq!(guard(view, view + BLOCK, None), Some(vec![view / PAGE]));
        title_memory_remapped(view, BLOCK);
        assert!(!known_view(view));
        // (Told before the view goes, as the runtime tells it: its address
        // may be another test's the moment it is free.)
        title_memory_remapped(view, BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
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
        title_memory_remapped(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
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
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
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
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
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

    /// A view of guarded pages mapped after they were taken is writable and
    /// nobody's: the guarded view's next check makes it read-only, so that
    /// a write through it faults and is reported at the guarded view. (The
    /// check noted the guarded view clean and left the new one writable:
    /// every write through it was lost on the copies over the other.)
    #[test]
    fn a_view_mapped_beside_a_guarded_one_is_guarded_with_it() {
        use windows_sys::Win32::System::Memory::VirtualProtect;
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        // The runtime maps the other view anew: told, and read-write.
        title_memory_remapped(other, SIZE);
        let mut old = 0;
        // SAFETY: the test's own view.
        assert_ne!(unsafe { VirtualProtect(other as *const _, SIZE, PAGE_READWRITE, &mut old) }, 0);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(Vec::new()));
        // SAFETY: the second view; the write faults and the handler opens it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(guarded, guarded + SIZE, None), Some(vec![pages(1)]), "a write through the new view");
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view a fault opened in part is made read-only whole again by the
    /// check that takes another view of its pages: a second write into the
    /// part that was open faults and is reported there too. (It stayed
    /// open until its own next check, and what was written through it
    /// meanwhile never reached the copies over the other view.)
    #[test]
    fn a_view_opened_in_part_is_armed_again_when_another_view_of_its_pages_is_taken() {
        if !guard_pages() {
            return;
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let page = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE, None), Some(vec![page(one, 0), page(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![page(two, 0), page(two, 1)]));
        // SAFETY: the second view; the write faults and opens its tail.
        unsafe { ((two + 8 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(one, one + SIZE, None, &mut Look::default()), Some(vec![(one / BLOCK, FULL << 8)]));
        // SAFETY: as above, in the part the first fault opened.
        unsafe { ((two + 12 * KERNEL_PAGE) as *mut u8).write_volatile(2) };
        assert_eq!(
            guard_fine(one, one + SIZE, None, &mut Look::default()),
            Some(vec![(one / BLOCK, FULL << 12)]),
            "a second write through the view the first one opened"
        );
        // The view written through reports both at its own check.
        assert_eq!(guard_fine(two, two + SIZE, None, &mut Look::default()), Some(vec![(two / BLOCK, FULL << 8)]));
        drop_views(section, &[one, two], SIZE);
    }

    /// A view opened for a write of ours is guarded again by the check
    /// that takes another view of its pages: a later write through it
    /// faults and is reported there.
    #[test]
    fn a_view_opened_for_a_write_is_guarded_again_when_another_view_of_its_pages_is_taken() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let page = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE, None), Some(vec![page(one, 0), page(one, 1)]));
        assert_eq!(guard(two, two + SIZE, None), Some(vec![page(two, 0), page(two, 1)]));
        // A write of a few bytes opens the whole block: what is stored
        // anywhere in it before the other view's next check is that
        // check's to report, not the write's kernel page alone.
        writing(two + 16, 32);
        // SAFETY: the second view, open.
        unsafe { ((two + 9 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        assert_eq!(guard_fine(one, one + SIZE, None, &mut Look::default()), Some(vec![(one / BLOCK, FULL)]));
        // SAFETY: the second view; the write faults and the handler opens it.
        unsafe { ((two + 16) as *mut u8).write_volatile(1) };
        assert_eq!(guard(one, one + SIZE, None), Some(vec![page(one, 0)]), "a write through the view opened before");
        drop_views(section, &[one, two], SIZE);
    }

    /// A landing that wrote beside the guards takes its own marks
    /// ([`take_landed_views`]): the blocks are noted clean and the pages in
    /// the journal, so that the view written at has nothing to report and
    /// every asking learns of the pages; the other view of the pages still
    /// reports them. A block a store of the title's opened meanwhile, here
    /// or at another view of its pages, is left to its check.
    #[test]
    fn a_landing_takes_its_own_marks_in_view_memory() {
        if !guard_pages() {
            return;
        }
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        crate::inplace::title_memory_section(section, SIZE as u64, true);
        let block = |view: usize, k: usize| view / BLOCK + k;
        assert!(guard(one, one + SIZE, None).is_some() && guard(two, two + SIZE, None).is_some());
        assert_eq!(guard(one, one + SIZE, None), Some(Vec::new()));
        let land = |at: usize, bytes: &[u8]| {
            let place = crate::beside::find(at, bytes.len()).expect("a place in a view of our own");
            assert_eq!(place.write(at, bytes), bytes.len());
            written_beside(at, bytes.len(), crate::writes::Cause::Back);
        };
        // The first block: written beside in two kernel pages, and taken.
        land(one + 2 * KERNEL_PAGE + 8, &[7; KERNEL_PAGE]);
        // The second: a store of the title's through the same view first.
        // SAFETY: inside the view; the store faults and opens the tail.
        unsafe { ((one + BLOCK + 5 * KERNEL_PAGE) as *mut u8).write_volatile(1) };
        land(one + BLOCK + 16, &[8; 32]);
        // The third: a store through the other view of its pages first.
        // SAFETY: as above, the other view.
        unsafe { ((two + 2 * BLOCK + 9 * KERNEL_PAGE) as *mut u8).write_volatile(2) };
        land(one + 2 * BLOCK + 16, &[9; 32]);
        let before = asked_now();
        let (now, noted) = take_landed_views(&[(one, one + SIZE)]);
        assert!(now > before);
        assert_eq!(noted.iter().map(|&(page, w)| (page, w.at, w.mask)).collect::<Vec<_>>(), vec![(block(one, 0), now, 0b1100)]);
        // The other blocks are their check's: where the title stored, and
        // armed again whole (the second), with its other view (the third).
        assert_eq!(
            guard_fine(one, one + SIZE, None, &mut Look::default()),
            Some(vec![(block(one, 1), FULL << 5 | 1), (block(one, 2), FULL << 9 | 1)])
        );
        // Every other asking learns of the pages taken from the journal.
        let (_, pages) = writes_fine(one, one + SIZE, true).expect("view memory");
        assert!(pages.iter().any(|&(page, w)| page == block(one, 0) && w.at == now && w.mask == 0b1100));
        // And the other view of the pages reports them at its own check.
        let other = guard_fine(two, two + SIZE, None, &mut Look::default()).expect("view memory");
        assert!(other.contains(&(block(two, 0), 0b1100)), "{other:x?}");
        // Nothing to take a second time.
        assert!(take_landed_views(&[(one, one + SIZE), (two, two + SIZE)]).1.is_empty());
        // What the check of the taking asks: is a block written again?
        assert_eq!(watched_dirty(one, BLOCK), Some(0));
        // SAFETY: as above.
        unsafe { ((one + 8) as *mut u8).write_volatile(3) };
        assert_eq!(watched_dirty(one, BLOCK), Some(1));
        drop_views(section, &[one, two], SIZE);
    }

    /// Stores through one view of some pages, and through the view that is
    /// checked, while that view is checked and copied as a refresh does, in
    /// three regions at once: when the stores are done and the view has
    /// been checked once more, the copy holds every byte. (A store that no
    /// check reports is a byte the copy lacks until a later store into its
    /// kernel page is reported, and the last ones are not: with the other
    /// views of a block left as they were when it is noted clean, some
    /// 1,900 of the 2,300 kernel pages here end with bytes the copy lacks.)
    #[test]
    fn stores_and_checks_in_several_regions_at_once_lose_no_write() {
        use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
        const BLOCKS: usize = 2 * REGION + 16;
        const SIZE: usize = BLOCKS * BLOCK;
        let (section, taken, stored) = two_views(SIZE);
        assert!(taken / BLOCK / REGION < (taken + SIZE - 1) / BLOCK / REGION);
        let stop = AtomicBool::new(false);
        let mut copy = vec![0u8; SIZE];
        // What a refresh does: the kernel pages the guards report, read
        // after the check.
        let take = |copy: &mut Vec<u8>| {
            let written = guard_fine(taken, taken + SIZE, None, &mut Look::default()).expect("view memory");
            for (block, mask) in written {
                for k in each_bit(u64::from(mask)) {
                    let at = block * BLOCK + k * KERNEL_PAGE;
                    // SAFETY: inside the view checked, which is readable.
                    let page = unsafe { std::slice::from_raw_parts(at as *const u8, KERNEL_PAGE) };
                    copy[at - taken..at - taken + KERNEL_PAGE].copy_from_slice(page);
                }
            }
        };
        let checks = std::thread::scope(|scope| {
            for writer in 0..3usize {
                let stop = &stop;
                scope.spawn(move || {
                    // Two through the other view, into all but the last
                    // blocks; one through the view checked, into the last
                    // sixteen and the eight before them, where both kinds
                    // meet. Each its own four bytes of sixteen.
                    let (view, from, blocks) = if writer == 2 { (taken, BLOCKS - 24, 24) } else { (stored, 0, BLOCKS - 16) };
                    let mut n = 0x9e37_79b9u32.wrapping_mul(writer as u32 + 1);
                    let mut round = 0u32;
                    while !stop.load(SeqCst) {
                        round += 1;
                        n = n.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let slot = (n >> 4) as usize % (blocks * BLOCK / 16);
                        // SAFETY: inside the test's view; a store into a
                        // guarded block faults and its handler opens it.
                        unsafe { ((view + from * BLOCK + slot * 16 + writer * 4) as *mut u32).write_volatile(round) };
                        // (Not all at once: checks come in between.)
                        if round.is_multiple_of(16) {
                            for _ in 0..2_000 {
                                std::hint::spin_loop();
                            }
                        }
                    }
                });
            }
            let started = std::time::Instant::now();
            let mut checks = 0;
            while started.elapsed() < std::time::Duration::from_millis(150) {
                take(&mut copy);
                checks += 1;
            }
            stop.store(true, SeqCst);
            checks
        });
        assert!(checks >= 16, "{checks} checks beside the stores");
        take(&mut copy);
        // SAFETY: the view checked, readable; nothing stores any more.
        let memory = unsafe { std::slice::from_raw_parts(taken as *const u8, SIZE) };
        let lost = copy.chunks(KERNEL_PAGE).zip(memory.chunks(KERNEL_PAGE)).filter(|(copy, memory)| copy != memory).count();
        assert_eq!(lost, 0, "kernel pages whose last stores no check reported");
        drop_views(section, &[taken, stored], SIZE);
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
