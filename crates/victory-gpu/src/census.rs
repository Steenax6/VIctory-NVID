//! What the pass over textures picked by index (`crate::draw`
//! `heap_textures`) uploads and why, what is known of the writes it only
//! finds by looking, where its time goes and what its queue waits for:
//! counts kept as the pass and the uploads' reads go, and a group of lines
//! with the periodic ones ([`lines`]).
//!
//! The pass is the layer's largest mover of texture bytes (some 200,000
//! uploads and 83 GB in the first 106 s of the title, each texture about
//! seven times), and what made a texture go up again was not known past
//! "found changed by a look, nothing told of it" (30,000 to 37,000 times a
//! run). These counts say, for every upload, what made its slot due and
//! what its read then found: which of its levels held bytes other than at
//! its last upload, which the same, which nothing; and for every change a
//! look finds, what the layer was told of the memory since the upload read
//! it.
//!
//! An upload's reason is known when the pass records it, what it reads
//! when its list is submitted (`crate::texture` `Staging::refresh`): the
//! reason rides in the staging buffer's state ([`mark`]) and the row is
//! counted at the read ([`read`]).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::writes::{CAUSES, Cause};

/// Why a slot's texture goes up, as the queue's entries and the staging
/// buffers carry it: the title wrote the slot's view (a slot new to the
/// queue, another view of the texture it showed, a record copied in).
pub(crate) const VIEW: u8 = 0;
/// More of its memory is mapped than when it was last uploaded in part.
pub(crate) const MORE: u8 = 1;
/// A texture object not all mapped, uploaded again every 120 frames.
pub(crate) const AGAIN: u8 = 2;
/// A look at it after its upload found other bytes (its fingerprint).
pub(crate) const RELOOK: u8 = 3;
/// An upload left a level the title holds without the card's memory, or a
/// level left out was brought back (`crate::tiles`).
pub(crate) const LEVELS: u8 = 4;
/// From here, one for each [`Cause`]: its memory was told as mapped,
/// unmapped, written or about to be.
pub(crate) const TOLD: u8 = 5;
/// The reasons.
pub(crate) const WHYS: usize = TOLD as usize + CAUSES;
/// The row of an upload of memory the pass has not uploaded before,
/// whatever made its slot due.
const FIRST: usize = WHYS;
const ROWS: usize = WHYS + 1;

/// The reason code of memory told by `cause`.
pub(crate) fn told(cause: Cause) -> u8 {
    TOLD + cause as u8
}

fn row_name(row: usize) -> &'static str {
    match row {
        FIRST => "first upload of the memory",
        r if r == VIEW as usize => "the slot's view written again",
        r if r == MORE as usize => "more of it mapped",
        r if r == AGAIN as usize => "an object not all mapped, every 120 frames",
        r if r == RELOOK as usize => "found changed by a look",
        r if r == LEVELS as usize => "a level without the card's memory",
        r => Cause::ALL.get(r - TOLD as usize).map_or("?", |c| c.name()),
    }
}

/// What an upload's row counts: uploads; the bytes of their textures in
/// title memory; the bytes staged (the host's layout, without the levels a
/// texture leaves out); of the title bytes read, those of levels whose bytes
/// were other than at the memory's last upload (or uploaded for the first
/// time), the same as then, and all zero (pages not there, or nothing
/// written yet); uploads none of whose levels had changed, and their title
/// bytes; uploads whose read took no digest (write-watched memory read by
/// its written pages); uploads that read nothing (the memory could not be
/// had).
const FIELDS: usize = 10;
static UPLOADS: [[AtomicU64; FIELDS]; ROWS] = [const { [const { AtomicU64::new(0) }; FIELDS] }; ROWS];

/// The most parts (levels, the smallest ones together) a texture's memory
/// is told apart in; one of more is one part.
pub(crate) const PARTS: usize = 16;

/// What the pass last uploaded from a texture's memory, by where it starts
/// and its bytes: a digest of each part (0: all zero), how many times it
/// went up, and the frame of the last.
struct Memo {
    parts: [u64; PARTS],
    uploads: u32,
    frame: u64,
}

static MEMO: Mutex<Option<HashMap<(u64, u64), Memo>>> = Mutex::new(None);

/// One part of a texture's memory as an upload read it: its bytes, a
/// digest of them (never 0), and whether they are all zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Part {
    pub(crate) bytes: u64,
    pub(crate) digest: u64,
    pub(crate) zero: bool,
}

/// What [`read`] found against the memory's last upload: whether the pass
/// uploaded it before, and the title bytes changed, the same and zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Found {
    pub(crate) before: bool,
    pub(crate) changed: u64,
    pub(crate) same: u64,
    pub(crate) zero: u64,
}

/// `parts` against `memo` (the memory's last upload, if any), which is
/// brought up to date.
fn compare(memo: Option<&mut Memo>, parts: &[Part]) -> Found {
    let mut found = Found { before: memo.is_some(), ..Found::default() };
    let mut none = [0u64; PARTS];
    let last = memo.map_or(&mut none, |m| &mut m.parts);
    for (part, last) in parts.iter().zip(last.iter_mut()) {
        // (A part that was not read keeps what is noted of it.)
        if part.bytes == 0 {
            continue;
        }
        let digest = if part.zero { 0 } else { part.digest.max(1) };
        if part.zero {
            found.zero += part.bytes;
        } else if found.before && *last == digest {
            found.same += part.bytes;
        } else {
            found.changed += part.bytes;
        }
        *last = digest;
    }
    found
}

/// An upload of the pass read `bytes` bytes of the texture memory
/// `[start, start + size)` for the reason `why`: `staged` bytes of staging
/// buffer, and its parts as read (`None`: read without a digest; empty:
/// nothing read; a part of no bytes was not read). Counted in its row.
pub(crate) fn read(why: u8, start: u64, size: u64, bytes: u64, staged: u64, parts: Option<&[Part]>) {
    /// Memory noted at most; past it the entries of uploads longest ago go.
    const MOST: usize = 1 << 17;
    let frame = crate::frame::current();
    let mut all = MEMO.lock().unwrap_or_else(|e| e.into_inner());
    let all = all.get_or_insert_with(HashMap::new);
    if all.len() >= MOST {
        let old = frame.saturating_sub(4096);
        all.retain(|_, m| m.frame >= old);
    }
    let found = match parts {
        Some(parts) if !parts.is_empty() => {
            let found = compare(all.get_mut(&(start, size)), parts);
            let memo = all.entry((start, size)).or_insert_with(|| {
                let mut memo = Memo { parts: [0; PARTS], uploads: 0, frame };
                compare(Some(&mut memo), parts);
                memo
            });
            (memo.uploads, memo.frame) = (memo.uploads.saturating_add(1), frame);
            found
        }
        _ => {
            let before = all.get_mut(&(start, size)).map(|m| (m.uploads, m.frame) = (m.uploads.saturating_add(1), frame));
            Found { before: before.is_some(), ..Found::default() }
        }
    };
    drop_row(why, bytes, staged, parts, found);
}

/// [`read`]'s counting, after the memo.
fn drop_row(why: u8, size: u64, staged: u64, parts: Option<&[Part]>, found: Found) {
    let row = &UPLOADS[if found.before { usize::from(why).min(WHYS - 1) } else { FIRST }];
    row[0].fetch_add(1, Relaxed);
    row[1].fetch_add(size, Relaxed);
    row[2].fetch_add(staged, Relaxed);
    row[3].fetch_add(found.changed, Relaxed);
    row[4].fetch_add(found.same, Relaxed);
    row[5].fetch_add(found.zero, Relaxed);
    match parts {
        None => {
            row[8].fetch_add(1, Relaxed);
        }
        Some([]) => {
            row[9].fetch_add(1, Relaxed);
        }
        Some(_) if found.before && found.changed == 0 => {
            row[6].fetch_add(1, Relaxed);
            row[7].fetch_add(size, Relaxed);
        }
        Some(_) => {}
    }
}

/// What the journal, the guards and the write watch knew of the parts of
/// a texture's memory that an upload of the pass read again, since the read
/// that last found them as the host copy holds them (`crate::texture`
/// `Staging::refresh_parts`), by the upload's reason: parts changed with
/// nothing seen (a screen by them would have left the bytes from before:
/// 0 if they tell of every store); parts changed and seen; and, of one
/// upload in sixteen, the bytes of unchanged parts with nothing seen (a
/// screen would not have read them) and of unchanged parts seen.
pub(crate) static SCREEN: [[AtomicU64; 4]; WHYS] = [const { [const { AtomicU64::new(0) }; 4] }; WHYS];

/// Counts one part in [`SCREEN`], for an upload made for the reason `why`.
pub(crate) fn screened(why: u8, changed: bool, seen: bool, bytes: u64) {
    let row = &SCREEN[usize::from(why).min(WHYS - 1)];
    match (changed, seen) {
        (true, false) => row[0].fetch_add(1, Relaxed),
        (true, true) => row[1].fetch_add(1, Relaxed),
        (false, false) => row[2].fetch_add(bytes, Relaxed),
        (false, true) => row[3].fetch_add(bytes, Relaxed),
    };
}

/// Diagnostics: the last number (`crate::writes::next`) of a tell of each
/// [`Cause`] by 64 KiB page of title memory, for [`unseen`]: the tells the
/// texture pass takes are gone once it took them, and the question is
/// whether one came after a read that is long past.
#[cfg(feature = "diagnostics")]
static TOLD_AT: Mutex<Option<HashMap<u64, [u64; CAUSES]>>> = Mutex::new(None);

/// Diagnostics: title memory `[start, end)` was told by the cause coded
/// `cause` under the number `number` (`crate::descriptor::host` `remapped`
/// and `written`, under their lock; this locks one mutex of its own and calls
/// nothing).
#[cfg(feature = "diagnostics")]
pub(crate) fn told_at(start: u64, end: u64, cause: u8, number: u64) {
    const PAGE: u64 = 64 << 10;
    if end <= start {
        return;
    }
    let cause = usize::from(cause).min(CAUSES - 1);
    let mut all = TOLD_AT.lock().unwrap_or_else(|e| e.into_inner());
    let all = all.get_or_insert_with(HashMap::new);
    // (No more than 4 GiB of it at once.)
    for page in (start / PAGE..end.div_ceil(PAGE)).take(1 << 16) {
        let last = &mut all.entry(page).or_default()[cause];
        *last = (*last).max(number);
    }
}

/// Diagnostics: the causes (a bit each) told of `[start, end)` under a
/// number above `stamp`.
#[cfg(feature = "diagnostics")]
fn told_since(start: u64, end: u64, stamp: u64) -> u32 {
    const PAGE: u64 = 64 << 10;
    let all = TOLD_AT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(all) = all.as_ref() else {
        return 0;
    };
    let mut causes = 0;
    for page in (start / PAGE..end.div_ceil(PAGE)).take(1 << 16) {
        if let Some(last) = all.get(&page) {
            for (cause, &n) in last.iter().enumerate() {
                if n > stamp {
                    causes |= 1 << cause;
                }
            }
        }
    }
    causes
}

/// Diagnostics: a part of a texture's memory that an upload of the pass
/// found changed while the journal, the guards and the write watch had seen
/// nothing there since the read that last found it as the host copy holds
/// it ([`screened`]): the upload's reason, where the part is, whether the
/// texture is made level by level, what that read knew of it
/// (`crate::texture` `unseen_flags`) and the number it began at
/// (`crate::writes::stamp`), and whether this read found pages of it
/// missing and a read of the title's still to write the texture.
#[cfg(feature = "diagnostics")]
pub(crate) struct Unseen {
    pub(crate) why: u8,
    pub(crate) at: usize,
    pub(crate) len: usize,
    pub(crate) level_by_level: bool,
    pub(crate) then: u64,
    pub(crate) stamp: u64,
    pub(crate) through: bool,
    pub(crate) pending: bool,
}

/// Diagnostics: [`UNSEEN`]'s places. Parts in all; of them, by the kind of
/// memory now (write-watched: the kernel answers; guarded view memory; some
/// blocks noted absent; some whose pages have other views the guards hold
/// with them); some pages missing at the earlier read, and at this one; a
/// read of the title's to write the texture under way then, and now; made
/// level by level; then one place for each [`Cause`] told there since the
/// earlier read; and parts none of these name.
#[cfg(feature = "diagnostics")]
const U_ALL: usize = 0;
#[cfg(feature = "diagnostics")]
const U_WATCHED: usize = 1;
#[cfg(feature = "diagnostics")]
const U_VIEW: usize = 2;
#[cfg(feature = "diagnostics")]
const U_ABSENT: usize = 3;
#[cfg(feature = "diagnostics")]
const U_ALIASED: usize = 4;
#[cfg(feature = "diagnostics")]
const U_MISSING_THEN: usize = 5;
#[cfg(feature = "diagnostics")]
const U_MISSING_NOW: usize = 6;
#[cfg(feature = "diagnostics")]
const U_PENDING_THEN: usize = 7;
#[cfg(feature = "diagnostics")]
const U_PENDING_NOW: usize = 8;
#[cfg(feature = "diagnostics")]
const U_LEVELS: usize = 9;
#[cfg(feature = "diagnostics")]
const U_TOLD: usize = 10;
#[cfg(feature = "diagnostics")]
const U_NONE: usize = U_TOLD + CAUSES;
/// Diagnostics: since the start ([`unseen`]).
#[cfg(feature = "diagnostics")]
static UNSEEN: [AtomicU64; U_NONE + 1] = [const { AtomicU64::new(0) }; U_NONE + 1];

/// Diagnostics: counts and names a part found changed with nothing seen.
#[cfg(feature = "diagnostics")]
pub(crate) fn unseen(u: Unseen) {
    let count = |k: usize| UNSEEN[k].fetch_add(1, Relaxed);
    let n = count(U_ALL);
    let state = crate::watch::part_state(u.at, u.at.saturating_add(u.len));
    let told = told_since(u.at as u64, u.at.saturating_add(u.len) as u64, u.stamp);
    let (missing_then, pending_then) = (u.then & 1 != 0, u.then & 2 != 0);
    let flags = [
        (U_WATCHED, state.watched),
        (U_VIEW, state.blocks > 0 && state.guarded + state.absent == state.blocks),
        (U_ABSENT, state.absent > 0),
        (U_ALIASED, state.aliased > 0),
        (U_MISSING_THEN, missing_then),
        (U_MISSING_NOW, u.through),
        (U_PENDING_THEN, pending_then),
        (U_PENDING_NOW, u.pending),
        (U_LEVELS, u.level_by_level),
    ];
    for &(k, on) in &flags {
        if on {
            count(k);
        }
    }
    for cause in 0..CAUSES {
        if told & 1 << cause != 0 {
            count(U_TOLD + cause);
        }
    }
    let named = told != 0 || state.absent > 0 || missing_then || u.through || pending_then || u.pending || u.level_by_level;
    if !named {
        count(U_NONE);
    }
    if n < 32 || n.is_multiple_of(200) {
        tracing::info!(
            why = row_name(usize::from(u.why)),
            at = format_args!("{:#x}", u.at),
            bytes = u.len,
            blocks = state.blocks,
            guarded_clean = state.guarded,
            absent = state.absent,
            aliased = state.aliased,
            write_watched = state.watched,
            told = format_args!("{told:#x}"),
            missing_then,
            missing_now = u.through,
            pending_then,
            pending_now = u.pending,
            level_by_level = u.level_by_level,
            frames_since = crate::frame::current().saturating_sub(u.then >> 8),
            so_far = n + 1,
            "a part of a texture changed with nothing seen since the read that found it as the host copy holds it"
        );
    }
}

/// What the pass knows of an upload when it records it, beside its reason
/// ([`FLAGS`]' places): its host copy was made anew after it had been let
/// go; made for the first time; another texture's view starts inside the
/// texture's span, and (the next place) the bytes of the span from there
/// on; one starts inside the part its own view holds (from the view's
/// first level on); the memory has host copies of several descriptions
/// now; several slots show the span; the upload is for a bake draw; the
/// upload staged nothing (its texture went up in this list already).
pub(crate) const ANEW: usize = 0;
pub(crate) const NEVER: usize = 1;
pub(crate) const INNER: usize = 2;
pub(crate) const INNER_BYTES: usize = 3;
pub(crate) const INNER_HELD: usize = 4;
pub(crate) const COPIES: usize = 5;
pub(crate) const SLOTS: usize = 6;
pub(crate) const BAKE: usize = 7;
pub(crate) const NOTHING: usize = 8;
/// Uploads whose texture's span was none (no bytes, or an end below its
/// start): asked about all the same, and nothing was inside.
pub(crate) const NO_SPAN: usize = 9;
/// Since the start, of the uploads the pass recorded.
pub(crate) static FLAGS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// What the uploads did level by level ([`BY_LEVEL`]'s places), every
/// upload's, the pass's and those of textures read through tables: bytes
/// of staging buffer detiled; bytes of levels not detiled because the host
/// copy held them as their memory is; bytes copied to the card; uploads
/// decided at a submission that copied something, and that copied nothing
/// (`crate::texture::decide_uploads`); bytes of title memory read; bytes
/// of it not read because they are not the title's (levels more detailed
/// than any view shows, where another texture starts), and because their
/// pages are not there; bytes of staging buffer zeroed for levels the
/// title views whose pages are not there; bytes of title memory not read
/// because the host copy is made without those levels (`crate::tiles`);
/// bytes read through the kernel of parts with some pages there and some
/// not; parts that went up as zeros because none of their pages was there,
/// and those of them that got their bytes (or the card's) since: the
/// difference is levels a view may show that are zeros on the host; and
/// subresources a view may show for which, at an upload's decision, the
/// host copy held nothing an upload or the card brought.
pub(crate) const L_DETILED: usize = 0;
pub(crate) const L_WITH_COPY: usize = 1;
pub(crate) const L_COPIED: usize = 2;
pub(crate) const L_SOME: usize = 3;
pub(crate) const L_NONE: usize = 4;
pub(crate) const L_READ: usize = 5;
pub(crate) const L_LEFT: usize = 6;
pub(crate) const L_AWAY: usize = 7;
pub(crate) const L_ZEROED: usize = 8;
pub(crate) const L_OUT: usize = 9;
pub(crate) const L_PARTLY: usize = 10;
pub(crate) const L_ZERO_PARTS: usize = 11;
pub(crate) const L_ZERO_FILLED: usize = 12;
pub(crate) const L_BARE: usize = 13;
/// Since the start.
pub(crate) static BY_LEVEL: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

/// The parts of the pass ([`PASS_US`]): taking what was told since the
/// last pass; the queue's order; looks at uploaded textures (their
/// fingerprints); asking what of a texture's memory is mapped; making the
/// view and the host texture; recording the upload (the staging buffer and
/// the copies); asking the staging's state and letting it go; the
/// fingerprint after an upload; the rest (the queue's bookkeeping).
pub(crate) const P_TOLD: usize = 0;
pub(crate) const P_ORDER: usize = 1;
pub(crate) const P_LOOK: usize = 2;
pub(crate) const P_MAPPED: usize = 3;
pub(crate) const P_MAKE: usize = 4;
pub(crate) const P_RECORD: usize = 5;
pub(crate) const P_STAGED: usize = 6;
pub(crate) const P_PRINT: usize = 7;
pub(crate) const P_REST: usize = 8;
const PASS_PARTS: usize = 9;
const PASS_NAMES: [&str; PASS_PARTS] = [
    "taking what was told",
    "the queue's order",
    "looks at uploaded ones",
    "asking what is mapped",
    "making views and host textures",
    "recording uploads",
    "the staging's state",
    "fingerprints after uploads",
    "the rest",
];
/// Microseconds of the pass by part, since the last line.
static PASS_US: [AtomicU64; PASS_PARTS] = [const { AtomicU64::new(0) }; PASS_PARTS];

/// Since the last line ([`PASS_COUNTS`]' places): passes; slots the queue
/// handed out; slots looked at before the pass stopped; fingerprints
/// taken, and the bytes read for them; askings of what is mapped; uploads
/// recorded; passes stopped by their time, and by their bytes.
pub(crate) const N_PASSES: usize = 0;
pub(crate) const N_DUE: usize = 1;
pub(crate) const N_SEEN: usize = 2;
pub(crate) const N_PRINTS: usize = 3;
pub(crate) const N_PRINT_BYTES: usize = 4;
pub(crate) const N_MAPPED: usize = 5;
pub(crate) const N_UPLOADS: usize = 6;
pub(crate) const N_TIME: usize = 7;
pub(crate) const N_BYTES: usize = 8;
pub(crate) static PASS_COUNTS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// A pass's clock: the time since the last [`Clock::lap`] goes to a part.
pub(crate) struct Clock {
    last: std::time::Instant,
    us: [u64; PASS_PARTS],
}

impl Clock {
    pub(crate) fn start() -> Clock {
        Clock { last: std::time::Instant::now(), us: [0; PASS_PARTS] }
    }

    /// The time since the last lap (or the start) was `part`'s.
    pub(crate) fn lap(&mut self, part: usize) {
        let now = std::time::Instant::now();
        self.us[part] += now.duration_since(self.last).as_micros() as u64;
        self.last = now;
    }
}

impl Drop for Clock {
    fn drop(&mut self) {
        self.lap(P_REST);
        for (all, us) in PASS_US.iter().zip(self.us) {
            all.fetch_add(us, Relaxed);
        }
    }
}

/// What is known of the memory of a texture a look found changed, since
/// its upload's read began ([`FOUND`]' places, after one for each
/// [`Cause`] of a numbered write that made the slot stale and one for each
/// of a tell that lay in the texture's span and was taken for another
/// texture's): the guards or the write watch saw a store there; some of
/// its 64 KiB blocks are not guarded, so nothing could tell; all of them
/// are guarded and none was written; its upload had not read yet.
pub(crate) const F_STALE: usize = 0;
pub(crate) const F_HIDDEN: usize = CAUSES;
pub(crate) const F_STORE: usize = 2 * CAUSES;
pub(crate) const F_OPEN: usize = 2 * CAUSES + 1;
pub(crate) const F_UNSEEN: usize = 2 * CAUSES + 2;
pub(crate) const F_UNREAD: usize = 2 * CAUSES + 3;
/// Since the start: textures found changed by a look, by what is known.
pub(crate) static FOUND: [AtomicU64; 2 * CAUSES + 4] = [const { AtomicU64::new(0) }; 2 * CAUSES + 4];

/// Since the start, by the look after an upload (the first to the fifth):
/// looks that found the memory as it was, and that found it changed.
pub(crate) static LOOKS: [[AtomicU64; 2]; 5] = [const { [const { AtomicU64::new(0) }; 2] }; 5];

/// Since the start, for each [`Cause`] ([`TELLS`]' places): ranges told
/// inside the spans of texture views; slots whose texture held bytes of
/// them when a pass took them; of those, the slots made due (memory mapped
/// or about to be written) or stale (a numbered write after their upload's
/// read); of the slots made due, those whose upload had begun its read
/// after the tell; slots whose span a tell lay in that was taken for
/// another texture's (one that starts between), uploaded before it.
pub(crate) const T_RANGES: usize = 0;
pub(crate) const T_HELD: usize = 1;
pub(crate) const T_DUE: usize = 2;
pub(crate) const T_READ_AFTER: usize = 3;
pub(crate) const T_HIDDEN: usize = 4;
pub(crate) static TELLS: [[AtomicU64; 5]; CAUSES] = [const { [const { AtomicU64::new(0) }; 5] }; CAUSES];

/// Counts `n` at `place` of the tells of the cause coded `cause`. A code
/// past the causes counts under the last and a place past the row is
/// dropped: these are called where the title's commands are recorded and
/// where the runtime maps memory, and must not be able to panic.
pub(crate) fn tell(cause: u8, place: usize, n: u64) {
    if let Some(count) = TELLS[usize::from(cause).min(CAUSES - 1)].get(place) {
        count.fetch_add(n, Relaxed);
    }
}

/// Counts a texture a look found changed under what was known of its
/// memory (a place of [`FOUND`]; one past them counts under the last).
pub(crate) fn found(known: usize) {
    FOUND[known.min(FOUND.len() - 1)].fetch_add(1, Relaxed);
}

/// The queue by what its slots wait for, as a pass last counted them
/// (`crate::descriptor::host::HostHeap::texture_queue_kinds`): never
/// looked at and due; never looked at, due later (a table's slot lately,
/// or left for a read the title submitted); held for want of room on the
/// card; memory not mapped, a view of address 0; memory not mapped,
/// another; uploaded in part, waiting for more; uploaded whole, waiting
/// for its first to fifth look; and what is none of these.
pub(crate) const KINDS: usize = 12;
const KIND_NAMES: [&str; KINDS] = [
    "never looked at and due",
    "never looked at, due later",
    "held for room on the card",
    "not mapped, a view of address 0",
    "not mapped",
    "uploaded in part, waiting for more",
    "waiting for the 1st look",
    "the 2nd",
    "the 3rd",
    "the 4th",
    "the 5th",
    "other",
];
pub(crate) static QUEUE: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// Placed textures by address, as a pass last counted them
/// (`crate::resource::placed_descriptions`): textures in all; addresses
/// with textures of several descriptions; the most at one address; such
/// addresses with several host copies alive.
pub(crate) static DESCRIPTIONS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether a pass counts the queue and the placed textures now: every
/// five seconds, as often as the lines are written (the counts walk the
/// queue and every placed texture, on a thread that records the frame).
pub(crate) fn counting_due() -> bool {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let next = NEXT.load(Relaxed);
    now >= next && NEXT.compare_exchange(next, now + 5000, Relaxed, Relaxed).is_ok()
}

/// Bytes of title memory that the checks of guarded textures read (the
/// textures draws read through tables, at their submissions), since the
/// last line, by why the check read: the guards saw a write; the memory was
/// mapped anew; one of the texture's first checks; a texture taken for one
/// written behind its guards (every check); the net's turn (the hash for
/// what the guards do not see, every eighth frame). `crate::texture`
/// `check_reads`.
pub(crate) static CHECK_READS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Since the start: slots of the queue of textures picked by index that
/// its index by due frame did not hold when the two were compared (at
/// every count of the queue, and by their sizes at every pass). 0: a pass
/// that asks the index for the due slots is handed every one. If not, the
/// index is made anew from the slots.
pub(crate) static INDEX_MISSES: AtomicU64 = AtomicU64::new(0);

/// The texture memories the pass uploaded by how many times: up to 1, 2,
/// 4, 8, 16, 32 times and more, each with the textures and their uploads.
fn times() -> [(u64, u64); 7] {
    let mut out = [(0u64, 0u64); 7];
    let all = MEMO.lock().unwrap_or_else(|e| e.into_inner());
    for memo in all.iter().flat_map(|all| all.values()) {
        let n = u64::from(memo.uploads.max(1));
        let bucket = (64 - (n - 1).leading_zeros() as usize).min(6);
        out[bucket].0 += 1;
        out[bucket].1 += n;
    }
    out
}

/// The lines, every few seconds; `frames`: the frames since the last.
pub(crate) fn lines(frames: f64) {
    let mb = |bytes: u64| bytes >> 20;
    let mut rows = String::new();
    let mut all = [0u64; FIELDS];
    for (row, counts) in UPLOADS.iter().enumerate() {
        let c: [u64; FIELDS] = std::array::from_fn(|k| counts[k].load(Relaxed));
        for (sum, n) in all.iter_mut().zip(c) {
            *sum += n;
        }
        if c[0] == 0 {
            continue;
        }
        use std::fmt::Write;
        let _ = write!(
            rows,
            "; {}: {} ({} MB of title memory read, {} MB their staging buffers; levels changed {} MB, the same {} MB, zero \
             {} MB; {} with no level changed {} MB; {} read without a digest, {} read nothing)",
            row_name(row),
            c[0],
            mb(c[1]),
            mb(c[2]),
            mb(c[3]),
            mb(c[4]),
            mb(c[5]),
            c[6],
            mb(c[7]),
            c[8],
            c[9]
        );
    }
    if all[0] == 0 && PASS_COUNTS[N_PASSES].load(Relaxed) == 0 {
        return;
    }
    tracing::info!(
        "texture uploads by reason, since the start (counted when the upload reads): {} in all ({} MB of title memory \
         read, {} MB their staging buffers; levels changed {} MB, the same {} MB, zero {} MB; {} with no level changed {} \
         MB){rows}",
        all[0],
        mb(all[1]),
        mb(all[2]),
        mb(all[3]),
        mb(all[4]),
        mb(all[5]),
        all[6],
        mb(all[7])
    );
    let f: [u64; 10] = std::array::from_fn(|k| FLAGS[k].load(Relaxed));
    let t = times();
    let d: [u64; 4] = std::array::from_fn(|k| DESCRIPTIONS[k].load(Relaxed));
    tracing::info!(
        "texture uploads as recorded, since the start: {} with a host copy made anew after it was let go, {} with a first \
         host copy; {} of textures with another texture's view starting inside their span ({} MB of them from there on), \
         {} with one inside the part their own view holds; {} of memory with host copies of several descriptions, {} of \
         spans several slots show; {} for bake draws; {} that staged nothing; {} of spans that were none. Memories by \
         times uploaded (textures/uploads): once {}/{}, twice {}/{}, to 4 {}/{}, to 8 \
         {}/{}, to 16 {}/{}, to 32 {}/{}, more {}/{}. Placed textures: {} in all, {} addresses with several \
         descriptions (at most {} at one), {} of them with several host copies alive",
        f[ANEW],
        f[NEVER],
        f[INNER],
        mb(f[INNER_BYTES]),
        f[INNER_HELD],
        f[COPIES],
        f[SLOTS],
        f[BAKE],
        f[NOTHING],
        f[NO_SPAN],
        t[0].0,
        t[0].1,
        t[1].0,
        t[1].1,
        t[2].0,
        t[2].1,
        t[3].0,
        t[3].1,
        t[4].0,
        t[4].1,
        t[5].0,
        t[5].1,
        t[6].0,
        t[6].1,
        d[0],
        d[1],
        d[2],
        d[3]
    );
    let r: [f64; 5] = std::array::from_fn(|k| CHECK_READS[k].swap(0, Relaxed) as f64 / f64::from(1u32 << 20) / frames.max(1.0));
    tracing::info!(
        "textures read through tables, MB of title memory their checks read per frame since the last such line, by why: \
         a write the guards saw {:.1}, memory mapped anew {:.1}, first checks {:.1}, textures taken for written behind their \
         guards {:.1}, the net's turn {:.1}",
        r[0],
        r[1],
        r[2],
        r[3],
        r[4]
    );
    let l: [u64; 14] = std::array::from_fn(|k| BY_LEVEL[k].load(Relaxed));
    tracing::info!(
        "texture uploads level by level, since the start (the pass's and those of textures read through tables): {} MB \
         of title memory read, {} MB not read because no view shows it (levels above the first one viewed, where \
         another texture starts or nothing is there), {} MB of levels a view shows not read because none of their pages \
         is there ({} MB of staging zeroed for those), \
         {} MB not read because the host copy is made without those levels, {} MB read through the kernel of levels \
         only partly there; {} levels went up as zeros because nothing of them was there, {} of them got their bytes \
         since; {} times a level a view may show was left with nothing on the host at an upload's decision though \
         other levels of its texture were brought (0: none may be); {} \
         uploads decided with levels to copy, {} with none (the host copy held every level as its memory is); {} MB \
         detiled into staging, {} MB of levels not detiled because the host copy held them, {} MB copied to the card",
        mb(l[L_READ]),
        mb(l[L_LEFT]),
        mb(l[L_AWAY]),
        mb(l[L_ZEROED]),
        mb(l[L_OUT]),
        mb(l[L_PARTLY]),
        l[L_ZERO_PARTS],
        l[L_ZERO_FILLED],
        l[L_BARE],
        l[L_SOME],
        l[L_NONE],
        mb(l[L_DETILED]),
        mb(l[L_WITH_COPY]),
        mb(l[L_COPIED])
    );
    {
        use std::fmt::Write;
        let mut text = String::new();
        for (row, counts) in SCREEN.iter().enumerate() {
            let c: [u64; 4] = std::array::from_fn(|k| counts[k].load(Relaxed));
            if c.iter().any(|&n| n != 0) {
                let _ = write!(
                    text,
                    "; {}: {} parts changed with nothing seen, {} changed and seen, {} MB unchanged with nothing seen, {} MB \
                     unchanged and seen",
                    row_name(row),
                    c[0],
                    c[1],
                    mb(c[2]),
                    mb(c[3])
                );
            }
        }
        if !text.is_empty() {
            tracing::info!(
                "what the journal, the guards and the write watch knew of the parts of textures an upload of the pass read \
                 again (parts the host copy held; every changed part, and the unchanged ones of one upload in 16), since \
                 the start, by the upload's reason{text}. A part changed with nothing seen is one a screen by them, in \
                 place of the read, would have left with the bytes from before"
            );
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        use std::fmt::Write;
        let u: [u64; U_NONE + 1] = std::array::from_fn(|k| UNSEEN[k].load(Relaxed));
        if u[U_ALL] > 0 {
            let mut told = String::new();
            for cause in Cause::ALL {
                let _ = write!(told, ", {} {}", cause.name(), u[U_TOLD + cause as usize]);
            }
            tracing::info!(
                "parts of textures an upload of the pass found changed with nothing seen (the line above), since the start, \
                 by what is known of them: {} in all; now in write-watched memory {}, in guarded view memory {}, with blocks \
                 noted absent {}, with blocks whose pages have other views {}; pages missing at the read that found them as \
                 the host copy holds them {}, at this read {}; a read of the title's to write the texture under way then {}, \
                 now {}; made level by level {}; told since that read{told}; none of these {}",
                u[U_ALL],
                u[U_WATCHED],
                u[U_VIEW],
                u[U_ABSENT],
                u[U_ALIASED],
                u[U_MISSING_THEN],
                u[U_MISSING_NOW],
                u[U_PENDING_THEN],
                u[U_PENDING_NOW],
                u[U_LEVELS],
                u[U_NONE]
            );
        }
    }
    {
        use std::fmt::Write;
        let found: [u64; 2 * CAUSES + 4] = std::array::from_fn(|k| FOUND[k].load(Relaxed));
        let mut text = String::new();
        for cause in Cause::ALL {
            let (stale, hidden) = (found[F_STALE + cause as usize], found[F_HIDDEN + cause as usize]);
            if stale + hidden > 0 {
                let _ = write!(text, "; {}: {stale} stale by it and not made due, {hidden} in a tell taken for another texture", cause.name());
            }
        }
        let mut looks = String::new();
        for (k, look) in LOOKS.iter().enumerate() {
            let _ = write!(looks, " {}:{}/{}", k + 1, look[1].load(Relaxed), look[0].load(Relaxed));
        }
        tracing::info!(
            "textures found changed by a look, since the start, by what was known of their memory since the upload's read: \
             {} in all; {} a store the guards or the write watch saw, {} with blocks not guarded (nothing could tell), {} \
             guarded and clean (nothing seen), {} whose upload had not read yet{text}; by look, changed/the same:{looks}",
            found.iter().sum::<u64>(),
            found[F_STORE],
            found[F_OPEN],
            found[F_UNSEEN],
            found[F_UNREAD]
        );
        let mut tells = String::new();
        for cause in Cause::ALL {
            let t: [u64; 5] = std::array::from_fn(|k| TELLS[cause as usize][k].load(Relaxed));
            if t[T_RANGES] > 0 {
                let _ = write!(
                    tells,
                    "; {}: {} ranges, {} slots held their bytes, {} made due or stale ({} of them read after the tell), {} \
                     more in spans the tell lay in",
                    cause.name(),
                    t[T_RANGES],
                    t[T_HELD],
                    t[T_DUE],
                    t[T_READ_AFTER],
                    t[T_HIDDEN]
                );
            }
        }
        if !tells.is_empty() {
            tracing::info!("told to the texture pass, since the start, by who told{tells}");
        }
    }
    {
        use std::fmt::Write;
        let n: [u64; 9] = std::array::from_fn(|k| PASS_COUNTS[k].swap(0, Relaxed));
        let mut parts = String::new();
        let mut total = 0.0;
        for (name, us) in PASS_NAMES.iter().zip(&PASS_US) {
            let ms = us.swap(0, Relaxed) as f64 / 1e3 / frames;
            total += ms;
            let _ = write!(parts, ", {name} {ms:.2}");
        }
        let per = |k: usize| n[k] as f64 / frames;
        tracing::info!(
            "the texture pass per frame, since the last such line: {:.1} passes, {total:.2} ms on its thread (ms{parts}); \
             {:.0} slots handed out, {:.0} looked at, {:.0} fingerprints over {:.1} MB, {:.0} askings of what is mapped, \
             {:.0} uploads recorded; {:.2} passes stopped by their time, {:.2} by their bytes",
            per(N_PASSES),
            per(N_DUE),
            per(N_SEEN),
            per(N_PRINTS),
            per(N_PRINT_BYTES) / 1e6,
            per(N_MAPPED),
            per(N_UPLOADS),
            per(N_TIME),
            per(N_BYTES)
        );
        let mut kinds = String::new();
        for (name, count) in KIND_NAMES.iter().zip(&QUEUE) {
            let _ = write!(kinds, ", {name} {}", count.load(Relaxed));
        }
        tracing::info!(
            "the texture queue by what its slots wait for, at a pass of the last five seconds: {}; {} slots its index by due \
             frame did not hold when compared with a walk over all (0: none can be missed)",
            kinds.trim_start_matches(", "),
            INDEX_MISSES.load(Relaxed)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(bytes: u64, digest: u64) -> Part {
        Part { bytes, digest, zero: false }
    }

    /// A memory's first upload counts every part with bytes as changed;
    /// its next says which parts hold other bytes, which the same and which
    /// nothing, by part; and one with nothing changed says so.
    #[test]
    fn an_upload_is_told_from_the_last_of_its_memory_by_part() {
        let mut memo = Memo { parts: [0; PARTS], uploads: 0, frame: 0 };
        let zero = Part { bytes: 400, digest: 9, zero: true };
        // The first: nothing to compare with.
        assert_eq!(compare(None, &[part(100, 5), zero]), Found { before: false, changed: 100, same: 0, zero: 400 });
        compare(Some(&mut memo), &[part(100, 5), zero]);
        // The large part arrived, the small one is as it was.
        let next = [part(100, 5), part(400, 7)];
        assert_eq!(compare(Some(&mut memo), &next), Found { before: true, changed: 400, same: 100, zero: 0 });
        // Nothing new.
        assert_eq!(compare(Some(&mut memo), &next), Found { before: true, changed: 0, same: 500, zero: 0 });
        // A part that went (its pages unmapped) is zero, not changed; when
        // it comes back with the same bytes it counts as changed (the
        // upload between held nothing there).
        assert_eq!(compare(Some(&mut memo), &[part(100, 5), zero]), Found { before: true, changed: 0, same: 100, zero: 400 });
        assert_eq!(compare(Some(&mut memo), &next), Found { before: true, changed: 400, same: 100, zero: 0 });
        // A part that was not read (it is not the title's now) counts for
        // nothing and keeps what was noted of it: read again with the same
        // bytes, it is the same.
        assert_eq!(compare(Some(&mut memo), &[part(100, 5), part(0, 0)]), Found { before: true, changed: 0, same: 100, zero: 0 });
        assert_eq!(compare(Some(&mut memo), &next), Found { before: true, changed: 0, same: 500, zero: 0 });
    }

    /// Counting takes any code: one past the causes, the places or what
    /// can be known counts under the last or is dropped, and a read of no
    /// parts, of more parts than are told apart or of no bytes is counted
    /// as any other.
    #[test]
    fn counting_takes_any_code() {
        tell(255, T_RANGES, 1);
        tell(0, 99, 1);
        found(usize::MAX);
        let many = [part(1, 1); PARTS + 9];
        read(255, u64::MAX, u64::MAX, u64::MAX, u64::MAX, Some(&many));
        let before = SCREEN[WHYS - 1][0].load(Relaxed);
        screened(255, true, false, u64::MAX);
        screened(VIEW, false, true, 7);
        assert!(SCREEN[WHYS - 1][0].load(Relaxed) > before);
        assert!(SCREEN[usize::from(VIEW)][3].load(Relaxed) >= 7);
        read(RELOOK, u64::MAX, 0, 0, 0, Some(&[]));
        read(VIEW, 0, 0, u64::MAX, 0, None);
        assert!(FOUND[FOUND.len() - 1].load(Relaxed) >= 1);
        assert!(TELLS[CAUSES - 1][T_RANGES].load(Relaxed) >= 1);
    }

    /// A reason code past the table counts in its last row, and the codes
    /// of the causes follow the fixed ones.
    #[test]
    fn reasons_have_rows() {
        assert_eq!(usize::from(told(Cause::ALL[CAUSES - 1])), WHYS - 1);
        assert_eq!(row_name(usize::from(told(Cause::Mapped))), Cause::Mapped.name());
        assert_eq!(row_name(FIRST), "first upload of the memory");
    }
}
