//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! A change detiles and copies the levels whose bytes changed (the memory
//! is told apart at its levels, [`Where`]; what the host copy holds of each
//! is kept with it, [`Sent`]).
//! shortcut: whole levels, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    bytes: u64,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// The levels that were the title's at the last detiling (a bit a
    /// level; the staging buffer holds nothing of the others, and none of
    /// them is copied), and what that was judged by
    /// ([`Staging::judgement`]). Levels on demand: the levels that get the
    /// card's memory.
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// The levels the last submitted upload wrote (a bit a level): all but
    /// the levels a texture made without them left out
    /// (`crate::tiles::Tiled::copied_now`). An upload that would write
    /// more now (a level was brought back) is recorded again.
    copied: std::sync::atomic::AtomicU32,
    /// The first subresource the buffer holds: the levels before it are
    /// left out of a texture made without them (a texture of one layer;
    /// `footprints[i]` is subresource `first + i`). An upload that would
    /// write one of them makes the staging anew ([`upload`]).
    first: u32,
    /// The buffer holds bytes that are neither zeros nor a level detiled
    /// whole (it came from the pool with what an earlier staging left, or
    /// a detiling failed): zeroed if an upload copies from it although
    /// nothing was detiled into it ([`Staging::wanted`]). Only the levels
    /// detiled into it are copied otherwise.
    stale: std::sync::atomic::AtomicBool,
    /// The size of the buffer if it came from the pool of spare staging
    /// buffers and goes back to it ([`staging_buffer`]); 0: a buffer of its
    /// own, released when this is dropped.
    pooled: u64,
    /// The runtime mapped or unmapped title memory under the texture since
    /// the last check ([`remapped`]): the next upload records its copy and
    /// the check before it reads the memory, whatever the guards, the
    /// write watch or the wait of a partly mapped texture say.
    remapped: Arc<std::sync::atomic::AtomicBool>,
    /// A detiling of title memory has filled the buffer (until then it
    /// holds zeros): a refresh that could not read memory leaves a texture
    /// that held bytes as it was, but must not leave this one empty.
    filled: std::sync::atomic::AtomicBool,
    /// The number the last read of title memory into the buffer began at
    /// (`crate::writes::stamp`); `crate::writes::NOT_READ` until the first
    /// refresh, 0 if that read nothing. The texture pass keeps it per slot
    /// (`crate::descriptor::host::HostHeap::texture_slot_stamp`).
    stamp: Arc<std::sync::atomic::AtomicU64>,
    /// The number the refresh under way began at (`crate::writes::stamp`),
    /// for what a part found as the host copy holds it notes ([`Sent`]).
    reading: std::sync::atomic::AtomicU64,
    /// Diagnostics: a read the title submitted was still to write the
    /// texture's memory when the refresh under way began; what the last
    /// read knew of each part ([`unseen_flags`]).
    #[cfg(feature = "diagnostics")]
    pending_read: std::sync::atomic::AtomicBool,
    #[cfg(feature = "diagnostics")]
    read_flags: [std::sync::atomic::AtomicU64; crate::census::PARTS],
    /// The first level its views show while its most detailed levels' pages
    /// are not there; 0: all ([`Staging::note_levels_there`]).
    shown_from: std::sync::atomic::AtomicU32,
    /// Where the texture's memory is and how its levels lie in it, part
    /// by part ([`Where`]): its digest is taken part by part.
    place: Where,
    /// What the staging buffer holds of each part, detiled: the key of the
    /// memory it was detiled from ([`part_key`]); 0: nothing an upload may
    /// copy.
    staged: [std::sync::atomic::AtomicU64; crate::census::PARTS],
    /// What the texture's host copy holds of each part (the host copy's
    /// own: [`Sent`]).
    sent: Arc<Sent>,
    /// For an upload of the texture pass: the level the view of the slot
    /// it is for starts at ([`picked`]); `u32::MAX`: not said.
    viewed: std::sync::atomic::AtomicU32,
    /// The texture is one of a view the title placed in memory
    /// (`crate::resource::texture_at`), not a texture object of its own:
    /// the title may hold only its less detailed levels, and an upload of
    /// the texture pass leaves the others alone ([`part_is`]).
    placed: bool,
    /// For an upload the texture pass recorded from this staging buffer:
    /// its reason and one (`crate::census`; 0: not the pass's), and
    /// whether its read has been counted.
    picked: std::sync::atomic::AtomicU32,
    counted: std::sync::atomic::AtomicBool,
    /// The number of the write journal's asking the last read of title
    /// memory came after (`crate::watch::asked_now`; 0 until a read).
    asked: Arc<std::sync::atomic::AtomicU64>,
    /// Diagnostics: the number of the last submission that decided a copy
    /// out of the buffer (crate::host::submissions_numbered; 0: none).
    #[cfg(feature = "diagnostics")]
    copy_in: std::sync::atomic::AtomicU64,
}

/// What a title texture's host copy holds of each part of the texture's
/// memory ([`Where`]): the key ([`part_key`]) of the bytes its levels of
/// that part were last copied from, 0 for a part nothing was copied of or
/// whose levels the card wrote since. It lives in the host copy and is
/// written only when a list is submitted, in the order the lists run
/// ([`decide_uploads`], [`CardWrite::publish`]).
///
/// An upload brings a host copy the levels whose bytes it lacks, and no
/// others. Every upload detiled and copied every level of its texture: the
/// texture pass keeps no staging buffer between two uploads of a texture,
/// so nothing knew what the card already had. In the opening cutscene
/// 78,510 of 218,624 uploads found no level changed since the memory's
/// last upload (41.9 GB of title memory), and of the levels of the others
/// 23.3 GB were the same.
///
/// Beside each key, the number of the write journal's asking that the
/// last read which found the part's memory as the host copy holds it came
/// after (`crate::watch::asked_now`; 0: none): what was known of the
/// memory since is counted when an upload reads the part again
/// (`crate::census::SCREEN`). Beside that, the number the same read began
/// at by the writes' count (`crate::writes::stamp`): a tell of the texture
/// pass's after it (memory mapped, written by DirectStorage or by the layer)
/// is numbered higher.
///
/// Diagnostics: what that read knew of the part ([`unseen_flags`]), for the
/// count of parts found changed with nothing seen (`crate::census::unseen`).
#[derive(Default)]
pub(crate) struct Sent(
    [std::sync::atomic::AtomicU64; crate::census::PARTS],
    [std::sync::atomic::AtomicU64; crate::census::PARTS],
    [std::sync::atomic::AtomicU64; crate::census::PARTS],
    #[cfg(feature = "diagnostics")] [std::sync::atomic::AtomicU64; crate::census::PARTS],
);

/// Diagnostics: what a read knew of one part of a texture's memory, for
/// [`Sent`]'s last field: some of its pages were not there (read through the
/// kernel, as zeros there), a read the title submitted was still to write
/// the texture's memory, and the frame of the read (from bit 8 on).
#[cfg(feature = "diagnostics")]
fn unseen_flags(through: bool, pending: bool) -> u64 {
    u64::from(through) | u64::from(pending) << 1 | crate::frame::current() << 8
}

impl Sent {
    /// The key of what the host copy holds of part `part` (0: nothing).
    pub(crate) fn key(&self, part: usize) -> u64 {
        self.0.get(part).map_or(0, |key| key.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The host copy holds nothing an upload brought.
    fn none(&self) -> bool {
        self.0.iter().all(|key| key.load(std::sync::atomic::Ordering::Relaxed) == 0)
    }

    /// The card wrote the subresources `subs` of the texture at `memory`:
    /// the parts they lie in hold no bytes of title memory any more.
    fn card_wrote(&self, memory: &Where, subs: u64) {
        for sub in (0..memory.total.min(64)).filter(|&sub| subs & memory.bit(sub) != 0) {
            let part = memory.part_of_level.get((sub % memory.levels) as usize).copied().unwrap_or(0);
            if let Some(key) = self.0.get(usize::from(part))
                && key.swap(0, std::sync::atomic::Ordering::Relaxed) == 1
            {
                // (Zeros no more: the card's bytes.)
                crate::census::BY_LEVEL[crate::census::L_ZERO_FILLED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// What a refresh does with one part of a texture's memory
/// ([`Staging::refresh_parts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartIs {
    /// The title's, and pages of it are there: read (a page that is not
    /// there as zeros), and its levels on the host brought up to date.
    Read,
    /// The title's, and none of its pages is there: its levels are zeros
    /// on the host until some come.
    Zeros,
    /// Not the title's, or shown by no view with none of its pages
    /// there: neither read nor written on the host.
    Left,
}

/// [`PartIs`] for a part that holds the levels `of_part` (a bit a level)
/// and ends `end` bytes from the texture's start, of a texture judged
/// `[first, limit]` ([`Staging::judgement`]): the lowest level a live view
/// of it shows, and where the next texture placed inside its span starts.
/// A level a view shows is the title's, whatever starts inside it. A more
/// detailed one is the title's only if it ends before the next texture
/// starts: the title holds a texture from the first level it views it
/// from, and what lies past that level's bytes is then another texture's.
/// Only an upload of the texture pass, of a texture the title placed, is
/// so cut short (`cut`): the views the judgement goes by are those in the
/// slots shaders pick by index, which that pass serves. One of the title's
/// texture objects is all its own, and so is any texture as a draw reads
/// it through a table: such a view is in none of those slots, and may
/// start at a more detailed level than they say. `there`: whether any of
/// the part's pages is there, asked only of a part that is the title's.
fn part_is(cut: bool, of_part: u32, end: u64, [first, limit]: [u64; 2], there: impl FnOnce() -> bool) -> PartIs {
    let viewed = reaches(of_part, first);
    if cut && !viewed && end > limit {
        PartIs::Left
    } else if there() {
        PartIs::Read
    } else if cut && !viewed {
        // No view shows it and none of it is there: nothing to bring, and
        // no zeros for it either. (It was staged and copied as zeros: of
        // the levels that so went up in two runs, a quarter ever got
        // bytes; the others were levels below the first one viewed that
        // the title had not mapped, 2.3 GB of staging in one of the runs.)
        PartIs::Left
    } else {
        PartIs::Zeros
    }
}

/// The first level an upload of the texture pass takes as shown: the
/// lowest of what the spans say a view in a slot picked by index shows
/// (`spans`) and where the view of the slot the upload is for starts
/// (`own`). Level 0 if neither says.
fn first_shown(spans: Option<u32>, own: Option<u32>) -> u64 {
    match (spans, own) {
        (Some(spans), Some(own)) => u64::from(spans.min(own)),
        (Some(first), None) | (None, Some(first)) => u64::from(first),
        (None, None) => 0,
    }
}

/// Whether a part that holds the levels `of_part` (a bit a level) holds
/// one from level `first` on.
fn reaches(of_part: u32, first: u64) -> bool {
    u64::from(of_part) >> first.min(32) != 0
}

/// The key of what a part of a texture's memory detiles to: of the digest
/// of its bytes (`digest`, never 0) and, of the levels it holds (`levels`,
/// a bit a level), the ones that are the title's (`held`); the others are
/// staged as zeros whatever lies in their memory. Never 0. One key, one
/// content of the part's levels in the staging buffer.
fn part_key(digest: u64, levels: u32, held: u32) -> u64 {
    let held = levels & held;
    if held == levels {
        digest.max(2)
    } else if held == 0 {
        1
    } else {
        (digest ^ u64::from(held).wrapping_mul(0x9e37_79b9_7f4a_7c15)).max(2)
    }
}

/// The memory of a texture of `size` bytes by level, as byte ranges in
/// memory order: `offsets` are where the levels before the shared block of
/// the smallest ones start (the console's layout puts that block first,
/// then the levels from the smallest up). One part for a texture of
/// several slices or of more than `crate::census::PARTS` levels.
fn memory_parts(size: u64, slices: u32, offsets: &[u64]) -> Vec<(usize, usize)> {
    let size = size as usize;
    let mut cuts: Vec<usize> = offsets.iter().map(|&o| o as usize).filter(|&o| o > 0 && o < size).collect();
    cuts.extend([0, size]);
    cuts.sort_unstable();
    cuts.dedup();
    // (Never none: a texture of no bytes is one part of none.)
    if slices != 1 || cuts.len() < 2 || cuts.len() - 1 > crate::census::PARTS {
        return vec![(0, size)];
    }
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Whether `bytes` are all zero, a word at a time.
fn all_zero(bytes: &[u8]) -> bool {
    // SAFETY: any bytes are valid words.
    let (head, words, tail) = unsafe { bytes.align_to::<u64>() };
    head.iter().all(|&b| b == 0) && words.iter().all(|&w| w == 0) && tail.iter().all(|&b| b == 0)
}

/// A staging buffer nothing holds: the buffer and where it is mapped.
struct Spare(ID3D12Resource, usize);

// SAFETY: a D3D12 buffer and its mapping, touched by nobody while it is
// spare.
unsafe impl Send for Spare {}

/// Spare staging buffers by size (a power of two), and their bytes in all.
static SPARE: Mutex<(std::collections::BTreeMap<u64, Vec<Spare>>, u64)> =
    Mutex::new((std::collections::BTreeMap::new(), 0));
/// The largest buffer that is pooled, and the most bytes kept spare when
/// `VICTORY_STAGING_POOL_MB` says nothing ([`spare_most`]).
const SPARE_LARGEST: u64 = 64 << 20;
const SPARE_MOST_DEFAULT_MB: u64 = 1024;
/// Since the start: staging buffers taken from the spare ones; made anew
/// (pooled sizes and not); spare ones the pool had no room to keep; and
/// buffers taken from the pool that were zeroed because an upload copied
/// from them with nothing detiled into them ([`Staging::wanted`]).
pub(crate) static SPARE_STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The most bytes of spare staging buffers kept: `VICTORY_STAGING_POOL_MB`
/// (1,024; it was 256 MB, with buffers of at most 8 MiB pooled).
fn spare_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_STAGING_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(SPARE_MOST_DEFAULT_MB)
            .min(16 << 10)
            << 20
    })
}

/// A mapped upload buffer of at least `total` bytes for a texture's
/// staging, its pool size (0: not pooled), and whether its bytes may be
/// what an earlier staging left (a spare buffer: only the levels detiled
/// into it are copied from it, and it is zeroed if an upload copies from it
/// with none detiled, [`Staging::wanted`]).
///
/// Every texture the title streams in got a buffer of its own from the
/// host and gave it back when its staging went: about a quarter of a
/// millisecond to make (run 213's profile: 25 ms a frame on the recording
/// threads for some 100 uploads, most of it the graphics kernel waiting)
/// and more to free on the thread that submits. A streaming burst is 200
/// to 500 uploads in a frame (run 222), and those are the frames that
/// dip. Buffers of up to 8 MiB were kept by size and used again, zeroed as
/// they were taken, within 256 MB; with textures made without their top
/// levels nothing waits for room and the opening cutscene staged 100 to
/// 230 MB a frame, so most buffers were made again (32.7 ms a frame in
/// `CreateCommittedResource` on the thread of the texture pass). Now up to
/// 64 MiB and within `VICTORY_STAGING_POOL_MB`, and not zeroed when taken:
/// an upload copies the levels detiled into the buffer and no others, and
/// a staging buffer nothing was detiled into before its first upload is
/// zeroed then.
/// `VICTORY_STAGING_POOL=0`: made and freed each time, as before.
fn staging_buffer(host: &crate::host::Host, total: u64) -> Option<(ID3D12Resource, *mut u8, u64, bool)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let size = total.max(64 << 10).next_power_of_two();
    if size > SPARE_LARGEST || !*ON.get_or_init(|| std::env::var_os("VICTORY_STAGING_POOL").is_none_or(|v| v != "0")) {
        SPARE_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return host.upload_buffer(total).map(|(buffer, data)| (buffer, data, 0, false));
    }
    let spare = {
        let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = all.0.get_mut(&size).and_then(Vec::pop);
        if taken.is_some() {
            all.1 -= size;
        }
        taken
    };
    if let Some(Spare(buffer, data)) = spare {
        SPARE_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return Some((buffer, data as *mut u8, size, true));
    }
    SPARE_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    host.upload_buffer(size).map(|(buffer, data)| (buffer, data, size, false))
}

/// For the 5-second line: staging buffers taken from the pool and made
/// anew since the start, spare ones the pool could not keep, pooled ones
/// zeroed late, and the bytes spare now.
pub(crate) fn staging_pool_stats() -> [u64; 5] {
    use std::sync::atomic::Ordering::Relaxed;
    let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).1;
    [
        SPARE_STATS[0].load(Relaxed),
        SPARE_STATS[1].load(Relaxed),
        SPARE_STATS[2].load(Relaxed),
        SPARE_STATS[3].load(Relaxed),
        spare,
    ]
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Digest>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// Its hash found it changed while its guard saw no write: the title
    /// writes it through another view, so it is hashed at every check.
    behind_guard: bool,
    /// Checks of it through the guards so far (counted up to a few).
    checks: u8,
    /// For a texture read by level ([`Staging::refresh_parts`]): the
    /// digest of each part as last read (0: not read), and how many
    /// refreshes found a part's bytes changed (`hash` is that count).
    parts: [u64; crate::census::PARTS],
    changes: u64,
    /// A read the title had submitted was still to write some of the
    /// texture's memory at the last check (`crate::writes::pending_over`).
    pending: bool,
    /// The frame of the last read through the guards, and one (0: none):
    /// the net's hash is taken once a frame ([`check_reads`]).
    hashed: u64,
    /// For a texture taken for one written behind its guards: checks in a
    /// row that found it unchanged.
    same: u16,
}

/// What a check of a guarded texture knows before it reads
/// ([`check_reads`]).
#[derive(Clone, Copy, Debug, Default)]
struct Check {
    /// The guards saw no write since the last check.
    unwritten: bool,
    /// Its memory was mapped anew since the last check.
    remapped: bool,
    /// Pages of it were missing at a read less than 30 frames ago.
    partly_lately: bool,
    /// One of its first checks.
    young: bool,
    /// It is taken for one written behind its guards.
    behind: bool,
    /// The net's turn for it this frame (every eighth).
    turn: bool,
    /// It was read through the guards this frame already.
    hashed_now: bool,
}

/// Whether a check of a guarded texture reads its memory. A write the
/// guards saw and a mapping told are read at once. Else: a texture with
/// pages missing is left for 30 frames after a read (nothing tells of its
/// bytes but those two, and it is read at most so often, as it was while
/// such a texture was read whole); its first checks and a texture written
/// behind its guards are read every time; and the net, the hash for what
/// the guards do not see, is taken on the texture's turn, once in that
/// frame however many of the frame's submissions sample the texture (it
/// was taken at each of them).
fn check_reads(c: Check) -> bool {
    if !c.unwritten || c.remapped {
        return true;
    }
    if c.partly_lately {
        return false;
    }
    c.young || c.behind || (c.turn && !c.hashed_now)
}

/// Why a check read ([`check_reads`]), as a place of
/// `crate::census::CHECK_READS`: a write seen, a mapping, a first check, a
/// texture behind its guards, the net's turn.
fn check_why(c: Check) -> usize {
    if !c.unwritten {
        0
    } else if c.remapped {
        1
    } else if c.young {
        2
    } else if c.behind {
        3
    } else {
        4
    }
}

/// Checks in a row that find a texture unchanged after which it is no
/// longer taken for one written behind its guards (`Last::behind_guard`):
/// it is hashed on its turn again, and taken for one anew when a change is
/// found that no guard saw. A texture caught once while it was filled was
/// hashed at every check for the rest of the run; the frames of a video
/// change every few checks and stay. (240 at first: after a scene loaded,
/// 119 textures so marked were 51 MB a frame of reads that took 25 seconds
/// to go, 3 MB a frame once they had.)
const BEHIND_CHECKS: u16 = 64;

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Every staging's [`Staging::remapped`] flag by where its texture's title
/// memory starts, with its bytes: [`STAGING`] is keyed by resource and
/// cannot say which textures lie in a range. Weak: the index keeps no
/// staging alive (and dropping what it upgrades frees a flag, nothing
/// more, which may be done with the runtime's locks held).
type RemapIndex = std::collections::BTreeMap<usize, Vec<(usize, std::sync::Weak<std::sync::atomic::AtomicBool>)>>;
static STAGED_AT: Mutex<RemapIndex> = Mutex::new(std::collections::BTreeMap::new());

/// The bytes of the largest texture staged so far: one that overlaps a
/// range starts at most that far below it, which bounds the search.
static LARGEST_STAGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn index_staged(start: usize, bytes: usize, flag: &Arc<std::sync::atomic::AtomicBool>) {
    LARGEST_STAGED.fetch_max(bytes, std::sync::atomic::Ordering::Relaxed);
    let mut index = STAGED_AT.lock().unwrap_or_else(|e| e.into_inner());
    let entries = index.entry(start).or_default();
    entries.retain(|(_, flag)| flag.strong_count() > 0);
    entries.push((bytes, Arc::downgrade(flag)));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: a texture
/// there may show other bytes now while its staging still says "uploaded
/// and unchanged", so the first list after the remap would read the old
/// texels and only the next one the new (fxceless/victory-fork pull
/// request #606, issue #574). Such textures are flagged
/// ([`Staging::remapped`]). Called with the runtime's locks held: locks one
/// mutex of its own and calls nothing else.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let earliest = at.saturating_sub(LARGEST_STAGED.load(Relaxed));
    let index = STAGED_AT.lock().unwrap_or_else(|e| e.into_inner());
    for (&start, entries) in index.range(earliest..end) {
        for (bytes, flag) in entries {
            if start.saturating_add(*bytes) > at
                && let Some(flag) = flag.upgrade()
            {
                flag.store(true, Relaxed);
            }
        }
    }
}

/// Takes the entries of stagings let go out of [`STAGED_AT`].
fn prune_staged_at() {
    STAGED_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, entries| {
        entries.retain(|(_, flag)| flag.strong_count() > 0);
        !entries.is_empty()
    });
}

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    // An upload from title memory earlier in the list (the pass over
    // textures picked by index) has the view noted as used already: only a
    // copy from the texture the host writes, done in this list, stands for
    // what the console reads there (fxceless/victory-fork pull request
    // #1241: the packed light atlas read the uploaded zeros, 0 on an
    // RTX 5080).
    list.first_use(placed as usize);
    let new_source = list.alias_source_changed(to, from);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !new_source && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line, with [`no_wait`]: first reads of a
/// texture never filled that waited for the runtime's remap instead of
/// being left for the next refresh, and the microseconds they waited.
pub(crate) static FIRST_WAITS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line: refreshes that read a texture whose
/// memory was not all mapped (pages not there read as zeros), and those of
/// them that took what they read (it changed and was not all zeros), so
/// that the texture went up with zeros where pages were missing. A texture
/// streaming in is read so while only some of its mips are there; the
/// second count rising where nothing streams is the fault of xmem's notice
/// after the unmap (textures with holes, other ones each run).
pub(crate) static PAGES_MISSING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Diagnostics, since the last line: detilings into a staging buffer; of
/// them, those while a copy out of the buffer that a submission decided had
/// not run (the submission had not finished); of those, the ones whose copy
/// is the submission's being prepared (a list of it refreshed the texture
/// again after one that copies it); and those that wrote zeros for a part
/// whose pages are not there.
#[cfg(feature = "diagnostics")]
static STAGING_UNDER: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Diagnostics: the line of [`STAGING_UNDER`].
#[cfg(feature = "diagnostics")]
pub(crate) fn staging_line() {
    let [all, under, own, zeros] = [0, 1, 2, 3].map(|i| STAGING_UNDER[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    tracing::info!(
        "staging buffers detiled into, since the last line: {all}; {under} while a copy out of the buffer a submission \
         decided had not run (it takes these bytes), {own} of them the copy of the submission being prepared, {zeros} \
         with zeros for pages not there"
    );
}

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    REFRESH_WAITS[slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Staging buffers made without the levels their texture leaves out, since
/// the start.
pub(crate) static STAGED_LEFT_OUT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // Its views' clamp goes with it: a texture made there next starts
        // from none (`Staging::note_levels_there`).
        if self.shown_from.load(Relaxed) != 0 {
            let mut map = SHOWN_FROM.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(map) = map.as_mut() {
                map.remove(&(self.memory as u64));
                SHOWN_ANY.store(!map.is_empty(), Relaxed);
            }
            crate::descriptor::host::levels_changed(self.memory as u64);
        }
        // Nothing holds the staging any more (lists keep it until they are
        // reset, after their submission has run): its buffer is spare.
        if self.pooled != 0 {
            let mut all = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            // Full: spare buffers of the size that holds the most bytes
            // make way (the sizes the title streams change from scene to
            // scene; a pool full of one size made every other size anew).
            let most = spare_most();
            let mut evicted = Vec::new();
            while all.1 + self.pooled > most && self.pooled <= most {
                let Some((&size, _)) = all
                    .0
                    .iter()
                    .filter(|(_, v)| !v.is_empty())
                    .max_by_key(|&(size, v)| *size * v.len() as u64)
                else {
                    break;
                };
                if let Some(spare) = all.0.get_mut(&size).and_then(Vec::pop) {
                    all.1 -= size;
                    evicted.push(spare);
                }
            }
            if !evicted.is_empty() {
                SPARE_STATS[2].fetch_add(evicted.len() as u64, Relaxed);
            }
            if all.1 + self.pooled <= most {
                all.1 += self.pooled;
                all.0.entry(self.pooled).or_default().push(Spare(self.buffer.clone(), self.data as usize));
            } else {
                SPARE_STATS[2].fetch_add(1, Relaxed);
            }
            // (Released after the lock.)
            drop(all);
            drop(evicted);
        }
    }
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_mut() else {
        return;
    };
    if map.is_empty() {
        return;
    }
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    // A list that recorded an upload or a check holds the staging until it
    // is reset: one only the table holds is named by no list, recorded or
    // running.
    let (mut went, mut bytes) = (0u64, 0u64);
    map.retain(|_, s| {
        let keep = Arc::strong_count(s) > 1
            || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
        if !keep {
            LET_GO[0].fetch_add(1, Relaxed);
            LET_GO[1].fetch_add(s.bytes, Relaxed);
            went += 1;
            bytes += s.bytes;
        }
        keep
    });
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
        prune_staged_at();
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// Since the start, for the textures the title fills (no target, depth or
/// unordered access flag) that the card itself wrote by a copy or a
/// resolve, such as the block-compressed textures the title's decoration
/// bake copies its result into, level by level:
/// - (0) such textures at their first card copy;
/// - (1) of them, those an upload from title memory had been recorded for
///   before (their memory holds nothing the card wrote: what its pages
///   held before);
/// - (2) subresources copied from title memory at a submission although
///   the card had written them last, by what the lists submitted before
///   had left (what the card wrote is overwritten): 27 uploads a run did
///   that while uploads were decided as the lists were recorded. Nothing
///   can now: the decision leaves those subresources out, and this counts
///   what it lets through all the same;
/// - (3) textures of views the title placed in memory that lay in the
///   memory of such a texture when the card first copied into it: each is
///   a host texture of its own, filled from title memory;
/// - (4) uploads from title memory recorded for the texture of a placed
///   view that lies in the memory of a texture the card copied into;
/// - (5) uploads withheld at their list's submission, in whole or in part,
///   because the card wrote those subresources last;
/// - (6) uploads not asked for because the card wrote every subresource
///   last (whether the title wrote since is asked when the list is
///   submitted);
/// - (7) times the title was found to have written the memory after the
///   card wrote the texture: the subresources it wrote are its own again
///   and go up from title memory;
/// - (8) subresources uploaded in the prelude of a list that itself then
///   copies into them on the card (the texture pass asked in the bake's
///   own list): the list's copy runs after its prelude, so the card's
///   bytes stand; the upload is work for nothing, and no fault. (The first
///   count's place held these for one run: 1,836.)
/// - (9) textures of placed views brought what the card wrote by a copy
///   on the card ([`Feed`]);
/// - (10) textures of placed views that start where a card-written
///   texture does and do not fit it (other texels, levels or format
///   family): no copy can bring them the card's bytes;
/// - (11), (12), (13) of (3): those that start where the texture does,
///   those with a host copy at that moment, and those a slot showed then.
pub(crate) static CARD_WRITTEN: [std::sync::atomic::AtomicU64; 14] = [const { std::sync::atomic::AtomicU64::new(0) }; 14];

/// Title textures an upload from title memory was ever recorded for
/// ([`upload`]), for [`CARD_WRITTEN`]'s second count.
static UPLOAD_RECORDED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// The line on [`CARD_WRITTEN`], with the periodic lines; nothing while
/// the card has copied into no texture the title fills.
pub(crate) fn card_written_line() {
    use std::sync::atomic::Ordering::Relaxed;
    let [first, uploaded_before, overtaken, placed_inside, placed_uploads, withheld, unasked, again, own, fed, unfit, at_start, made, shown] =
        CARD_WRITTEN.each_ref().map(|c| c.load(Relaxed));
    if first == 0 {
        return;
    }
    tracing::info!(
        "textures the title fills that the card copied into, since the start: {first} at their first copy, {uploaded_before} \
         of them with an upload from title memory recorded before it; {overtaken} subresources copied from title memory \
         over what the card had written last (0: none can be); {own} uploaded ahead of a card copy of their own list \
         (the copy runs after: nothing lost); {withheld} uploads withheld at submission because the card wrote last, \
         {unasked} not asked for because it wrote all of the texture last, {again} times the title was found to have \
         written the memory after the card (uploaded again); {placed_inside} textures of placed views lay in the memory \
         of such a texture at its first copy ({at_start} at its start, {made} with a host copy, {shown} shown by a slot), \
         {fed} such textures brought the card's bytes by a copy on the card, {unfit} at its start that do not fit it, \
         {placed_uploads} uploads from title memory into such textures"
    );
}

/// Where a title texture's memory is and how its subresources lie in it,
/// for what the card wrote of it ([`CardState`]). Taken when a list records
/// a card copy into the texture, while the title's object is surely there:
/// the list may be submitted after the title let the object go.
#[derive(Clone, Debug)]
pub(crate) struct Where {
    start: usize,
    size: usize,
    /// The memory by level ([`memory_parts`]) and the part each level's
    /// bytes lie in.
    parts: Vec<(usize, usize)>,
    part_of_level: Vec<u8>,
    /// The levels each part holds, a bit a level.
    levels_of_part: Vec<u32>,
    /// Levels a layer, and subresources in all.
    levels: u32,
    total: u32,
}

impl Where {
    /// Of a texture of `size` bytes at `start` with `levels` levels a
    /// layer and `total` subresources, laid out in `slices` slices with
    /// the levels before the shared block of the smallest ones at
    /// `offsets`.
    fn new(start: usize, size: u64, slices: u32, offsets: &[u64], levels: u32, total: u32) -> Where {
        let parts = memory_parts(size, slices, offsets);
        let part_of_level: Vec<u8> = (0..levels as usize)
            .map(|level| match offsets.get(level) {
                Some(&at) if parts.len() > 1 => parts.iter().position(|p| p.0 as u64 == at).unwrap_or(0) as u8,
                _ => 0,
            })
            .collect();
        let mut levels_of_part = vec![0u32; parts.len()];
        for (level, &part) in part_of_level.iter().enumerate() {
            // (A level past the thirty-second is told by no bit: every part
            // of such a texture is taken with every other, see `fits`.)
            if let Some(of_part) = levels_of_part.get_mut(usize::from(part)) {
                *of_part |= 1u32.checked_shl(level as u32).unwrap_or(0);
            }
        }
        Where { start, size: size as usize, parts, part_of_level, levels_of_part, levels: levels.max(1), total: total.max(1) }
    }

    /// Of the texture at `start` that `layout` lays out, with `levels`
    /// levels a layer and `total` subresources on the host: by level if
    /// the layout itself puts every level whole into the part taken for
    /// it ([`Where::fits`]), else one part of everything.
    fn of(start: usize, layout: &victory_formats::texture::Layout, levels: u32, total: u32) -> Where {
        let tail = (layout.first_tail_mip as usize).min(layout.mips.len());
        let offsets: Vec<u64> = layout.mips[..tail].iter().map(|m| m.offset).collect();
        let by_level = Where::new(start, layout.size, layout.slices, &offsets, levels, total);
        if by_level.parts.len() < 2 || by_level.fits(layout) {
            by_level
        } else {
            // (No slices: one part.)
            Where::new(start, layout.size, 0, &[], levels, total)
        }
    }

    /// Whether every level's bytes lie in the part taken for it, by where
    /// the layout puts the level's corner texels: a level is detiled and
    /// copied when its part's bytes changed, so no byte of it may lie in
    /// another.
    fn fits(&self, layout: &victory_formats::texture::Layout) -> bool {
        let n = u64::from(layout.desc.element_bytes);
        self.levels as usize == layout.mips.len()
            && self.levels <= 32
            && self.levels_of_part.iter().all(|&levels| levels != 0)
            && layout.mips.iter().enumerate().all(|(level, m)| {
                let part = self.part_of_level.get(level).and_then(|&part| self.parts.get(usize::from(part)));
                let Some(&(from, to)) = part else {
                    return false;
                };
                let (w, h) = (m.width.saturating_sub(1), m.height.saturating_sub(1));
                [(0, 0), (w, 0), (0, h), (w, h)].into_iter().all(|(x, y)| {
                    layout
                        .offset(level as u32, x, y, 0)
                        .is_ok_and(|at| at >= from as u64 && at.saturating_add(n) <= to as u64)
                })
            })
    }

    /// Subresource `sub` as a bit of a mask; every bit for a texture of
    /// more subresources than a mask tells apart (it is then all the
    /// card's or all the title's).
    fn bit(&self, sub: u32) -> u64 {
        subresource_bit(sub, self.total)
    }

    /// Every subresource.
    fn all(&self) -> u64 {
        if self.total >= 64 { u64::MAX } else { (1u64 << self.total) - 1 }
    }

    /// A digest of each part of the memory as it is now (never 0), read
    /// through the kernel so that pages not there count as zeros. All 0 for
    /// a texture too long to be one (nothing is then ever told changed).
    fn digests(&self) -> [u64; crate::census::PARTS] {
        let mut out = [0u64; crate::census::PARTS];
        if self.size == 0 || self.size > 1 << 30 {
            return out;
        }
        // (A buffer of its own: this runs between a list's reads, and the
        // thread's read buffer may be lent out.)
        let mut bytes = vec![0u8; self.size];
        crate::host::copy_mapped(bytes.as_mut_ptr(), self.start as *const u8, self.size, false);
        for (digest_of, &(from, to)) in out.iter_mut().zip(&self.parts) {
            let d = digest(bytes.get(from..to.min(bytes.len())).unwrap_or(&[])).0;
            *digest_of = (d as u64 ^ (d >> 64) as u64).max(1);
        }
        out
    }

    /// The subresources whose bytes lie in a part that differs between the
    /// digests `a` and `b`.
    fn changed(&self, a: &[u64; crate::census::PARTS], b: &[u64; crate::census::PARTS]) -> u64 {
        let mut subs = 0;
        for sub in 0..self.total.min(64) {
            let part = self.part_of_level.get((sub % self.levels) as usize).copied().unwrap_or(0) as usize;
            if a.get(part) != b.get(part) {
                subs |= self.bit(sub);
            }
        }
        subs
    }
}

/// Subresource `sub` of `total` as a bit of a mask (see [`Where::bit`]).
fn subresource_bit(sub: u32, total: u32) -> u64 {
    if total > 64 { u64::MAX } else { 1u64 << sub.min(63) }
}

/// [`Where`] for title texture `resource` with the host copy `texture`;
/// `None` for one without memory or a console layout.
fn where_of(resource: This, texture: &ID3D12Resource) -> Option<Where> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())?;
    if memory.is_null() || layout.size == 0 {
        return None;
    }
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let levels = u32::from(d.MipLevels).max(1);
    let total = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        levels
    } else {
        levels * u32::from(d.DepthOrArraySize).max(1)
    };
    Some(Where::of(memory as usize, &layout, levels, total))
}

/// What the card wrote of a title texture's host copy since the title last
/// wrote the texture's memory: for a texture both can write, who wrote
/// last is right, and an upload from title memory must not run after a
/// card copy into the same subresource unless the title wrote the memory
/// in between. It lives in the host copy (as upstream's flag does,
/// Marvisak/VIctory pull request #1934) and changes only when a list is
/// submitted, in the order the lists run ([`decide_uploads`],
/// [`CardWrite::publish`]): a list that is recorded and never sent changes
/// nothing.
///
/// "The title wrote since" is told by the memory itself: a digest of each
/// level's bytes is kept from the card's write, and a subresource whose
/// bytes are other than then is the title's again (as AndyLatimer's branch
/// tells it for memory nothing watches, d5344a8; here for all of it, by
/// level). The bytes are read only when the write journal or the guards
/// say something may have written them ([`crate::watch::seen_since`]).
#[derive(Default)]
pub(crate) struct CardState {
    /// The subresources the card wrote last (a bit each), and every
    /// subresource the texture has (0 until the card first wrote it), for
    /// those who ask without the lock.
    subs: std::sync::atomic::AtomicU64,
    all: std::sync::atomic::AtomicU64,
    card: Mutex<Option<Card>>,
}

/// See [`CardState`]: where the memory is, the digests of its parts when
/// they were last compared, and the journal's asking that was after.
struct Card {
    memory: Where,
    digests: [u64; crate::census::PARTS],
    asked: u64,
}

/// Has the write watch or the guards take `memory` anew, so that a store
/// after this is seen: the asking's number (0: nothing watches it).
fn watch_anew(memory: &Where) -> u64 {
    crate::watch::writes(memory.start, memory.start.saturating_add(memory.size), false)
        .or_else(|| crate::watch::writes(memory.start, memory.start.saturating_add(memory.size), true))
        .map_or(0, |(now, _)| now)
}

impl CardState {
    /// Whether something may have written the memory since its bytes were
    /// last compared: asked of the journal and the guards, no byte read.
    fn maybe_written(card: &Card) -> bool {
        let end = card.memory.start.saturating_add(card.memory.size);
        let (noted, written, open) = crate::watch::seen_since(card.memory.start, end, card.asked);
        noted || written > 0 || open > 0
    }

    /// Compares the memory with what it held when last compared, if
    /// something may have written it: the card-written subresources whose
    /// bytes are other now are the title's again. Returns them.
    fn settle(&self, card: &mut Card) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        if !Self::maybe_written(card) {
            return 0;
        }
        let asked = watch_anew(&card.memory);
        let now = card.memory.digests();
        let rewritten = card.memory.changed(&card.digests, &now) & self.subs.load(Relaxed);
        (card.digests, card.asked) = (now, asked);
        if rewritten != 0 {
            self.subs.fetch_and(!rewritten, Relaxed);
        }
        rewritten
    }

    /// At the submission of a list that asks for an upload of the
    /// subresources `wanted` from title memory: those to copy (the title's
    /// bytes are the last written), those withheld (the card's are), and
    /// those of the first that the title wrote after the card.
    fn decide(&self, wanted: u64) -> (u64, u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.subs.load(Relaxed) & wanted == 0 {
            return (wanted, 0, 0);
        }
        let mut card = self.card.lock().unwrap_or_else(|e| e.into_inner());
        let rewritten = card.as_mut().map_or(0, |card| self.settle(card));
        let subs = self.subs.load(Relaxed);
        (wanted & !subs, wanted & subs, rewritten & wanted)
    }

    /// At the submission of a list that writes the subresources `subs` of
    /// the texture at `memory` on the card: from here on the card's bytes
    /// stand for them. What the title wrote before is settled first, so
    /// that a subresource it wrote and this list does not write stays the
    /// title's.
    fn wrote(&self, memory: &Where, subs: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let mut card = self.card.lock().unwrap_or_else(|e| e.into_inner());
        match card.as_mut() {
            Some(card) => {
                self.settle(card);
            }
            None => {
                let asked = watch_anew(memory);
                self.all.store(memory.all(), Relaxed);
                *card = Some(Card { digests: memory.digests(), memory: memory.clone(), asked });
            }
        }
        self.subs.fetch_or(subs, Relaxed);
    }

    /// For a list being recorded: whether the card wrote every subresource
    /// last, as far as the lists submitted so far say. Two loads: nothing
    /// is asked of the journal or the guards on a thread that records.
    /// Such a list asks for no upload and has the memory looked at when it
    /// is submitted ([`CardState::check`]).
    fn all_card(&self) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let all = self.all.load(Relaxed);
        all != 0 && self.subs.load(Relaxed) & all == all
    }

    /// At the submission of a list that asked for no upload because the
    /// card wrote all of the texture last: whether the title wrote its
    /// memory since. The subresources it wrote are its own again, and the
    /// next list that samples the texture uploads them.
    fn check(&self) -> u64 {
        let mut card = self.card.lock().unwrap_or_else(|e| e.into_inner());
        card.as_mut().map_or(0, |card| self.settle(card))
    }

    /// The subresources the card wrote last (tests).
    #[cfg(test)]
    pub(crate) fn subs(&self) -> u64 {
        self.subs.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A card copy or resolve a list recorded into a texture the title fills:
/// noted for the lists after it when the list is submitted
/// ([`CardWrite::publish`]).
pub(crate) struct CardWrite {
    copy: Arc<crate::resource::HostTexture>,
    memory: Where,
    subs: u64,
}

impl CardWrite {
    /// The list is submitted: the card's bytes stand for what it writes,
    /// in every host texture of that memory ([`feed_twins`]).
    pub(crate) fn publish(&self) {
        card_wrote(&self.copy, &self.memory, self.subs);
        feed_twins(&self.copy, &self.memory);
    }

    /// The host copy written and the subresources.
    fn written(&self) -> (&Arc<crate::resource::HostTexture>, u64) {
        (&self.copy, self.subs)
    }

    /// If this is the note of `copy`: the list writes `subs` of it too.
    pub(crate) fn more(&mut self, copy: &Arc<crate::resource::HostTexture>, subs: u64) -> bool {
        let same = Arc::ptr_eq(&self.copy, copy);
        if same {
            self.subs |= subs;
        }
        same
    }
}

/// From here on the card's bytes stand for the subresources `subs` of host
/// copy `copy`, whose memory is `memory`: noted for the uploads decided
/// after ([`CardState::wrote`]), and what the copy held of title memory
/// there is gone ([`Sent::card_wrote`]).
fn card_wrote(copy: &crate::resource::HostTexture, memory: &Where, subs: u64) {
    copy.card.wrote(memory, subs);
    copy.sent.card_wrote(memory, subs);
}

/// The textures the card wrote, by where their memory starts: its bytes
/// and the host copy (weak: a copy let go is not kept by this). For the
/// textures of views the title placed in the same memory ([`card_twin`]).
/// Written and read at submissions only.
static CARD_AT: Mutex<std::collections::BTreeMap<u64, (u64, std::sync::Weak<crate::resource::HostTexture>)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A copy on the card that brings the texture of a view placed in a
/// card-written texture's memory what the card wrote there: the
/// subresources `subs` of `from` into the same ones of `to`. The title
/// may view one memory through several descriptions (the formats of one
/// family, say), each a host texture of its own here, where the console
/// has one set of bytes: what the card writes into one is in none of the
/// others, and title memory never holds it. Recorded into the prelude of
/// the next list submitted, so after the list whose copy it follows and
/// before whatever comes next ([`take_feeds`]).
pub(crate) struct Feed {
    from: Arc<crate::resource::HostTexture>,
    to: Arc<crate::resource::HostTexture>,
    subs: u64,
}

/// The feeds waiting for the next prelude.
static FEEDS: Mutex<Vec<Feed>> = Mutex::new(Vec::new());

/// The feeds noted since the last call: for the prelude being recorded.
pub(crate) fn take_feeds() -> Vec<Feed> {
    std::mem::take(&mut *FEEDS.lock().unwrap_or_else(|e| e.into_inner()))
}

impl Feed {
    /// The texture read and the texture written.
    pub(crate) fn textures(&self) -> [&Arc<crate::resource::HostTexture>; 2] {
        [&self.from, &self.to]
    }

    /// Records the copies into `list`, the one texture a copy source and
    /// the other a copy destination.
    pub(crate) fn record(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: a live resource.
        let d = unsafe { self.from.texture.GetDesc() };
        let levels = u32::from(d.MipLevels).max(1);
        let total = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            levels
        } else {
            levels * u32::from(d.DepthOrArraySize).max(1)
        };
        let part = |texture: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        for sub in (0..total).filter(|&sub| self.subs & subresource_bit(sub, total) != 0) {
            // SAFETY: an open list; live textures of the same subresources
            // ([`fits`]) in the copy states.
            unsafe { list.CopyTextureRegion(&part(&self.to.texture, sub), 0, 0, 0, &part(&self.from.texture, sub), None) };
        }
    }
}

/// The family of formats a format belongs to, as its typeless member's
/// number: the host copies between two textures of one family as they
/// are. A format of no family here is its own.
fn format_family(format: u32) -> u32 {
    match format {
        1..=4 => 1,
        5..=8 => 5,
        9..=14 => 9,
        15..=18 => 15,
        19..=22 => 19,
        23..=25 => 23,
        27..=32 => 27,
        33..=38 => 33,
        39..=43 => 39,
        44..=47 => 44,
        48..=52 => 48,
        53..=59 => 53,
        60..=64 => 60,
        70..=72 => 70,
        73..=75 => 73,
        76..=78 => 76,
        79..=81 => 79,
        82..=84 => 82,
        87 | 90 | 91 => 90,
        88 | 92 | 93 => 92,
        94..=96 => 94,
        97..=99 => 97,
        other => other,
    }
}

/// Whether a copy on the card can bring `to` what `from` holds,
/// subresource for subresource: the same texels, levels and layers, in
/// formats of one family.
fn fits(from: &ID3D12Resource, to: &ID3D12Resource) -> bool {
    // SAFETY: live resources.
    let (a, b) = unsafe { (from.GetDesc(), to.GetDesc()) };
    (a.Dimension, a.Width, a.Height, a.DepthOrArraySize, a.MipLevels, a.SampleDesc.Count)
        == (b.Dimension, b.Width, b.Height, b.DepthOrArraySize, b.MipLevels, b.SampleDesc.Count)
        && format_family(a.Format.0 as u32) == format_family(b.Format.0 as u32)
}

/// At the submission of a list that wrote `source` on the card, whose
/// memory is `memory`: the host copies of the views the title placed at
/// the start of that memory are brought what the card wrote last
/// ([`Feed`], in the next prelude), and count as written by the card
/// themselves from here on, so that no upload from title memory goes over
/// it. One that does not fit is left as it is, and counted.
fn feed_twins(source: &Arc<crate::resource::HostTexture>, memory: &Where) {
    use std::sync::atomic::Ordering::Relaxed;
    CARD_AT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(memory.start as u64, (memory.size as u64, Arc::downgrade(source)));
    let subs = source.card.subs.load(Relaxed);
    for twin in crate::resource::placed_copies_at(memory.start as u64) {
        if Arc::ptr_eq(&twin, source) {
            continue;
        }
        if !fits(&source.texture, &twin.texture) {
            CARD_WRITTEN[10].fetch_add(1, Relaxed);
            continue;
        }
        CARD_WRITTEN[9].fetch_add(1, Relaxed);
        card_wrote(&twin, memory, subs);
        FEEDS.lock().unwrap_or_else(|e| e.into_inner()).push(Feed { from: source.clone(), to: twin, subs });
    }
}

/// For an upload asked of `staging` into `copy`: the texture the card
/// wrote in the same memory, if `copy` is another one's (a view placed
/// there) and fits it.
fn card_twin(staging: &Staging, copy: &Arc<crate::resource::HostTexture>) -> Option<Arc<crate::resource::HostTexture>> {
    let mut all = CARD_AT.lock().unwrap_or_else(|e| e.into_inner());
    let start = staging.memory as u64;
    let &(size, ref source) = all.get(&start)?;
    let Some(source) = source.upgrade() else {
        all.remove(&start);
        return None;
    };
    drop(all);
    if size != staging.layout.size || Arc::ptr_eq(&source, copy) || !fits(&source.texture, &copy.texture) {
        return None;
    }
    // Only what is still the card's there: if the title wrote the memory
    // since (a texture streamed into the place of the one the card wrote),
    // its bytes are the truth for every view of it, this one too.
    source.card.check();
    Some(source)
}

/// An upload from title memory a list asked for ([`upload`]): decided when
/// the list is submitted ([`decide_uploads`]) and recorded into its prelude
/// ([`record_copies`]). `copied`: the levels it would write (all but those
/// a texture made without them leaves out).
pub(crate) struct TextureUpload {
    copy: Arc<crate::resource::HostTexture>,
    staging: Arc<Staging>,
    copied: u32,
}

/// What [`decide_uploads`] found to copy: the host copy, the staging
/// buffer, and the subresources (a bit each, see [`Where::bit`]).
pub(crate) type Copies = (Arc<crate::resource::HostTexture>, Arc<Staging>, u64);

/// At a list's submission: the textures it asked no upload for because the
/// card wrote all of them last are looked at ([`CardState::check`]).
pub(crate) fn check_card_written(copies: &[Arc<crate::resource::HostTexture>]) {
    for copy in copies {
        if copy.card.check() != 0 {
            CARD_WRITTEN[7].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// At a list's submission, before the card writes it records are noted:
/// which of the uploads it asked for are made. One is made if the staging
/// buffer holds bytes its host copy lacks (never uploaded, or detiled anew
/// since: its refresh ran with the list's steps), and of it the
/// subresources the card did not write last ([`CardState::decide`]).
/// Decided here and not while the list is recorded, because lists run in
/// the order they are submitted: an upload recorded before a card copy
/// into the same texture ran after it when its list was submitted later,
/// and what the card had written was gone (Marvisak/VIctory issue #1891
/// and pull request #1934, zondy1i, whose design this is).
pub(crate) fn decide_uploads(uploads: &[TextureUpload]) -> Vec<Copies> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut copies = Vec::new();
    for upload in uploads {
        let staging = &upload.staging;
        let sent = &upload.copy.sent;
        // Every subresource the upload may write.
        let all = staging.subresources(upload.copied);
        // The texture of a view placed in the memory of a texture the card
        // wrote (its host copy made, or first asked for, after that): what
        // the card wrote is brought by a copy on the card in this prelude,
        // and is the card's here too from now on.
        if CARD_WRITTEN[0].load(Relaxed) != 0
            && let Some(source) = card_twin(staging, &upload.copy)
        {
            let subs = source.card.subs.load(Relaxed) & all & !upload.copy.card.subs.load(Relaxed);
            if subs != 0 {
                CARD_WRITTEN[9].fetch_add(1, Relaxed);
                card_wrote(&upload.copy, &staging.place, subs);
                FEEDS.lock().unwrap_or_else(|e| e.into_inner()).push(Feed { from: source, to: upload.copy.clone(), subs });
            }
        }
        // Of them, those of the levels whose bytes the host copy lacks:
        // the staging buffer holds them detiled (its refresh ran with the
        // list's steps), and what the host copy holds is other ([`Sent`]).
        let wanted = all & staging.wanted(sent);
        let (subs, withheld, rewritten) = upload.copy.card.decide(wanted);
        CARD_WRITTEN[5].fetch_add(u64::from(withheld != 0), Relaxed);
        CARD_WRITTEN[7].fetch_add(u64::from(rewritten != 0), Relaxed);
        // What is copied and what the card wrote last have nothing in
        // common (the state is what the lists before this one left; this
        // list's own card copies are noted after its uploads are decided).
        let over = subs & upload.copy.card.subs.load(Relaxed);
        CARD_WRITTEN[2].fetch_add(u64::from(over.count_ones()), Relaxed);
        // The host copy then holds what the staging buffer holds, or what
        // the card wrote after it.
        staging.uploaded.store(true, Relaxed);
        staging.dirty.store(false, Relaxed);
        staging.copied.store(upload.copied, Relaxed);
        // What a slot that shows the texture reads is settled once this
        // submission has run, copied or not (`crate::descriptor::after`).
        crate::descriptor::after::uploaded(&upload.copy.texture);
        let by_level = &crate::census::BY_LEVEL;
        if subs == 0 {
            by_level[crate::census::L_NONE].fetch_add(1, Relaxed);
        } else {
            by_level[crate::census::L_SOME].fetch_add(1, Relaxed);
            by_level[crate::census::L_COPIED].fetch_add(staging.bytes_of(subs), Relaxed);
            staging.mark_sent(sent, subs);
            #[cfg(feature = "diagnostics")]
            staging.copy_in.store(crate::host::submissions_numbered()[0], Relaxed);
            copies.push((upload.copy.clone(), staging.clone(), subs));
        }
        // What a view may show of the texture and the host copy holds
        // nothing for, neither the title's bytes nor the card's: counted,
        // and the first of them said. Where the upload brings levels: one
        // that brings none found the texture's memory gone or all zeros
        // and leaves the host copy as it is, with whatever view it had.
        let bare = if subs == 0 { 0 } else { staging.bare(sent) & !upload.copy.card.subs.load(Relaxed) };
        if bare != 0 {
            by_level[crate::census::L_BARE].fetch_add(u64::from(bare.count_ones()), Relaxed);
            static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if SAID.fetch_add(1, Relaxed) < 8 {
                tracing::warn!(
                    memory = ?staging.memory,
                    bytes = staging.layout.size,
                    levels = staging.mips,
                    subresources = format_args!("{bare:#x}"),
                    copied = format_args!("{subs:#x}"),
                    first_shown = staging.judged[0].load(Relaxed),
                    next_texture_at = staging.judged[1].load(Relaxed),
                    slot_view_from = staging.viewed.load(Relaxed),
                    by_the_pass = staging.picked.load(Relaxed) != 0,
                    "an upload was decided and levels a view may show hold nothing on the host, neither the title's bytes \
                     nor the card's, though other levels of the texture were brought"
                );
            }
        }
    }
    copies
}

/// Counts the subresources a list uploads in its prelude (`copies`) and
/// then itself copies into on the card (`writes`): [`CARD_WRITTEN`]'s
/// ninth count. The copy runs after the prelude.
pub(crate) fn count_own_writes(copies: &[Copies], writes: &[CardWrite]) {
    for (copy, _, subs) in copies {
        let own = writes.iter().filter(|w| Arc::ptr_eq(w.written().0, copy)).fold(0u64, |all, w| all | w.written().1);
        CARD_WRITTEN[8].fetch_add(u64::from((subs & own).count_ones()), std::sync::atomic::Ordering::Relaxed);
    }
}

/// Records into `list` (a list's prelude, the texture in the copy
/// destination state) the copies of the staging buffer's subresources
/// `subs` into the host copy.
pub(crate) fn record_copies(list: &ID3D12GraphicsCommandList, copy: &crate::resource::HostTexture, staging: &Staging, subs: u64) {
    let total = staging.first + staging.footprints.len() as u32;
    for (i, footprint) in staging.footprints.iter().enumerate() {
        let sub = staging.first + i as u32;
        let bit = subresource_bit(sub, total);
        if subs & bit == 0 {
            continue;
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        // SAFETY: an open list; both locations name resources the list
        // that asked for the upload holds until it is reset.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// A host copy or resolve recorded in `list` writes subresource `sub` of
/// `resource` (`None`: every one), whose host copy is `texture`. For a
/// texture the title fills, the card's bytes stand for them from the
/// list's submission on ([`CardState`]).
pub(crate) fn host_written(list: &HostList, resource: This, texture: &ID3D12Resource, sub: Option<u32>) {
    use std::sync::atomic::Ordering::Relaxed;
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    if title_written(texture) {
        if new {
            CARD_WRITTEN[0].fetch_add(1, Relaxed);
            let uploaded = UPLOAD_RECORDED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_some_and(|set| set.contains(&(resource as usize)));
            CARD_WRITTEN[1].fetch_add(u64::from(uploaded), Relaxed);
        }
        // One note a list for each texture: the bake copies a dozen
        // levels into one, each by a call of its own.
        if let Some(copy) = crate::resource::host_copy(resource) {
            // SAFETY: a live resource.
            let d = unsafe { texture.GetDesc() };
            let levels = u32::from(d.MipLevels).max(1);
            let total = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                levels
            } else {
                levels * u32::from(d.DepthOrArraySize).max(1)
            };
            let subs = sub.map_or(if total >= 64 { u64::MAX } else { (1u64 << total) - 1 }, |sub| subresource_bit(sub, total));
            if !list.card_write_more(&copy, subs)
                && let Some(memory) = where_of(resource, texture)
            {
                list.card_write(CardWrite { copy, memory, subs });
            }
        }
    }
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
            // The textures of placed views made in its memory so far (the
            // table keeps every one ever made, so most are of textures that
            // lay there before the memory was this texture's): those that
            // start where it does, those with a host copy now, those a slot
            // shows now. One at its start that fits it is brought the
            // card's bytes at the submission ([`feed_twins`]).
            let inside = crate::resource::placed_in(at as u64, bytes, resource);
            if !inside.is_empty() {
                let at_start = inside.iter().filter(|p| p.0 == at as u64).count() as u64;
                let made = inside.iter().filter(|p| crate::resource::copy_state(p.1).is_some_and(|s| s.0 == 2)).count() as u64;
                let shown = inside.iter().filter(|p| crate::descriptor::host::first_viewed(p.0).is_some()).count() as u64;
                for (place, n) in [(3, inside.len() as u64), (11, at_start), (12, made), (13, shown)] {
                    CARD_WRITTEN[place].fetch_add(n, Relaxed);
                }
                static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if SAID.fetch_add(1, Relaxed) < 16 {
                    tracing::warn!(
                        texture = ?resource,
                        memory = format_args!("{:#x}", at as u64),
                        bytes,
                        width = words[4],
                        height = words[6],
                        format = words[8],
                        placed_views_textures = inside.len(),
                        at_its_start = at_start,
                        with_a_host_copy = made,
                        shown_by_a_slot = shown,
                        "the card copies into a texture in whose memory the title placed views before: one that starts \
                         where it does and fits it is brought what the card writes, the others keep what they have"
                    );
                }
            }
        }
    }
}

/// For [`upload`]: an upload from title memory of `resource` is recorded.
/// Counts the one recorded for the texture of a placed view that lies in
/// the memory of a texture the card copied into ([`CARD_WRITTEN`]), and
/// notes the texture as uploaded.
fn upload_recorded(resource: This) {
    use std::sync::atomic::Ordering::Relaxed;
    let new = UPLOAD_RECORDED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert(resource as usize);
    // Nothing the card copied into: no memory to lie in.
    if CARD_WRITTEN[0].load(Relaxed) == 0 {
        return;
    }
    let Some((at, _)) = crate::resource::texture_memory(resource) else {
        return;
    };
    let address = at as u64;
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(start, d)| address < start.saturating_add(d.0) && d.5 != resource as usize)
            .map(|(start, d)| (*start, *d))
    };
    if let Some((start, (bytes, width, height, _, format, written))) = hit {
        CARD_WRITTEN[4].fetch_add(1, Relaxed);
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if new && SAID.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(
                texture = ?resource,
                memory = format_args!("{address:#x}"),
                card_written_texture = format_args!("{written:#x}"),
                its_memory = format_args!("{start:#x}"),
                its_bytes = bytes,
                its_width = width,
                its_height = height,
                its_format = format,
                "an upload from title memory into the texture of a view placed in the memory of a texture the card copied \
                 into: it shows title memory, not what the card wrote"
            );
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    drop(set);
    if let Some(set) = UPLOAD_RECORDED.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        set.remove(&resource);
    }
    // No view placed in its memory is fed from it any more ([`CARD_AT`]).
    {
        let at = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        let mut card = CARD_AT.lock().unwrap_or_else(|e| e.into_inner());
        for (start, _) in at.iter().filter(|(_, d)| d.5 == resource) {
            card.remove(start);
        }
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    if !list.first_use(resource as usize) {
        return;
    }
    // A texture the card wrote all of last: the host copy has what is
    // right unless the title wrote the memory since, which is asked when
    // the list is submitted (its next reader then uploads); no staging
    // buffer is made and no byte read for it. (Whether the card wrote is
    // what lists submitted so far did; a card copy only recorded, in any
    // list, changes nothing here: an upload asked for is decided when its
    // own list is submitted.)
    let card_copied = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    if card_copied
        && let Some(copy) = crate::resource::host_copy(resource)
        && copy.card.all_card()
    {
        CARD_WRITTEN[6].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        list.card_check(copy);
        return;
    }
    let Some(mut staging) = staging(resource, texture) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    // A staging buffer made for a host copy that was let go since: made
    // anew for the one there is now, whose levels it brings ([`Sent`]).
    let copy = crate::resource::host_copy(resource);
    if let Some(copy) = &copy
        && !Arc::ptr_eq(&copy.sent, &staging.sent)
    {
        unstage(resource);
        let Some(again) = self::staging(resource, texture) else {
            return;
        };
        staging = again;
    }
    // A staging made without the first levels of a texture that leaves
    // fewer out now (a level brought back): made anew with them.
    if staging.first > 0 {
        let wants = crate::resource::host_tiled(resource)
            .as_ref()
            .and_then(|t| t.tiled())
            .map_or(u32::MAX, crate::tiles::Tiled::copied_now);
        if wants & ((1u32 << staging.first) - 1) != 0 {
            unstage(resource);
            let Some(again) = self::staging(resource, texture) else {
                return;
            };
            staging = again;
        }
    }
    staging.used.store(crate::host::moment(), Relaxed);
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    // (Only for those: no other texture that keeps its staging buffer is
    // judged by the views. The pass's uploads are, and make theirs anew.)
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // A texture made without its most detailed levels writes all but
    // those that have no tiles (`crate::tiles::Tiled::copied_now`); one
    // brought back since the last upload (`crate::resource::promote_reduced`)
    // is uploaded again, which gives it its tiles.
    let tiled = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    let copied = tiled.as_ref().and_then(|t| t.tiled()).map_or(u32::MAX, crate::tiles::Tiled::copied_now);
    // (Never a level the staging does not hold.)
    let copied = copied & !((1u32 << staging.first.min(31)) - 1);
    if copied & !staging.copied.load(Relaxed) != 0 {
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // Memory under it remapped ([`remapped`]): this list uploads it, and the
    // check at its submission reads the memory first.
    if staging.remapped.load(Relaxed) {
        staging.dirty.store(true, Relaxed);
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left. What it finds changed goes up in that
    // submission, since the copies are decided there ([`decide_uploads`]):
    // the list asks for the upload all the same. (It was the next list
    // that samples the texture that uploaded it, a frame late where every
    // list of the frame was recorded before that check. Still so for a
    // texture made level by level, whose levels get their memory by a step
    // of the list that uploads them.)
    if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
        if !staging.on_demand
            && let Some(copy) = copy
        {
            list.defer_upload(TextureUpload { copy, staging: staging.clone(), copied });
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        // The list reads the texture all the same (through a table copied
        // as it is recorded): it holds it until it is reset, as it does a
        // texture it uploads (through its transition). Without, nothing but
        // the idle time kept the sweep from letting the host copy go while
        // the list was recorded and not yet sent (`VICTORY_ASYNC_SUBMIT`
        // sends later); a texture released before the list that reads it is
        // sent is a page fault when it runs. `VICTORY_TEXTURE_HOLD_READS=0`:
        // not held, as before.
        static HOLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *HOLD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_HOLD_READS").is_none_or(|v| v != "0")) {
            crate::tiles::ORDER[6].fetch_add(1, Relaxed);
            list.keep(texture);
        }
        list.defer_texture(staging);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    upload_recorded(resource);
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The copies out of the staging buffer are not recorded here: whether
    // they are made, and of which subresources, is decided when the list
    // is submitted ([`decide_uploads`]), and they go into the list's
    // prelude, ahead of the list. The staging buffer lives as long as
    // `staging`, which the deferred refresh below holds until the list is
    // reset.
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        format!(
            "texture upload {resource:?} ({}x{}x{} f{} mips {}) asked for: {} subresources from a staging of {:#x} bytes, \
             levels {copied:#x}, copied in the prelude if title memory is what was written last",
            d.Width,
            d.Height,
            d.DepthOrArraySize,
            d.Format.0,
            d.MipLevels,
            staging.footprints.len(),
            staging.size
        )
    });
    if let Some(copy) = copy {
        list.defer_upload(TextureUpload { copy, staging: staging.clone(), copied });
    }
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level but those left
    // out ([`crate::tiles::Tiled::copied_now`]), so each one given here is
    // written whole by it (the rest are copies into no memory, which the
    // card drops).
    let Some(on_demand) = tiled else {
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging);
        return;
    };
    list.defer_texture(staging.clone());
    list.defer_host(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), copied, staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.layout.size as usize)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::{Relaxed, SeqCst};
        let started = std::time::Instant::now();
        // The writes numbered so far are in memory before the read begins
        // (`crate::writes`); a read the title has submitted into this memory
        // and not done reads what is there before it.
        let stamp = crate::writes::stamp();
        let (start, size) = (self.memory as u64, self.layout.size);
        let under = crate::writes::pending_over(start, start + size);
        let asked = crate::watch::asked_now();
        self.reading.store(stamp, SeqCst);
        #[cfg(feature = "diagnostics")]
        self.pending_read.store(under, Relaxed);
        let read = self.refresh_now();
        if read > 0 {
            self.stamp.store(stamp, SeqCst);
            // What the read's own asking was numbered, or the last before.
            let own = self.last.lock().unwrap_or_else(|e| e.into_inner()).asked;
            self.asked.store(own.max(asked), SeqCst);
            if under {
                READ_UNDER_PENDING.fetch_add(1, Relaxed);
            }
        } else if self.stamp.load(SeqCst) == crate::writes::NOT_READ {
            // Nothing read (the memory could not be had): as if read before
            // any write.
            self.stamp.store(0, SeqCst);
        }
        // An upload of the texture pass whose read took no digest: counted
        // as such (`crate::census`).
        let why = self.picked.load(Relaxed);
        if why != 0 && !self.counted.swap(true, Relaxed) {
            let none: &[crate::census::Part] = &[];
            crate::census::read((why - 1) as u8, start, size, read as u64, self.bytes, if read > 0 { None } else { Some(none) });
        }
        // (A spare buffer nothing filled is zeroed only if an upload copies
        // from it all the same: `Staging::wanted`.)
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    fn refresh_timed(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        // Remapped since the last check ([`remapped`]): read whatever the
        // write watch, the guards or the partly mapped wait say.
        let remapped = self.remapped.swap(false, std::sync::atomic::Ordering::Relaxed);
        if let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
            if !remapped && last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            if self.place.parts.len() > 1 {
                self.refresh_parts(last, false);
            } else {
                with_read(size, |read| {
                    crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                    let (_, digests) = self.digest_of(read);
                    self.detile(read, &digests);
                });
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken every eighth frame only, by the
        // texture's address, so the textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        // A texture whose hash found it changed while the guard saw nothing
        // is hashed at every check from then on (`Last::behind_guard`): the
        // title's video player decodes into a ring of three frames written
        // through another view, and on its eighth-frame turn each frame of
        // the ring was refreshed about every 24 frames, so two presented
        // frames in three showed video from up to two seconds before (the
        // opening video stuttered: fxceless/victory-fork pull requests #701,
        // with 0 of 318 frames out of order after it, and #602, #489).
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mut unseen = false;
        let mut why_read = None;
        if *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
        {
            const HASH_EVERY: u64 = 8;
            let frame = crate::frame::current();
            let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
            last.asked = now;
            // A texture's first checks hash it whatever the turn, so one
            // written behind its guard is found at its second use, not up
            // to a second later (its memory moves from run to run, so it
            // cannot be known by address).
            const FIRST_CHECKS: u8 = 3;
            let young = last.checks < FIRST_CHECKS;
            last.checks = last.checks.saturating_add(1);
            let check = Check {
                unwritten,
                remapped,
                partly_lately: last.partial.is_some_and(|at| frame < at.saturating_add(30)),
                young,
                behind: last.behind_guard,
                turn: frame.wrapping_add((start >> 16) as u64).is_multiple_of(HASH_EVERY),
                hashed_now: last.hashed == frame.wrapping_add(1),
            };
            if !check_reads(check) {
                return 0;
            }
            last.hashed = frame.wrapping_add(1);
            why_read = Some(check_why(check));
            unseen = unwritten;
        }
        // A read the title has submitted into this memory writes it with
        // no guard seeing it until its bytes are all there (then they are
        // told, and the guarded blocks under them opened).
        let pending = crate::writes::pending_over(start as u64, start.saturating_add(size) as u64);
        let flight = [std::mem::replace(&mut last.pending, pending), pending];
        let hash_before = last.hash;
        let read = self.refresh_hashed(last, size, remapped);
        let changed = hash_before.is_some() && last.hash != hash_before;
        if unseen && changed && !last.behind_guard && !behind_guards(flight, remapped) {
            BEHIND_KNOWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if unseen && changed && !last.behind_guard && behind_guards(flight, remapped) {
            last.behind_guard = true;
            let n = BEHIND_GUARD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // The first of them, and one in a hundred after, with what the
            // check knew: which textures these are is not known from the
            // counts.
            if n < 24 || n.is_multiple_of(100) {
                tracing::info!(
                    memory = ?self.memory,
                    bytes = size,
                    levels = self.mips,
                    level_by_level = self.on_demand,
                    placed = self.placed,
                    pages_missing = last.partial.is_some(),
                    check = last.checks,
                    read = ["a write seen", "mapped anew", "a first check", "behind its guards", "its turn"][why_read.unwrap_or(4).min(4)],
                    so_far = n + 1,
                    "a texture changed with no write its guard saw and no read of the title's or mapping to explain it: taken \
                     for one written through another view, hashed at every check until it stays unchanged"
                );
            }
        }
        // Taken for one written behind its guards, and unchanged check
        // after check: no longer.
        if last.behind_guard {
            last.same = if changed { 0 } else { last.same.saturating_add(1) };
            if last.same >= BEHIND_CHECKS {
                (last.behind_guard, last.same) = (false, 0);
                BEHIND_GONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // What the checks read, by why (`crate::census::CHECK_READS`).
        if let Some(why) = why_read {
            crate::census::CHECK_READS[why].fetch_add(read as u64, std::sync::atomic::Ordering::Relaxed);
        }
        read
    }

    /// [`Staging::refresh_timed`] past the write watch and the guards: the
    /// texture's memory read, hashed and detiled if it changed. Returns the
    /// bytes of title memory read.
    fn refresh_hashed(&self, last: &mut Last, size: usize, remapped: bool) -> usize {
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            match crate::watch::title_memory_reading_now() {
                Some(mapped) => mapped,
                None => {
                    let read = self.refresh_unlocked(last, size);
                    if read > 0 || self.filled.load(std::sync::atomic::Ordering::Relaxed) {
                        return read;
                    }
                    // Never filled: left for the next refresh, it went up
                    // as zeros, and the frame's draws and a bake that
                    // samples it (once, at load: skin and clothing) took
                    // them (a black face that never came back). Its first
                    // read waits for the runtime's remap instead.
                    let waiting = std::time::Instant::now();
                    let mapped = waited(2, crate::watch::title_memory_reading);
                    FIRST_WAITS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    FIRST_WAITS[1]
                        .fetch_add(waiting.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                    mapped
                }
            }
        } else {
            waited(2, crate::watch::title_memory_reading)
        };
        // By level, where the memory is told apart at its levels: what of
        // it is there is asked level by level, and read where it lies.
        if self.place.parts.len() > 1 && !through_copy() {
            return self.refresh_parts(last, true);
        }
        let mapped = committed(self.memory, size);
        self.note_levels_there(mapped);
        if !mapped {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames (reading around
            // unmapped pages is slow, and nothing tells us when pages
            // arrive): a mip that streams in shows up to 30 frames late.
            let frame = crate::frame::current();
            if !remapped && last.partial.is_some_and(|at| frame < at + 30) {
                return 0;
            }
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                PAGES_MISSING[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let (hash, digests) = self.digest_of(read);
                if last.hash == Some(hash) {
                    return false;
                }
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                PAGES_MISSING[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.detile(read, &digests);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let (hash, digests) = self.digest_of(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    return size;
                }
                last.hash = Some(hash);
                self.detile(tiled, &digests);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let (hash, digests) = self.digest_of(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled, &digests);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then nothing is taken and the texture
    /// is looked at again at its next refresh (what was learnt about writes
    /// to it is forgotten, so that it is not passed over as unwritten).
    /// Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let (hash, digests) = self.digest_of(read);
            if last.hash != Some(hash) {
                last.hash = Some(hash);
                self.detile(read, &digests);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// Detiles into the staging buffer the levels of `tiled` (the
    /// texture's memory, all of it read) that neither it nor the host copy
    /// holds as they are now. `digests`: those of the memory's parts
    /// ([`Staging::digest_of`]). For a texture whose memory is one part,
    /// and for the reads that take a texture whole; the others read it by
    /// level ([`Staging::refresh_parts`]).
    fn detile(&self, tiled: &[u8], digests: &[u64; crate::census::PARTS]) {
        use std::sync::atomic::Ordering::Relaxed;
        // Which levels are the title's is judged here, once, for what the
        // staging buffer holds from now on. Levels on demand: the step that
        // gives levels their memory goes by this answer.
        let judged = if self.on_demand { self.judgement() } else { [0, u64::MAX] };
        self.judged[0].store(judged[0], Relaxed);
        self.judged[1].store(judged[1], Relaxed);
        let held = if self.on_demand { self.held_levels(judged) } else { u32::MAX };
        self.detiled.store(held, Relaxed);
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        let is = [PartIs::Read; crate::census::PARTS];
        let levels = self.stage_keys(digests, &is, held);
        if levels != 0 {
            self.detile_levels(tiled, levels, &is);
        }
        self.filled.store(true, Relaxed);
    }

    /// Notes what the staging buffer is to hold of each part after this
    /// refresh, and returns the levels to write into it for that: a part
    /// read (`digests`) is told by its key ([`part_key`]), one whose pages
    /// are not there is zeros, one left alone is nothing. A part the host
    /// copy holds as it is now is not staged, and nothing of it is copied;
    /// one the buffer holds as it is now is not written again. `held`:
    /// the levels that are the title's; no other is written or copied.
    fn stage_keys(&self, digests: &[u64; crate::census::PARTS], is: &[PartIs; crate::census::PARTS], held: u32) -> u32 {
        use std::sync::atomic::Ordering::Relaxed;
        let (mut levels, mut with_copy) = (0u32, 0u32);
        for (k, &of_part) in self.place.levels_of_part.iter().enumerate().take(crate::census::PARTS) {
            let key = match is[k] {
                PartIs::Left => 0,
                PartIs::Zeros => 1,
                PartIs::Read => part_key(digests[k], of_part, held),
            };
            if key == 0 || of_part & held == 0 {
                self.staged[k].store(0, Relaxed);
            } else if key == self.sent.key(k) {
                self.staged[k].store(0, Relaxed);
                with_copy |= of_part & held;
            } else if self.staged[k].swap(key, Relaxed) != key {
                levels |= of_part & held;
            }
        }
        if with_copy != 0 {
            let bytes = self.bytes_of(self.subresources(with_copy));
            crate::census::BY_LEVEL[crate::census::L_WITH_COPY].fetch_add(bytes, Relaxed);
        }
        levels
    }

    /// Writes the levels `levels` into the staging buffer: detiled from
    /// `tiled` (the texture's memory, or a copy of it that holds the parts
    /// those levels lie in), or zeros for those of a part whose pages are
    /// not there.
    fn detile_levels(&self, tiled: &[u8], levels: u32, is: &[PartIs; crate::census::PARTS]) {
        use std::sync::atomic::Ordering::Relaxed;
        // Diagnostics: the buffer is written now, on the CPU, while a copy
        // out of it that a submission decided may not have run yet: that
        // copy takes these bytes, not those of its own moment.
        #[cfg(feature = "diagnostics")]
        {
            let [started, finished] = crate::host::submissions_numbered();
            let copy = self.copy_in.load(Relaxed);
            STAGING_UNDER[0].fetch_add(1, Relaxed);
            if copy > finished {
                STAGING_UNDER[1].fetch_add(1, Relaxed);
                STAGING_UNDER[2].fetch_add(u64::from(copy == started), Relaxed);
                STAGING_UNDER[3].fetch_add(u64::from(is.contains(&PartIs::Zeros)), Relaxed);
            }
        }
        let by_level = &crate::census::BY_LEVEL;
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let sub = self.first + i as u32;
            let (mip, slice) = (sub % self.mips, sub / self.mips);
            if mip < 32 && levels & (1 << mip) == 0 {
                continue;
            }
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            let part = usize::from(self.place.part_of_level.get(mip as usize).copied().unwrap_or(0));
            if is[part.min(crate::census::PARTS - 1)] == PartIs::Zeros {
                linear.fill(0);
                by_level[crate::census::L_ZEROED].fetch_add(len as u64, Relaxed);
                continue;
            }
            by_level[crate::census::L_DETILED].fetch_add(len as u64, Relaxed);
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                // Nothing of the buffer is known to hold a level whole.
                for staged in &self.staged {
                    staged.store(0, Relaxed);
                }
                self.stale.store(true, Relaxed);
                return;
            }
        }
        self.dirty.store(true, Relaxed);
    }

    /// The refresh of a texture whose memory is told apart by level
    /// ([`Where`]): what is the title's of it is read, hashed and brought
    /// up to date level by level, and nothing else is touched.
    ///
    /// A texture was read, hashed and staged over its full size whatever
    /// the title held of it. The title views a texture from the first
    /// level it holds, and the memory past that level's bytes is other
    /// textures' or not mapped: in the opening cutscene 122,900 of 218,624
    /// uploads were of textures with another texture's view starting
    /// inside their span (31.9 GB of them from there on, none inside the
    /// part their own view holds), and 37.5 GB of what was read was all
    /// zeros. Other textures' bytes went up as this one's levels, and
    /// zeros for levels not there.
    ///
    /// Now each part is one of three things ([`PartIs`]): the title's and
    /// there (read; detiled and copied if its bytes are other than the
    /// host copy's), the title's with pages not there (zeros on the host,
    /// once), or not the title's (left alone: not read, not staged, not
    /// copied). Levels on demand: what is the title's is what
    /// [`Staging::held_levels`] says, as before, and the others get
    /// neither memory nor zeros.
    ///
    /// `in_place`: the caller holds title memory still
    /// (`crate::watch::title_memory_reading`): a part is read where it
    /// lies, if all its pages are there. Otherwise it is read through the
    /// kernel, pages that are not there as zeros.
    /// Returns the bytes of title memory read.
    fn refresh_parts(&self, last: &mut Last, in_place: bool) -> usize {
        use crate::census as c;
        use std::sync::atomic::Ordering::Relaxed;
        let parts = &self.place.parts;
        let n = parts.len().min(c::PARTS);
        // What is the title's of it is judged by the views in slots that
        // shaders pick by index: for a texture made level by level, as
        // before, and for an upload of the pass that serves those slots,
        // of a texture the title placed. A texture a draw reads through a
        // table is read whole: the table's view is in none of those slots,
        // and where an older view of the same memory still stood in one,
        // from a less detailed level, the levels the table's view shows
        // were left alone and never went up.
        let why = self.picked.load(Relaxed);
        let cut = self.placed && why != 0;
        let mut judged = if self.on_demand || cut { self.judgement() } else { [0, u64::MAX] };
        if cut && !self.on_demand {
            // The slot the upload is for shows the texture from its own
            // view's first level, whether the spans hold that view or not
            // (a view written over, or one the spans left out, is in none:
            // every level then counted as shown, and those the title does
            // not hold were read from the next textures' memory).
            let own = self.viewed.load(Relaxed);
            let spans = crate::descriptor::host::first_viewed(self.memory as u64);
            judged[0] = first_shown(spans, (own != u32::MAX).then_some(own));
        }
        self.judged[0].store(judged[0], Relaxed);
        self.judged[1].store(judged[1], Relaxed);
        let on_demand = self.on_demand.then(|| self.held_levels(judged));
        let mut is = [PartIs::Left; c::PARTS];
        // The parts read through the kernel: the title's, with some of
        // their pages there and some not. Such a part is not zeros: a level
        // takes the bytes that are there. (For one refresh's worth of
        // commits a part with a page missing was all zeros on the host:
        // the texture's memory by the layout ends in padding the title
        // need not map, and the level before it then stayed black for
        // good. The read of the whole texture through the kernel, which
        // this replaces, took what was there.)
        let mut through = [false; c::PARTS];
        let (mut held, mut left, mut away, mut out, mut partly) = (0u32, 0u64, 0u64, 0u64, 0u64);
        for k in 0..n {
            let ((from, to), of_part) = (parts[k], self.place.levels_of_part[k]);
            is[k] = match on_demand {
                Some(levels) if of_part & levels != 0 => PartIs::Read,
                Some(_) => PartIs::Left,
                None => {
                    let at = self.memory.wrapping_add(from);
                    let mut whole = true;
                    let state = part_is(cut, of_part, to as u64, judged, || {
                        whole = !in_place || committed(at, to - from);
                        whole || some_there(at, to - from)
                    });
                    through[k] = state == PartIs::Read && !whole;
                    state
                }
            };
            match is[k] {
                PartIs::Left => left += (to - from) as u64,
                PartIs::Zeros => {
                    away += (to - from) as u64;
                    held |= of_part;
                }
                PartIs::Read => {
                    held |= on_demand.map_or(of_part, |levels| of_part & levels);
                    partly += if through[k] { (to - from) as u64 } else { 0 };
                }
            }
            // Every level of it is left out of the host copy, and so of
            // the staging buffer (a texture made without its most detailed
            // levels, `crate::tiles`): the title's all the same (the levels
            // stay held), with nothing to read it for. It was read and
            // hashed at every upload: the levels left out are three
            // quarters of such a texture's bytes or more, and the uploads
            // read 91.2 GB of title memory for 47.5 GB of staging.
            if is[k] != PartIs::Left && !reaches(of_part, u64::from(self.first)) {
                (is[k], through[k]) = (PartIs::Left, false);
                out += (to - from) as u64;
            }
        }
        self.detiled.store(held, Relaxed);
        if in_place {
            // Its views are clamped to the first level from which every
            // level has all its texels' pages (`Staging::note_levels_there`),
            // asked by the texels, as before, and only when a part of the
            // title's has pages missing. A level left alone is no view's.
            let incomplete = (0..n).any(|k| through[k] || is[k] == PartIs::Zeros);
            let alone = (0..n).filter(|&k| is[k] == PartIs::Left).fold(0, |all, k| all | self.place.levels_of_part[k]);
            self.note_shown_from(if incomplete { self.first_level_there(alone) } else { 0 });
            // For the checks: read again in 30 frames at the earliest
            // unless a write is seen or a mapping told ([`check_reads`]).
            // So also a texture made level by level of which the title
            // does not hold every level: its span has pages missing or
            // another texture's, and while it was read whole it was read
            // that seldom. (Without the wait such a texture, the largest
            // kind there is, was read at each of its first checks while
            // it streamed in, and marked as written behind its guards.)
            let unheld = on_demand.is_some_and(|levels| (0..n).any(|k| self.place.levels_of_part[k] & levels == 0));
            last.partial = (incomplete || unheld).then(crate::frame::current);
        }
        let by_level = &c::BY_LEVEL;
        by_level[c::L_LEFT].fetch_add(left, Relaxed);
        by_level[c::L_AWAY].fetch_add(away, Relaxed);
        by_level[c::L_OUT].fetch_add(out, Relaxed);
        by_level[c::L_PARTLY].fetch_add(partly, Relaxed);
        if away + partly > 0 {
            PAGES_MISSING[0].fetch_add(1, Relaxed);
        }
        let size = self.layout.size as usize;
        let count = why != 0 && !self.counted.swap(true, Relaxed);
        let mut digests = [0u64; c::PARTS];
        let mut census = [c::Part { bytes: 0, digest: 0, zero: false }; c::PARTS];
        let mut read = 0usize;
        // Pages of what is the title's are missing.
        let partial = in_place && (0..n).any(|k| through[k] || is[k] == PartIs::Zeros);
        let taken = |bytes: &[u8]| {
            let d = digest(bytes).0;
            let d = (d as u64 ^ (d >> 64) as u64).max(1);
            (d, c::Part { bytes: bytes.len() as u64, digest: d, zero: (count || partial) && all_zero(bytes) })
        };
        let reading = |k: usize| is[k] == PartIs::Read;
        if in_place && (0..n).all(|k| reading(k) && !through[k]) {
            // All of the texture is there: read where it lies.
            for (k, &(from, to)) in parts.iter().enumerate().take(n) {
                // SAFETY: the part's pages are all there (asked above), and
                // stay while the caller holds title memory still.
                let bytes = unsafe { std::slice::from_raw_parts(self.memory.add(from), to - from) };
                (digests[k], census[k]) = taken(bytes);
                read += to - from;
            }
            let levels = self.stage_keys(&digests, &is, held);
            if levels != 0 {
                // SAFETY: as above, every part of it.
                self.detile_levels(unsafe { std::slice::from_raw_parts(self.memory, size) }, levels, &is);
            }
        } else if in_place {
            // Not all of the texture is there to be read where it lies. A
            // part all of whose pages are there is read in place; one with
            // pages missing is read through the kernel, which stands them
            // (zeros for those pages, the bytes of the others). The parts
            // to detile are then all in the buffer, each at its own place.
            with_read(size, |buffer| {
                for k in (0..n).filter(|&k| reading(k)) {
                    let (from, to) = parts[k];
                    let bytes: &[u8] = if through[k] {
                        crate::host::copy_mapped(buffer[from..to].as_mut_ptr(), self.memory.wrapping_add(from), to - from, false);
                        &buffer[from..to]
                    } else {
                        // SAFETY: as above.
                        unsafe { std::slice::from_raw_parts(self.memory.add(from), to - from) }
                    };
                    (digests[k], census[k]) = taken(bytes);
                    read += to - from;
                }
                // Pages are missing and what is there is nothing but
                // zeros, or nothing is there at all: the texture is left
                // as it is, on the host too. Its memory went, or has not
                // come; so it was while such a texture was read whole
                // ("nothing at all mapped changes nothing"), and a
                // texture whose memory goes before other memory comes in
                // its place kept its texels meanwhile.
                if partial && (0..n).filter(|&k| reading(k)).all(|k| census[k].zero) {
                    return;
                }
                if partial {
                    PAGES_MISSING[1].fetch_add(1, Relaxed);
                }
                let levels = self.stage_keys(&digests, &is, held);
                if levels != 0 {
                    for k in (0..n).filter(|&k| reading(k) && !through[k] && self.place.levels_of_part[k] & levels != 0) {
                        let (from, to) = parts[k];
                        // SAFETY: as above; the buffer holds `size` bytes,
                        // and the part lies inside them.
                        unsafe {
                            std::ptr::copy_nonoverlapping(self.memory.add(from), buffer[from..to].as_mut_ptr(), to - from);
                        }
                    }
                    self.detile_levels(buffer, levels, &is);
                }
            });
        } else {
            with_read(size, |buffer| {
                for k in (0..n).filter(|&k| reading(k)) {
                    let (from, to) = parts[k];
                    crate::host::copy_mapped(buffer[from..to].as_mut_ptr(), self.memory.wrapping_add(from), to - from, false);
                    (digests[k], census[k]) = taken(&buffer[from..to]);
                    read += to - from;
                }
                let levels = self.stage_keys(&digests, &is, held);
                if levels != 0 {
                    self.detile_levels(buffer, levels, &is);
                }
            });
        }
        // What the journal, the guards and the write watch knew of the
        // parts the host copy held, since the read that last found them as
        // it holds them: counted for the pass's uploads, every changed
        // part and the unchanged ones of one upload in sixteen
        // (`crate::census::SCREEN`). A part changed with nothing seen is
        // one that a screen by them, in place of this read, would have
        // left with the bytes from before.
        static TURN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sampled = count && TURN.fetch_add(1, Relaxed).is_multiple_of(16);
        #[cfg(feature = "diagnostics")]
        let pending_read = self.pending_read.load(Relaxed);
        for k in (0..n).filter(|&k| reading(k)) {
            #[cfg(feature = "diagnostics")]
            self.read_flags[k].store(unseen_flags(through[k], pending_read), Relaxed);
            let (was, since) = (self.sent.key(k), self.sent.1[k].load(Relaxed));
            let changed = part_key(digests[k], self.place.levels_of_part[k], held) != was;
            if count && was > 1 && since != 0 && (changed || sampled) {
                let ((from, to), at) = (parts[k], self.memory as usize);
                let (noted, written, open) = crate::watch::seen_since(at.saturating_add(from), at.saturating_add(to), since);
                let seen = noted || written > 0 || open > 0;
                c::screened((why - 1) as u8, changed, seen, (to - from) as u64);
                #[cfg(feature = "diagnostics")]
                if changed && !seen {
                    c::unseen(c::Unseen {
                        why: (why - 1) as u8,
                        at: at.saturating_add(from),
                        len: to - from,
                        level_by_level: self.on_demand,
                        then: self.sent.3[k].load(Relaxed),
                        stamp: self.sent.2[k].load(Relaxed),
                        through: through[k],
                        pending: pending_read,
                    });
                }
            }
            // Found as the host copy holds it, by a read after this asking.
            if !changed && last.asked != 0 {
                self.sent.1[k].fetch_max(last.asked, Relaxed);
                self.sent.2[k].fetch_max(self.reading.load(Relaxed), Relaxed);
                #[cfg(feature = "diagnostics")]
                self.sent.3[k].store(self.read_flags[k].load(Relaxed), Relaxed);
            }
        }
        by_level[c::L_READ].fetch_add(read as u64, Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, read as u64);
        if count {
            c::read((why - 1) as u8, self.memory as u64, self.layout.size, read as u64, self.bytes, Some(&census[..n]));
        }
        // A part read before holds other bytes now: the texture changed
        // (for those who ask whether it did with no write seen,
        // `Staging::refresh_timed`). A part read for the first time, or no
        // longer, is no change of the title's bytes.
        let changed = (0..n).any(|k| last.parts[k] != 0 && digests[k] != 0 && last.parts[k] != digests[k]);
        if changed || last.hash.is_none() {
            last.changes += 1;
        }
        last.hash = Some(Digest(u128::from(last.changes)));
        last.parts = digests;
        if read > 0 {
            self.filled.store(true, Relaxed);
        }
        read
    }

    /// The subresources an upload from this staging buffer has to copy
    /// for the host copy to hold the texture as its memory was at the last
    /// refresh, a bit each: those of the levels the buffer holds detiled
    /// and the host copy (`sent`) holds otherwise or not at all.
    fn wanted(&self, sent: &Sent) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        let (mut levels, mut any) = (0u32, false);
        let held = self.detiled.load(Relaxed);
        for (k, &of_part) in self.place.levels_of_part.iter().enumerate().take(crate::census::PARTS) {
            let key = self.staged[k].load(Relaxed);
            any |= key != 0;
            if key != 0 && key != sent.key(k) {
                levels |= of_part & held;
            }
        }
        // Nothing was ever read of the texture (its memory could not be
        // had, or was all zeros) and nothing brought to its host copy,
        // which may hold what the last owner of its place on the card
        // left: zeros, once, as before.
        if !any && sent.none() && !self.uploaded.load(Relaxed) {
            let _last = self.last.lock().unwrap_or_else(|e| e.into_inner());
            if self.stale.swap(false, Relaxed) {
                // SAFETY: the buffer's mapping holds at least `bytes`
                // bytes; no list has read it yet (the copies decided here
                // run after this).
                unsafe { std::ptr::write_bytes(self.data, 0, self.bytes as usize) };
                SPARE_STATS[3].fetch_add(1, Relaxed);
            }
            return u64::MAX;
        }
        if levels == 0 { 0 } else { self.subresources(levels) }
    }

    /// The upload copied the subresources `subs`: the host copy (`sent`)
    /// holds from here on what the buffer holds of every part all of whose
    /// staged levels were among them. A part of which a level was left out
    /// (the card wrote it last, or it has no memory on the card) is asked
    /// for again by the next upload.
    fn mark_sent(&self, sent: &Sent, subs: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        // Levels on demand: a held level was left without the card's
        // memory, and the copy into it went nowhere.
        if self.on_demand && self.waiting.load(Relaxed) != 0 {
            return;
        }
        let held = self.detiled.load(Relaxed);
        for (k, &of_part) in self.place.levels_of_part.iter().enumerate().take(crate::census::PARTS) {
            let key = self.staged[k].load(Relaxed);
            let need = if of_part & held == 0 { 0 } else { self.subresources(of_part & held) };
            if key != 0 && need != 0 && need & !subs == 0 {
                // A part that is zeros on the host because nothing of it
                // was there, and one that got its bytes after that: counted.
                let before = sent.0[k].swap(key, Relaxed);
                let by_level = &crate::census::BY_LEVEL;
                if key == 1 && before != 1 {
                    by_level[crate::census::L_ZERO_PARTS].fetch_add(1, Relaxed);
                } else if key != 1 && before == 1 {
                    by_level[crate::census::L_ZERO_FILLED].fetch_add(1, Relaxed);
                }
                sent.1[k].store(self.asked.load(std::sync::atomic::Ordering::SeqCst), Relaxed);
                sent.2[k].store(self.stamp.load(std::sync::atomic::Ordering::SeqCst), Relaxed);
                #[cfg(feature = "diagnostics")]
                sent.3[k].store(self.read_flags[k].load(Relaxed), Relaxed);
            }
        }
    }

    /// The subresources a view of the texture may show for which the host
    /// copy (`sent`) holds nothing an upload brought, a bit each: for the
    /// count of them (`crate::census::L_BARE`). A view may show every
    /// level of a texture a draw reads through a table, and those from the
    /// first one viewed of one the pass uploads. Not asked of a texture
    /// made level by level, whose views stop at the levels with memory.
    fn bare(&self, sent: &Sent) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        // (Nor of one nothing was ever read of: it is zeros all over.)
        if self.on_demand || (sent.none() && self.staged.iter().all(|key| key.load(Relaxed) == 0)) {
            return 0;
        }
        let first = if self.placed && self.picked.load(Relaxed) != 0 { self.judged[0].load(Relaxed) } else { 0 };
        let shown = u32::MAX.checked_shl(first.min(32) as u32).unwrap_or(0);
        let mut levels = 0u32;
        for (k, &of_part) in self.place.levels_of_part.iter().enumerate().take(crate::census::PARTS) {
            if sent.key(k) == 0 {
                levels |= of_part & shown;
            }
        }
        if levels == 0 { 0 } else { self.subresources(levels) }
    }

    /// The bytes of the staging buffer the subresources `subs` take.
    fn bytes_of(&self, subs: u64) -> u64 {
        let total = self.first + self.footprints.len() as u32;
        let mut bytes = 0;
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            if subs & subresource_bit(self.first + i as u32, total) != 0 {
                let f = footprint.Footprint;
                bytes += u64::from(f.RowPitch) * u64::from(rows) * u64::from(f.Depth);
            }
        }
        bytes
    }

    /// The digest of the texture's memory as read (`tiled`), taken part by
    /// part ([`memory_parts`]) and folded into one. For an upload of the
    /// texture pass the parts are counted against the memory's last upload
    /// (`crate::census::read`), once: which hold other bytes, which the
    /// same, which nothing.
    fn digest_of(&self, tiled: &[u8]) -> (Digest, [u64; crate::census::PARTS]) {
        use std::sync::atomic::Ordering::Relaxed;
        let why = self.picked.load(Relaxed);
        let count = why != 0 && !self.counted.swap(true, Relaxed);
        let fold = |d: u128| (d as u64 ^ (d >> 64) as u64).max(1);
        let mut digests = [0u64; crate::census::PARTS];
        let parts = &self.place.parts;
        crate::census::BY_LEVEL[crate::census::L_READ].fetch_add(tiled.len() as u64, Relaxed);
        if parts.len() <= 1 && !count {
            let whole = digest(tiled);
            digests[0] = fold(whole.0);
            return (whole, digests);
        }
        let mut read = [crate::census::Part { bytes: 0, digest: 0, zero: false }; crate::census::PARTS];
        let mut folded = [0u8; 16 * crate::census::PARTS];
        let n = parts.len().min(crate::census::PARTS);
        for (k, &(from, to)) in parts.iter().take(n).enumerate() {
            let bytes = tiled.get(from..to.min(tiled.len())).unwrap_or(&[]);
            let d = digest(bytes).0;
            folded[16 * k..16 * k + 16].copy_from_slice(&d.to_le_bytes());
            digests[k] = fold(d);
            read[k] = crate::census::Part { bytes: bytes.len() as u64, digest: digests[k], zero: count && all_zero(bytes) };
        }
        if count {
            let all = tiled.len() as u64;
            crate::census::read((why - 1) as u8, self.memory as u64, self.layout.size, all, self.bytes, Some(&read[..n]));
        }
        if n <= 1 {
            let whole = digest(tiled);
            digests[0] = fold(whole.0);
            (whole, digests)
        } else {
            (digest(&folded[..16 * n]), digests)
        }
    }

    /// The subresources an upload of the levels `copied` writes from this
    /// staging buffer, a bit each ([`subresource_bit`]).
    fn subresources(&self, copied: u32) -> u64 {
        let total = self.first + self.footprints.len() as u32;
        let mut subs = 0;
        for i in 0..self.footprints.len() as u32 {
            let sub = self.first + i;
            let level = sub % self.mips.max(1);
            if level >= 32 || copied & (1 << level) != 0 {
                subs |= subresource_bit(sub, total);
            }
        }
        subs
    }

    /// The bytes from the start of the texture's memory that the levels
    /// from `first` on take: what a view that starts at that level says
    /// the title holds (all of it for level 0, and for a texture of
    /// several slices).
    fn own_bytes(&self, first: u32) -> u64 {
        let (layout, mips) = (&self.layout, &self.layout.mips);
        if layout.slices != 1 || first == 0 || mips.is_empty() {
            return layout.size;
        }
        let tail = (layout.first_tail_mip as usize).min(mips.len());
        let above = (first as usize - 1).min(mips.len() - 1);
        if above < tail { mips[above].offset.min(layout.size) } else { layout.alignment.min(layout.slice_size) }
    }

    /// A texture uploaded whole while it streams in has its small levels
    /// first; the large ones are pages not there yet and read as zeros.
    /// Sampled, the zeros came out black, and only triangle edges (whose
    /// wide derivatives pick a smaller level) showed the texture: a wire net
    /// over black clothes and props (AntoninMichelq/VIctory-forkAMD
    /// 440e19e5ee). Which levels have their pages is known without looking
    /// at a byte: the first level from which every level is all mapped, in
    /// every layer. Views of the texture are clamped there
    /// ([`levels_shown_from`]), as the views of textures made level by level
    /// are, and the clamp goes when the pages come; nothing is filled on the
    /// processor. Not for a texture made level by level (`crate::tiles`
    /// clamps its own views).
    fn note_levels_there(&self, mapped: bool) {
        if self.on_demand || self.mips < 2 {
            return;
        }
        self.note_shown_from(if mapped { 0 } else { self.first_level_there(0) });
    }

    /// [`Staging::note_levels_there`], given the first level from which
    /// every level has all its pages.
    fn note_shown_from(&self, first: u32) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.on_demand || self.mips < 2 {
            return;
        }
        if self.shown_from.swap(first, Relaxed) == first {
            return;
        }
        let start = self.memory as u64;
        {
            let mut map = SHOWN_FROM.lock().unwrap_or_else(|e| e.into_inner());
            let map = map.get_or_insert_with(HashMap::new);
            if first == 0 {
                map.remove(&start);
            } else {
                map.insert(start, first);
            }
            SHOWN_ANY.store(!map.is_empty(), Relaxed);
        }
        // The texture's slots are made again at the next texture pass.
        crate::descriptor::host::levels_changed(start);
    }

    /// The first level from which every level of every layer has all its
    /// pages ([`Staging::note_levels_there`]); the level count if even the
    /// last has not. `taken`: levels taken as there without asking (a bit a
    /// level).
    fn first_level_there(&self, taken: u32) -> u32 {
        let layout = &self.layout;
        let n = u64::from(layout.desc.element_bytes);
        let slices = layout.slices.max(1);
        let there = |mip: u32| {
            let Some(m) = layout.mips.get(mip as usize) else {
                return false;
            };
            (0..slices).all(|slice| {
                let (Ok(first), Ok(last)) = (
                    layout.offset(mip, 0, 0, slice),
                    layout.offset(mip, m.width.saturating_sub(1), m.height.saturating_sub(1), slice),
                ) else {
                    return false;
                };
                let (from, to) = (first.min(last), first.max(last) + n);
                committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
            })
        };
        let mut first = self.mips;
        while first > 0 && (1u32.checked_shl(first - 1).is_some_and(|bit| taken & bit != 0) || there(first - 1)) {
            first -= 1;
        }
        first
    }
}

/// Where a texture uploaded whole while it streams in has its first level
/// with every page there, by where its title memory starts, for those that
/// do not have all ([`Staging::note_levels_there`]).
static SHOWN_FROM: Mutex<Option<HashMap<u64, u32>>> = Mutex::new(None);
/// Whether [`SHOWN_FROM`] holds any: views are made by the thousand a frame
/// and look at it without its lock while it is empty.
static SHOWN_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The level views of title texture `resource` are clamped to (above 0)
/// while its most detailed levels' pages are not there
/// ([`Staging::note_levels_there`]).
pub(crate) fn levels_shown_from(resource: This) -> Option<u32> {
    if !SHOWN_ANY.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let (memory, _) = crate::resource::texture_memory(resource)?;
    let map = SHOWN_FROM.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()?.get(&(memory as u64)).copied()
}

/// Textures found changed with no write their guard saw, hashed at every
/// check since (`Last::behind_guard`), since the start.
static BEHIND_GUARD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Checks that found a texture changed with no write its guard saw while
/// what changed it was known ([`behind_guards`]), since the start.
static BEHIND_KNOWN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Textures no longer taken for ones written behind their guards, after
/// [`BEHIND_CHECKS`] checks that found them unchanged, since the start.
static BEHIND_GONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a change of a texture's bytes that its guard saw no write for
/// is the work of a writer the guards do not see, so that the texture is
/// hashed at every check from then on. Not if a read the title submitted
/// was still to write its memory at this check or the one before
/// (`flight`): DirectStorage fills the pages with no guard seeing it and
/// tells when it is done. Not if the runtime mapped or unmapped memory
/// under it since the last check (`remapped`): other pages show there.
///
/// Both were taken for such a writer. A texture read while its pages were
/// being filled was then hashed at every check for good: 139 textures
/// after a prologue run in which the reading by level had the textures
/// read more often while they streamed in, where 23 had been; in the room
/// after it they were 62 of the 63 MB read a frame, 5.5 ms where 2.2 had
/// been.
fn behind_guards(flight: [bool; 2], remapped: bool) -> bool {
    !flight[0] && !flight[1] && !remapped
}

/// The line on textures shown from a smaller level while their large
/// levels' pages are not there ([`SHOWN_FROM`]) and on textures written
/// behind their guards ([`BEHIND_GUARD`]), every few seconds; nothing while
/// there are none.
pub(crate) fn filled_line() {
    use std::sync::atomic::Ordering::Relaxed;
    let clamped = SHOWN_FROM.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, HashMap::len);
    let behind = BEHIND_GUARD.load(Relaxed);
    let known = BEHIND_KNOWN.load(Relaxed);
    let gone = BEHIND_GONE.load(Relaxed);
    if clamped as u64 + behind + known > 0 {
        tracing::info!(
            "textures streaming in, shown from their first level whose pages are all there (their views clamped), now: \
             {clamped}; since the start: {behind} textures found written through another view, behind their guards, \
             hashed at every check since, {gone} of them no longer (unchanged for {BEHIND_CHECKS} checks: on their turn \
             again); {known} checks found a texture changed with no write seen while a read of the title's was writing \
             it or its memory had been mapped anew (not taken for that)"
        );
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
}

/// [`unstage`] for the sweep of idle host copies: the staging state is
/// dropped later, off the thread that submits (`crate::host::retire_later`).
pub(crate) fn unstage_later(resource: This) {
    let gone = {
        let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_mut().and_then(|map| map.remove(&(resource as usize)))
    };
    if let Some(gone) = gone {
        crate::host::retire_later(gone);
    }
}

/// What the texture pass keeps of an upload it recorded
/// ([`picked`]): the read stamp ([`Staging::stamp`]) and the journal's
/// asking the read comes after ([`Staging::asked`]), both shared and
/// filled in by the refresh at the list's submission; where the texture's
/// memory is and its bytes; and the bytes of it a view from the level
/// asked about holds.
pub(crate) struct Picked {
    pub(crate) stamp: Arc<std::sync::atomic::AtomicU64>,
    pub(crate) asked: Arc<std::sync::atomic::AtomicU64>,
    pub(crate) memory: (u64, u64),
    pub(crate) own: u64,
}

/// The texture pass recorded an upload from title texture `resource`'s
/// staging buffer for the reason `why` (`crate::census`), for a view that
/// starts at level `first`: the upload's read counts itself under that
/// reason. `None`: the texture has no staging buffer.
pub(crate) fn picked(resource: This, why: u8, first: u32) -> Option<Picked> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let s = map.as_ref()?.get(&(resource as usize))?;
    s.picked.store(u32::from(why) + 1, std::sync::atomic::Ordering::Relaxed);
    s.viewed.store(first, std::sync::atomic::Ordering::Relaxed);
    Some(Picked {
        stamp: s.stamp.clone(),
        asked: s.asked.clone(),
        memory: (s.memory as u64, s.layout.size),
        own: s.own_bytes(first),
    })
}

/// For tests: the levels a refresh of title texture `resource` reads and
/// stages (a bit a level), as an upload of the texture pass (`picked`) or as
/// a draw's read through a table; `None` without a host.
#[cfg(test)]
pub(crate) fn levels_staged(resource: This, picked: Option<u32>) -> Option<u32> {
    use std::sync::atomic::Ordering::Relaxed;
    let texture = crate::resource::host_resource(resource)?;
    unstage(resource);
    let staging = staging(resource, texture)?;
    if let Some(first) = picked {
        staging.picked.store(u32::from(crate::census::VIEW) + 1, Relaxed);
        staging.viewed.store(first, Relaxed);
    }
    staging.refresh();
    let held = staging.detiled.load(Relaxed);
    let staged = |k: &usize| staging.staged[*k].load(Relaxed) != 0;
    let levels = (0..staging.place.parts.len()).filter(staged).fold(0, |all, k| all | staging.place.levels_of_part[k] & held);
    unstage(resource);
    Some(levels)
}

/// Since the start: reads of texture memory into a staging buffer that a
/// read the title had submitted was still to write
/// (`crate::writes::pending_over`): they took what was there before it.
pub(crate) static READ_UNDER_PENDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut print = 0u64;
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            let d = digest(&read[..bytes]).0;
            print = print.rotate_left(37) ^ d as u64 ^ (d >> 64) as u64;
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    print | 1
}

/// A 128-bit digest of `bytes`, to tell whether a texture's memory changed:
/// four lanes of folded 64-bit multiplications over 64 bytes at a time, the
/// length folded in at the end. Not a cryptographic hash, which is not
/// wanted here (nothing makes up textures to collide) and costs about three
/// times as much: on the Ryzen 7 9800X3D blake3 took 0.174 ms a megabyte,
/// this 0.060 (`hash_costs`), and the texture pass moves 84 to 219 MB a
/// frame in the opening cutscene. From avatarmanz/VIctory-7900XTX
/// 2dfba4dbd.
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap_or([0; 8]));
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane).wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41)) ^ b;
        }
    };
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        step(block);
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high =
        fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len) ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
pub(crate) fn staged(resource: This) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    let size = s.layout.size as usize;
    Some((size as u64, committed(s.memory, size)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// Whether the 64 KiB block of title memory at `block` is there, asked at
/// `at` inside it: the answer is kept until the runtime says the block was
/// mapped anew ([`blocks_remapped`]). The pass over the heap's textures
/// asked for every block of every texture still due, every frame (6,000
/// textures waiting for their memory in the intro: 12 ms a frame of a
/// recording thread in probes, most of them of blocks that had not changed
/// since the frame before). `VICTORY_MAPPED_CACHE=0`: asked every time.
fn block_there(block: usize, at: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_MAPPED_CACHE").is_none_or(|v| v != "0")) {
        return readable(at, 1, true).0 == 1;
    }
    let key = block >> 16;
    if let Some(&there) = BLOCKS_THERE.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return there;
    }
    // Asked without the lock: a remapping meanwhile takes the entry out
    // again only if it comes after the insert below, so its number is
    // looked at before and after.
    let before = BLOCKS_REMAPPED.load(std::sync::atomic::Ordering::Acquire);
    let there = readable(at, 1, true).0 == 1;
    let mut blocks = BLOCKS_THERE.lock().unwrap_or_else(|e| e.into_inner());
    if BLOCKS_REMAPPED.load(std::sync::atomic::Ordering::Acquire) == before {
        blocks.insert(key, there);
    }
    there
}

/// The answers [`block_there`] keeps, by block number.
static BLOCKS_THERE: std::sync::Mutex<
    std::collections::HashMap<usize, bool, std::hash::BuildHasherDefault<crate::descriptor::slots::SlotHasher>>,
> = std::sync::Mutex::new(std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()));

/// Counts the remappings told to [`blocks_remapped`].
static BLOCKS_REMAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The runtime mapped, unmapped or freed title memory in `[at, at + len)`:
/// what [`block_there`] kept for its blocks is asked anew.
pub(crate) fn blocks_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut blocks = BLOCKS_THERE.lock().unwrap_or_else(|e| e.into_inner());
    BLOCKS_REMAPPED.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    if blocks.is_empty() {
        return;
    }
    let (first, last) = (at >> 16, (at + len - 1) >> 16);
    if last - first > blocks.len() {
        blocks.retain(|&key, _| key < first || key > last);
    } else {
        for key in first..=last {
            blocks.remove(&key);
        }
    }
}

/// Whether any of `[at, at + len)` is there, by 64 KiB block as
/// [`mapped_bytes`] asks it (one page of each block, the answer kept until
/// the runtime maps the block anew). For a part of a texture not all of
/// whose pages are there: something of it, or nothing.
fn some_there(at: *const u8, len: usize) -> bool {
    const BLOCK: usize = 64 << 10;
    let start = at as usize;
    let end = start.saturating_add(len);
    if end <= start {
        return false;
    }
    // Its first and last page are asked afresh, as [`mapped_bytes`] asks
    // them: what is kept of a block is from the last mapping the runtime
    // told of, and "nothing of it is there" makes a level zeros on the
    // host without a read. A level's bytes start at its first page.
    if committed(at, 1) || committed((end - 1) as *const u8, 1) {
        return true;
    }
    let mut block = start & !(BLOCK - 1);
    while block < end {
        if block_there(block, block.max(start)) {
            return true;
        }
        let Some(next) = block.checked_add(BLOCK) else {
            break;
        };
        block = next;
    }
    false
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let end = start.saturating_add(len);
    if !committed(at, 1) && !committed((end - 1) as *const u8, 1) {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if block_there(block, lo) {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let tiled = crate::resource::host_tiled(resource);
    let on_demand = tiled.is_some();
    // What the host copy holds of the texture (its own note, shared).
    let sent = crate::resource::host_copy(resource).map_or_else(Arc::default, |copy| copy.sent.clone());
    // The levels an upload leaves out now (`crate::tiles::Tiled::copied_now`):
    // the first ones are not staged at all (they are three quarters of the
    // texture's bytes or more), when `VICTORY_STAGE_LEFT_OUT` is not `1`.
    static STAGE_ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let stage_all = *STAGE_ALL.get_or_init(|| std::env::var_os("VICTORY_STAGE_LEFT_OUT").is_some_and(|v| v == "1"));
    let left_out = if stage_all {
        0
    } else {
        tiled.as_ref().and_then(|t| t.tiled()).map_or(0, |t| (!t.copied_now()).trailing_ones().min(31))
    };
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let all = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    // Only a texture of one layer leaves levels out: its subresources are
    // its levels, and those left out are the first ones.
    let first = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.DepthOrArraySize == 1 {
        left_out.min(all.saturating_sub(1))
    } else {
        0
    };
    let count = all - first;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources from `first` on.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    if first > 0 {
        STAGED_LEFT_OUT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let (buffer, data, pooled, stale) = staging_buffer(host, total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let remapped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    index_staged(memory as usize, layout.size as usize, &remapped);
    let s = Arc::new(Staging {
        place: Where::of(memory as usize, &layout, u32::from(d.MipLevels).max(1), all),
        staged: Default::default(),
        sent,
        placed: crate::resource::is_placed(resource),
        viewed: std::sync::atomic::AtomicU32::new(u32::MAX),
        picked: std::sync::atomic::AtomicU32::new(0),
        counted: std::sync::atomic::AtomicBool::new(false),
        asked: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        remapped,
        filled: std::sync::atomic::AtomicBool::new(false),
        stamp: Arc::new(std::sync::atomic::AtomicU64::new(crate::writes::NOT_READ)),
        reading: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "diagnostics")]
        pending_read: std::sync::atomic::AtomicBool::new(false),
        #[cfg(feature = "diagnostics")]
        read_flags: Default::default(),
        shown_from: std::sync::atomic::AtomicU32::new(0),
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        bytes: total,
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        copied: std::sync::atomic::AtomicU32::new(0),
        first,
        stale: std::sync::atomic::AtomicBool::new(stale),
        #[cfg(feature = "diagnostics")]
        copy_in: std::sync::atomic::AtomicU64::new(0),
        pooled,
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod remap_tests {
    use std::sync::atomic::Ordering::Relaxed;

    /// A remap under a texture already uploaded flags it, so the next list
    /// uploads it again; a remap of memory beside it, or of nothing, does
    /// not (fxceless/victory-fork pull request #606).
    #[test]
    fn a_remap_under_a_staged_texture_flags_it() {
        use windows_sys::Win32::System::Memory::*;
        const RUN: usize = 0x1_0000;
        // SAFETY: a private allocation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * RUN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!base.is_null());
        let start = base as usize;
        // A linear 4x4 RGBA8 texture of one level.
        let description = [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(start as u64, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            super::staging(resource, host_texture)
        });
        // None: no host GPU here.
        if let Some(staging) = staged {
            super::remapped(start + RUN, RUN);
            super::remapped(start, 0);
            assert!(!staging.remapped.load(Relaxed), "a remap beside it flagged the texture");
            super::remapped(start + 0x10, 0x10);
            assert!(staging.remapped.load(Relaxed), "a remap under it did not flag the texture");
            staging.refresh();
            assert!(!staging.remapped.load(Relaxed), "the check took the flag");
        }
        // SAFETY: the allocation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// A texture uploaded whole while its most detailed level's pages are
    /// not there is shown from the first level that has them, and from all
    /// once they come.
    #[test]
    fn a_texture_streaming_in_is_shown_from_the_levels_that_are_there() {
        use windows_sys::Win32::System::Memory::*;
        const RUN: usize = 0x1_0000;
        // A linear 256x256 RGBA8 texture of nine levels: the console puts
        // the small levels first (level 8 at 0, level 1 ends at 98,044) and
        // level 0 behind them, up to 360,192; only the first two runs are
        // there.
        // SAFETY: a private reservation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 6 * RUN, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: the first two runs of the reservation above.
        assert!(!unsafe { VirtualAlloc(base, 2 * RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let description = [3, 0, 0, 0, 256, 0, 256, 0x9_0001, 28, 1, 0, 1, 0, 0];
        let made = crate::resource::texture_at(base as u64, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            Some((resource, super::staging(resource, host_texture)?))
        });
        // What is there holds one byte value all over.
        // SAFETY: the two runs committed above.
        unsafe { std::ptr::write_bytes(base.cast::<u8>(), 0x5a, 2 * RUN) };
        // (As the runtime says when it maps memory: what was known of
        // these blocks, from a test before at this address, is asked anew.)
        super::blocks_remapped(base as usize, 6 * RUN);
        if let Some((resource, staging)) = made {
            staging.refresh();
            assert_eq!(super::levels_shown_from(resource), Some(1), "level 0's pages are not there");
            // Level 0 starts in the second run and reaches past it: its
            // first rows are there and its last are not. It is not zeros
            // for that: the rows that are there go up as they are, the
            // others as zeros (a level with a page missing was zeros all
            // over, and stayed so when the page was one the title never
            // maps).
            let level = |n: usize, row: usize| {
                let footprint = staging.footprints[n];
                let at = footprint.Offset as usize + row * footprint.Footprint.RowPitch as usize;
                // SAFETY: inside the mapped staging buffer, which lives as
                // long as `staging`.
                unsafe { std::slice::from_raw_parts(staging.data.add(at), 16).to_vec() }
            };
            assert_eq!(level(0, 0), [0x5a; 16], "the rows of level 0 that are there");
            assert_eq!(level(0, 31), [0x5a; 16]);
            assert_eq!(level(0, 200), [0; 16], "the rows whose pages are not there");
            assert_eq!(level(1, 0), [0x5a; 16], "a level all there");
            assert!(staging.staged[usize::from(staging.place.part_of_level[0])].load(std::sync::atomic::Ordering::Relaxed) > 1);
            // SAFETY: the whole reservation above.
            assert!(!unsafe { VirtualAlloc(base, 6 * RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
            super::remapped(base as usize, 6 * RUN);
            super::blocks_remapped(base as usize, 6 * RUN);
            staging.refresh();
            assert_eq!(super::levels_shown_from(resource), None, "all there: no clamp");
            assert_eq!(level(0, 0), [0x5a; 16], "what was there is there still");
        }
        // SAFETY: the reservation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// With `VICTORY_TEXTURE_NO_WAIT=1`, a texture never filled whose first
    /// refresh meets the runtime remapping (and cannot be read through the
    /// kernel: a page is not there) waits for the remap and is read, not
    /// left for the next refresh with zeros.
    #[test]
    fn a_first_read_waits_for_a_remap() {
        use windows_sys::Win32::System::Memory::*;
        const CHILD: &str = "VICTORY_TEST_FIRST_READ";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", std::thread::current().name().unwrap(), "--nocapture"])
                .env(CHILD, "1")
                .env("VICTORY_TEXTURE_NO_WAIT", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        const RUN: usize = 0x1_0000;
        // A linear 256x256 RGBA8 texture: four runs, only the first there.
        // SAFETY: a private reservation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * RUN, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: the first run of the reservation above.
        assert!(!unsafe { VirtualAlloc(base, RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: committed memory of ours.
        unsafe { base.cast::<u8>().write(0x33) };
        let description = [3, 0, 0, 0, 256, 0, 256, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(base as u64, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            super::staging(resource, host_texture)
        });
        if let Some(staging) = staged {
            let (held, release) = std::sync::mpsc::channel();
            let remapping = std::thread::spawn(move || {
                let _changing = crate::watch::title_memory_remapping();
                held.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(100));
            });
            release.recv().unwrap();
            staging.refresh();
            remapping.join().unwrap();
            // SAFETY: the staging buffer stays mapped while `staging` lives.
            assert_eq!(unsafe { staging.data.read_volatile() }, 0x33, "the first read was left for later");
            assert_eq!(super::FIRST_WAITS[0].load(Relaxed), 1);
        }
        // SAFETY: the reservation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    /// A level in memory nothing watches and nobody told of is read and
    /// staged, and again when its bytes change; the level before it, with
    /// no page there, is zeros beside it. (The case of Marvisak/VIctory
    /// pull request #2639, where a mip off the runtime's memory was never
    /// read: here nothing is concluded from what is watched, a level is
    /// read if it is the title's and its pages are there.)
    #[test]
    fn a_level_in_memory_nothing_watches_is_brought_and_brought_again() {
        use windows_sys::Win32::System::Memory::*;
        const RUN: usize = 0x1_0000;
        // Two runs, the second alone there.
        // SAFETY: a private reservation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 2 * RUN, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        let there = base.cast::<u8>().wrapping_add(RUN);
        // SAFETY: the second run of the reservation above.
        assert!(!unsafe { VirtualAlloc(there.cast(), RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        // SAFETY: the run committed above.
        unsafe { std::ptr::write_bytes(there, 0x40, RUN) };
        super::blocks_remapped(base as usize, 2 * RUN);
        // A linear 128x128 RGBA8 texture of two levels: level 1 (16 KiB)
        // lies first, in the run that is not there; level 0 (64 KiB) after
        // it, the committed run whole.
        let address = base as u64 + RUN as u64 - (16 << 10);
        let description = [3, 0, 0, 0, 128, 0, 128, 0x2_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(address, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            super::unstage(resource);
            super::staging(resource, host_texture)
        });
        if let Some(staging) = staged {
            assert_eq!(staging.place.parts, [(0, 16 << 10), (16 << 10, 80 << 10)], "level 1 first, then level 0");
            let key = |level: usize| staging.staged[usize::from(staging.place.part_of_level[level])].load(Relaxed);
            let texel = |level: usize| {
                // SAFETY: inside the mapped staging buffer.
                unsafe { staging.data.add(staging.footprints[level].Offset as usize).read_volatile() }
            };
            staging.refresh();
            assert_eq!(key(1), 1, "the level with no page there: zeros");
            let first = key(0);
            assert!(first > 1, "the level in memory nothing watches is read");
            assert_eq!((texel(0), texel(1)), (0x40, 0));
            // Its bytes change, and nobody is told.
            // SAFETY: the committed run.
            unsafe { std::ptr::write_bytes(there, 0x80, RUN) };
            staging.refresh();
            assert!(key(0) > 1 && key(0) != first, "and read again when it changed");
            assert_eq!(texel(0), 0x80);
        }
        // SAFETY: the reservation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
        super::blocks_remapped(base as usize, 2 * RUN);
    }

    /// A texture with pages missing whose memory holds nothing but zeros
    /// where it is there is left as it is: nothing is staged for it and
    /// its host copy keeps what it has. When bytes come, they go up.
    #[test]
    fn a_texture_partly_there_with_nothing_but_zeros_changes_nothing() {
        use windows_sys::Win32::System::Memory::*;
        const RUN: usize = 0x1_0000;
        // The linear 256x256 texture of nine levels again, its first two
        // runs there (the small levels, level 1 and the start of level 0).
        // SAFETY: a private reservation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 6 * RUN, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: the first two runs of the reservation above.
        assert!(!unsafe { VirtualAlloc(base, 2 * RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        super::blocks_remapped(base as usize, 6 * RUN);
        let description = [3, 0, 0, 0, 256, 0, 256, 0x9_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(base as u64, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            super::unstage(resource);
            super::staging(resource, host_texture)
        });
        if let Some(staging) = staged {
            let any = || staging.staged.iter().any(|key| key.load(Relaxed) != 0);
            staging.refresh();
            assert!(!any(), "zeros where it is there, pages missing: nothing staged");
            // SAFETY: the two runs committed above.
            unsafe { std::ptr::write_bytes(base.cast::<u8>(), 0x5a, 2 * RUN) };
            super::remapped(base as usize, 2 * RUN);
            staging.refresh();
            assert!(any(), "its bytes came");
            assert!(staging.staged[usize::from(staging.place.part_of_level[1])].load(Relaxed) > 1);
        }
        // SAFETY: the reservation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
        super::blocks_remapped(base as usize, 6 * RUN);
    }

    /// Whether anything of a range is there: by what is kept of its 64 KiB
    /// blocks, and afresh at its first and last page, so that pages mapped
    /// with nobody told of it are found where a level's bytes start.
    #[test]
    fn something_of_a_range_is_there_or_nothing() {
        use windows_sys::Win32::System::Memory::*;
        const RUN: usize = 0x1_0000;
        // Four runs, the second alone there.
        // SAFETY: a private reservation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * RUN, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        let at = |run: usize, byte: usize| base.cast::<u8>().wrapping_add(run * RUN + byte).cast_const();
        // SAFETY: the second run of the reservation above.
        assert!(!unsafe { VirtualAlloc(at(1, 0).cast(), RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        super::blocks_remapped(base as usize, 4 * RUN);
        assert!(super::some_there(at(0, 0), 4 * RUN), "a block in the middle");
        assert!(super::some_there(at(1, 0x800), 0x100));
        assert!(super::some_there(at(0, 0xff00), 0x200), "its last page");
        assert!(!super::some_there(at(0, 0), RUN));
        assert!(!super::some_there(at(2, 0), 2 * RUN));
        // The third run comes and nobody tells: what is kept of its block
        // says it is not there, its first page says it is.
        // SAFETY: the third run of the reservation above.
        assert!(!unsafe { VirtualAlloc(at(2, 0).cast(), RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        assert!(super::some_there(at(2, 0), 2 * RUN), "pages mapped with nobody told");
        // No bytes, and a range at the top of the address space: nothing.
        assert!(!super::some_there(at(1, 0), 0));
        assert!(!super::some_there(usize::MAX as *const u8, 16));
        // SAFETY: the reservation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
        super::blocks_remapped(base as usize, 4 * RUN);
    }

    /// A partly mapped texture is read at most every 30 frames; after a
    /// remap under it, at once.
    #[test]
    fn a_remap_under_a_partly_mapped_texture_is_read_at_once() {
        use windows_sys::Win32::System::Memory::*;
        const RUN: usize = 0x1_0000;
        // A linear 256x256 RGBA8 texture is four runs of 64 KiB; only the
        // first is committed.
        // SAFETY: a private reservation of our own, released at the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * RUN, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        // SAFETY: the first run of the reservation above.
        assert!(!unsafe { VirtualAlloc(base, RUN, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let first = base.cast::<u8>();
        let description = [3, 0, 0, 0, 256, 0, 256, 0x1_0001, 28, 1, 0, 1, 0, 0];
        let staged = crate::resource::texture_at(base as u64, &description).and_then(|resource| {
            let host_texture = crate::resource::host_resource(resource)?;
            super::staging(resource, host_texture)
        });
        if let Some(staging) = staged {
            // SAFETY: the staging buffer stays mapped while `staging` lives.
            let staged_byte = || unsafe { staging.data.read_volatile() };
            // SAFETY: committed memory of ours.
            unsafe { first.write(0x11) };
            staging.refresh();
            assert_eq!(staged_byte(), 0x11, "the first read");
            // The same frame, other memory shown there.
            // SAFETY: as above.
            unsafe { first.write(0x22) };
            super::remapped(base as usize, RUN);
            staging.refresh();
            assert_eq!(staged_byte(), 0x22, "the read after the remap waited");
        }
        // SAFETY: the reservation above, released once.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }
}

#[cfg(test)]
mod digest_tests {
    use super::{Digest, digest};

    /// The digest tells every length and every changed byte apart, zeros
    /// among them (avatarmanz/VIctory-7900XTX 2dfba4dbd).
    /// What the card wrote last stays the card's until the title writes
    /// that level's bytes, level by level: a level the card did not write
    /// comes from title memory, one the title wrote after the card is the
    /// title's again, and a card write after that is the card's. (The
    /// memory here is a buffer of the test's: nothing watches it, so it is
    /// compared every time.)
    #[test]
    fn the_card_keeps_a_level_until_the_title_writes_it() {
        use super::{CardState, Where, subresource_bit};
        // Two levels in 0x3000 bytes: the block of the small one first,
        // then level 0 from 0x1000.
        let mut memory = vec![7u8; 0x3000];
        let at = Where::new(memory.as_ptr() as usize, 0x3000, 1, &[0x1000], 2, 2);
        assert_eq!((at.all(), at.bit(0), at.bit(1)), (3, 1, 2));
        let card = CardState::default();
        // Nothing is the card's: every level asked for is copied.
        assert!(!card.all_card());
        assert_eq!(card.decide(3), (3, 0, 0));
        assert_eq!(card.check(), 0);
        // The card writes level 0: it is withheld, level 1 is copied.
        card.wrote(&at, 1);
        assert_eq!(card.subs(), 1);
        assert!(!card.all_card());
        assert_eq!(card.decide(3), (2, 1, 0));
        assert_eq!(card.decide(2), (2, 0, 0));
        // The title writes level 0's bytes: its own again, and said so.
        memory[0x2000] = 9;
        assert_eq!(card.decide(3), (3, 0, 1));
        assert_eq!(card.subs(), 0);
        assert_eq!(card.decide(3), (3, 0, 0));
        // The card writes both, the title then the small level alone.
        card.wrote(&at, 3);
        assert!(card.all_card());
        assert_eq!(card.check(), 0);
        memory[0x10] = 1;
        assert_eq!(card.check(), 2);
        assert_eq!(card.subs(), 1);
        // The title writes level 0 and the card then writes level 1: what
        // the title wrote of the level the card does not write stays the
        // title's.
        memory[0x2800] = 3;
        card.wrote(&at, 2);
        assert_eq!(card.subs(), 2);
        assert_eq!(card.decide(3), (1, 2, 0));
        // A texture of more subresources than a mask tells apart is all
        // the card's or all the title's; one of several slices is one part.
        let many = Where::new(memory.as_ptr() as usize, 0x3000, 6, &[0x1000], 12, 72);
        assert_eq!((many.all(), many.bit(3), many.bit(71)), (u64::MAX, u64::MAX, u64::MAX));
        assert_eq!(many.changed(&[1; crate::census::PARTS], &[2; crate::census::PARTS]), u64::MAX);
        assert_eq!(many.changed(&[1; crate::census::PARTS], &[1; crate::census::PARTS]), 0);
        assert_eq!((subresource_bit(0, 1), subresource_bit(63, 64), subresource_bit(200, 64)), (1, 1 << 63, 1 << 63));
        // Formats of one family are copied as they are: the two of a
        // block-compressed kind, the 8-bit colours; not one into another.
        use super::format_family;
        assert_eq!(format_family(77), format_family(78));
        assert_eq!(format_family(28), format_family(29));
        assert_eq!(format_family(87), format_family(91));
        assert_ne!(format_family(78), format_family(71));
        assert_ne!(format_family(28), format_family(87));
        assert_eq!(format_family(26), 26);
        assert_eq!(format_family(u32::MAX), u32::MAX);
        // No bytes, or more than a texture has: nothing is read, nothing
        // is ever told changed.
        let none = Where::new(0, 0, 1, &[], 1, 1);
        assert_eq!(none.digests(), [0; crate::census::PARTS]);
        let huge = Where::new(0, 1 << 40, 1, &[], 1, 1);
        assert_eq!(huge.digests(), [0; crate::census::PARTS]);
        drop(memory);
    }

    /// What a host copy holds is noted part by part: a part's key tells
    /// its bytes and which of its levels are the title's; the layout itself
    /// says whether each level lies whole in the part taken for it; and a
    /// card write into a level takes the note of that level's part alone.
    #[test]
    fn a_host_copy_is_told_what_it_holds_part_by_part() {
        use super::{Sent, Where, part_key};
        use std::sync::atomic::Ordering::Relaxed;
        assert_eq!(part_key(77, 0b100, u32::MAX), 77);
        assert_eq!(part_key(77, 0b100, 0b011), 1, "a level that is not the title's is zeros whatever its bytes");
        assert_ne!(part_key(77, 0b1100, 0b0100), part_key(77, 0b1100, 0b1000));
        assert_ne!(part_key(77, 0b1100, 0b0100), part_key(77, 0b1100, 0b1100));
        assert_eq!(part_key(77, 0b1100, 0b0011), 1);
        assert!(part_key(0, 1, 1) >= 2 && part_key(1, 1, 1) >= 2, "never the key of nothing, nor that of zeros");
        // A linear 256x256 texture of nine levels, of 0x9_0001 layers and
        // levels: every level in one part, each part with a level, level
        // 0 in the last one.
        let layout = |words: [u32; 14]| {
            // SAFETY: 56 bytes in the description's layout.
            let desc = unsafe { words.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
            crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
        };
        let nine = layout([3, 0, 0, 0, 256, 0, 256, 0x9_0001, 28, 1, 0, 1, 0, 0]).unwrap();
        let at = Where::of(0x10_0000, &nine, 9, 9);
        assert!(at.parts.len() > 1, "told apart by level: {:?}", at.parts);
        assert!(at.fits(&nine));
        assert_eq!(at.levels_of_part.iter().fold(0, |all, levels| all | levels), 0x1ff);
        assert_eq!(at.levels_of_part.iter().map(|levels| levels.count_ones()).sum::<u32>(), 9, "a level in two parts");
        assert_eq!(at.parts.first().map(|p| p.0), Some(0));
        assert_eq!(at.parts.last().map(|p| p.1 as u64), Some(nine.size));
        assert_eq!(at.levels_of_part.last().copied(), Some(1), "level 0 lies last, alone");
        for (level, &part) in at.part_of_level.iter().enumerate() {
            assert_ne!(at.levels_of_part[usize::from(part)] & (1 << level), 0);
        }
        // Other levels on the host than the layout has: nothing says where
        // they lie, so one part of everything. Several layers: one part.
        let other = Where::of(0x10_0000, &nine, 5, 5);
        assert_eq!((other.parts.len(), other.levels_of_part[0]), (1, 0x1f));
        let six = layout([3, 0, 0, 0, 256, 0, 256, 0x9_0006, 28, 1, 0, 1, 0, 0]).unwrap();
        let cube = Where::of(0x10_0000, &six, 9, 54);
        assert_eq!((cube.parts.len(), cube.levels_of_part[0]), (1, 0x1ff));
        // The host copy holds every part; the card writes level 0 and the
        // smallest level: their parts are forgotten, the others are not.
        let sent = Sent::default();
        assert!(sent.none());
        for key in &sent.0[..at.parts.len()] {
            key.store(5, Relaxed);
        }
        assert!(!sent.none());
        sent.card_wrote(&at, at.bit(0) | at.bit(8));
        let (of_0, of_8) = (usize::from(at.part_of_level[0]), usize::from(at.part_of_level[8]));
        for part in 0..at.parts.len() {
            assert_eq!(sent.key(part), if part == of_0 || part == of_8 { 0 } else { 5 }, "part {part}");
        }
        // A part there is none of holds nothing; a write of nothing, or to
        // a texture of more subresources than are told apart, forgets all
        // it may have touched and never reaches past the notes.
        assert_eq!(sent.key(usize::MAX), 0);
        sent.card_wrote(&at, 0);
        assert_eq!(sent.key(usize::from(at.part_of_level[1])), 5);
        sent.card_wrote(&cube, u64::MAX);
        assert_eq!(sent.key(0), 0);
    }

    /// What of a texture's memory is the title's, part by part: every
    /// level a live view shows, whatever starts inside it; a more detailed
    /// one only if it ends before the next texture placed in the span
    /// starts; all of a texture object. A part that is the title's is read
    /// if its pages are there, and is zeros on the host if not; what is
    /// not the title's is never asked about.
    #[test]
    fn a_part_is_the_titles_from_the_first_level_viewed() {
        use super::{PartIs, part_is};
        let never = || -> bool { panic!("a part that is not the title's was asked about") };
        // Viewed from level 3, the next texture starting 0x5000 in: the
        // parts of levels 3 and up are the title's wherever they end; level
        // 2 ends before the neighbour and is the title's; levels 1 and 0
        // reach past its start and are left.
        let judged = [3, 0x5000];
        assert_eq!(part_is(true, 0b1111_1000, 0x1000, judged, || true), PartIs::Read);
        assert_eq!(part_is(true, 0b1000, 0x9000, judged, || true), PartIs::Read, "a level viewed, with a view starting inside it");
        assert_eq!(part_is(true, 0b0100, 0x5000, judged, || true), PartIs::Read, "a level that ends where the next texture starts");
        assert_eq!(part_is(true, 0b0010, 0x5001, judged, never), PartIs::Left);
        assert_eq!(part_is(true, 0b0001, 0x40_0000, judged, never), PartIs::Left);
        // Its pages not there: zeros for a level a view shows; a level no
        // view shows is left, whether another texture starts in it or not.
        assert_eq!(part_is(true, 0b1000, 0x2000, judged, || false), PartIs::Zeros);
        assert_eq!(part_is(true, 0b1111_1000, 0x1000, judged, || false), PartIs::Zeros);
        assert_eq!(part_is(true, 0b0100, 0x5000, judged, || false), PartIs::Left);
        assert_eq!(part_is(true, 0b0001, 0x40_0000, [3, u64::MAX], || false), PartIs::Left);
        // No neighbour inside: every level is the title's.
        assert_eq!(part_is(true, 0b0001, 0x40_0000, [3, u64::MAX], || true), PartIs::Read);
        // No view says from where (level 0): all of it, wherever others start.
        assert_eq!(part_is(true, 0b0001, 0x40_0000, [0, 0x1000], || true), PartIs::Read);
        // A texture object of the title's is all its own, and so is any
        // texture as a draw reads it through a table.
        assert_eq!(part_is(false, 0b0001, 0x40_0000, judged, || true), PartIs::Read);
        assert_eq!(part_is(false, 0b0001, 0x40_0000, judged, || false), PartIs::Zeros);
        // A first level past every level there is, a part of no level, an
        // end at the top of the address space: left, or read; no overflow.
        assert_eq!(part_is(true, u32::MAX, u64::MAX, [u64::MAX, 0], never), PartIs::Left);
        assert_eq!(part_is(true, 0, 0, [0, 0], || true), PartIs::Read);
        assert_eq!(part_is(true, 0, 1, [0, 0], never), PartIs::Left);
        assert_eq!(part_is(true, 1 << 31, u64::MAX, [31, 0], || true), PartIs::Read);
        assert_eq!(part_is(true, 1 << 31, u64::MAX, [32, u64::MAX], || true), PartIs::Read);
        // The first level the pass takes as shown: the lowest of what the
        // spans say and where the slot's own view starts.
        use super::first_shown;
        assert_eq!(first_shown(Some(3), Some(1)), 1);
        assert_eq!(first_shown(Some(1), Some(3)), 1);
        assert_eq!(first_shown(None, Some(2)), 2);
        assert_eq!(first_shown(Some(4), None), 4);
        assert_eq!(first_shown(None, None), 0);
        // When a check of a guarded texture reads.
        use super::{Check, check_reads, check_why};
        let quiet = Check { unwritten: true, ..Check::default() };
        assert!(!check_reads(quiet), "nothing written, not its turn");
        assert!(check_reads(Check { unwritten: false, ..quiet }) && check_why(Check { unwritten: false, ..quiet }) == 0);
        assert!(check_reads(Check { remapped: true, ..quiet }) && check_why(Check { remapped: true, ..quiet }) == 1);
        assert!(check_reads(Check { young: true, ..quiet }) && check_why(Check { young: true, ..quiet }) == 2);
        assert!(check_reads(Check { behind: true, ..quiet }) && check_why(Check { behind: true, ..quiet }) == 3);
        assert!(check_reads(Check { turn: true, ..quiet }) && check_why(Check { turn: true, ..quiet }) == 4);
        // The net once a frame; a texture behind its guards every time.
        assert!(!check_reads(Check { turn: true, hashed_now: true, ..quiet }));
        assert!(check_reads(Check { behind: true, hashed_now: true, ..quiet }));
        // Pages missing at a read lately: only a write or a mapping.
        let partly = Check { partly_lately: true, ..quiet };
        assert!(!check_reads(Check { turn: true, young: true, behind: true, ..partly }));
        assert!(check_reads(Check { unwritten: false, ..partly }) && check_reads(Check { remapped: true, ..partly }));
        // A change no guard saw is a writer's the guards do not see,
        // unless a read of the title's was writing the memory at one of
        // the two checks, or the memory was mapped anew between them.
        use super::behind_guards;
        assert!(behind_guards([false, false], false));
        assert!(!behind_guards([true, false], false) && !behind_guards([false, true], false));
        assert!(!behind_guards([false, false], true) && !behind_guards([true, true], true));
        // A part holds a level from a given one on, or none: the part of
        // level 1 alone holds none from level 2 on (a host copy made
        // without its first two levels has nothing of it).
        use super::reaches;
        assert!(reaches(0b0010, 0) && reaches(0b0010, 1) && !reaches(0b0010, 2));
        assert!(reaches(0b1111_0000, 7) && !reaches(0b1111_0000, 8));
        assert!(!reaches(0, 0) && !reaches(u32::MAX, 32) && !reaches(u32::MAX, u64::MAX) && reaches(u32::MAX, 31));
    }

    /// A texture's memory is told apart at the starts of its levels, in
    /// memory order, the block of the smallest ones first; one of several
    /// slices, or of more levels than are told apart, is one part.
    #[test]
    fn memory_parts_follow_the_levels() {
        // 0x1000 of small levels, then levels of 0x1000, 0x4000, 0x10000.
        let offsets = [0x6000, 0x2000, 0x1000];
        assert_eq!(super::memory_parts(0x16000, 1, &offsets), [(0, 0x1000), (0x1000, 0x2000), (0x2000, 0x6000), (0x6000, 0x16000)]);
        assert_eq!(super::memory_parts(0x16000, 6, &offsets), [(0, 0x16000)]);
        assert_eq!(super::memory_parts(0x1000, 1, &[]), [(0, 0x1000)]);
        // An offset of 0 or past the end cuts nothing.
        assert_eq!(super::memory_parts(0x3000, 1, &[0, 0x2000, 0x9000]), [(0, 0x2000), (0x2000, 0x3000)]);
        let many: Vec<u64> = (1..=20).map(|k| k * 0x1000).collect();
        assert_eq!(super::memory_parts(0x20_0000, 1, &many), [(0, 0x20_0000)]);
        // No bytes, no slices, offsets at the top of the address space: one
        // part, never none.
        assert_eq!(super::memory_parts(0, 1, &[]), [(0, 0)]);
        assert_eq!(super::memory_parts(0, 0, &[0, u64::MAX]), [(0, 0)]);
        assert_eq!(super::memory_parts(0x1000, 1, &[u64::MAX, 0x800]), [(0, 0x800), (0x800, 0x1000)]);
        assert!(super::all_zero(&[0u8; 37]) && !super::all_zero(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]));
    }

    #[test]
    fn a_digest_tells_bytes_apart() {
        let zeros = [0u8; 400];
        let mut seen: Vec<Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }

    /// What the digest and blake3 cost on this machine, in milliseconds a
    /// megabyte over 64 MiB. Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture hash_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn hash_costs() {
        let bytes: Vec<u8> = (0..64usize << 20).map(|i| ((i * 2_654_435_761) >> 13) as u8).collect();
        let timed = |what: &str, f: &dyn Fn()| {
            f();
            let started = std::time::Instant::now();
            for _ in 0..8 {
                f();
            }
            println!("{what}: {:.3} ms a MB", started.elapsed().as_secs_f64() * 1e3 / 8.0 / 64.0);
        };
        timed("blake3", &|| {
            std::hint::black_box(blake3::hash(&bytes));
        });
        timed("digest", &|| {
            std::hint::black_box(digest(&bytes));
        });
    }
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }
}
