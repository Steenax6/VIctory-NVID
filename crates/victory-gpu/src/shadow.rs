//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Pages (by index, as `copied`) a refresh took in part: the asking
    /// after which the bytes at these offsets in the page (in order, apart)
    /// were last the title's ([`refresh_parts`]). Locked inside `copied`'s
    /// lock.
    partial: Mutex<std::collections::HashMap<usize, (u64, Vec<(u32, u32)>)>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// For a copy in the card's memory, which has no snapshot: one lock a
    /// 64 KiB piece (cut as [`pieces_of`] cuts them) over its CPU side,
    /// made when first taken ([`Shadow::side_of`]).
    side_pieces: std::sync::OnceLock<Box<[Mutex<()>]>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// Diagnostics: who asked for a refresh of the copy since it was last
    /// brought up to date ([`ASKED_BY_DRAW`] and the others), for the
    /// census (`crate::copystat::buffers`).
    #[cfg(feature = "diagnostics")]
    asked_by: AtomicU8,
    /// The range last brought up to date by asking, with the events counted
    /// in it just before (`crate::watch::Seen`): with
    /// `VICTORY_REFRESH_EVENTS`, a refresh inside it at the same count has
    /// nothing to ask.
    events_seen: Mutex<crate::watch::Seen>,
    /// The check of `VICTORY_LAND_TAKEN=2`: pages of title memory a
    /// landing would have noted taken by this copy, with the landing's
    /// asking ([`Shadow::landed_pages`]); looked at by the refresh that
    /// reads them.
    landed_check: Mutex<std::collections::HashMap<usize, u64>>,
    /// Diagnostics, for the report of a removed device: the last refresh of
    /// a range of the copy where no shader writes ([`Shadow::update_free`]),
    /// as [`Refreshed`] says.
    #[cfg(feature = "diagnostics")]
    refreshed: Mutex<Refreshed>,
}

/// Diagnostics: what a copy's last refresh did ([`Shadow::lag_report`]).
#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy, Debug, Default)]
struct Refreshed {
    /// The frame, the range, and how it went: 0 none yet, 1 left out by
    /// events, 2 asked (with the asking's number and the bytes copied),
    /// 3 copied whole (memory not watched).
    frame: u64,
    from: usize,
    to: usize,
    how: u8,
    asking: u64,
    bytes: usize,
    /// The last refresh that asked, as above (frame, range, asking, bytes).
    asked: (u64, usize, usize, u64, usize),
    /// When the last refresh and the last that asked were made
    /// (`crate::host::now_us`).
    us: (u64, u64),
}

/// Who asks for a refresh of a copy ([`Shadow::refresh_as`]): a draw for
/// the views it binds; the pass over the buffer views shaders pick by
/// index, for every copy one of them names; a program that reads title
/// memory by address, for every numbered copy. Bits: a copy keeps the
/// askers it had until it is brought up to date.
pub(crate) const ASKED_BY_DRAW: u8 = 1;
pub(crate) const ASKED_BY_PICKED: u8 = 2;
pub(crate) const ASKED_BY_ADDRESS: u8 = 4;

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
    /// One bit per 4 KiB block of title memory (block 0 starts the copy's
    /// piece 0, `crate::card_fine::BLOCK`): set once a compare on the card
    /// flagged the block (or the piece, without fine flags, or a range that
    /// was not compared), and never cleared. Where no bit is set no shader
    /// has been seen to write ([`Shadow::merge_under_writes`],
    /// [`merge_pages`]).
    flagged: Mutex<Vec<u64>>,
    /// The copy itself, for [`DIRTY`]: set once the copy is shared (a copy
    /// a test builds by hand has none, and is asked for its ranges itself).
    me: std::sync::OnceLock<std::sync::Weak<Shadow>>,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_KB` (64, every title buffer; 1024 before, 16
/// MiB once) and `VICTORY_CARD_BUFFERS_MB` (1280; 1024 before, 512 once).
/// Run 69's gameplay has ten
/// title buffers of 16 MiB or more, 494 MB, and 1,650 MB of the card's
/// budget free. From 1 MiB up the shader writes of the smaller buffers are
/// found by the card's hashes too, and with that the first room's walls
/// were drawn in every run.
///
/// Under the least size a copy stays in the PC's memory, and every load,
/// store and atomic a shader does on it crosses the bus. Those copies are
/// most of them: some 3,700 of 3,800 and 30 MB in all in the source's first
/// rooms, where the passes that were slow by the thread were small
/// dispatches on such buffers (one tile-reset pass: 737,000 one-dword
/// requests on a 288-byte and a 900 KiB buffer in 1.4 to 2 ms). With
/// `VICTORY_CARD_BUFFERS_MIN_KB=64` (every title buffer: a copy is rounded
/// up to 64 KiB) and `VICTORY_CARD_BUFFERS_MB=1280` the title's lists took
/// 49.8 ms of card time a frame where they took 68.0, single passes 1.6 ms
/// where they took 6.7, on an RTX 4070 (fxceless/victory-fork c36971b,
/// runs 274 and 275). What it costs: 64 KiB of the card's memory a copy
/// and as much again for one shaders write, and the passes that send and
/// hash before each submission go through more copies (11.6 to 23.0 ms of
/// card time there; [`Shadow::card_hashed`] hashes a copy's own bytes for
/// that). Measured here (RTX 4070 Ti, prologue, with the own-bytes hash):
/// the first room 77.2 ms a frame against 87.8 and 89.0 with 1 MiB, the
/// card busy 56 % of the time against 74 to 78, the frame line's copies 26
/// ms against 17.5, the opening cutscene 108 ms, the picture the same; a
/// 22-minute run in the open world with it had no trouble. Where the
/// copies went is said with the memory lines ([`placed_line`]).
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let number = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (
            (number("VICTORY_CARD_BUFFERS_MIN_KB", 64) << 10).max(PIECE as u64),
            number("VICTORY_CARD_BUFFERS_MB", 1280) << 20,
        )
    })
}

/// Title buffers' host copies made since the start, by where they went and
/// by size: in the card's memory; in the PC's memory of 64 KiB (the least a
/// copy is), up to 256 KiB, up to 1 MiB, larger; each as (copies, bytes).
static PLACED: [[AtomicU64; 2]; 5] = [const { [const { AtomicU64::new(0) }; 2] }; 5];

fn placed_count(size: u64, on_card: bool) {
    let class = match (on_card, size) {
        (true, _) => 0,
        (false, ..=0x1_0000) => 1,
        (false, ..=0x4_0000) => 2,
        (false, ..=0x10_0000) => 3,
        _ => 4,
    };
    PLACED[class][0].fetch_add(1, Ordering::Relaxed);
    PLACED[class][1].fetch_add(size, Ordering::Relaxed);
}

/// Where title buffers' host copies went, since the start: a line with the
/// frame-rate line when a copy was made since the last one.
pub(crate) fn placed_line() {
    static SAID: AtomicU64 = AtomicU64::new(0);
    let v: Vec<(u64, u64)> = PLACED.iter().map(|c| (c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed))).collect();
    let made: u64 = v.iter().map(|c| c.0).sum();
    if SAID.swap(made, Ordering::Relaxed) == made {
        return;
    }
    let mb = |b: u64| b as f64 / (1 << 20) as f64;
    tracing::info!(
        "title buffers' host copies made so far, by where shaders read them: {} in the card's memory {:.0} MB; in the PC's \
         memory (read over the bus): {} of 64 KiB {:.0} MB, {} up to 256 KiB {:.0} MB, {} up to 1 MiB {:.0} MB, {} larger {:.0} MB \
         (the least size for the card: {} KiB, VICTORY_CARD_BUFFERS_MIN_KB; 64 is every title buffer)",
        v[0].0,
        mb(v[0].1),
        v[1].0,
        mb(v[1].1),
        v[2].0,
        mb(v[2].1),
        v[3].0,
        mb(v[3].1),
        v[4].0,
        mb(v[4].1),
        card_limits().0 >> 10,
    );
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_limits().1.saturating_sub(CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// The copies in the card's memory whose CPU side changed and has not gone
/// up yet: a copy is put here when its first such range is noted
/// ([`Shadow::card_changed`]) and taken out when its ranges are
/// ([`card_dirty`]). A copy may be here more than once, or with nothing
/// left to take.
static DIRTY: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = {
        // The copies that are gone leave the list here, once a frame-rate
        // line (`card_dirty` let them go while it still walked them all).
        let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
        carded.retain(|w| w.strong_count() > 0);
        carded.len() as u64
    };
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// Diagnostics: the copies in the card's memory, as (start, bytes) of
/// their title memory.
#[cfg(feature = "diagnostics")]
pub(crate) fn card_spans() -> Vec<(u64, u64)> {
    let carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    carded.iter().filter_map(|w| w.upgrade()).map(|s| (s.start, s.len)).collect()
}

/// Diagnostics: the copies in the card's memory by when they were last
/// asked for (`Shadow::used`): in this frame or the one before, within 60
/// frames, within 600, longer ago; each as (copies, bytes of their card
/// side).
#[cfg(feature = "diagnostics")]
pub(crate) fn card_use() -> [[u64; 2]; 4] {
    let now = crate::host::moment() >> 32;
    let mut out = [[0u64; 2]; 4];
    let carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    for shadow in carded.iter().filter_map(|w| w.upgrade()) {
        let ago = now.saturating_sub(shadow.used.load(Ordering::Relaxed) >> 32);
        let class = match ago {
            0..=1 => 0,
            2..=60 => 1,
            61..=600 => 2,
            _ => 3,
        };
        out[class][0] += 1;
        out[class][1] += shadow.card.as_ref().map_or(0, |c| c.size);
    }
    out
}

/// Diagnostics: every host copy of title memory that is alive, as (start,
/// bytes): title buffers' copies and windows outside buffers.
#[cfg(feature = "diagnostics")]
pub(crate) fn copy_spans() -> Vec<(u64, u64)> {
    let mut spans: Vec<(u64, u64)> = {
        let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
        map.iter().filter(|(_, entry)| entry.shadow.is_some()).map(|(key, entry)| (key.0, entry.end.saturating_sub(key.0))).collect()
    };
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.values().map(|s| (s.start, s.len)));
    spans
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more). With fine flags (`crate::card_fine::on`) 16 bytes
/// for each 4 KiB block of a piece, 256 a piece.
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    let per_piece = if crate::card_fine::on() { 16 * u64::from(crate::card_fine::BLOCKS) } else { 16 };
    (size / PIECE as u64 + 2) * per_piece
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`): textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
///
/// Only the copies with something to send are looked at ([`DIRTY`]). Every
/// copy in the card's memory was, at every submission: 3,700 copies 29
/// times a frame, 3 ms a frame of the submitting thread (profile of
/// 2026-11-26) to find the few that had changed.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let changed = std::mem::take(&mut *DIRTY.lock().unwrap_or_else(|e| e.into_inner()));
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    let mut out: Vec<(Arc<Shadow>, Vec<(usize, usize)>)> = Vec::with_capacity(changed.len());
    for shadow in changed.iter().filter_map(|weak| weak.upgrade()) {
        if let Some(ranges) = shadow.card_take_dirty(sent_as, fetched) {
            out.push((shadow, ranges));
        }
    }
    out
}

/// Notes `[from, to)` in a copy's changed ranges (joined with the last one
/// noted where they touch); whether it is the first since the ranges were
/// taken.
fn note_changed(dirty: &mut Vec<(usize, usize)>, from: usize, to: usize) -> bool {
    let first = dirty.is_empty();
    match dirty.last_mut() {
        Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
        _ => dirty.push((from, to)),
    }
    first
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// With fine flags (`crate::card_fine`): the offsets of the 4 KiB blocks
/// of piece `i` of `span` set in `mask` (bit `b`: block `b`, cut at
/// multiples of 4 KiB of title memory as `CARD_COMPARE_FINE` cuts them),
/// within what [`card_piece_of`] gives for the piece, pushed onto `out` as
/// runs (blocks that touch, also across calls, as one). Every byte of the
/// piece's part of the span that lies in a set block is in a run, and no
/// other byte is.
pub(crate) fn card_blocks_of(start: usize, len: usize, span: &CardSpan, i: u32, mask: u32, out: &mut Vec<(usize, usize)>) {
    use crate::card_fine::{BLOCK, BLOCKS};
    let (plo, phi) = card_piece_of(start, len, span, i);
    if plo >= phi || mask == 0 {
        return;
    }
    // The title address of the piece's first byte.
    let base = (start / PIECE + (span.first + i) as usize) * PIECE;
    for b in 0..BLOCKS as usize {
        if mask & (1 << b) == 0 {
            continue;
        }
        let lo = (base + b * BLOCK).saturating_sub(start).max(plo);
        let hi = (base + (b + 1) * BLOCK).saturating_sub(start).min(phi);
        if lo >= hi {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.1 == lo => last.1 = hi,
            _ => out.push((lo, hi)),
        }
    }
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// The fetch inside flights (`CARD_FETCH` in host.rs) copies each piece
    /// from `inflight_piece_from` on: what the merge takes of the piece lies
    /// in that slot.
    #[test]
    fn a_piece_fetched_in_flight_fits_its_slot() {
        let slot = crate::host::INFLIGHT_SLOT as usize;
        for start in [0usize, 4096, 7, PIECE - 16, PIECE - 3, 3 * PIECE + 0x1f09, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    let at = crate::host::inflight_piece_from(&span, i);
                    assert_eq!(at % 16, 0);
                    assert!(lo >= at && hi - at <= slot, "{start:#x} {from} {to} piece {i}: {lo}..{hi} from {at}");
                }
            }
        }
    }

    /// The slots are a power of two, one per 64 KiB asked, within the
    /// shader's 32-bit offsets.
    #[test]
    fn inflight_slots_are_a_power_of_two() {
        use crate::host::{INFLIGHT_SLOT, inflight_slots};
        assert_eq!(inflight_slots(64 << 20), 1024);
        assert_eq!(inflight_slots(100 << 20), 1024);
        assert_eq!(inflight_slots(1), 1);
        assert_eq!(inflight_slots(1 << 20), 16);
        let most = inflight_slots(u64::MAX);
        assert!(most.is_power_of_two() && u64::from(most) * INFLIGHT_SLOT <= 1 << 32);
    }

    /// Tickets drawn and landed in order, the mark raised as the landing
    /// raises it (every ticket a landed flight drew), through the counter's
    /// wrap with the slots the default gives: no two tickets in the air
    /// that got a slot share one, slots are still given after the wrap, and
    /// the mark follows the counter.
    #[test]
    fn inflight_tickets_never_share_a_slot_across_the_wrap() {
        use crate::host::{inflight_gets_slot, inflight_raise, inflight_slots};
        let slots = inflight_slots(64 << 20);
        let mut landed = u32::MAX - 3000;
        let mut next = landed;
        // Flights of 1 to 700 flagged pieces, up to three in the air:
        // (tickets drawn, those that got a slot).
        let mut air: std::collections::VecDeque<(Vec<u32>, Vec<u32>)> = Default::default();
        let mut after_wrap = 0;
        for flight in 0..40u32 {
            let drawn: Vec<u32> = (0..1 + flight * 37 % 700).map(|i| next.wrapping_add(i)).collect();
            next = next.wrapping_add(drawn.len() as u32);
            let slotted: Vec<u32> = drawn.iter().copied().filter(|&k| inflight_gets_slot(k, landed, slots)).collect();
            after_wrap += slotted.iter().filter(|&&k| k < 1 << 20).count();
            air.push_back((drawn, slotted));
            let mut used: Vec<u32> = air.iter().flat_map(|a| a.1.iter()).map(|k| k % slots).collect();
            let n = used.len();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), n, "flight {flight}: two tickets in the air share a slot");
            if air.len() == 3
                && let Some((drawn, _)) = air.pop_front()
            {
                for k in drawn {
                    landed = inflight_raise(landed, k);
                }
            }
        }
        while let Some((drawn, _)) = air.pop_front() {
            for k in drawn {
                landed = inflight_raise(landed, k);
            }
        }
        assert!(next < 1 << 20, "the counter wrapped");
        assert!(after_wrap > 0, "slots after the wrap");
        assert_eq!(landed, next);
        // Raised in wrapping order, never back.
        assert_eq!(inflight_raise(u32::MAX - 1, u32::MAX), 0);
        assert_eq!(inflight_raise(5, 2), 5);
        assert_eq!(inflight_raise(u32::MAX - 1, 3), 4);
        assert!(!inflight_gets_slot(3, 4, 8));
    }

    /// Flights 5, 6, 7 in the air with the fetch inside flights: 5 lands
    /// the old way with the queue's count at 7. The title's byte that went
    /// up in send 7 must still be told apart when 6 lands from its slot
    /// (its merge leaves out sends after 6), so send 7's note must outlive
    /// the mark (`card_dirty` keeps notes above it, `card_merge_relay`
    /// leaves out those above its "fetched after").
    #[test]
    fn a_fetch_the_old_way_keeps_the_notes_flights_in_the_air_need() {
        use crate::host::fallback_fetched_mark;
        let notes = [5u64, 6, 7];
        let mark = fallback_fetched_mark(true, 7, Some(5));
        let kept: Vec<u64> = notes.iter().copied().filter(|&s| s > mark).collect();
        assert!(kept.iter().any(|&s| s > 6), "send 7's note was dropped before flight 6 merged");
        assert_eq!(mark, 5);
        // Without the switch every later merge fetches after 7: as before.
        assert_eq!(fallback_fetched_mark(false, 7, Some(5)), 7);
        assert_eq!(fallback_fetched_mark(true, 7, None), 0);
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs of 4-byte words of title memory (cut on its addresses;
/// `read` and `merged` are of the memory at `address`) that hold a byte in
/// which `merged` differs from `read`, as (offset, bytes) in order to
/// `put`. A shader stores words, so these are what the card wrote; every
/// other byte of `merged` is the title's as it was read.
fn word_runs(address: usize, read: &[u8], merged: &[u8], mut put: impl FnMut(usize, usize)) {
    let n = read.len().min(merged.len());
    let mut i = 0;
    while i < n {
        // Equal bytes are passed over, 32 at a time while they last.
        match first_difference(&read[i..n], &merged[i..n]) {
            Some(d) => i += d,
            None => break,
        }
        // From the word that holds this byte (or the stretch's first
        // byte) to the end of the last word in a row that holds one.
        let first = i - ((address + i) & 3).min(i);
        let mut end = first;
        while end < n {
            let word = (((address + end) | 3) + 1 - address).min(n);
            if read[end..word] == merged[end..word] {
                break;
            }
            end = word;
        }
        put(first, end - first);
        i = end;
    }
}

/// Whether the merges that write a stretch of title memory back
/// ([`Shadow::relay_merge`], [`Shadow::card_merge`],
/// [`Shadow::card_merge_relay`]) look, right before that write, for title
/// bytes it is about to put back to older values (`VICTORY_MERGE_RACE=1`;
/// off by default). Logging only: [`merge_race`].
fn merge_race_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_RACE").is_some_and(|v| v == "1"))
}

/// [`merge_race`]'s sums. Of frame `frame`, and in `total` since the
/// start: stretches written back, their bytes, title bytes put back to
/// older values, bytes the title and the card both changed, stretches with
/// either, the runs a write of the card's words alone makes
/// ([`word_runs`]) and their bytes. `said`: the frame's first stretches
/// with either. `quiet`: frames since the last line.
struct MergeRace {
    frame: u64,
    now: [u64; 7],
    said: Vec<String>,
    total: [u64; 7],
    quiet: u64,
}

static MERGE_RACE: Mutex<MergeRace> =
    Mutex::new(MergeRace { frame: 0, now: [0; 7], said: Vec::new(), total: [0; 7], quiet: 0 });

/// With [`merge_race_on`], right before a merge writes `merged` over title
/// memory at `address` (a stretch of the copy of the title buffer `copy`:
/// start and bytes): title memory is read once more and compared with
/// `read`, what the merge read of it before. A byte that changed in
/// between and that the write-back stores with its value as first read
/// (the card did not change it) is a title byte put back to an older
/// value; one the write-back stores with the card's byte is one both
/// changed. Only bytes the write-back stores are looked at. Summed per
/// frame; a line for every frame that has any, and one every 100 frames
/// whatever it holds, so that "none" is said, not assumed.
///
/// A lower bound: a title write that lands after this read (while the
/// write-back itself runs) is not seen, nor one that wrote a byte and then
/// its old value again. Costs one more read and two compares of every
/// stretch written back, on the thread that merges.
fn merge_race(read: Option<&[u8]>, copy: (u64, u64), address: usize, merged: &[u8], which: &'static str) {
    if !merge_race_on() {
        return;
    }
    let Some(read) = read else {
        return;
    };
    let n = read.len().min(merged.len());
    let (read, merged) = (&read[..n], &merged[..n]);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    word_runs(address, read, merged, |off, len| runs.push((off, len)));
    let whole = [(0, n)];
    let looked: &[(usize, usize)] = &whole;
    let mut again = vec![0u8; n];
    let (mut put_back, mut both, mut first) = (0u64, 0u64, None);
    if crate::host::read_into(address as *const u8, &mut again) {
        for &(off, len) in looked {
            for i in off..off + len {
                if again[i] == read[i] || again[i] == merged[i] {
                    continue;
                }
                if merged[i] == read[i] {
                    put_back += 1;
                    first.get_or_insert(i);
                } else {
                    both += 1;
                }
            }
        }
    }
    let frame = crate::frame::current();
    let mut race = MERGE_RACE.lock().unwrap_or_else(|e| e.into_inner());
    if race.frame != frame {
        race.quiet += 1;
        if race.now[4] > 0 || race.quiet >= 100 {
            tracing::info!(
                frame = race.frame,
                stretches = race.now[0],
                stretch_bytes = race.now[1],
                put_back = race.now[2],
                both_changed = race.now[3],
                stretches_hit = race.now[4],
                word_runs = race.now[5],
                word_run_bytes = race.now[6],
                stretches_since_start = race.total[0],
                stretch_bytes_since_start = race.total[1],
                put_back_since_start = race.total[2],
                both_changed_since_start = race.total[3],
                stretches_hit_since_start = race.total[4],
                word_runs_since_start = race.total[5],
                word_run_bytes_since_start = race.total[6],
                first = ?race.said,
                "merge race: title bytes that changed between a merge's read of title memory and its write-back of what shaders wrote, in this frame (VICTORY_MERGE_RACE=1)"
            );
            race.quiet = 0;
        }
        race.frame = frame;
        race.now = [0; 7];
        race.said.clear();
    }
    let hit = u64::from(put_back + both > 0);
    let run_bytes: u64 = runs.iter().map(|r| r.1 as u64).sum();
    let add = [1, n as u64, put_back, both, hit, runs.len() as u64, run_bytes];
    for (i, v) in add.iter().enumerate() {
        race.now[i] += v;
        race.total[i] += v;
    }
    if hit > 0 && race.said.len() < 8 {
        race.said.push(format!(
            "{which}: copy {:#x}+{:#x} stretch {address:#x}+{n:#x}: {put_back} put back (first at {:#x}), {both} both changed",
            copy.0,
            copy.1,
            address + first.unwrap_or(0)
        ));
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6);
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                crate::read_census::layer_thread(false);
                let workers = WORKERS.get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    job();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_KB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (copies of the least size VICTORY_CARD_BUFFERS_MIN_KB or more are in the card's memory)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// How far apart two pieces that differ may be and still be written with
/// one stretch (`coalesce_pieces`): a 64 KiB piece here is one kernel read
/// and one kernel write of title memory.
const PIECE_GAP: usize = 64 << 10;

/// The most one merged stretch may hold (longer ones are cut): the read and
/// the write of it are one call each, but 75 MB a frame arriving in a
/// handful of very long stretches would stall the submitting thread.
const PIECE_MOST: usize = 4 << 20;

/// Spans of a copy that differ, as `Shadow::differing` returns them (one per
/// 64 KiB piece): neighbouring ones within [`PIECE_GAP`] become one stretch.
/// `back_piece` already reads and writes title memory from the first byte
/// that differs to the last, and writes only what the GPU wrote, so the
/// bytes between two such spans cost a read of title memory and no write.
fn coalesce_pieces(mut spans: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if spans.len() < 2 {
        return spans;
    }
    spans.sort_unstable_by_key(|&(at, _)| at);
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
    for (at, n) in spans {
        let mut joined = false;
        if let Some(last) = out.last_mut() {
            let end = last.0 + last.1;
            if at <= end.saturating_add(gap) && (at + n).saturating_sub(last.0) <= PIECE_MOST {
                last.1 = end.max(at + n) - last.0;
                joined = true;
            }
        }
        if !joined {
            out.push((at, n));
        }
    }
    out
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Joins runs (offset, length; in order, apart) that have fewer than `gap`
/// bytes between them: one run from the first's start to the last's end.
fn join_runs(runs: &mut Vec<(usize, usize)>, gap: usize) {
    let mut kept = 0;
    for i in 1..runs.len() {
        let (off, len) = runs[i];
        let end = runs[kept].0 + runs[kept].1;
        if off - end < gap {
            runs[kept].1 = off + len - runs[kept].0;
        } else {
            kept += 1;
            runs[kept] = (off, len);
        }
    }
    runs.truncate(kept + 1);
}

/// How close two runs of a piece [`Shadow::copy_back`] puts back must lie
/// to be written as one, in bytes between them (`VICTORY_BACK_RUN_GAP`, 64;
/// `0`: every run is a write of its own, as before).
///
/// A shader that writes records leaves many short runs: every byte of a
/// record that comes out as it was ends one. In the world frames of the
/// opening scene (RX 7900 XT, Wine) about 250 pieces a frame held 240,000
/// to 280,000 runs, each a call into the kernel: 0.8 to 1.2 s of a frame of
/// 1.4 s where that call goes through Wine. Joined over gaps under 64
/// bytes they are about 4,000 writes a frame and 20 ms (under 8 bytes about
/// 15,000 writes, under 512 about 1,100).
///
/// What it gives up: the bytes between two joined runs are written as they
/// were read a moment before, so a title write landing on one of them
/// between the read and the write is lost, as it was for the whole stretch
/// before only the GPU's bytes were written. That is now a byte less than
/// `gap` bytes from one the shader wrote, not anywhere in the stretch.
fn back_run_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_BACK_RUN_GAP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(64))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes; of those runs, the ones
/// joined over blocks no compare ever flagged ([`merge_pages`]) and their
/// bytes.
static MERGED: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 7] {
    [0, 1, 2, 3, 4, 5, 6].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// Whether [`Shadow::merge_under_writes`] sends the title's changes to a
/// copy in the card's memory up as one run per stretch of 4 KiB blocks
/// that no compare on the card ever flagged (`Card::flagged`), from the
/// first changed byte to the last, and byte-exact only in blocks that one
/// did (`VICTORY_MERGE_PAGES=0`: byte-exact everywhere, as before; on by
/// default).
///
/// Why: the title rewrites per-draw constants in pages of its big buffer
/// over which a shader has an unordered view but in which shaders in fact
/// never write; byte by byte they were 37,339 runs a frame (0.32 MB) for
/// 257 pages, each a copy of its own, and above `VICTORY_MERGE_RUNS` every
/// flight was settled (1-2.6 full waits for the card a frame). The joined
/// runs do not count against that cap.
///
/// The risk: the first time shaders write a block while the title also
/// changed bytes of it under the same flights (before any compare flagged
/// it: flags are noted when a flight lands), the joined run puts the CPU
/// side's bytes over the shaders' on the card once, and that write is
/// lost. From the next compare on the block is flagged and merged exactly.
pub(crate) fn merge_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_PAGES").is_none_or(|v| v != "0"))
}

/// [`Shadow::card_page_runs`]: `title` against `held` (one length), the
/// first byte `pos` bytes from the start of the copy's piece 0;
/// `flagged(g)` says whether 4 KiB block `g` (counted from there) was ever
/// flagged. The exact runs (in flagged blocks, joined over `gap`), then one
/// run per stretch of unflagged blocks that changed, first to last changed
/// byte; each as (offset, bytes) into `title`. Every differing byte is in
/// one of them.
fn page_runs(pos: usize, title: &[u8], held: &[u8], gap: usize, flagged: impl Fn(usize) -> bool) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    use crate::card_fine::BLOCK;
    let n = title.len().min(held.len());
    let (mut exact, mut joined) = (Vec::new(), Vec::new());
    let mut s = 0;
    while s < n {
        let f = flagged((pos + s) / BLOCK);
        // The stretch: blocks on from here with the same mark.
        let mut e = (((pos + s) / BLOCK + 1) * BLOCK - pos).min(n);
        while e < n && flagged((pos + e) / BLOCK) == f {
            e = (((pos + e) / BLOCK + 1) * BLOCK - pos).min(n);
        }
        let runs = changed_runs(&title[s..e], &held[s..e]);
        if f {
            exact.extend(joined_over(runs, gap).into_iter().map(|(r, m)| (r + s, m)));
        } else if let (Some(a), Some(b)) = (runs.first(), runs.last()) {
            joined.push((a.0 + s, b.0 + b.1 - a.0));
        }
        s = e;
    }
    (exact, joined)
}

/// The most changed runs with which a 4 KiB block that a compare on the
/// card once flagged goes up byte by byte when the title rewrites it under
/// flights' shader writes ([`Shadow::merge_under_writes`]); a block with
/// more goes up as one run, first changed byte to last, as a block no
/// compare ever flagged does. And where a page's exact runs are still more
/// than its submission may send (`VICTORY_MERGE_RUNS`), its blocks with the
/// most runs go up whole too, until they are not: the flights are no longer
/// settled for it. `VICTORY_MERGE_BLOCK_RUNS` (32; `0`: every flagged block
/// byte by byte and the flights settled above the cap, as before).
///
/// Why: the flags are for good (`Card::flagged`), and what shaders write in
/// the title's big buffer moves through its ring of per-draw constants, so
/// in time most of the ring is flagged. The title then rewrites a chunk of
/// the ring for the next frame's draws while a flight that names the whole
/// buffer is in the air, and byte by byte the new constants differ from
/// the old in runs of some six bytes: 42,700 runs a frame for 0.25 MB with
/// the cap out of the way (run `s_x5_runs`: 42,000 copies a frame recorded
/// for them, preparing 10 ms longer, the landing thread 6), and with the cap
/// one full wait for the card and the landings a frame, 12.5 ms of the
/// submitting thread standing (profile of 2026-11-26). A block rewritten
/// in more places than this is the title's block again, not one a shader
/// and the title share a few bytes of.
///
/// The risk is [`merge_pages`]': a flight in the air that writes such a
/// block has its bytes between the block's first and last changed byte put
/// back to what the copy held, once. Diagnostics builds count the blocks
/// that took a run of this kind and that a flight then in the air turned
/// out to have written (`crate::copystat::buffers::flagged`): the count is
/// the measure of this rule, and of the joined runs before it.
fn block_runs() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_MERGE_BLOCK_RUNS").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(32))
}

/// `exact` (a page's exact runs, in order, as [`page_runs`] gives them; the
/// page's first byte lies `pos` bytes from the start of the copy's piece
/// 0) by 4 KiB block, a run counted with the block it starts in: each
/// block's first run in `exact` and how many it has.
fn runs_by_block(pos: usize, exact: &[(usize, usize)]) -> Vec<(usize, usize)> {
    use crate::card_fine::BLOCK;
    let mut blocks: Vec<(usize, usize)> = Vec::new();
    let mut last = usize::MAX;
    for (i, &(r, _)) in exact.iter().enumerate() {
        let block = pos.saturating_add(r) / BLOCK;
        match blocks.last_mut() {
            Some(entry) if block == last => entry.1 += 1,
            _ => blocks.push((i, 1)),
        }
        last = block;
    }
    blocks
}

/// The rule of [`block_runs`] for a page's exact runs: the runs of every
/// block with more than `most` of them become one run for the block, from
/// its first changed byte to its last; and while more than `left` exact
/// runs remain, so do those of the block with the most runs left. The
/// exact runs that stay, then the runs of the blocks taken whole, both in
/// order. Every byte of `exact` is in one of them.
fn rewritten_blocks(pos: usize, exact: Vec<(usize, usize)>, most: usize, left: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let blocks = runs_by_block(pos, &exact);
    let mut whole: Vec<bool> = blocks.iter().map(|b| b.1 > most).collect();
    let mut kept: usize = blocks.iter().zip(&whole).filter(|(_, whole)| !**whole).map(|(b, _)| b.1).sum();
    if kept <= left && !whole.contains(&true) {
        return (exact, Vec::new());
    }
    if kept > left {
        // The fullest first; among equals the one that lies first.
        let mut order: Vec<usize> = (0..blocks.len()).filter(|&i| !whole[i]).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(blocks[i].1));
        for i in order {
            if kept <= left {
                break;
            }
            whole[i] = true;
            kept -= blocks[i].1;
        }
    }
    let (mut stay, mut taken) = (Vec::with_capacity(kept), Vec::new());
    for (&(first, n), &whole) in blocks.iter().zip(&whole) {
        let Some(runs) = exact.get(first..first + n) else {
            continue;
        };
        match (whole, runs.first(), runs.last()) {
            (true, Some(a), Some(b)) => taken.push((a.0, b.0 + b.1 - a.0)),
            _ => stay.extend_from_slice(runs),
        }
    }
    (stay, taken)
}

/// `runs` (in order) with those a gap of `gap` bytes or fewer apart made
/// one run, the gap's bytes included.
fn joined_over(runs: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if gap == 0 || runs.len() < 2 {
        return runs;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at - (last.0 + last.1) <= gap => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

/// The gap, in bytes, over which the title bytes put into a copy in the
/// card's memory under flights' shader writes are joined into one run
/// (`VICTORY_MERGE_GAP`, 0 to 64; 0 by default: only the bytes the title
/// changed go up, as before).
///
/// Run 219: the title rewrites some hundred pages of its big buffer a
/// frame while flights that may write that buffer are in the air, and
/// byte by byte the new content differs from the old in 38,000 runs a
/// frame (equal bytes in between, mostly zeros), far over the 512 runs a
/// submission may send, so the flights were settled twice a frame, 50 ms
/// of the submitting thread. Joined over small gaps the same pages are a
/// few runs.
///
/// What it gives up: a gap byte is equal in title memory and on the CPU
/// side of the copy, and goes up with its run. If a flight in the air
/// wrote exactly that byte on the card (a shader store within `gap` bytes
/// between two bytes the title changed in the same frame), the store is
/// overwritten with the old value, and no compare finds it. Exact (0) is
/// the default for that reason.
fn merge_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_MERGE_GAP").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(0, |n| n.min(64)))
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

#[cfg(test)]
mod page_run_tests {
    use super::page_runs;
    use crate::card_fine::BLOCK;

    /// Every byte that differs is in a run; exact runs lie in flagged
    /// blocks and hold only differing bytes; each stretch of unflagged
    /// blocks has at most one run, from its first changed byte to its last.
    #[test]
    fn page_runs_cover_every_change_and_are_exact_where_flagged() {
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..300 {
            let n = 1 + (next() as usize) % (64 << 10);
            let pos = (next() as usize) % (4 * BLOCK);
            let held: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut title = held.clone();
            for _ in 0..(next() % 40) {
                let at = (next() as usize) % n;
                title[at] ^= 1 + (next() % 255) as u8;
            }
            let marks: Vec<bool> = (0..64).map(|_| next() % 3 == 0).collect();
            let flagged = |g: usize| marks[g % 64];
            let (exact, joined) = page_runs(pos, &title, &held, 0, flagged);
            let block = |i: usize| (pos + i) / BLOCK;
            for &(r, m) in &exact {
                for i in r..r + m {
                    assert!(flagged(block(i)));
                    assert_ne!(title[i], held[i]);
                }
            }
            for &(r, m) in &joined {
                assert!(m > 0);
                assert_ne!(title[r], held[r]);
                assert_ne!(title[r + m - 1], held[r + m - 1]);
                for i in r..r + m {
                    assert!(!flagged(block(i)));
                }
            }
            // Joined runs are one per stretch: no two in a stretch.
            for w in joined.windows(2) {
                let between = (w[0].0 + w[0].1..w[1].0).any(|i| flagged(block(i)));
                assert!(between, "two joined runs in one stretch of unflagged blocks");
            }
            for i in (0..n).filter(|&i| title[i] != held[i]) {
                let inside = |runs: &[(usize, usize)]| runs.iter().any(|&(r, m)| i >= r && i < r + m);
                assert!(inside(&exact) || inside(&joined), "byte {i} left out");
            }
        }
    }

    /// Nothing ever flagged: one run for the page, first to last change.
    #[test]
    fn a_page_never_flagged_goes_up_as_one_run() {
        let held = vec![0u8; 64 << 10];
        let mut title = held.clone();
        for at in (100..60_000).step_by(37) {
            title[at] = 1;
        }
        let (exact, joined) = page_runs(0, &title, &held, 0, |_| false);
        assert!(exact.is_empty());
        let last = (100..60_000).step_by(37).next_back().unwrap();
        assert_eq!(joined, [(100, last + 1 - 100)]);
        // All flagged: the exact runs, as before.
        let (exact, joined) = page_runs(0, &title, &held, 0, |_| true);
        assert!(joined.is_empty());
        assert_eq!(exact.len(), (100..60_000).step_by(37).count());
    }

    /// `n` runs of four bytes, 64 apart, from byte 8 of `block` on.
    fn runs_in(block: usize, n: usize) -> impl Iterator<Item = (usize, usize)> {
        (0..n).map(move |k| (block * BLOCK + 8 + k * 64, 4))
    }

    /// Whether every byte of `runs` lies in a run of `stay` or of `taken`.
    fn covered(runs: &[(usize, usize)], stay: &[(usize, usize)], taken: &[(usize, usize)]) -> bool {
        runs.iter().all(|&(r, m)| stay.contains(&(r, m)) || taken.iter().any(|&(a, n)| a <= r && r + m <= a + n))
    }

    /// A flagged block with more runs than the rule allows goes up as one
    /// run, first changed byte to last; the others stay byte-exact.
    #[test]
    fn a_block_rewritten_in_many_places_is_one_run() {
        use super::rewritten_blocks;
        // Blocks 0 to 3 with 3, 40, 10 and 5 runs.
        let exact: Vec<(usize, usize)> = runs_in(0, 3).chain(runs_in(1, 40)).chain(runs_in(2, 10)).chain(runs_in(3, 5)).collect();
        let (stay, taken) = rewritten_blocks(0, exact.clone(), 32, 512);
        assert_eq!(taken, [(BLOCK + 8, 39 * 64 + 4)]);
        assert_eq!(stay.len(), 18);
        assert!(stay.iter().all(|r| exact.contains(r)) && covered(&exact, &stay, &taken));
        // None over the line and room for all: as they were.
        let (stay, taken) = rewritten_blocks(0, exact.clone(), 64, 512);
        assert_eq!(stay, exact);
        assert!(taken.is_empty());
    }

    /// Where the submission has room for fewer exact runs than remain, the
    /// blocks with the most runs go up whole until they fit: never more
    /// exact runs than there is room for.
    #[test]
    fn the_fullest_blocks_go_whole_until_the_runs_fit() {
        use super::rewritten_blocks;
        let exact: Vec<(usize, usize)> = runs_in(0, 3).chain(runs_in(1, 40)).chain(runs_in(2, 10)).chain(runs_in(3, 5)).collect();
        // Room for 9: block 1 goes by the line, block 2 (the fullest left)
        // for room; 3 + 5 stay.
        let (stay, taken) = rewritten_blocks(0, exact.clone(), 32, 9);
        assert_eq!(stay.len(), 8);
        assert_eq!(taken, [(BLOCK + 8, 39 * 64 + 4), (2 * BLOCK + 8, 9 * 64 + 4)]);
        assert!(covered(&exact, &stay, &taken));
        // Room for 4: block 3 goes too. No room: every block.
        let (stay, taken) = rewritten_blocks(0, exact.clone(), 32, 4);
        assert_eq!((stay.len(), taken.len()), (3, 3));
        let (stay, taken) = rewritten_blocks(0, exact.clone(), 32, 0);
        assert!(stay.is_empty() && taken.len() == 4 && covered(&exact, &stay, &taken));
        for left in 0..60 {
            let (stay, taken) = rewritten_blocks(0, exact.clone(), 1000, left);
            assert!(stay.len() <= left && covered(&exact, &stay, &taken), "{left}");
        }
        assert_eq!(rewritten_blocks(0, Vec::new(), 32, 0), (Vec::new(), Vec::new()));
    }

    /// A run is its first byte's block's, and blocks are cut where the
    /// card's compare cuts them: at multiples of 4 KiB of title memory,
    /// `pos` bytes into the copy's first piece.
    #[test]
    fn runs_are_counted_with_the_block_they_start_in() {
        use super::runs_by_block;
        // The page's first byte lies 100 bytes into a block.
        let pos = 100;
        let exact = [(0, 4), (BLOCK - 104, 8), (BLOCK - 96, 4), (BLOCK, 4), (3 * BLOCK, 1)];
        // Bytes 0 and BLOCK - 104 lie in the first block (the second run
        // ends in the next); BLOCK - 96 and BLOCK in the second.
        assert_eq!(runs_by_block(pos, &exact), [(0, 2), (2, 2), (4, 1)]);
        assert!(runs_by_block(usize::MAX, &exact).len() <= exact.len());
    }
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_runs, join_runs};

    /// Runs with fewer bytes between them than the gap become one, from the
    /// first's start to the last's end; the others stay as they are.
    #[test]
    fn runs_closer_than_the_gap_are_joined() {
        let runs = vec![(0, 1), (4, 1), (8, 2), (74, 3), (100, 1), (1000, 5)];
        let joined = |gap| {
            let mut r = runs.clone();
            join_runs(&mut r, gap);
            r
        };
        // No gap: every run its own.
        assert_eq!(joined(0), runs);
        // 3 bytes lie between the first two and between the next two.
        assert_eq!(joined(3), runs);
        assert_eq!(joined(4), [(0, 10), (74, 3), (100, 1), (1000, 5)]);
        // 64 lie between 10 and 74: not under 64.
        assert_eq!(joined(64), [(0, 10), (74, 27), (1000, 5)]);
        assert_eq!(joined(65), [(0, 101), (1000, 5)]);
        assert_eq!(joined(usize::MAX), [(0, 1005)]);
        let mut none: Vec<(usize, usize)> = Vec::new();
        join_runs(&mut none, 64);
        assert!(none.is_empty());
        let mut one = vec![(7, 2)];
        join_runs(&mut one, 64);
        assert_eq!(one, [(7, 2)]);
    }

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

/// Whether a refresh remembers what it took of a page it took only in part
/// (`VICTORY_REFRESH_PARTS=0`: it does not, as before).
///
/// A refresh cuts its range around what shaders are writing and to the
/// range asked for, and a 64 KiB page only partly in the cut range was
/// copied in part and left stale (only a whole page is noted taken): every
/// later refresh copied that part again, written or not. The ranges a list
/// reads and the ranges shaders write start anywhere, so most pages at
/// their edges were such pages: the frame's 80,000 copies averaged 12 KB.
/// With the switch the bytes taken are noted with the asking ([`page_plan`])
/// and are copied again only when the write watch says the page was
/// written since.
fn refresh_parts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_PARTS").is_none_or(|v| v != "0"))
}

/// Whether a refresh of a page the title wrote copies only the kernel's
/// 4 KiB pages the write watch found written (`VICTORY_REFRESH_4K=0`: the
/// whole 64 KiB page, as before). Where the journal cannot say (the copy
/// took the page before the write before last, view memory, our own writes)
/// the whole page is copied.
fn refresh_kernel_pages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_4K").is_none_or(|v| v != "0"))
}

/// Whether a refresh of a copy in the card's memory writes and sends up
/// only the 4 KiB blocks whose title bytes differ from the copy's CPU side
/// (`VICTORY_SEND_CHANGED=0`: every byte refreshed is written and sent up,
/// as before).
///
/// The copy-back merges what shaders wrote into title memory and into the
/// CPU's side (`Shadow::card_merge`, `card_merge_relay`). That write sets
/// the pages' write-watch bits or opens their guards, so the next refresh
/// takes them for the title's, copies them into the CPU's side and sends
/// them up again (some 55-90 MB a frame), though the card holds exactly
/// those bytes. Compared first, such blocks are equal and stay as they
/// are; a block the title wrote as well differs and goes up, as before.
fn send_changed_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_CHANGED").is_none_or(|v| v != "0"))
}

/// Since the last log line: blocks a refresh of a copy in the card's
/// memory found equal and did not send up ([`send_changed_only`]), and
/// their bytes.
pub(crate) static SENT_SAME: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Byte runs inside one 64 KiB page: offsets from the page's first byte,
/// `[from, to)`, in order and apart.
type Runs = Vec<(u32, u32)>;

/// `a` without `b` (both in order and apart).
fn runs_minus(a: &[(u32, u32)], b: &[(u32, u32)]) -> Runs {
    let mut out = Runs::new();
    for &(mut lo, hi) in a {
        for &(x, y) in b {
            if y <= lo || x >= hi {
                continue;
            }
            if x > lo {
                out.push((lo, x));
            }
            lo = lo.max(y);
            if lo >= hi {
                break;
            }
        }
        if lo < hi {
            out.push((lo, hi));
        }
    }
    out
}

/// `a` and `b` together, in order and apart (runs that touch become one).
fn runs_union(a: &[(u32, u32)], b: &[(u32, u32)]) -> Runs {
    let mut all: Runs = a.iter().chain(b).copied().filter(|r| r.0 < r.1).collect();
    all.sort_unstable();
    let mut out = Runs::new();
    for (lo, hi) in all {
        match out.last_mut() {
            Some((_, e)) if *e >= lo => *e = (*e).max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// What `a` and `b` share.
fn runs_and(a: &[(u32, u32)], b: &[(u32, u32)]) -> Runs {
    let mut out = Runs::new();
    for &(lo, hi) in a {
        for &(x, y) in b {
            let (l, h) = (lo.max(x), hi.min(y));
            if l < h {
                out.push((l, h));
            }
        }
    }
    runs_union(&out, &[])
}

/// The bytes of a page (its part in the copy, `inside`) that are the
/// title's as of now if they were as of asking `x`: all of them if the page
/// was not written since (`last <= x`); those outside the kernel pages
/// `mask` if every write since lies in them (`before <= x`); else none.
fn current_since(inside: (u32, u32), x: u64, last: u64, before: u64, mask: u16) -> Runs {
    if x == 0 {
        return Runs::new();
    }
    if last <= x {
        return vec![inside];
    }
    if mask == crate::watch::FULL || before > x {
        return Runs::new();
    }
    const KP: u32 = 4096;
    let written: Runs = (0..16u32).filter(|k| mask & (1 << k) != 0).map(|k| (k * KP, (k + 1) * KP)).collect();
    runs_minus(&[inside], &runs_union(&written, &[]))
}

/// For a refresh of the bytes `seg` of a page (offsets in the page, inside
/// `inside`, its part in the copy): the runs to copy from title memory, and
/// the bytes of the page that are the title's once they are copied.
///
/// The copy took the whole page after asking `taken` (0: never), and
/// `held` (if any) the bytes it lists after asking `.0`. The journal says
/// the page was last found written at `last` (0: never), and that every
/// write found after `before` lies in the kernel pages `mask`
/// ([`crate::watch::Written`]). A byte is current if it was taken after an
/// asking since which no write touched it; the rest of `seg` is copied.
fn page_plan(
    inside: (u32, u32),
    seg: (u32, u32),
    taken: u64,
    held: Option<(u64, &[(u32, u32)])>,
    last: u64,
    before: u64,
    mask: u16,
) -> (Runs, Runs) {
    let mut current = current_since(inside, taken, last, before, mask);
    if let Some((at, runs)) = held {
        let still = runs_and(runs, &current_since(inside, at, last, before, mask));
        current = runs_union(&current, &still);
    }
    let need = runs_minus(&[seg], &current);
    let after = runs_and(&runs_union(&current, &need), &[inside]);
    (need, after)
}

#[cfg(test)]
mod plan_tests {
    use super::{page_plan, runs_and, runs_minus, runs_union};
    use crate::watch::FULL;

    const K: u32 = 4096;
    const P: u32 = 16 * K;

    #[test]
    fn runs_take_away_join_and_meet() {
        assert_eq!(runs_minus(&[(0, 100)], &[(10, 20), (30, 40)]), [(0, 10), (20, 30), (40, 100)]);
        assert_eq!(runs_minus(&[(0, 100)], &[(0, 100)]), []);
        assert_eq!(runs_minus(&[(10, 20)], &[(0, 5), (25, 30)]), [(10, 20)]);
        assert_eq!(runs_minus(&[(0, 10), (20, 30)], &[(5, 25)]), [(0, 5), (25, 30)]);
        assert_eq!(runs_union(&[(0, 10)], &[(10, 20), (30, 40)]), [(0, 20), (30, 40)]);
        assert_eq!(runs_union(&[(5, 8), (0, 6)], &[]), [(0, 8)]);
        assert_eq!(runs_and(&[(0, 10), (20, 30)], &[(5, 25)]), [(5, 10), (20, 25)]);
    }

    /// A page never taken is copied as asked, and is whole only if asked
    /// whole.
    #[test]
    fn a_page_never_taken_is_copied_as_asked() {
        assert_eq!(page_plan((0, P), (0, P), 0, None, 0, 0, FULL), (vec![(0, P)], vec![(0, P)]));
        assert_eq!(page_plan((0, P), (100, 300), 0, None, 7, 0, FULL), (vec![(100, 300)], vec![(100, 300)]));
        // The copy's part of the page only (a buffer starting inside it).
        assert_eq!(page_plan((50, P), (50, P), 0, None, 0, 0, FULL), (vec![(50, P)], vec![(50, P)]));
    }

    /// What a part took is not copied again while the page is not written;
    /// the rest of the page is, once.
    #[test]
    fn a_part_taken_is_not_copied_again_until_written() {
        let held: &[(u32, u32)] = &[(100, 300)];
        // Not written since asking 5, which took the part: nothing to copy.
        assert_eq!(page_plan((0, P), (100, 300), 0, Some((5, held)), 4, 0, FULL), (vec![], vec![(100, 300)]));
        // Asked for more of the page: only what the part left out.
        assert_eq!(page_plan((0, P), (0, 1000), 0, Some((5, held)), 4, 0, FULL), (vec![(0, 100), (300, 1000)], vec![(0, 1000)]));
        // The rest of the page too: whole now.
        assert_eq!(page_plan((0, P), (0, P), 0, Some((5, held)), 0, 0, FULL), (vec![(0, 100), (300, P)], vec![(0, P)]));
        // Only what follows the part: bytes before it are still not taken.
        assert_eq!(page_plan((0, P), (300, P), 0, Some((5, held)), 0, 0, FULL), (vec![(300, P)], vec![(100, P)]));
        // Written after the part was taken (anywhere): all of it again.
        assert_eq!(page_plan((0, P), (100, 300), 0, Some((5, held)), 6, 5, FULL), (vec![(100, 300)], vec![(100, 300)]));
    }

    /// A page written in one kernel page since it was taken: only that one.
    #[test]
    fn only_the_kernel_pages_written_are_copied() {
        let mask = 1 << 3;
        // Taken at 5; written at 6 in kernel page 3, the write before at 4.
        assert_eq!(page_plan((0, P), (0, P), 5, None, 6, 4, mask), (vec![(3 * K, 4 * K)], vec![(0, P)]));
        // Taken before the write before last: it cannot be said where.
        assert_eq!(page_plan((0, P), (0, P), 3, None, 6, 4, mask), (vec![(0, P)], vec![(0, P)]));
        // A part of the page around the written kernel page.
        // (Its first bytes stay to be taken.)
        assert_eq!(
            page_plan((0, P), (3 * K + 10, 5 * K), 5, None, 6, 4, mask),
            (vec![(3 * K + 10, 4 * K)], vec![(0, 3 * K), (3 * K + 10, P)])
        );
        // Kernel pages 1 and 2 written, the part held covers kernel page 2 and is from after the write before.
        let held: &[(u32, u32)] = &[(2 * K, 3 * K)];
        assert_eq!(page_plan((0, P), (0, 4 * K), 0, Some((5, held)), 6, 4, 0b110), (vec![(0, 4 * K)], vec![(0, 4 * K)]));
    }
}

/// [`stale_pages_in`] with what the journal knows of where in each page:
/// for each page of `[a, e)` the copy must take, the runs of it (offsets in
/// the page, inside `[a, e)` and the copy `bounds`) not known to be the
/// title's: with `kernel_pages`, only the kernel pages written since the
/// copy took the page, where the journal can say ([`current_since`]).
fn stale_runs_in(
    base: usize,
    copied: &[u64],
    bounds: (usize, usize),
    a: usize,
    e: usize,
    written: &[(usize, crate::watch::Written)],
    kernel_pages: bool,
) -> Vec<(usize, Runs)> {
    let mut out = Vec::new();
    let mut w = written.iter().peekable();
    for page in (a / PAGE..e.div_ceil(PAGE)).filter(|&page| page >= base) {
        let Some(&taken) = copied.get(page - base) else {
            break;
        };
        while w.peek().is_some_and(|&&(p, _)| p < page) {
            w.next();
        }
        let entry = w.peek().filter(|&&&(p, _)| p == page).map(|&&(_, x)| x);
        let (last, before, mask) = entry.map_or((0, 0, crate::watch::FULL), |x| {
            (x.at, x.before, if kernel_pages { x.mask } else { crate::watch::FULL })
        });
        if taken != 0 && last <= taken {
            continue;
        }
        let at = page * PAGE;
        let inside = ((at.max(bounds.0) - at) as u32, ((at + PAGE).min(bounds.1) - at) as u32);
        let seg = ((at.max(a) - at) as u32, ((at + PAGE).min(e) - at) as u32);
        if seg.0 >= seg.1 {
            continue;
        }
        let need = runs_minus(&[seg], &current_since(inside, taken, last, before, mask));
        if !need.is_empty() {
            out.push((page, need));
        }
    }
    out
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }

    /// The runs are the stale pages' bytes in the range, less the kernel
    /// pages the journal says were not written since the copy took them.
    #[test]
    fn stale_runs_are_the_written_kernel_pages_of_stale_pages() {
        use crate::watch::Written;
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        let bounds = (at(0x100), at(0x104));
        let p = PAGE as u32;
        let w = |at, before, mask| Written { at, before, mask };
        // 0x101 never taken: all of it; 0x102 written at 6 in kernel page 2,
        // the write before at 4 (before the copy took it at 5): that page;
        // 0x103 written at 7, the write before at 6 (after 5): it cannot be
        // said where, all of it.
        let written = [(0x102, w(6, 4, 1 << 2)), (0x103, w(7, 6, 1))];
        let pages = |kernel| stale_runs_in(base, &copied, bounds, at(0x100), at(0x104), &written, kernel);
        assert_eq!(pages(true), [(0x101, vec![(0, p)]), (0x102, vec![(2 * 4096, 3 * 4096)]), (0x103, vec![(0, p)])]);
        assert_eq!(pages(false), [(0x101, vec![(0, p)]), (0x102, vec![(0, p)]), (0x103, vec![(0, p)])]);
        // As `stale_pages_in` would say which, always.
        let same: Vec<usize> = pages(true).into_iter().map(|(p, _)| p).collect();
        let pairs: Vec<(usize, u64)> = written.iter().map(|&(p, x)| (p, x.at)).collect();
        assert_eq!(same, stale_pages_in(base, &copied, at(0x100), at(0x104), &pairs));
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.iter().filter(|&&at| at != 0).count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        let gone: Vec<Arc<Shadow>> = idle.into_iter().filter_map(|key| map.remove(&key)).collect();
        if !gone.is_empty() {
            COPIES_GONE.fetch_add(1, Ordering::Release);
        }
        gone
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

/// [`Shadow::side_of`]: the locks held.
enum SideGuard<'a> {
    /// A copy in the card's memory: its pieces' locks, in order (held,
    /// never read).
    Pieces { _held: Vec<std::sync::MutexGuard<'a, ()>> },
    /// Any other copy: its snapshot, locked.
    Snapshot(std::sync::MutexGuard<'a, Option<Vec<u8>>>),
}

impl SideGuard<'_> {
    /// The snapshot of a copy that has one.
    fn snapshot(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            SideGuard::Snapshot(before) => before.as_mut(),
            SideGuard::Pieces { .. } => None,
        }
    }
}

impl Shadow {
    /// The lock of the copy's reference (`before`: its snapshot, or what
    /// stands for the CPU's side of a copy in the card's memory), taken; a
    /// wait of 0.2 ms or more for it is an event of `crate::timeline`.
    fn side(&self) -> std::sync::MutexGuard<'_, Option<Vec<u8>>> {
        crate::timeline::waited("L:side", || self.before.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// What holds offsets `[from, to)` of the copy's CPU side still for a
    /// merge (rule R2: no byte goes into it inside a range shaders write
    /// but through the merge of a landing or [`Shadow::merge_under_writes`],
    /// and the two never see each other half done): for a copy in the
    /// card's memory the locks of the 64 KiB pieces that hold the offsets,
    /// taken in the order of the pieces; for any other copy its snapshot's
    /// lock ([`Shadow::side`]).
    ///
    /// One lock for the whole copy had every merge of a landing into the
    /// heap's copy (174 MB) queue on it: the helper threads that merged
    /// beside the landing's thread waited 15.4 ms a frame for it. Pieces of
    /// one copy are its own bytes and title memory's; what a merge or a
    /// putting in of title bytes reads and writes lies in the pieces it
    /// holds. A wait of 0.2 ms or more is `L:side` of the timeline.
    fn side_of(&self, from: usize, to: usize) -> SideGuard<'_> {
        if self.card.is_none() {
            return SideGuard::Snapshot(self.side());
        }
        let locks = self.side_pieces.get_or_init(|| {
            let n = pieces_of(self.start as usize, 0, (self.len as usize).max(1)).end;
            (0..n).map(|_| Mutex::new(())).collect()
        });
        let pieces = if from < to { self.pieces(from, to) } else { 0..0 };
        SideGuard::Pieces {
            _held: crate::timeline::waited("L:side", || {
                pieces.filter_map(|p| locks.get(p)).map(|lock| lock.lock().unwrap_or_else(|e| e.into_inner())).collect()
            }),
        }
    }

    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        self.refresh_as(list, address, len, ASKED_BY_DRAW);
    }

    /// [`Shadow::refresh`] for `who` (one of the `ASKED_BY_` bits): what a
    /// diagnostics build counts the refresh under. Nothing else differs.
    pub(crate) fn refresh_as(self: &Arc<Self>, list: &HostList, address: u64, len: u64, who: u8) {
        let len = self.asked(address, len, who);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// The bytes a refresh of `len` bytes at `address` asked by `who` brings
    /// up to date: `len` cut at the copy's end. A diagnostics build counts
    /// the asking under `who`.
    pub(crate) fn asked(&self, address: u64, len: u64, who: u8) -> u64 {
        let len = len.min(self.start + self.len - address);
        #[cfg(feature = "diagnostics")]
        {
            // (Loaded first: recording threads ask for the same copies, and
            // a store each time would pass the copy's line between them.)
            if self.asked_by.load(Ordering::Relaxed) & who != who {
                self.asked_by.fetch_or(who, Ordering::Relaxed);
            }
            crate::copystat::buffers::asked(who, len);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = who;
        len
    }

    /// Diagnostics: the askers the copy had ([`Shadow::refresh_as`]), taken.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn asked_taken(&self) -> u8 {
        self.asked_by.swap(0, Ordering::Relaxed)
    }

    /// Diagnostics: at the landing of flight `flight`, its compare flagged
    /// the blocks `mask` of the copy's piece `piece` (as
    /// [`Shadow::card_flagged`] counts pieces): told to the census by the
    /// title address of the piece.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn census_flagged(&self, flight: u64, piece: u32, mask: u32) {
        if mask == 0 {
            return;
        }
        let at = (self.start as usize / PIECE * PIECE).saturating_add((piece as usize).saturating_mul(PIECE));
        crate::copystat::buffers::flagged(flight, at, mask);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            partial: Mutex::default(),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: Mutex::default(),
            side_pieces: std::sync::OnceLock::new(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            place: None,
            #[cfg(feature = "diagnostics")]
            asked_by: AtomicU8::new(0),
            events_seen: Mutex::default(),
            landed_check: Mutex::default(),
            #[cfg(feature = "diagnostics")]
            refreshed: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        list.hoist_wrote(address as usize, len.min(self.left(address)) as usize);
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.side();
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.side();
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            // Pieces that lie together are one stretch: at 970 pieces and
            // ~300 ms a frame (75 MB), the calls are most of the cost.
            let differing = coalesce_pieces(differing, PIECE_GAP);
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                let at = at + first;
                crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                crate::watch::writing_as(start + at, n, crate::writes::Cause::Back);
                // Only the GPU's bytes (upstream #2009): the rest of the
                // stretch is the title's as read, and a title write landing
                // there since stays. Runs a few bytes apart go in one write
                // (`back_run_gap`).
                join_runs(&mut runs, back_run_gap());
                for &(off, len) in &runs {
                    crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                    written += len;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.side();
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = merge_race_on().then(|| title.clone());
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "pc relay");
        // Beside the guards where one view of ours shows the stretch
        // (`crate::beside`), the blocks counted written once the bytes
        // are there; else, and for what a missing page left, the guards
        // are opened first, as ever.
        let place = if crate::beside::mode() != 0 && fast_merge() { crate::beside::find(start + at, n) } else { None };
        let done = place.as_ref().map_or(0, |place| place.write(start + at, &title[..]));
        if done > 0 {
            crate::watch::written_beside(start + at, done, crate::writes::Cause::Back);
        }
        if done < n {
            crate::watch::writing_as(start + at + done, n - done, crate::writes::Cause::Back);
            if fast_merge() {
                // The processor's copy, the kernel's for what faults.
                write_title(start + at + done, &title[done..]);
            } else {
                crate::host::write_mapped((start + at + done) as *mut u8, title[done..].as_ptr(), n - done);
            }
        }
        n
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.side().take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => !stale_pages_in(base, &copied, a, e, &written).is_empty(),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        // Title memory is read into one buffer a thread, kept between
        // calls: a zeroed 64 KiB made for every call that looks at a page
        // was most of such a call's own time (Marvisak/VIctory pull request
        // 3246, lhnizdo, does the same for its marked pages' copy-back).
        // Every byte of it that is used was read from title memory first.
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        TITLE.with(|title| self.merge_under_writes_into(address, len, runs_left, &mut title.borrow_mut()))
    }

    /// [`Shadow::merge_under_writes`]; `title` is room to read title
    /// memory into.
    fn merge_under_writes_into(&self, address: usize, len: usize, runs_left: &mut usize, title: &mut Vec<u8>) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            #[cfg(feature = "diagnostics")]
            crate::copystat::buffers::gave_up(crate::copystat::buffers::UNWATCHED, self.start, self.len, from, 0, *runs_left);
            return true;
        }
        let base = start / PAGE;
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes_fine(a, e, true) else {
                #[cfg(feature = "diagnostics")]
                crate::copystat::buffers::gave_up(crate::copystat::buffers::NO_JOURNAL, self.start, self.len, a, 0, *runs_left);
                return true;
            };
            let stale: Vec<(usize, Runs)> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_runs_in(base, &copied, (start, end), a, e, &written, refresh_kernel_pages())
            };
            if !stale.is_empty() && title.len() < PAGE {
                title.resize(PAGE, 0);
            }
            for (page, runs) in stale {
                let mut taken = true;
                for (rlo, rhi) in runs {
                    let (lo, hi) = (page * PAGE + rlo as usize, page * PAGE + rhi as usize);
                    if lo >= hi {
                        continue;
                    }
                    let n = hi - lo;
                    let title = &mut title[..n];
                    // The lock of the copy's reference (the snapshot, or the
                    // CPU's side of a copy in the card's memory), from before
                    // title memory is read until the title's bytes are in: with
                    // the copy-back on its own thread ([`relay`]) that thread
                    // puts what the card wrote into title memory and into the
                    // reference under the same lock, and a page read before
                    // such a merge and compared after it would take the card's
                    // bytes for the title's. (For a copy in the card's memory
                    // the lock of the piece that holds the page.)
                    let at = lo - start;
                    let mut before = self.side_of(at, at + n);
                    // A page that is not there is left as it is, and looked at
                    // again next time.
                    if !crate::host::read_into(lo as *const u8, title) {
                        taken = false;
                        continue;
                    }
                    MERGED[0].fetch_add(1, Ordering::Relaxed);
                    if self.card.is_some() {
                        // SAFETY: inside the mapping of the copy's CPU side,
                        // which only the submitting thread writes.
                        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                        // With `merge_pages`: byte-exact only where a compare
                        // ever flagged the block; elsewhere one run a stretch,
                        // outside the cap.
                        let (runs, joined, rewritten) = if merge_pages() {
                            self.card_page_runs(at, title, held, *runs_left)
                        } else {
                            (joined_over(changed_runs(title, held), merge_gap()), Vec::new(), Vec::new())
                        };
                        if runs.len() > *runs_left {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::buffers::gave_up(crate::copystat::buffers::RUN_CAP, self.start, self.len, lo, runs.len(), *runs_left);
                            return true;
                        }
                        *runs_left -= runs.len();
                        for (r, m) in runs {
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            self.card_changed(at + r, at + r + m);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::buffers::title_wrote(lo + r, m);
                        }
                        for (r, m) in joined {
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            self.card_changed(at + r, at + r + m);
                            MERGED[3].fetch_add(1, Ordering::Relaxed);
                            MERGED[4].fetch_add(m as u64, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            {
                                crate::copystat::buffers::title_wrote(lo + r, m);
                                crate::copystat::buffers::joined(lo + r, m, PREPARED.load(Ordering::Relaxed));
                            }
                        }
                        for (r, m) in rewritten {
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            self.card_changed(at + r, at + r + m);
                            MERGED[5].fetch_add(1, Ordering::Relaxed);
                            MERGED[6].fetch_add(m as u64, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            {
                                crate::copystat::buffers::title_wrote(lo + r, m);
                                crate::copystat::buffers::joined(lo + r, m, PREPARED.load(Ordering::Relaxed));
                            }
                        }
                    } else {
                        let Some(snapshot) = before.snapshot() else {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::buffers::gave_up(crate::copystat::buffers::NO_SNAPSHOT, self.start, self.len, lo, 0, *runs_left);
                            return true;
                        };
                        // SAFETY: the snapshot's allocation holds the copy's
                        // bytes at the copy's offsets (`writing`), and these
                        // are in a declared range.
                        let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                        for (r, m) in changed_runs(title, held) {
                            // SAFETY: inside the copy's mapping. The card may be
                            // writing the copy beside these bytes; they are
                            // written here and nothing around them.
                            unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                    }
                    drop(before);
                }
                // Taken, if the range holds all of the page that is in the
                // copy (the bytes not looked at are the title's: unwritten
                // since the copy took them); else the rest of it is the
                // refresh's, which notes only whole pages, and the page is
                // looked at again.
                if taken && (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        // By event (`crate::watch::events_mode`), for a range in view
        // memory of a copy that is asked about at all (one whose memory
        // nothing watches is copied whole every time, whatever the
        // counts): whether anything happened in the range last brought up
        // to date since it was, if this one lies inside it.
        use crate::watch::{Checked, Look, Seen};
        let by_event = crate::watch::events_mode();
        let watched = self.watched.load(Ordering::Relaxed) != NOT_WATCHED;
        let looked = (watched && crate::inplace::view_at(from).is_some() && crate::inplace::view_at(to - 1).is_some())
            .then(|| {
                let seen = *self.events_seen.lock().unwrap_or_else(|e| e.into_inner());
                let inside = seen.holds(from, to);
                // (The counts of this range, from before its asking.)
                (seen, inside, inside && seen.quiet(), Seen::before(from, to))
            });
        if by_event == 1 && matches!(looked, Some((_, true, true, _))) {
            let left_out = Checked { inside: true, quiet: true, quiet_locked: true, uncounted: true, copied: None };
            crate::watch::event_checked(left_out, (self.start, self.len), from, to, None);
            #[cfg(feature = "diagnostics")]
            self.note_refreshed(from, to, 1, 0, 0);
            return 0;
        }
        if watched {
            // The asking looks at the counts again with its locks held:
            // what it finds is then what they stand for.
            let mut look = Look { seen: looked.map(|l| l.0), ..Look::default() };
            // (The asking's number, for the diagnostics' record.)
            let mut asking = 0u64;
            let copied = if refresh_parts() || refresh_kernel_pages() {
                crate::watch::writes_fine_look(from, to, true, &mut look).map(|(now, written)| {
                    asking = now;
                    self.copy_stale_fine(from, to, now, &written)
                })
            } else {
                crate::watch::writes_fine_look(from, to, true, &mut look)
                    .map(|(now, written)| (now, written.into_iter().map(|(p, w)| (p, w.at)).collect::<Vec<_>>()))
                    .map(|(now, written)| {
                        asking = now;
                        self.copy_stale(from, to, now, &written)
                    })
            };
            #[cfg(not(feature = "diagnostics"))]
            let _ = asking;
            if let Some(copied) = copied {
                #[cfg(feature = "diagnostics")]
                self.note_refreshed(from, to, 2, asking, copied);
                if let Some((_, inside, quiet, before)) = looked {
                    let checked = Checked {
                        inside,
                        quiet,
                        quiet_locked: inside && look.quiet,
                        uncounted: inside && look.uncounted,
                        copied: Some(copied),
                    };
                    crate::watch::event_checked(checked, (self.start, self.len), from, to, Some(&look));
                    // The counts from before the asking: what the asking
                    // itself noted has this range asked once more.
                    let mut seen = self.events_seen.lock().unwrap_or_else(|e| e.into_inner());
                    // A range inside the one known keeps the wider one
                    // while nothing happened in it.
                    if !(inside && quiet) {
                        *seen = before.after(&look);
                    }
                }
                self.watched.store(1, Ordering::Relaxed);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        self.note_refreshed(from, to, 3, 0, to - from);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. One that differs once is a write
    /// of the title's that arrived between the refresh and the look.
    ///
    /// One that differs on three refreshes in a row was said to have
    /// changed "with nothing to tell of it". That is not so for a block a
    /// read of the title's is filling: DirectStorage writes a destination's
    /// pages through a view of the runtime's own for as long as the read
    /// and its decoding take, and the guards are told when it is done
    /// (`crate::watch::title_memory_written`); until then every host copy
    /// over the pages holds the bytes from before, and a copy is brought up
    /// to date several times in one submission. The open world's entry had
    /// 1,024 such blocks in 130 s (run s_we_y2_check), all of them in
    /// 4 MiB windows, at the rate the reads came in. So the looks are told
    /// apart ([`REFRESH_CHECKED`], [`refresh_check_line`]): those under a
    /// read the title has submitted and that is not done
    /// (`crate::writes::in_flight_over`) count for nothing; a block that
    /// differs three times in a row with no such read is said, with what
    /// the guards hold of it; when it matches again it is counted by
    /// whether a write behind the guards was told there meanwhile; and one
    /// that has differed for two seconds with nothing told is said once
    /// more: that one is a copy that keeps old bytes.
    fn refresh_check(&self, from: usize, to: usize) {
        if !refresh_checked() {
            return;
        }
        const SAMPLE: usize = 256;
        static SAID: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                if copy == &title[..n] {
                    if let Some(was) = map.remove(&at)
                        && was.times >= 3
                    {
                        REFRESH_CHECKED[if was.told { 3 } else { 4 }].fetch_add(1, Ordering::Relaxed);
                    }
                } else if crate::writes::in_flight_over(at as u64, (at + n) as u64) {
                    REFRESH_CHECKED[0].fetch_add(1, Ordering::Relaxed);
                } else {
                    REFRESH_CHECKED[1].fetch_add(1, Ordering::Relaxed);
                    let d = map.entry(at).or_insert_with(|| Differing {
                        times: 0,
                        since: std::time::Instant::now(),
                        told: false,
                        long: false,
                    });
                    d.times += 1;
                    let named = |what: &str, said: u64, d: &Differing| {
                        tracing::warn!(
                            said,
                            copy = self.id,
                            copy_start = format_args!("{start:#x}"),
                            copy_bytes = self.len,
                            address = format_args!("{at:#x}"),
                            refreshed = format_args!("{from:#x}..{to:#x}"),
                            on_the_card = self.card.is_some(),
                            window = self.window,
                            refreshes = d.times,
                            ms = d.since.elapsed().as_millis() as u64,
                            guards = crate::watch::guard_state(at),
                            copy_zeros = copy.iter().all(|&b| b == 0),
                            title_zeros = title[..n].iter().all(|&b| b == 0),
                            "{what} (VICTORY_REFRESH_CHECK=1)"
                        );
                    };
                    if d.times == 3 {
                        REFRESH_CHECKED[2].fetch_add(1, Ordering::Relaxed);
                        let said = SAID[0].fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            named(
                                "a host copy of title memory differs from it after its refresh, three times in a row, with no read of the title's known in flight there",
                                said,
                                d,
                            );
                        }
                    }
                    if d.times >= 3 && !d.long && !d.told && d.since.elapsed() >= std::time::Duration::from_secs(2) {
                        d.long = true;
                        REFRESH_CHECKED[5].fetch_add(1, Ordering::Relaxed);
                        let said = SAID[1].fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            named(
                                "a host copy of title memory has differed from it for two seconds of refreshes, and nothing was told of a write there: the copy keeps old bytes",
                                said,
                                d,
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Diagnostics: notes a refresh of `[from, to)` ([`Refreshed`]).
    #[cfg(feature = "diagnostics")]
    fn note_refreshed(&self, from: usize, to: usize, how: u8, asking: u64, bytes: usize) {
        let frame = crate::frame::current();
        let mut r = self.refreshed.lock().unwrap_or_else(|e| e.into_inner());
        let now = crate::host::now_us();
        let asked = if how == 2 { (frame, from, to, asking, bytes) } else { r.asked };
        let us = (now, if how == 2 { now } else { r.us.1 });
        *r = Refreshed { frame, from, to, how, asking, bytes, asked, us };
    }

    /// Diagnostics, for the report of a removed device: what the copy holds
    /// of the 64 KiB page at title address `at` and of how it was brought
    /// up to date: the page's stamp (the asking after which it was last
    /// copied) and the part taken of it, whether the memory is watched,
    /// what the copy remembers by events, the ranges shaders may write over
    /// the page, and the last refreshes. Locks are only taken if they can
    /// be had at once.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn lag_report(&self, at: usize) -> String {
        use std::fmt::Write;
        let page = at / PAGE;
        let i = page.wrapping_sub(self.start as usize / PAGE);
        let mut out = format!("copy {} {:#x}+{:#x}", self.id, self.start, self.len);
        let _ = write!(
            out,
            " (watched {}, card {}, window {}, frame now {})",
            self.watched.load(Ordering::Relaxed),
            self.card.is_some(),
            self.window,
            crate::frame::current()
        );
        match self.copied.try_lock() {
            Ok(c) => {
                let _ = write!(out, ": page stamp {:?}", c.get(i));
            }
            Err(_) => out.push_str(": stamps locked"),
        }
        if let Ok(p) = self.partial.try_lock()
            && let Some((stamp, runs)) = p.get(&i)
        {
            let _ = write!(out, ", taken in part after {stamp}: {runs:?}");
        }
        match self.events_seen.try_lock() {
            Ok(seen) => {
                let _ = write!(out, "; {}", seen.describe());
            }
            Err(_) => out.push_str("; seen locked"),
        }
        if let Ok(p) = self.pending.try_lock() {
            let over: Vec<String> =
                p.iter().filter(|&&(a, e)| a < (page + 1) * PAGE && e > page * PAGE).map(|(a, e)| format!("{a:#x}..{e:#x}")).collect();
            if !over.is_empty() {
                let _ = write!(out, "; shaders may write {}", over.join(" "));
            }
        }
        if let Ok(r) = self.refreshed.try_lock() {
            let how = ["none", "left out by events", "asked", "copied whole"][usize::from(r.how.min(3))];
            let now = crate::host::now_us();
            let ago = |us: u64| now.saturating_sub(us) as f64 / 1000.0;
            let _ = write!(
                out,
                "; last refresh frame {} {:#x}..{:#x} {how} (asking {}, {} bytes) {:.1} ms ago; last that asked frame {} {:#x}..{:#x} asking {} {} bytes {:.1} ms ago",
                r.frame,
                r.from,
                r.to,
                r.asking,
                r.bytes,
                ago(r.us.0),
                r.asked.0,
                r.asked.1,
                r.asked.2,
                r.asked.3,
                r.asked.4,
                ago(r.us.1)
            );
        }
        out
    }

    /// Diagnostics, for the report of a removed device: `len` bytes of the
    /// copy from `address` on against title memory as it is now. The 4-byte
    /// words (from `address` on) in which they differ, whether a range
    /// shaders may be writing holds part of it, and the first `first` such
    /// words as (address, the title's value, the copy's value). For a copy
    /// in the card's memory the copy's side is the CPU's, which is what
    /// went up to the card. `None` for title memory in place (no copy) and
    /// a range outside the buffer. Parts of title memory that cannot be
    /// read are passed over.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn against_title(&self, address: usize, len: usize, first: usize) -> Option<(usize, bool, Vec<(usize, u32, u32)>)> {
        if self.place.is_some() {
            return None;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return None;
        }
        let start = self.start as usize;
        let pending = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|&(a, e)| a < start + to && e > start + from);
        let (mut words, mut said) = (0, Vec::new());
        let mut title = vec![0u8; PAGE];
        let mut at = from;
        while at < to {
            // Up to the end of the title page (a page that cannot be read
            // is passed over whole).
            let (lo, hi) = (at, (((start + at) / PAGE + 1) * PAGE - start).min(to));
            let n = hi - lo;
            let title = &mut title[..n];
            if crate::host::read_into((start + lo) as *const u8, title) {
                // SAFETY: `[lo, hi)` lies inside `[from, to)`, which
                // `inside` cut to the copy, so inside the mapping of its bytes.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(lo).cast_const(), n) };
                for (w, (t, c)) in title.chunks(4).zip(copy.chunks(4)).enumerate() {
                    if t != c {
                        words += 1;
                        if said.len() < first {
                            let word = |b: &[u8]| {
                                let mut w = [0u8; 4];
                                w[..b.len()].copy_from_slice(b);
                                u32::from_le_bytes(w)
                            };
                            said.push((start + lo + 4 * w, word(t), word(c)));
                        }
                    }
                }
            }
            at = hi;
        }
        Some((words, pending, said))
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// [`Shadow::copy_stale`] that takes only the bytes of each page that
    /// are not known current: with [`refresh_kernel_pages`], of a page
    /// written after it was taken only the kernel's 4 KiB pages the write
    /// watch found written since; with [`refresh_parts`], of a page taken in
    /// part before, only what that part left out ([`page_plan`]).
    fn copy_stale_fine(&self, from: usize, to: usize, now: u64, written: &[(usize, crate::watch::Written)]) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let (parts, kernel_pages) = (refresh_parts(), refresh_kernel_pages());
        let landed_check = land_taken() == 2;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut partial = self.partial.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        // What to copy, as title addresses, joined where they touch.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for page in from / PAGE..=(to - 1) / PAGE {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    Some(w)
                }
                _ => None,
            };
            let (last, before, mask) = match write {
                Some(w) => (w.at, w.before, if kernel_pages { w.mask } else { crate::watch::FULL }),
                None => (0, 0, crate::watch::FULL),
            };
            let i = page - base;
            let taken = copied[i];
            if taken != 0 && last <= taken {
                continue;
            }
            let at = page * PAGE;
            let inside = ((at.max(start) - at) as u32, ((at + PAGE).min(end) - at) as u32);
            let seg = ((at.max(from) - at) as u32, ((at + PAGE).min(to) - at) as u32);
            if landed_check {
                self.landed_checked(page, last, write.map_or(0, |w| w.mask), seg);
            }
            let held = if parts { partial.get(&i).map(|(p, r)| (*p, r.as_slice())) } else { None };
            let (need, after) = page_plan(inside, seg, taken, held, last, before, mask);
            for (lo, hi) in need {
                let (lo, hi) = (at + lo as usize, at + hi as usize);
                match runs.last_mut() {
                    Some((_, e)) if *e == lo => *e = hi,
                    _ => runs.push((lo, hi)),
                }
            }
            // The copy is taken after every asking in the journal, the one
            // that found this page's last write among them: a stamp below
            // that write would have the page copied again at every refresh
            // while the submission's asking stands.
            let stamp = now.max(last);
            if after.len() == 1 && after[0] == inside {
                copied[i] = stamp;
                partial.remove(&i);
            } else if parts && !after.is_empty() {
                partial.insert(i, (stamp, after));
            }
        }
        // Each run is inside `[from, to)`: next to it shaders may be
        // writing the copy.
        runs.into_iter().map(|(a, e)| self.copy(a, e)).sum()
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if self.card.is_some() && send_changed_only() {
            self.copy_compared(from, to);
            return to - from;
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// [`Shadow::copy`] for a copy in the card's memory with
    /// [`send_changed_only`]: title memory is read into scratch memory and
    /// compared with the CPU's side a 4 KiB block (of title memory) at a
    /// time; only the blocks that differ are written there and sent up.
    fn copy_compared(&self, from: usize, to: usize) {
        const STEP: usize = 64 << 10;
        const SMALL: usize = 4 << 10;
        thread_local! {
            static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.resize(STEP, 0);
            let mut at = from;
            while at < to {
                let n = (STEP - at % STEP).min(to - at);
                let read = &mut scratch[..n];
                // Not zeroed first (0.5 ms a frame of the submitting
                // thread in the profile of 2026-11-26): the copy writes
                // every byte of it, the title's or, for a page that is not
                // there, zero. Only the diagnostics mode that measures by
                // leaving the copy out writes nothing.
                #[cfg(feature = "diagnostics")]
                if crate::copystat::mode() == 2 {
                    read.fill(0);
                }
                crate::host::copy_mapped(read.as_mut_ptr(), at as *const u8, n, false);
                // SAFETY: inside the mapping (`[at, at + n)` is within the
                // buffer); only the submitting thread writes these bytes.
                let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), n) };
                let mut k = 0;
                while k < n {
                    let m = (SMALL - (at + k) % SMALL).min(n - k);
                    let equal = read[k..k + m] == held[k..k + m];
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::buffers::refreshed_block(at + k, m, equal);
                    if equal {
                        SENT_SAME[0].fetch_add(1, Ordering::Relaxed);
                        SENT_SAME[1].fetch_add(m as u64, Ordering::Relaxed);
                    } else {
                        held[k..k + m].copy_from_slice(&read[k..k + m]);
                        let off = at + k - start;
                        self.changed(off, off + m);
                        self.card_changed(off, off + m);
                    }
                    k += m;
                }
                at += n;
            }
        });
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let first = note_changed(&mut card.dirty.lock().unwrap_or_else(|e| e.into_inner()), from, to);
        // The copy's first range since its ranges were last taken: it has
        // something to send (outside the ranges' lock: `card_dirty` takes
        // the list, then each copy's ranges).
        if first && let Some(me) = card.me.get() {
            DIRTY.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
        }
    }

    /// For [`card_dirty`]: the offsets of the CPU side that changed and
    /// have not gone up, taken, in order and joined where they touch (none:
    /// nothing changed); noted as sent with send `sent_as`, and the notes
    /// of sends up to `fetched` dropped (every fetch still to be merged
    /// came after them).
    fn card_take_dirty(&self, sent_as: u64, fetched: u64) -> Option<Vec<(usize, usize)>> {
        let card = self.card.as_ref()?;
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let ranges = std::mem::take(&mut *dirty);
        if relay() {
            let mut sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            sent.retain(|s| s.2 > fetched);
            sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
        }
        (!ranges.is_empty()).then(|| joined(ranges))
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    /// The bytes from the copy's start that the card's hashes cover, for a
    /// copy of `size` bytes in the card's memory: the copy's whole buffer;
    /// by default the title buffer's own bytes, rounded up to the hash's
    /// 16-byte step (`VICTORY_CARD_HASH_LEN=0`: the whole buffer). A copy's buffer is the title buffer
    /// rounded up to 64 KiB, so a title buffer of 288 bytes was hashed over
    /// 4,096 steps where 18 hold everything that can go back to the title
    /// (what is fetched is cut to the title buffer, [`Shadow::card_piece`]).
    /// With every title buffer on the card (`VICTORY_CARD_BUFFERS_MIN_KB=64`)
    /// the source's send-and-hash lists took 19.7 ms of card time a frame
    /// where they had taken 10.9 (fxceless/victory-fork c36971b). Every hash
    /// of a copy, sent and compared, takes the same length. The default with
    /// every title buffer on the card ([`card_limits`]).
    pub(crate) fn card_hashed(&self, size: u64) -> u64 {
        static OWN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*OWN.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_LEN").is_none_or(|v| v != "0")) {
            return size;
        }
        self.len.next_multiple_of(16).clamp(16, size.max(16))
    }

    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// [`card_blocks_of`] for this copy: the flagged blocks of piece `i` of
    /// `span` as runs of offsets, onto `out`.
    pub(crate) fn card_blocks(&self, span: &CardSpan, i: u32, mask: u32, out: &mut Vec<(usize, usize)>) {
        card_blocks_of(self.start as usize, self.len as usize, span, i, mask, out);
    }

    /// The card's compare flagged the blocks `mask` of piece `piece` (of
    /// the copy's pieces, as [`CardSpan::first`] counts them): noted for
    /// good in `Card::flagged`, whatever the merge then finds.
    pub(crate) fn card_flagged(&self, piece: u32, mask: u32) {
        let Some(card) = &self.card else {
            return;
        };
        if mask == 0 {
            return;
        }
        let first = piece as usize * crate::card_fine::BLOCKS as usize;
        let mut bits = card.flagged.lock().unwrap_or_else(|e| e.into_inner());
        for b in 0..crate::card_fine::BLOCKS as usize {
            if mask & (1 << b) != 0 {
                let g = first + b;
                if bits.len() <= g / 64 {
                    bits.resize(g / 64 + 1, 0);
                }
                bits[g / 64] |= 1 << (g % 64);
            }
        }
    }

    /// For [`Shadow::merge_under_writes`] with [`merge_pages`], of a copy
    /// in the card's memory: the runs of offsets `at..at + title.len()`
    /// (within one 64 KiB page) where `title` differs from `held`, as
    /// (offset from `at`, bytes): first the exact runs, in the blocks a
    /// compare ever flagged; then one run per stretch of blocks none ever
    /// flagged, from its first changed byte to its last; then one run for
    /// each flagged block taken whole by the rule of [`block_runs`], which
    /// leaves at most `left` exact runs.
    #[allow(clippy::type_complexity)]
    fn card_page_runs(&self, at: usize, title: &[u8], held: &[u8], left: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let Some(card) = &self.card else {
            return (joined_over(changed_runs(title, held), merge_gap()), Vec::new(), Vec::new());
        };
        use crate::card_fine::BLOCK;
        let start = self.start as usize;
        // Where `title[0]` lies from the start of the copy's piece 0.
        let pos = start + at - start / PIECE * PIECE;
        let first = pos / BLOCK;
        let last = (pos + title.len().max(1) - 1) / BLOCK;
        let marks: Vec<bool> = {
            let bits = card.flagged.lock().unwrap_or_else(|e| e.into_inner());
            (first..=last).map(|g| bits.get(g / 64).is_some_and(|w| w & (1 << (g % 64)) != 0)).collect()
        };
        let (exact, joined) = page_runs(pos, title, held, merge_gap(), |g| marks.get(g - first).copied().unwrap_or(true));
        // For the census: the page's exact runs as found, and block by
        // block (a run is counted with the block it starts in).
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::buffers::under_page(self.start, self.len, start.saturating_add(at), exact.len());
            let mut runs = exact.iter().map(|r| (pos + r.0) / BLOCK).peekable();
            while let Some(block) = runs.next() {
                let mut n = 1;
                while runs.next_if_eq(&block).is_some() {
                    n += 1;
                }
                crate::copystat::buffers::under_block(n);
            }
        }
        match block_runs() {
            0 => (exact, joined, Vec::new()),
            most => {
                let (exact, rewritten) = rewritten_blocks(pos, exact, most, left);
                (exact, joined, rewritten)
            }
        }
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            let as_read = merge_race_on().then(|| title.clone());
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                crate::host::shaders_write(start + at, &title[..]);
                crate::watch::writing_as(start + at, n, crate::writes::Cause::Back);
                merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card");
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                #[cfg(feature = "diagnostics")]
                crate::copystat::buffers::landed(start + at, n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        if fast_merge() {
            return self.card_merge_relay_fast(at, gpu, fetched_after, title);
        }
        self.card_merge_relay_whole(at, gpu, fetched_after, title)
    }

    /// [`Shadow::card_merge_relay`] as it was (`VICTORY_FAST_MERGE=0`): the
    /// span from the first changed byte to the last, read and written back
    /// whole, through the kernel.
    fn card_merge_relay_whole(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.side_of(at, at + gpu.len());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = (mapped && merge_race_on()).then(|| title.clone());
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        crate::host::shaders_write(start + at, &title[..]);
        crate::watch::writing_as(start + at, n, crate::writes::Cause::Back);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        #[cfg(feature = "diagnostics")]
        crate::copystat::buffers::landed(start + at, n);
        n
    }

    /// For [`landed`]: `noted` are the 64 KiB pages a landing's taking of
    /// the kernel's bits noted written at asking `now` (in page order, with
    /// the kernel pages of each), `runs` the title memory the landing
    /// wrote of this copy (address, end; closed to refreshes still). A
    /// page of the copy whose noted kernel pages all lie in the runs and
    /// hold the bytes the CPU side holds, and that the copy had taken
    /// whole after the write before, is noted taken at `now` (`mode` 1) or
    /// kept for the check (`mode` 2).
    fn landed_pages(&self, noted: &[(usize, crate::watch::Written)], runs: &[(usize, usize)], now: u64, mode: u8) {
        thread_local! {
            static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let first = noted.partition_point(|&(page, _)| page < base);
        // The pages that hold what the CPU side holds: (page, the asking
        // that found it written before this one, kernel pages).
        let mut same: Vec<(usize, u64, u32)> = Vec::new();
        let (mut outside, mut differed) = (0u64, 0u64);
        {
            // The CPU side's lock, as the merge holds it, for each kernel
            // page looked at: what else writes these bytes of it (title
            // bytes put in under shader writes) holds it too, and no
            // refresh writes a range not done with.
            READ.with(|read| {
                let mut read = read.borrow_mut();
                read.resize(KERNEL_PAGE, 0);
                for &(page, w) in noted[first..].iter().take_while(|&&(page, _)| page * PAGE < end) {
                    let mut holds = w.mask != 0;
                    for k in (0..16).filter(|k| w.mask & (1 << k) != 0) {
                        let (lo, hi) = (page * PAGE + k * KERNEL_PAGE, page * PAGE + (k + 1) * KERNEL_PAGE);
                        if lo < start || hi > end || !runs.iter().any(|&(a, e)| a <= lo && hi <= e) {
                            outside += 1;
                            holds = false;
                            break;
                        }
                        let _side = self.side_of(lo - start, hi - start);
                        // SAFETY: `[lo, hi)` lies inside the copy (checked
                        // above), so inside the mapping of its CPU side.
                        let held = unsafe { std::slice::from_raw_parts(self.data.add(lo - start).cast_const(), KERNEL_PAGE) };
                        if !crate::host::read_into(lo as *const u8, &mut read[..]) || read[..] != *held {
                            differed += 1;
                            holds = false;
                            break;
                        }
                    }
                    if holds {
                        same.push((page, w.before, w.mask.count_ones()));
                    }
                }
            });
        }
        LANDED[4].fetch_add(outside, Ordering::Relaxed);
        LANDED[5].fetch_add(differed, Ordering::Relaxed);
        if same.is_empty() {
            return;
        }
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut check = (mode == 2).then(|| self.landed_check.lock().unwrap_or_else(|e| e.into_inner()));
        for (page, before, kernel_pages) in same {
            let Some(taken) = copied.get_mut(page - base) else {
                continue;
            };
            // Taken whole after the write before this one: with the noted
            // pages equal, the copy holds the page as of this asking.
            if *taken == 0 || *taken < before {
                LANDED[6].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            LANDED[2].fetch_add(1, Ordering::Relaxed);
            LANDED[3].fetch_add(u64::from(kernel_pages) * KERNEL_PAGE as u64, Ordering::Relaxed);
            match check.as_mut() {
                Some(check) => {
                    check.insert(page, now);
                }
                None => *taken = (*taken).max(now),
            }
        }
    }

    /// The check of [`land_taken`], at a refresh that is about to read
    /// `page` (last noted written at `last`, in the kernel pages `mask`)
    /// between offsets `seg` of it: if the landing that noted it would
    /// have had the copy pass the page by, title memory there is compared
    /// with the CPU side first.
    fn landed_checked(&self, page: usize, last: u64, mask: u16, seg: (u32, u32)) {
        let Some(stamp) = self.landed_check.lock().unwrap_or_else(|e| e.into_inner()).remove(&page) else {
            return;
        };
        LANDED[8].fetch_add(1, Ordering::Relaxed);
        // A later note: the page was written again and found so; with the
        // switch on it would be read now as well.
        if last != stamp {
            LANDED[9].fetch_add(1, Ordering::Relaxed);
            return;
        }
        let start = self.start as usize;
        let mut read = [0u8; KERNEL_PAGE];
        for k in (0..16usize).filter(|k| mask & (1 << k) != 0) {
            let (lo, hi) = ((k * KERNEL_PAGE).max(seg.0 as usize), ((k + 1) * KERNEL_PAGE).min(seg.1 as usize));
            if lo >= hi {
                continue;
            }
            let at = page * PAGE + lo;
            // SAFETY: `seg` lies inside the copy's part of the page
            // (`copy_stale_fine`), so inside the mapping of its CPU side.
            let held = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), hi - lo) };
            if !crate::host::read_into(at as *const u8, &mut read[..hi - lo]) || read[..hi - lo] == *held {
                continue;
            }
            // Other bytes than the copy's. Written since the taking (the
            // kernel holds the bit, and the next asking notes it)?
            if crate::watch::watched_dirty(page * PAGE + k * KERNEL_PAGE, KERNEL_PAGE).is_some_and(|n| n > 0) {
                LANDED[9].fetch_add(1, Ordering::Relaxed);
                return;
            }
            static SAID: AtomicU64 = AtomicU64::new(0);
            if LANDED[10].fetch_add(1, Ordering::Relaxed) < 32 && SAID.fetch_add(1, Ordering::Relaxed) < 32 {
                tracing::warn!(
                    copy = format_args!("{:#x}+{:#x}", self.start, self.len),
                    at = format_args!("{at:#x}"),
                    "a page a landing's copy would have passed by holds other bytes than the copy, and nothing wrote it since the landing took its bits: with VICTORY_LAND_TAKEN=1 the copy would have kept its own"
                );
            }
            return;
        }
    }

    /// [`Shadow::card_merge_relay`] with [`fast_merge`]: the same merge,
    /// byte for byte, of what the card changed; but only the stretches
    /// where the fetched bytes differ from the CPU's side
    /// ([`crate::card_fine::changed_stretches`], joined over
    /// [`MERGE_JOIN`] equal bytes) are read from title memory, merged and
    /// written back, each with the processor's own copy
    /// ([`crate::watch::copy_title`], the kernel's write for what faults);
    /// guards are opened once over all of them, and a run with nothing
    /// changed touches neither the guards nor title memory. Before, the
    /// whole stretch from the first changed byte to the last was read into
    /// a zeroed buffer and written back through the kernel.
    ///
    /// Differs from the old merge in one way: bytes between stretches (the
    /// card did not change them) are no longer read from title memory and
    /// written back unchanged, so a title write that lands there meanwhile
    /// is no longer put back to the value read (the CPU's side never took
    /// such bytes, in either merge). A stretch the card's bytes leave as
    /// title memory holds it, or that is not all mapped, is not written.
    fn card_merge_relay_fast(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.side_of(at, at + gpu.len());
        // SAFETY: as in `card_merge_relay`.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        MERGE_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let (found, stretches, write) = &mut *scratch;
            // A run that changed almost everywhere (a buffer rewritten whole)
            // is merged whole: no looking for stretches in it.
            let mostly = crate::card_fine::wide_compare() && crate::card_fine::mostly_changed(gpu, held);
            if mostly {
                found.clear();
                found.push((0, gpu.len()));
            } else {
                crate::card_fine::changed_stretches(gpu, held, MERGE_JOIN, found);
            }
            let (Some(&(first, _)), Some(&(last, last_n))) = (found.first(), found.last()) else {
                return 0;
            };
            // Title bytes the card had not received when the fetch ran
            // stay out (as in `card_merge_relay`).
            let (lo, hi) = (at + first, at + last + last_n);
            let skip: Vec<(usize, usize)> = {
                let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
                dirty
                    .iter()
                    .copied()
                    .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                    .filter(|&(a, b)| a < hi && b > lo)
                    .collect()
            };
            stretches.clear();
            // Such a run with nothing of it waiting to go up is not read from
            // title memory first: the bytes the card changed are written, and
            // title memory is left as it is around them (the merge's result,
            // byte for byte, without reading what it keeps).
            // Where the changed bytes lie in more than a few stretches (bytes
            // the card left as they were, between them), one read of the run
            // and one write cost less than a write a stretch: merged as
            // before.
            let mut whole = mostly && skip.is_empty();
            if whole {
                crate::card_fine::changed_stretches(gpu, held, 0, found);
                stretches.extend(found.iter().map(|&(a, n)| (at + a, at + a + n)));
                if stretches.is_empty() {
                    return 0;
                }
                if stretches.len() > WHOLE_MOST {
                    stretches.clear();
                    stretches.push((at, at + gpu.len()));
                    whole = false;
                    WHOLE_RUNS[3].fetch_add(1, Ordering::Relaxed);
                } else {
                    WHOLE_RUNS[0].fetch_add(1, Ordering::Relaxed);
                    WHOLE_RUNS[1].fetch_add(gpu.len() as u64, Ordering::Relaxed);
                    WHOLE_RUNS[2].fetch_add(stretches.len() as u64, Ordering::Relaxed);
                }
            } else if skip.is_empty() {
                stretches.extend(found.iter().map(|&(a, n)| (at + a, at + a + n)));
            } else {
                for &(a, n) in found.iter() {
                    stretches.extend(outside(at + a, at + a + n, skip.clone()));
                }
            }
            // Read and merged, each stretch at its place in `title`.
            let total: usize = stretches.iter().map(|&(a, b)| b - a).sum();
            if title.len() < total {
                title.resize(total, 0);
            }
            let mut put = 0;
            let mut any = false;
            write.clear();
            let mut races: Vec<Vec<u8>> = Vec::new();
            for &(a, b) in stretches.iter() {
                let (o, n) = (a - at, b - a);
                let t = &mut title[put..put + n];
                if whole {
                    // Every byte of these stretches is one the card changed:
                    // the merge puts the card's byte there.
                    t.copy_from_slice(&gpu[o..o + n]);
                    held[o..o + n].copy_from_slice(&gpu[o..o + n]);
                    if merge_race_on() {
                        races.push(Vec::new());
                    }
                    write.push(true);
                    any = true;
                    put += n;
                    continue;
                }
                let mapped = crate::host::read_into((start + a) as *const u8, t);
                if merge_race_on() {
                    races.push(if mapped { t.to_vec() } else { Vec::new() });
                }
                // Title memory that is not mapped: the CPU's side keeps
                // what the card wrote, as before.
                let changed = merge_piece(&gpu[o..o + n], &mut held[o..o + n], t);
                write.push(changed && mapped);
                any |= changed && mapped;
                put += n;
            }
            if !any {
                return 0;
            }
            // A stretch that is not all mapped is left out of the write (as
            // the old merge left out its whole run), and so is one the card's
            // bytes did not change in title memory.
            let (hull_lo, hull_hi) = (stretches[0].0, stretches[stretches.len() - 1].1);
            // Beside the guards (`crate::beside`), where views of ours show
            // every stretch of it: the guards stay as they are, and the
            // blocks are counted written once the bytes are there. Else
            // the guards are opened first, as ever.
            let places = if crate::beside::mode() != 0 {
                let to_write: Vec<(usize, usize)> =
                    stretches.iter().zip(write.iter()).filter(|(_, w)| **w).map(|(&(a, b), _)| (start + a, b - a)).collect();
                crate::beside::find_all(start + hull_lo, hull_hi - hull_lo, &to_write)
            } else {
                None
            };
            let mut went_beside = false;
            if places.is_none() {
                crate::watch::writing_as(start + hull_lo, hull_hi - hull_lo, crate::writes::Cause::Back);
            }
            let mut written = 0;
            let mut put = 0;
            // Which of the stretches written this one is.
            let mut nth = 0;
            let mut race = races.into_iter();
            for (&(a, b), &write) in stretches.iter().zip(write.iter()) {
                let n = b - a;
                let t = &title[put..put + n];
                put += n;
                let as_read = race.next();
                if !write {
                    continue;
                }
                crate::host::shaders_write(start + a, t);
                merge_race(as_read.as_deref(), (self.start, self.len), start + a, t, "card relay");
                let place = places.as_ref().and_then(|places| places.of(nth));
                nth += 1;
                match place {
                    Some(place) => {
                        let done = place.write(start + a, t);
                        went_beside |= done > 0;
                        if done < n {
                            // (A page of our view that is not there, or
                            // the check's doubt: this stretch as before.)
                            crate::watch::writing_as(start + a + done, n - done, crate::writes::Cause::Back);
                            write_title(start + a + done, &t[done..]);
                        }
                    }
                    None => {
                        write_title(start + a, t);
                        crate::read_census::wrote(start + a, n);
                    }
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::buffers::landed(start + a, n);
                written += n;
            }
            if went_beside {
                crate::watch::written_beside(start + hull_lo, hull_hi - hull_lo, crate::writes::Cause::Back);
            }
            if SAMPLE_NEXT.take() {
                // The last stretch written, up to 256 bytes of it, as written
                // and as the card held it.
                let mut put = 0;
                let mut last = None;
                for (&(a, b), &write) in stretches.iter().zip(write.iter()) {
                    if write {
                        last = Some((a, put, (b - a).min(256)));
                    }
                    put += b - a;
                }
                if let Some((a, put, m)) = last {
                    LAST_MERGED.set(Some((a, title[put..put + m].to_vec(), gpu[a - at..a - at + m].to_vec())));
                }
            }
            written
        })
    }

    /// The check at a landing's telling (`VICTORY_LAND_SAMPLE=1`), for a
    /// stretch a merge wrote at offset `at` of the copy (`merged`: what it
    /// wrote; `card`: what the card held there): the bytes the stretch
    /// landed (those it wrote as the card held them) that title memory and
    /// the copy's CPU side do not both hold now, looked at under the lock
    /// of their pieces. The stretch's other bytes are the title's as the
    /// merge read them, which the CPU side takes only with a refresh: not
    /// looked at. (Title memory that is not there: 0.)
    pub(crate) fn landed_apart(&self, at: usize, merged: &[u8], card: &[u8]) -> usize {
        let n = merged.len().min(card.len());
        if n == 0 || at + n > self.len as usize {
            return 0;
        }
        let _side = self.side_of(at, at + n);
        let mut read = vec![0u8; n];
        if !crate::host::read_into((self.start as usize + at) as *const u8, &mut read) {
            return 0;
        }
        // SAFETY: inside the copy's mapping (checked above).
        let held = unsafe { std::slice::from_raw_parts(self.data.add(at).cast_const(), n) };
        (0..n).filter(|&i| merged[i] == card[i] && read[i] != held[i]).count()
    }
}

/// Whether the relay's merge into title memory of what shaders wrote to
/// copies in the card's memory works stretch by stretch with the
/// processor's copy ([`Shadow::card_merge_relay_fast`];
/// `VICTORY_FAST_MERGE=0`: the whole changed span through the kernel, as
/// before). On by default.
///
/// In e16's profile the relay thread spent 31 ms a frame in that merge:
/// 12.3 in the kernel's write, 5.2 opening guards, 3.9 zeroing its buffer,
/// 3.5 reading title memory, 5.4 comparing byte by byte.
fn fast_merge() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAST_MERGE").is_none_or(|v| v != "0"))
}

/// Whether every refresh's copy is sampled against title memory afterwards
/// (`VICTORY_REFRESH_CHECK=1`, a diagnostic; `Shadow::refresh_check`).
pub(crate) fn refresh_checked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1"))
}

/// What the refresh check holds of a block (by the address of its sample)
/// that a copy was found to differ in, with no read of the title's in
/// flight over it.
struct Differing {
    /// Refreshes in a row that found it so.
    times: u32,
    /// Since the first of them.
    since: std::time::Instant,
    /// Whether a write behind the guards was told there meanwhile
    /// ([`refresh_check_told`]).
    told: bool,
    /// Whether it was said to have differed for long.
    long: bool,
}

static DIFFERING: Mutex<BTreeMap<usize, Differing>> = Mutex::new(BTreeMap::new());

/// The refresh check, since the start: looks that found a block other than
/// title memory under a read of the title's in flight, and with none;
/// blocks that differed three times in a row with none; of them, those
/// that matched again after a write behind the guards was told there, and
/// those that matched again with none told; blocks that differed for two
/// seconds with nothing told.
static REFRESH_CHECKED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// A write behind the guards into title memory `[at, at + len)` is done
/// and told (`crate::watch::title_memory_written`): the blocks the refresh
/// check holds as differing there were under it. Not to be called with the
/// guards' lock held.
pub(crate) fn refresh_check_told(at: usize, len: usize) {
    if !refresh_checked() || len == 0 {
        return;
    }
    let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
    // (A sample is a block's first bytes, or a refresh's.)
    for (_, d) in map.range_mut(at.saturating_sub(255)..at.saturating_add(len)) {
        d.told = true;
    }
}

/// With the frame line, while the refresh check is on.
pub(crate) fn refresh_check_line() {
    if !refresh_checked() {
        return;
    }
    let [flying, none, thrice, told, untold, long] = [0, 1, 2, 3, 4, 5].map(|i| REFRESH_CHECKED[i].load(Ordering::Relaxed));
    let still = DIFFERING.lock().unwrap_or_else(|e| e.into_inner()).values().filter(|d| d.times >= 3).count();
    tracing::info!(
        "refresh check, since the start (VICTORY_REFRESH_CHECK=1): {flying} looks found a copy's block other than title memory \
         while a read the title had submitted was writing there (the guards are told when it is done); {none} with no such read \
         known; {thrice} blocks differed three times in a row with none, of them {told} matched again once a write behind the \
         guards was told there, {untold} matched again with none told, {still} differ now; {long} differed for two seconds with \
         nothing told (those are copies that keep old bytes; each is named)"
    );
}

/// Whether a landing takes the kernel's written bits of the private title
/// memory it wrote, right behind its write, and counts the pages that then
/// hold what its copy holds as taken by that copy (1: the way, unset or any
/// value but 2; `VICTORY_LAND_TAKEN=2`: the check, below). Leaving the bits
/// to the refresh (the old way) is gone: the check counted 0 of 743,252
/// pages in the prologue and 0 of 181,911 at the open world's entry.
///
/// Private title memory is write-watched by the kernel, not guarded. A
/// landing's write sets the pages' bits like any write, the next refresh's
/// asking finds them, and the landing's own copy reads the pages it has
/// just been given and compares them with its CPU side: the same bytes,
/// 22.8 MB a frame in the first room (the census of 2026-11-26, "read and
/// found equal"), on the submitting thread.
///
/// With the switch ([`landed`]): once a landing's pieces are merged, the
/// bits of what it wrote in private memory are taken (one asking for the
/// pieces that lie together, `crate::watch::take_landed`) and the pages
/// they name are noted in the journal at once, as the next asking would
/// have noted them: every other host copy over those pages learns of the
/// landing there, as before. The landing's copy then reads each noted
/// 4 KiB page on the landing thread and compares it with its CPU side; a
/// 64 KiB page whose noted pages all lie in what the landing wrote, all
/// compare equal, and which the copy had taken before, is noted taken by
/// that copy at the landing's asking, and its next refresh passes it by.
/// Any other page is left to the refresh, as before. A write of the
/// title's behind the taking sets its bit again; one before it is in the
/// bytes compared. So there is no moment in which a write is lost, which
/// `VICTORY_MERGE_FORGET` had (it dropped the bits, the title's of before
/// the landing among them; it was never run and is gone).
///
/// The cost is the taking: a kernel call that resets written bits takes
/// 30 to 70 microseconds whatever the range, with the journal's lock held
/// as for every asking. So only landings that fetched [`LANDED_MIN`] bytes
/// of private memory or more take them (the buffers shaders rewrite whole
/// are most of the echo); the rest is left to the refresh.
///
/// The check (`=2`) takes and notes as `=1` and marks nothing: the refresh
/// reads the pages as before, and for each page `=1` would have marked it
/// compares title memory with the CPU side first. A page that differs,
/// whose bit the kernel does not hold (nothing wrote it since the taking)
/// and of which the journal holds no later note, is one `=1` would have
/// left with other bytes than the title's: counted and named, 0 is what
/// `=1` needs. With `VICTORY_REFRESH_CHECK=1` beside it every copy's
/// refreshes are sampled against title memory as well: the note made at
/// the landing must leave no other copy behind.
pub(crate) fn land_taken() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LAND_TAKEN").as_deref().map(str::trim) {
        Ok("2") => 2,
        _ => 1,
    })
}

/// The bytes of private title memory a landing must have fetched for it
/// to take the kernel's bits ([`land_taken`]): a taking costs about what
/// reading and comparing that much costs.
const LANDED_MIN: usize = 512 << 10;

/// Since the last frame line ([`land_taken`]): landings that took the
/// bits, kernel pages noted by them, 64 KiB pages noted taken (or, with
/// the check, that would have been), the bytes of the kernel pages in
/// them (the echo that is not read again), pages left to the refresh
/// because a noted kernel page lay outside what the landing wrote, because
/// one differed from the CPU side, because the copy had not taken the page
/// before; landings below [`LANDED_MIN`]; and for the check, since the
/// start: pages a refresh looked at, those written again since (the
/// kernel's bit, or a later note), and the finds.
static LANDED: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// A landing merged the fetched pieces `pieces` (copy, offsets in it)
/// into title memory and the copies' CPU sides, and their ranges are
/// still closed to refreshes: see [`land_taken`].
pub(crate) fn landed(pieces: &[(Arc<Shadow>, usize, usize)]) {
    let mode = land_taken();
    // The pieces in view memory: the guards' marks of them are taken as the
    // kernel's bits are below, and counted by the same copy.
    crate::watch::landed_views(
        pieces
            .iter()
            .filter(|(shadow, _, _)| shadow.card.is_some() && shadow.place.is_none())
            .map(|(shadow, lo, hi)| (Arc::as_ptr(shadow) as usize, shadow.start as usize + lo, shadow.start as usize + hi)),
        |copy, noted, runs, now| {
            if let Some((shadow, _, _)) = pieces.iter().find(|piece| Arc::as_ptr(&piece.0) as usize == copy) {
                shadow.landed_pages(noted, runs, now, mode);
            }
        },
    );
    // The pieces in private memory, as title addresses.
    let mut private: Vec<(&Arc<Shadow>, usize, usize)> = pieces
        .iter()
        .filter(|(shadow, lo, hi)| lo < hi && shadow.card.is_some() && shadow.place.is_none())
        .map(|(shadow, lo, hi)| (shadow, shadow.start as usize + lo, shadow.start as usize + hi))
        .filter(|&(_, lo, _)| crate::inplace::view_at(lo).is_none())
        .collect();
    let bytes: usize = private.iter().map(|p| p.2 - p.1).sum();
    if bytes < LANDED_MIN {
        LANDED[7].fetch_add(u64::from(bytes > 0), Ordering::Relaxed);
        return;
    }
    let ranges: Vec<(usize, usize)> =
        private.iter().map(|&(_, lo, hi)| (lo / KERNEL_PAGE * KERNEL_PAGE, hi.next_multiple_of(KERNEL_PAGE))).collect();
    let (now, noted) = crate::watch::take_landed(&ranges);
    LANDED[0].fetch_add(1, Ordering::Relaxed);
    LANDED[1].fetch_add(noted.iter().map(|(_, w)| u64::from(w.mask.count_ones())).sum(), Ordering::Relaxed);
    if noted.is_empty() {
        return;
    }
    // Copy by copy, each with what the landing wrote of it.
    private.sort_unstable_by_key(|&(shadow, lo, _)| (Arc::as_ptr(shadow) as usize, lo));
    for of_one in private.chunk_by(|a, b| Arc::ptr_eq(a.0, b.0)) {
        let runs: Vec<(usize, usize)> = of_one.iter().map(|&(_, lo, hi)| (lo, hi)).collect();
        of_one[0].0.landed_pages(&noted, &runs, now, mode);
    }
}

/// The kernel's page: what its written bits stand for.
const KERNEL_PAGE: usize = 4096;

/// With the frame line.
pub(crate) fn landed_line(frames: f64) {
    let mode = land_taken();
    let [takings, kernel_pages, marked, bytes, outside, differed, not_taken, small] =
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| LANDED[i].swap(0, Ordering::Relaxed) as f64 / frames);
    let [looked, written_since, finds] = [8, 9, 10].map(|i| LANDED[i].load(Ordering::Relaxed));
    tracing::info!(
        "landings' written bits taken in private title memory, per frame (VICTORY_LAND_TAKEN={mode}): {takings:.1} landings took \
         them ({small:.1} fetched too little there and left them to the refresh), {kernel_pages:.0} 4 KiB pages noted; {marked:.0} \
         pages of 64 KiB {} by the landing's copy, {:.2} MB it does not read again; left to the refresh: {outside:.0} with a noted \
         page outside what the landing wrote, {differed:.0} that differ from the copy (the title wrote there), {not_taken:.0} the \
         copy had not taken before; the check, since the start: {looked} such pages looked at by a refresh, {written_since} \
         written again since, {finds} with other bytes than the copy's and nothing to say so (0 is what =1 needs)",
        if mode == 1 { "noted taken" } else { "that would be noted taken" },
        bytes / 1e6,
    );
}

/// Equal bytes over which [`Shadow::card_merge_relay_fast`] joins changed
/// stretches: fewer, longer writes for bytes a shader wrote in a strided
/// pattern. An equal byte inside a stretch is put back as the title holds
/// it (as the old merge did for its whole span).
const MERGE_JOIN: usize = 64;

/// Since the start: runs merged whole without reading title memory first
/// (`Shadow::card_merge_relay_fast`), their bytes, the stretches of
/// changed bytes written of them; and the runs that changed almost whole
/// in more than [`WHOLE_MOST`] stretches, merged as before.
pub(crate) static WHOLE_RUNS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The most stretches of changed bytes a run that changed almost whole is
/// written in without reading title memory first.
const WHOLE_MOST: usize = 8;

thread_local! {
    /// Whether [`Shadow::card_merge_relay_fast`] on this thread keeps the
    /// last stretch it writes in [`LAST_MERGED`] ([`sample_next`]).
    static SAMPLE_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The last stretch [`Shadow::card_merge_relay_fast`] wrote on this
    /// thread when asked to keep it (offset in its copy, up to 256 bytes as
    /// written, the same bytes as the card held them): for the sample at a
    /// landing's telling ([`last_merged`]).
    static LAST_MERGED: std::cell::Cell<Option<(usize, Vec<u8>, Vec<u8>)>> = const { std::cell::Cell::new(None) };
}

/// The next merge on this thread keeps the last stretch it writes.
pub(crate) fn sample_next() {
    SAMPLE_NEXT.set(true);
}

/// The stretch kept ([`LAST_MERGED`]), taken; and no more is kept.
pub(crate) fn last_merged() -> Option<(usize, Vec<u8>, Vec<u8>)> {
    SAMPLE_NEXT.set(false);
    LAST_MERGED.take()
}

thread_local! {
    /// [`Shadow::card_merge_relay_fast`]'s lists, kept between calls.
    static MERGE_SCRATCH: std::cell::RefCell<(Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<bool>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

/// Writes `bytes` into title memory at `at` with the processor's copy
/// (`crate::watch::copy_title`) when `VICTORY_USER_COPY` is on, and what
/// faults (a page not there, or not writable) through the kernel from
/// there on ([`crate::host::write_mapped`], which leaves pages that are
/// not mapped alone). The caller opened the guards (`crate::watch::writing`).
fn write_title(at: usize, bytes: &[u8]) {
    let n = bytes.len();
    if crate::watch::user_copies() {
        // SAFETY: the destination is title memory whose guards are open
        // and which the copy's handler survives where it is not there; the
        // source is our own buffer, apart from it.
        let done = unsafe { crate::watch::copy_title(at as *mut u8, bytes.as_ptr(), n) };
        if done < n {
            // Up to 64 bytes past `done` may be written: written again.
            crate::host::write_mapped((at + done) as *mut u8, bytes[done..].as_ptr(), n - done);
        }
        return;
    }
    crate::host::write_mapped(at as *mut u8, bytes.as_ptr(), n);
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
/// Read shared by [`lookup`] (every view of every draw, on every recording
/// thread: as a mutex it was 32 ms a frame of theirs, 15 of them waiting,
/// run e18); everything that changes it takes it exclusive, as
/// [`buffers`] does.
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

/// Counts the changes to [`BUFFERS`] that can change what [`lookup`] answers
/// for an ask it answered from a buffer's copy before: a buffer added or
/// taken out, a copy taken away. Bumped under the exclusive lock, after
/// the change. (A copy made where there was none changes no such answer:
/// [`lookup`] remembers only answers that were a copy.)
static BUFFERS_CHANGED: AtomicU64 = AtomicU64::new(0);

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

fn buffers_changed() {
    BUFFERS_CHANGED.fetch_add(1, Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let mut map = buffers();
    let old = map.insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
    buffers_changed();
    drop(map);
    crate::memory::ownership_changed(start, old.map_or(len, |e| len.max(e.end - start)));
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let mut map = buffers();
    let old = map.remove(&(start, resource));
    buffers_changed();
    drop(map);
    if let Some(old) = old {
        crate::memory::ownership_changed(start, old.end - start);
    }
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
        if !gone.is_empty() {
            COPIES_GONE.fetch_add(1, Ordering::Release);
        }
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
        buffers_changed();
    }
    let n = gone.len() as u64;
    if n != 0 {
        crate::memory::ownership_changed(at, len);
    }
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)` if it has one already: nothing is made (a window outside buffers
/// is not looked for).
pub(crate) fn existing(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
    let key = buffer_for(&map, address, end)?;
    map.get(&key).and_then(|e| e.shadow.clone())
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

/// The buffer an ask of `[address, end)` takes: of the 64 that start
/// highest at or below `address`, the first that reaches `end`.
fn buffer_for(map: &BTreeMap<(u64, usize), Entry>, address: u64, end: u64) -> Option<(u64, usize)> {
    map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end).map(|(k, _)| *k)
}

/// Whether [`lookup`] remembers, per thread, the copies it answered asks
/// with ([`answered`]); `VICTORY_SHADOW_CACHE=0`: every ask looks in the
/// table.
fn lookup_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHADOW_CACHE").is_none_or(|v| v != "0"))
}

/// An answer [`lookup`] gave from a buffer's copy: the ask, the count of
/// [`BUFFERS_CHANGED`] it was given at, and the copy (weak: a remembered
/// answer holds nothing, so a copy goes when it went before).
struct Answer {
    address: u64,
    end: u64,
    changes: u64,
    shadow: std::sync::Weak<Shadow>,
}

/// Answers remembered per thread, by ask.
const ANSWERS: usize = 512;

thread_local! {
    static ANSWERED: std::cell::RefCell<Vec<Option<Answer>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Where an ask is remembered.
fn answer_slot(address: u64, end: u64) -> usize {
    let h = (address ^ end.rotate_left(29)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (h >> 40) as usize % ANSWERS
}

/// The copy [`lookup`] answered `[address, end)` with on this thread, if
/// [`BUFFERS`] has not changed since in a way that could change the answer
/// and the copy is still there. It is then the answer the table gives:
/// what the table answers depends on the buffers in it and on which have a
/// copy, and only [`BUFFERS_CHANGED`]'s changes take a copy away or change
/// which buffer an ask takes.
fn answered(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let changes = BUFFERS_CHANGED.load(Ordering::Acquire);
    ANSWERED.with(|a| {
        let a = a.borrow();
        let hit = a.get(answer_slot(address, end))?.as_ref()?;
        (hit.changes == changes && hit.address == address && hit.end == end).then(|| hit.shadow.upgrade()).flatten()
    })
}

fn remember(address: u64, end: u64, changes: u64, shadow: &Arc<Shadow>) {
    ANSWERED.with(|a| {
        let mut a = a.borrow_mut();
        if a.is_empty() {
            a.resize_with(ANSWERS, || None);
        }
        a[answer_slot(address, end)] = Some(Answer { address, end, changes, shadow: Arc::downgrade(shadow) });
    });
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let cache = lookup_cache();
    if cache && let Some(shadow) = answered(address, end) {
        return Some(shadow);
    }
    {
        // Shared, for the ask whose buffer has its copy already (all but
        // the first ask of each buffer).
        let map = BUFFERS.read().unwrap_or_else(|e| e.into_inner());
        // (Changed only under the exclusive lock.)
        let changes = BUFFERS_CHANGED.load(Ordering::Acquire);
        match buffer_for(&map, address, end) {
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
            Some(key) => {
                if let Some(shadow) = map.get(&key).and_then(|e| e.shadow.clone()) {
                    if cache {
                        remember(address, end, changes, &shadow);
                    }
                    return Some(shadow);
                }
            }
        }
    }
    // The copy is to be made: exclusive, and looked for again (the table
    // may have changed between the locks).
    let mut map = buffers();
    let Some(key) = buffer_for(&map, address, end) else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && size >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            placed_count(size, false);
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        let (buffer, data) = host()?.shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(size) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                // The small ones are counted only (`placed_line`): with
                // `VICTORY_CARD_BUFFERS_MIN_KB=64` they are thousands.
                if size >= 1 << 20 {
                    tracing::info!(
                        start = format_args!("{:#x}", key.0),
                        len,
                        "a buffer's host copy is in the card's memory"
                    );
                }
                let card = Card {
                    staging: buffer,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    size,
                    flagged: Mutex::default(),
                    me: std::sync::OnceLock::new(),
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        placed_count(size, card.is_some());
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
            if let Some(card) = &shadow.card {
                let _ = card.me.set(Arc::downgrade(&shadow));
            }
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: std::sync::RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = std::sync::RwLock::new(BTreeMap::new());

/// How often a copy went out of [`LOOSE`] or [`EXACT`]: the next ask for its
/// memory finds or makes another. Bumped under the table's lock, after the
/// change. (A copy made changes no answer given before it.)
static COPIES_GONE: AtomicU64 = AtomicU64::new(0);

/// A count that moves whenever an ask for the copy of some title memory
/// ([`find`], [`exact`]) may be answered with another copy than before:
/// [`BUFFERS_CHANGED`] and [`COPIES_GONE`] together. What remembers the
/// copy a view found (`crate::draw::Seen`) asks this before it trusts that.
pub(crate) fn changes() -> u64 {
    BUFFERS_CHANGED.load(Ordering::Acquire) + COPIES_GONE.load(Ordering::Acquire)
}

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    // Shared, for the ask whose window is there (`let_go_idle` takes the
    // lock exclusive from its look to its removal, so none is handed out in
    // between, as before).
    {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let key = loose_key(&map, address, end, unsized_ask, any);
        if let Some(shadow) = map.get(&key) {
            loose_note(shadow, address, end, unsized_ask);
            return Some(shadow.clone());
        }
    }
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    let (start, len) = loose_key(&map, address, end, unsized_ask, any);
    let note = |shadow: &Shadow| loose_note(shadow, address, end, unsized_ask);
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    loose_make(&mut map, start, len, note)
}

/// What [`loose`] notes of an ask in the window it hands out, under the
/// lock (see `let_go_idle`).
fn loose_note(shadow: &Shadow, address: u64, end: u64, unsized_ask: bool) {
    shadow.used.store(crate::host::moment(), Ordering::Relaxed);
    shadow.asked[0].fetch_min(address, Ordering::Relaxed);
    shadow.asked[1].fetch_max(end, Ordering::Relaxed);
    if unsized_ask {
        shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
    }
}

/// The window (start, bytes) [`loose`] takes for an ask, given the windows
/// in `map`.
fn loose_key(map: &BTreeMap<(u64, u64), Arc<Shadow>>, address: u64, end: u64, unsized_ask: bool, any: bool) -> (u64, u64) {
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    }
}

/// [`loose`]'s new window `(start, len)`, under the exclusive lock.
fn loose_make(
    map: &mut BTreeMap<(u64, u64), Arc<Shadow>>,
    start: u64,
    len: u64,
    note: impl Fn(&Shadow),
) -> Option<Arc<Shadow>> {
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the one just made is the CPU's side.
    let (buffer, card) = match window_card_for(len) {
        Some(on_card) => {
            crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
            let card = Card {
                staging: buffer,
                before: Mutex::default(),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                size: len,
                flagged: Mutex::default(),
                me: std::sync::OnceLock::new(),
            };
            (on_card, Some(card))
        }
        None => (buffer, None),
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    let shadow = Arc::new(shadow);
    if shadow.card.is_some() {
        CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        if let Some(card) = &shadow.card {
            let _ = card.me.set(Arc::downgrade(&shadow));
        }
    }
    note(&shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        let before = map.len();
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
        if map.len() != before {
            COPIES_GONE.fetch_add(1, Ordering::Release);
        }
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_OWN_COPY=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

#[cfg(test)]
mod card_merge_tests {
    use super::{Card, Shadow};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    /// A plain buffer on the software adapter: what a copy's `buffer` and
    /// its card's `staging` are; the merge never touches them.
    fn warp_buffer() -> Option<ID3D12Resource> {
        // SAFETY: plain DXGI and D3D12 creation calls on the software adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).ok()?;
            let device = device?;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 10,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
            buffer
        }
    }

    /// A copy's changed ranges are taken once, in order and joined, and
    /// noted as sent with the send that took them; its first range since
    /// they were last taken is what puts it on the list `card_dirty` walks.
    #[test]
    fn a_copys_changed_ranges_are_taken_once_and_its_first_one_lists_it() {
        // One test on the software adapter at a time, to the end of this one.
        let _alone = crate::host::warp_alone();
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let mut side = vec![0u8; LEN];
        let title = vec![0u8; LEN];
        let mut shadow = Shadow::new(buffer, side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
        // (No reference to itself: a copy built by hand stays off the list
        // the submissions of other tests take their copies from.)
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            size: 0,
            flagged: Mutex::default(),
            me: std::sync::OnceLock::new(),
        });
        assert_eq!(shadow.card_take_dirty(1, 0), None);
        shadow.card_changed(0x100, 0x200);
        shadow.card_changed(0x1000, 0x1100);
        shadow.card_changed(0x200, 0x300);
        // An empty range notes nothing.
        shadow.card_changed(0x5000, 0x5000);
        assert_eq!(shadow.card_take_dirty(2, 0), Some(vec![(0x100, 0x300), (0x1000, 0x1100)]));
        assert_eq!(shadow.card_take_dirty(3, 0), None);
        shadow.card_changed(0x40, 0x80);
        assert_eq!(shadow.card_take_dirty(4, 2), Some(vec![(0x40, 0x80)]));
        // With the relay the ranges are noted with their send, and the
        // notes of sends up to the fetched mark go.
        if super::relay() {
            let sent = shadow.card.as_ref().map(|c| c.sent.lock().unwrap().clone());
            assert_eq!(sent, Some(vec![(0x40, 0x80, 4)]));
        }
        // The first range noted says so, the ones after it do not.
        let mut dirty = Vec::new();
        assert!(super::note_changed(&mut dirty, 8, 16));
        assert!(!super::note_changed(&mut dirty, 16, 32));
        assert!(!super::note_changed(&mut dirty, 100, 200));
        assert_eq!(dirty, [(8, 32), (100, 200)]);
    }

    /// A landing's taking of the kernel's written bits
    /// (`VICTORY_LAND_TAKEN`), on a copy of write-watched memory: the page
    /// that holds only what the landing wrote is noted taken by its copy;
    /// one the title wrote too, and one the landing did not write, are left
    /// to the refresh; the journal names all three to every other asker.
    /// The check marks nothing, and finds a page whose bytes are not the
    /// copy's with nothing written since, and not one written since.
    #[test]
    fn a_landings_pages_are_taken_where_they_hold_the_copys_bytes() {
        use super::{KERNEL_PAGE, LANDED, PAGE};
        use std::sync::atomic::Ordering::Relaxed;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        // One test on the software adapter at a time, to the end of this one.
        let _alone = crate::host::warp_alone();
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 4 * PAGE;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(title, 0);
        let mut side = vec![0u8; LEN];
        let mut shadow = Shadow::new(buffer, side.as_mut_ptr(), 0, title as u64, LEN as u64);
        shadow.card = Some(Card {
            staging,
            before: Mutex::new(None),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            size: 0,
            flagged: Mutex::default(),
            me: std::sync::OnceLock::new(),
        });
        // Both sides get `value` at `at`, as a landing's merge leaves them;
        // or title memory alone, as a write of the title's does.
        let put = |at: usize, value: u8, both: bool| {
            // SAFETY: inside the allocation and inside `side`, which nothing
            // else touches in this test.
            unsafe {
                ((title + at) as *mut u8).write_volatile(value);
                if both {
                    shadow.data.add(at).write(value);
                }
            }
        };
        // The copy has taken everything, as of an asking that found nothing.
        let (taken, found) = crate::watch::writes_fine(title, title + LEN, false).expect("watched");
        assert_eq!(found, []);
        shadow.copied.lock().unwrap().fill(taken);
        // A landing writes pages 1 and 2; the title writes page 2 beside
        // it, and page 3.
        put(PAGE + 2 * KERNEL_PAGE + 5, 7, true);
        put(PAGE + 3 * KERNEL_PAGE, 8, true);
        put(2 * PAGE + 4 * KERNEL_PAGE, 9, true);
        put(2 * PAGE + 5 * KERNEL_PAGE + 1, 10, false);
        put(3 * PAGE + 1, 11, false);
        let runs = [(title + PAGE, title + 3 * PAGE)];
        let (now, noted) = crate::watch::take_landed(&runs);
        assert!(now > taken);
        let page = |k: usize| title / PAGE + k;
        assert_eq!(noted.iter().map(|n| (n.0, n.1.mask)).collect::<Vec<_>>(), [(page(1), 0b1100), (page(2), 0b11_0000)]);
        shadow.landed_pages(&noted, &runs, now, 1);
        assert_eq!(*shadow.copied.lock().unwrap(), [taken, now, taken, taken], "page 1 alone holds the copy's bytes");
        // Every other asker finds the landing's pages in the journal, and
        // the title's page 3 by the kernel's bit, which nobody took.
        let (later, found) = crate::watch::writes_fine(title, title + LEN, false).expect("watched");
        assert_eq!(found.iter().map(|f| (f.0, f.1.at)).collect::<Vec<_>>(), [(page(1), now), (page(2), now), (page(3), later)]);
        // The check: nothing is marked, the page is kept to be looked at.
        put(PAGE + 8, 12, true);
        let (again, noted) = crate::watch::take_landed(&runs);
        shadow.landed_pages(&noted, &runs, again, 2);
        assert_eq!(shadow.copied.lock().unwrap()[1], now, "the check marks nothing");
        let counts = || [8, 9, 10].map(|i| LANDED[i].load(Relaxed));
        let before = counts();
        // The page as the landing left it: looked at, no find.
        shadow.landed_checked(page(1), again, 0b1, (0, PAGE as u32));
        assert_eq!(counts(), [before[0] + 1, before[1], before[2]]);
        // Other bytes in the copy than in title memory, and nothing wrote
        // the page since the taking: a find.
        shadow.landed_pages(&noted, &runs, again, 2);
        // SAFETY: inside `side`.
        unsafe { shadow.data.add(PAGE + 8).write(13) };
        shadow.landed_checked(page(1), again, 0b1, (0, PAGE as u32));
        assert_eq!(counts(), [before[0] + 2, before[1], before[2] + 1]);
        // Written by the title since the taking: the kernel holds the bit,
        // and the next asking will name it. No find.
        // SAFETY: as above.
        unsafe { shadow.data.add(PAGE + 8).write(12) };
        shadow.landed_pages(&noted, &runs, again, 2);
        put(PAGE + 9, 14, false);
        shadow.landed_checked(page(1), again, 0b1, (0, PAGE as u32));
        assert_eq!(counts(), [before[0] + 3, before[1] + 1, before[2] + 1]);
        drop(shadow);
        // SAFETY: our allocation.
        unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) };
    }

    /// The mixed landing the fetch inside flights must survive, on a real
    /// copy in the card's memory and its merge (`Shadow::card_merge_relay`):
    /// flights 5, 6, 7 in the air; 5 lands the old way with the queue's
    /// count at 7; the title wrote a byte that went up in send 7 into a
    /// piece flight 6's shaders write; 6 then lands from its own slot, which
    /// holds that byte as it was before send 7. The notes of sends are
    /// dropped as `card_dirty` drops them, below the mark the old-way fetch
    /// sets (`crate::host::fallback_fetched_mark`): with the mark at the
    /// landed plans' send the title's byte survives and the shader's write
    /// lands; with the mark at the fetch's count (as before the fix) the
    /// title's byte is put back to the card's older one.
    #[test]
    fn a_title_byte_sent_after_a_flight_survives_its_slot() {
        // One test on the software adapter at a time, to the end of this one.
        let _alone = crate::host::warp_alone();
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let (title_byte, shader_byte) = (100usize, 200usize);
        let content: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        let land = |mark: u64| -> Vec<u8> {
            let mut title = content.clone();
            let mut side = content.clone();
            // The title's byte, refreshed into the CPU's side and sent up in send 7.
            title[title_byte] ^= 0x5a;
            side[title_byte] = title[title_byte];
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: staging.clone(),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
                flagged: Mutex::default(),
                me: std::sync::OnceLock::new(),
            });
            // As `card_dirty` with the mark set by the old-way fetch.
            if let Some(card) = &shadow.card {
                card.sent.lock().unwrap().retain(|s| s.2 > mark);
            }
            // Flight 6's slot: the card's bytes behind its lists, before send 7.
            let mut slot = content.clone();
            slot[shader_byte] ^= 0xc3;
            let written = shadow.card_merge_relay(0, &slot, 6, &mut Vec::new());
            assert!(written > 0);
            drop(shadow);
            title
        };
        let fixed = land(crate::host::fallback_fetched_mark(true, 7, Some(5)));
        assert_eq!(fixed[title_byte], content[title_byte] ^ 0x5a, "the title's byte was lost");
        assert_eq!(fixed[shader_byte], content[shader_byte] ^ 0xc3, "the shader's write did not land");
        // The mark at the fetch's count: the note of send 7 is gone and the
        // older card byte is taken for a shader's write.
        let before = land(7);
        assert_eq!(before[title_byte], content[title_byte]);
    }

    /// The stretch-by-stretch merge (`card_merge_relay_fast`) leaves title
    /// memory and the CPU's side exactly as the whole-span merge does, on
    /// random runs: shader writes alone, in clusters and everywhere, title
    /// bytes the card did not change, and a range sent up after the fetch
    /// (skipped by both).
    #[test]
    fn the_fast_merge_leaves_what_the_whole_merge_leaves() {
        // One test on the software adapter at a time, to the end of this one.
        let _alone = crate::host::warp_alone();
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let mut rng = 0x0bad_cafe_1234_5678u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for round in 0..60 {
            let content: Vec<u8> = (0..LEN).map(|_| next() as u8).collect();
            // What the card holds: the shaders' writes.
            let mut gpu = content.clone();
            let writes = match round % 4 {
                0 => 3,
                1 => 200,
                2 => LEN / 2,
                _ => 0,
            };
            for _ in 0..writes {
                let at = (next() as usize) % LEN;
                let len = 1 + (next() as usize) % 40;
                for b in &mut gpu[at..(at + len).min(LEN)] {
                    *b = b.wrapping_add(1 + (next() % 255) as u8);
                }
            }
            // Title bytes written since (not on the card).
            let mut title0 = content.clone();
            for _ in 0..(next() % 50) {
                let at = (next() as usize) % LEN;
                title0[at] ^= 0x5a;
            }
            // A range the title wrote that went up after the fetch: in
            // the CPU's side and title memory, not in what was fetched.
            let skip_at = (next() as usize) % (LEN - 600);
            let mut side0 = content.clone();
            for i in skip_at..skip_at + 500 {
                title0[i] = title0[i].wrapping_add(3);
                side0[i] = title0[i];
            }
            let land = |fast: bool| -> (Vec<u8>, Vec<u8>, usize) {
                let mut title = title0.clone();
                let mut side = side0.clone();
                // (The merge writes title memory through its address.)
                let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_mut_ptr() as u64, LEN as u64);
                shadow.card = Some(Card {
                    staging: staging.clone(),
                    before: Mutex::new(None),
                    dirty: Mutex::default(),
                    sent: Mutex::new(vec![(skip_at, skip_at + 500, 9)]),
                    size: 0,
                    flagged: Mutex::default(),
                    me: std::sync::OnceLock::new(),
                });
                // Two runs of the piece, as fetched blocks.
                let mut scratch = Vec::new();
                let mut written = 0;
                for (a, b) in [(0, LEN / 2), (LEN / 2, LEN)] {
                    written += if fast {
                        shadow.card_merge_relay_fast(a, &gpu[a..b], 5, &mut scratch)
                    } else {
                        shadow.card_merge_relay_whole(a, &gpu[a..b], 5, &mut scratch)
                    };
                }
                drop(shadow);
                (title, side, written)
            };
            let (title_whole, side_whole, written_whole) = land(false);
            let (title_fast, side_fast, written_fast) = land(true);
            assert!(title_fast == title_whole, "round {round}: title memory differs");
            assert!(side_fast == side_whole, "round {round}: the CPU's side differs");
            assert!(written_fast <= written_whole, "round {round}: {written_fast} > {written_whole}");
            assert_eq!(written_fast == 0, written_whole == 0, "round {round}");
        }
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod lookup_tests {
    use super::{find, register, unregister};
    use std::sync::Arc;

    /// The answers a thread remembers are the table's: the same copy while
    /// the table stands, the inner buffer once one is added, the outer one
    /// again when it goes, and not the copy of a buffer taken out.
    #[test]
    fn remembered_answers_follow_the_table() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        if crate::host::host().is_none() {
            return; // no host GPU
        }
        const LEN: usize = 1 << 16;
        // SAFETY: a plain allocation of our own.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!memory.is_null());
        let at = memory as u64;
        let (outer, inner) = (0x71_0001usize, 0x71_0002usize);
        register(outer, at, LEN as u64);
        let one = find(at + 0x100, 0x40).expect("a host copy");
        for _ in 0..3 {
            assert!(Arc::ptr_eq(&find(at + 0x100, 0x40).unwrap(), &one));
        }
        register(inner, at + 0x100, 0x100);
        let two = find(at + 0x100, 0x40).expect("the inner buffer's copy");
        assert!(!Arc::ptr_eq(&two, &one), "the inner buffer, added since");
        assert!(Arc::ptr_eq(&find(at + 0x100, 0x40).unwrap(), &two));
        unregister(inner, at + 0x100);
        assert!(Arc::ptr_eq(&find(at + 0x100, 0x40).unwrap(), &one), "the outer again");
        unregister(outer, at);
        assert!(find(at + 0x100, 0x40).is_none_or(|s| !Arc::ptr_eq(&s, &one)), "not the copy of a buffer gone");
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }
}
