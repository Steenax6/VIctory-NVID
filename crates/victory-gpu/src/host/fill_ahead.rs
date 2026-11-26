//! Fills of upload memory (a constant buffer a table names, copied into
//! the list's own memory when the list is submitted) that read title memory
//! flights "may write" are no longer waited for; what the flights then
//! write is checked against them, and the first fill that read bytes a
//! flight changed afterwards turns the waiting back on for good.
//!
//! Why: a flight declares every range its shaders can store to, and the
//! title has an unordered view over the whole of its 174 MB buffer, whose
//! copy is in the card's memory. Its per-draw constants lie in the same
//! buffer, at new addresses every frame. So the first such fill of a frame
//! waited until those flights had landed: one wait a frame, 56 to 95 ms of
//! the submitting thread standing still (runs 202 and 206; 145 ms while
//! the world loaded in 208), the largest single wait of the frame. Run 208
//! looked at every one of them: 44 fills of 480 bytes, and after the wait
//! the bytes were the same 44 times.
//!
//! What it risks, and how it is bounded: a fill that reads bytes a flight
//! in the air is about to change takes the old bytes, and the lists of
//! that one submission draw with them. `wrote` sees it when the flight
//! lands (the bytes going into title memory are compared with the ones
//! there, over every fill that read ahead of them) and from then on fills
//! wait as before. `VICTORY_FILL_WAIT=1`: they wait from the start.
//!
//! Moved out of `host.rs`, which the asset policy keeps under 512 KiB;
//! nothing in it changed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Fills are looked up by the 64 KiB of title memory they read.
const PIECE: usize = 1 << 16;
/// A fill is checked against flights that land this many frames after
/// it at most (flights are settled at 256 lists in the air, and by
/// every present that waits).
const KEEP: u64 = 16;

/// A flight changed bytes a fill had read ahead of it and
/// `VICTORY_FILL_HOT=0` is set: every fill waits, as it did before
/// the pieces were told apart.
static WAIT: AtomicBool = AtomicBool::new(false);
/// The 64 KiB pieces of title memory in which a flight changed bytes a
/// fill had read ahead of it: fills and copies that read them wait for
/// the flights; the others go on reading ahead. (The first such fill
/// turned the waiting on for every fill: in the intro that was one
/// fill of 2,048 bytes of a small buffer shaders do write each frame,
/// and from then on two fills a frame waited 54 ms for flights, 1,229
/// of them looked at, each of 480 bytes of per-draw constants
/// elsewhere, and after the wait their bytes were the same 1,229
/// times.)
static HOT: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
/// How many pieces `HOT` holds (read without the lock).
static HOT_PIECES: AtomicU64 = AtomicU64::new(0);

fn by_piece() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_HOT").is_none_or(|v| v != "0"))
}
/// Since the last frame line: fills that read ahead; landings' writes
/// that touched one's piece. Since the start: fills that read bytes a
/// flight then changed.
static COUNT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

#[derive(Default)]
struct Piece {
    /// The frame its old fills were last dropped in.
    pruned: u64,
    /// From, to, frame, and the number of the last submission started
    /// when it read: the flights in the air then have that number or
    /// a lower one, and only their writes can be ones it missed.
    fills: Vec<(usize, usize, u64, u64)>,
}

/// The lowest submission number among the flights the landing thread
/// is landing now; 0: not known (a landing by another way), and then
/// every write counts. (Run 211: a read of frame 1381 was taken for
/// missed because shaders wrote the same 80 bytes in frame 1397, in a
/// flight sent sixteen frames after it.)
static LANDING: AtomicU64 = AtomicU64::new(0);

pub(super) fn landing(first: u64) {
    LANDING.store(first, Relaxed);
}

static READ: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_WAIT").is_none_or(|v| v != "1")) && !WAIT.load(Relaxed)
}

/// Whether a flight was seen to change bytes read ahead of it in a
/// 64 KiB piece that `[src, src + len)` touches.
pub(super) fn hot(src: usize, len: usize) -> bool {
    if len == 0 || HOT_PIECES.load(Relaxed) == 0 {
        return false;
    }
    let hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
    hot.as_ref().is_some_and(|hot| (src / PIECE..=(src + len - 1) / PIECE).any(|piece| hot.contains(&piece)))
}

/// Whether a read of `len` bytes at `src` that flights may write goes
/// ahead of them: fills do not wait, and no flight was seen to change
/// bytes read ahead in the 64 KiB pieces it reads.
pub(super) fn ahead(src: usize, len: usize) -> bool {
    if !on() {
        return false;
    }
    if len == 0 || HOT_PIECES.load(Relaxed) == 0 {
        return true;
    }
    let hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
    let Some(hot) = hot.as_ref() else { return true };
    !(src / PIECE..=(src + len - 1) / PIECE).any(|piece| hot.contains(&piece))
}

/// A fill reads `len` bytes at `src` ahead of flights that may write
/// them.
pub(super) fn read(src: usize, len: usize) {
    if len == 0 {
        return;
    }
    COUNT[0].fetch_add(1, Relaxed);
    let frame = crate::frame::current();
    let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
    let map = read.get_or_insert_with(HashMap::new);
    for piece in src / PIECE..=(src + len - 1) / PIECE {
        let p = map.entry(piece).or_default();
        if p.pruned != frame {
            p.pruned = frame;
            p.fills.retain(|f| f.2 + KEEP >= frame);
        }
        p.fills.push((src, src + len, frame, super::STARTED.load(Relaxed)));
    }
}

/// A landing is about to put `new` into title memory at `at`: bytes
/// shaders wrote, merged into the title's.
pub(crate) fn wrote(at: usize, new: &[u8]) {
    if new.is_empty() || WAIT.load(Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    let landing = LANDING.load(Relaxed);
    let hits: Vec<(usize, usize, u64)> = {
        let read = READ.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = read.as_ref() else { return };
        let end = at + new.len();
        (at / PIECE..=(end - 1) / PIECE)
            .filter_map(|piece| map.get(&piece))
            .flat_map(|p| p.fills.iter().copied())
            .filter(|f| f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3))
            .map(|f| (f.0, f.1, f.2))
            .collect()
    };
    if hits.is_empty() {
        return;
    }
    COUNT[1].fetch_add(1, Relaxed);
    let mut now = Vec::new();
    for (from, to, read_in) in hits {
        let (a, b) = (from.max(at), to.min(at + new.len()));
        now.clear();
        now.resize(b - a, 0);
        // Title memory as it is, which is what the fill read unless the
        // title wrote it since; not there: nothing to say.
        if !super::read_into(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
            continue;
        }
        let n = COUNT[2].fetch_add(1, Relaxed) + 1;
        let first = now.iter().zip(&new[a - at..b - at]).position(|(x, y)| x != y).unwrap_or(0);
        if by_piece() {
            // Only reads of the pieces this fill read wait from here on.
            let fresh = {
                let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
                let hot = hot.get_or_insert_with(Default::default);
                let before = hot.len();
                hot.extend(from / PIECE..=(to - 1) / PIECE);
                HOT_PIECES.store(hot.len() as u64, Relaxed);
                hot.len() - before
            };
            if fresh > 0 {
                tracing::warn!(
                    "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders \
                     changed them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the \
                     bytes from before. Reads of that 64 KiB of title memory wait for flights from here on \
                     (VICTORY_FILL_HOT=0: every fill waits; {n} such fills)",
                    to - from,
                    a + first
                );
            }
            continue;
        }
        if !WAIT.swap(true, Relaxed) {
            tracing::warn!(
                "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                 them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                 Fills wait for flights again from here on, as with VICTORY_FILL_WAIT=1 ({n} such fills)",
                to - from,
                a + first
            );
        }
        return;
    }
}

/// With the frame line.
pub(super) fn line(frames: f64) {
    let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Relaxed));
    if ahead == 0 && looked == 0 {
        return;
    }
    let kept = {
        let frame = crate::frame::current();
        let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
        read.as_mut().map_or(0, |map| {
            map.retain(|_, p| {
                p.fills.retain(|f| f.2 + KEEP >= frame);
                !p.fills.is_empty()
            });
            map.values().map(|p| p.fills.len()).sum::<usize>()
        })
    };
    tracing::info!(
        "fills of upload memory that read title memory flights may write, not waited for (VICTORY_FILL_WAIT=1 waits): {:.1} per \
         frame; {:.1} landings' writes per frame fell in the same 64 KiB as one and were compared with it; fills that read \
         bytes a flight then changed, since the start: {}{}; {kept} fills kept to check",
        ahead as f64 / frames,
        looked as f64 / frames,
        COUNT[2].load(Relaxed),
        if WAIT.load(Relaxed) {
            " (fills wait again)".to_owned()
        } else {
            format!(" ({} pieces of 64 KiB whose reads wait)", HOT_PIECES.load(Relaxed))
        }
    );
}
