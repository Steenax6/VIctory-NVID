//! The heap sync ([`super::sync_heaps`]) on a thread of its own, beside a
//! submission's refreshes (`VICTORY_HEAP_SYNC_BESIDE`, on; `=0`: on the
//! submitting thread before them, as before).
//!
//! Why: the submitting thread ran the sync (8 to 9 ms a frame in the first
//! room, 29 submissions: some 40,000 records the title copied into its
//! heaps as bytes, mirrored), then the refreshes of host copies, the
//! lists' steps and the snapshots, one after the other. The sync reads
//! records and makes host descriptors; the refreshes read title memory
//! into host copies; neither reads what the other writes. Only the lists
//! need both, when they go to the card. So the sync starts where it
//! started (after the sweep, under the submission lock) on its own thread,
//! and is joined before the lists are sent: the submission takes about the
//! longer of the two instead of their sum.
//!
//! What stays as it was:
//! - Its askings (the write watch's, the guards') of every heap come first,
//!   in the order of the heaps, as when it ran here: the submitting thread
//!   waits for them ([`Beside::asked`]) before the submission's own asking
//!   (`crate::watch::ask_once`), so the journal numbers them as before.
//! - It never meets the sweep: it starts after it and is joined in the same
//!   submission, under the submission lock, which the sweep needs.
//! - The slots it leaves waiting for a texture's host copy are made after
//!   the join (`super::make_waiting` once more), before the lists go.
//! - It runs on a thread of its own, never among the threads the
//!   submitting thread shares out work to and waits for
//!   (`crate::shadow::wide_run`): waiting for it from there could stand
//!   behind its own work.
//!
//! What it touches that the submitting thread touches meanwhile is behind
//! the locks the recording threads already share with it (the journal, the
//! guards, the slots' stripes, the views): the recording threads mirror
//! views at every draw while this runs, as before.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex};

/// Whether the sync runs beside the refreshes (`VICTORY_HEAP_SYNC_BESIDE=0`:
/// on the submitting thread, as before).
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_BESIDE").is_none_or(|v| v != "0"))
}

/// Where the thread is with the one sync it is given at a time.
enum Phase {
    /// Nothing to do.
    Idle,
    /// Given, not begun.
    Wanted,
    /// Asking the write watch and the guards about the heaps' pages.
    Asking,
    /// Reading and mirroring records.
    Walking,
    /// Done; a panic of the sync, to be raised where it is joined.
    Done(Option<Box<dyn std::any::Any + Send>>),
}

struct Lane {
    phase: Mutex<Phase>,
    moved: Condvar,
}

static LANE: Lane = Lane { phase: Mutex::new(Phase::Idle), moved: Condvar::new() };

fn lock() -> std::sync::MutexGuard<'static, Phase> {
    LANE.phase.lock().unwrap_or_else(|e| e.into_inner())
}

fn set(phase: Phase) {
    *lock() = phase;
    LANE.moved.notify_all();
}

/// The thread, made at the first sync given.
fn thread() {
    loop {
        {
            let mut phase = lock();
            while !matches!(*phase, Phase::Wanted) {
                phase = LANE.moved.wait(phase).unwrap_or_else(|e| e.into_inner());
            }
            *phase = Phase::Asking;
        }
        let _span = crate::timeline::span("hsync>", "hsync<", 0, 0);
        let done = std::panic::catch_unwind(|| super::sync_heaps_then(|| set(Phase::Walking)));
        set(Phase::Done(done.err()));
    }
}

/// Since the last frame line: microseconds the submitting thread waited
/// for the sync's askings and at the join, and the syncs run beside.
pub(crate) static WAITED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A sync begun by [`begin`]; joined when dropped, if not before.
pub(crate) struct Beside {
    /// Running on the thread (false: it ran in [`begin`]).
    away: bool,
}

/// Begins the heap sync: on its thread, or here and to its end without
/// the switch.
pub(crate) fn begin() -> Beside {
    if !on() || !super::heap_sync_on() {
        super::sync_heaps();
        return Beside { away: false };
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    let mut made = true;
    STARTED.call_once(|| {
        made = std::thread::Builder::new().name("victory-heap-sync".into()).spawn(thread).is_ok();
    });
    if !made {
        super::sync_heaps();
        return Beside { away: false };
    }
    {
        let mut phase = lock();
        // (Joined at every submission: nothing else can stand here.)
        debug_assert!(matches!(*phase, Phase::Idle));
        *phase = Phase::Wanted;
    }
    LANE.moved.notify_all();
    WAITED[2].fetch_add(1, Relaxed);
    Beside { away: true }
}

impl Beside {
    /// Waits until the sync has asked about every heap's pages.
    pub(crate) fn asked(&self) {
        if !self.away {
            return;
        }
        let started = std::time::Instant::now();
        let mut phase = lock();
        while matches!(*phase, Phase::Wanted | Phase::Asking) {
            phase = LANE.moved.wait(phase).unwrap_or_else(|e| e.into_inner());
        }
        WAITED[0].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    }

    /// Waits until the sync is done, and raises its panic here if it had one.
    pub(crate) fn join(mut self) {
        self.wait();
    }

    fn wait(&mut self) {
        if !std::mem::replace(&mut self.away, false) {
            return;
        }
        let _span = crate::timeline::span("hjoin>", "hjoin<", 0, 0);
        let started = std::time::Instant::now();
        let done = {
            let mut phase = lock();
            loop {
                if let Phase::Done(_) = &*phase {
                    break std::mem::replace(&mut *phase, Phase::Idle);
                }
                phase = LANE.moved.wait(phase).unwrap_or_else(|e| e.into_inner());
            }
        };
        WAITED[1].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        if let Phase::Done(Some(panic)) = done
            && !std::thread::panicking()
        {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Drop for Beside {
    fn drop(&mut self) {
        self.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The phases go round: given, asked, done, idle again, twice.
    #[test]
    fn a_sync_beside_is_asked_then_joined() {
        for _ in 0..2 {
            let sync = begin();
            sync.asked();
            assert!(!matches!(*lock(), Phase::Wanted | Phase::Asking));
            sync.join();
            assert!(matches!(*lock(), Phase::Idle));
        }
    }
}
