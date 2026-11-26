//! Title lists submitted again while an earlier submission of them may still
//! run: a count, nothing more.
//!
//! D3D12 forbids submitting a command list again while an earlier
//! submission of it still runs (debug layer message 553); a list may go
//! again after the fence the application signalled behind the earlier
//! submission was reached. It was reported elsewhere, and is not proven for
//! this title, that in the open world the title submits one recorded list
//! about once a second, and that a fault followed after 128 such
//! submissions in one run. Until now only "the same list twice in one
//! submission" was known here (`HostList::prelude`, the second and later
//! runs of a list in one submission).
//!
//! What is counted is the title's side: a title list (the object the title
//! records into and submits) named by a later `ExecuteCommandLists` than
//! the one before, and no `Reset` of it in between (a reset list is
//! recorded anew: that is not the same recording again). The earlier
//! submission's fence is the first one the title signalled on the same
//! queue after it; whether that was reached is what the title's own fence
//! says (`crate::fence::completed`), which is as far as the title can know.
//!
//! The host side cannot run one host list twice at once across
//! submissions: see the notes on [`crate::list::detach`]. A submission that
//! is not waited for takes the host list out of the title's list and leaves
//! it a new, closed, empty one; a submission that is waited for is over
//! before the next can begin.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::fence::FenceRef;
use crate::object::This;

/// What the title lists submitted so far are known by, per queue and list
/// (their addresses).
struct Tracker<F> {
    /// Per list submitted, and not reset since: the fence the title
    /// signalled behind its last submission and the value, once it did.
    lists: HashMap<usize, Option<(F, u64)>>,
    /// Per queue: the lists submitted on it since its last signal, which
    /// that signal follows.
    pending: HashMap<usize, HashSet<usize>>,
}

/// How a submission found the lists it names again.
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
struct Again {
    /// The fence behind the earlier submission was not reached yet.
    flying: u64,
    /// The title had signalled nothing behind the earlier submission yet.
    unfenced: u64,
    /// The fence was reached: the title may submit it again.
    after: u64,
}

impl<F> Tracker<F> {
    fn new() -> Tracker<F> {
        Tracker { lists: HashMap::new(), pending: HashMap::new() }
    }

    /// A submission of `lists` to `queue`. A list named twice in it counts
    /// once (that case is the prelude's, not this one's).
    fn submit(&mut self, queue: usize, lists: &[usize], reached: impl Fn(&F, u64) -> bool) -> Again {
        let mut again = Again::default();
        let mut named = HashSet::new();
        for &list in lists {
            if list == 0 || !named.insert(list) {
                continue;
            }
            match self.lists.get(&list) {
                Some(Some((fence, value))) if reached(fence, *value) => again.after += 1,
                Some(Some(_)) => again.flying += 1,
                Some(None) => again.unfenced += 1,
                None => {}
            }
            self.lists.insert(list, None);
            // The signal behind this submission is the one it waits for.
            self.pending.values_mut().for_each(|p| {
                p.remove(&list);
            });
            self.pending.entry(queue).or_default().insert(list);
        }
        again
    }

    /// The title signalled a fence on `queue`: behind the lists submitted
    /// on it since the last signal.
    fn signal(&mut self, queue: usize, value: u64, fence: impl Fn() -> F) {
        let Some(lists) = self.pending.get_mut(&queue) else { return };
        for list in lists.drain() {
            if let Some(entry) = self.lists.get_mut(&list) {
                *entry = Some((fence(), value));
            }
        }
    }

    /// The title reset or released `list`: what it records next is another
    /// recording.
    fn forget(&mut self, list: usize) {
        self.lists.remove(&list);
        for pending in self.pending.values_mut() {
            pending.remove(&list);
        }
    }
}

static TRACKER: Mutex<Option<Tracker<FenceRef>>> = Mutex::new(None);

/// Since the last periodic line, and since the start: lists submitted again
/// with the earlier fence not reached, with no fence behind the earlier
/// submission yet, and after the fence was reached.
static INTERVAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static TOTAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A title submission of `lists` to `queue`, before it is run.
pub(crate) fn submitted(queue: This, lists: &[This]) {
    let ids: Vec<usize> = lists.iter().map(|&l| l as usize).collect();
    let again = TRACKER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Tracker::new)
        .submit(queue as usize, &ids, |fence, value| crate::fence::completed(fence.this()) >= value);
    let counts = [again.flying, again.unfenced, again.after];
    if counts.iter().all(|&n| n == 0) {
        return;
    }
    for (i, n) in counts.into_iter().enumerate() {
        INTERVAL[i].fetch_add(n, Relaxed);
        TOTAL[i].fetch_add(n, Relaxed);
    }
    if again.flying + again.unfenced > 0 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                frame = crate::frame::current(),
                fence_not_reached = again.flying,
                nothing_signalled_since = again.unfenced,
                "a title list was submitted again while an earlier submission of it may still run (D3D12 forbids this: debug \
                 layer message 553); counted from now on, logged once"
            );
        });
    }
}

/// The title signalled `fence` to `value` on `queue`.
pub(crate) fn signalled(queue: This, fence: This, value: u64) {
    if let Some(tracker) = TRACKER.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        tracker.signal(queue as usize, value, || FenceRef::new(fence));
    }
}

/// The title reset or released `list`.
pub(crate) fn forget(list: This) {
    if let Some(tracker) = TRACKER.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        tracker.forget(list as usize);
    }
}

/// What a title list's last recording held, kept when its host side is
/// taken (`crate::list::note_detached`).
pub(crate) struct Summary {
    /// The title list's type.
    pub(crate) kind: u32,
    /// Draws (mesh dispatches too), indexed draws, indirect executions,
    /// dispatches, clears, copies, resolves.
    pub(crate) counts: [u32; 7],
    /// Whether it set render targets.
    pub(crate) targets: bool,
    /// Host lists the recording is in (cut at waits for memory values).
    pub(crate) parts: u32,
    /// The first pipelines it bound, in order (shader stages and hashes).
    pub(crate) pipelines: Vec<String>,
}

impl Summary {
    fn text(&self) -> String {
        let [draws, indexed, indirect, dispatches, clears, copies, resolves] = self.counts;
        format!(
            "{draws} draws, {indexed} indexed draws, {indirect} indirect, {dispatches} dispatches, {clears} clears, {copies} copies, \
             {resolves} resolves, {} host lists, render targets {}, first pipelines {:?}",
            self.parts,
            if self.targets { "set" } else { "not set" },
            self.pipelines
        )
    }
}

/// Lists run empty, and the draws and dispatches they did not run: since the
/// last periodic line, and since the start.
static EMPTY_INTERVAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static EMPTY_TOTAL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// A title list was submitted again, without a reset, on a path that takes
/// its host list away: what runs is the empty list left in its place, and
/// the recording `recorded` is not replayed.
pub(crate) fn ran_empty(recorded: &Summary) {
    let [draws, indexed, indirect, dispatches, ..] = recorded.counts;
    let lost = [1, u64::from(draws + indexed + indirect), u64::from(dispatches)];
    for (i, n) in lost.into_iter().enumerate() {
        EMPTY_INTERVAL[i].fetch_add(n, Relaxed);
        EMPTY_TOTAL[i].fetch_add(n, Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let seen = LOGGED.fetch_add(1, Relaxed);
    if seen < 8 {
        tracing::warn!(
            "a title list ran empty ({} of the first 8 logged): submitted again without a reset after its host list was taken \
             away, frame {}, list type {}; its recording held {}",
            seen + 1,
            crate::frame::current(),
            recorded.kind,
            recorded.text()
        );
    }
}

/// The line with the others every few seconds.
pub(crate) fn line() {
    let [empty, empty_draws, empty_dispatches] = std::array::from_fn(|i| EMPTY_INTERVAL[i].swap(0, Relaxed));
    let [empty_all, draws_all, dispatches_all] = std::array::from_fn(|i| EMPTY_TOTAL[i].load(Relaxed));
    tracing::info!(
        "title lists that ran empty (submitted again without a reset on a path that gives the host list away): {empty} in the \
         interval, {empty_draws} draws and {empty_dispatches} dispatches not replayed; since the start: {empty_all} lists, \
         {draws_all} draws, {dispatches_all} dispatches"
    );
    let [flying, unfenced, after] = std::array::from_fn(|i| INTERVAL[i].swap(0, Relaxed));
    let [flying_all, unfenced_all, after_all] = std::array::from_fn(|i| TOTAL[i].load(Relaxed));
    tracing::info!(
        "title lists submitted again in a later submission (not reset between): {flying} in the interval while the fence of the \
         earlier submission was not reached yet, {unfenced} with no fence signalled behind the earlier one yet, {after} after \
         it was reached (harmless); since the start: {flying_all}, {unfenced_all}, {after_all}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reached(now: u64) -> impl Fn(&u32, u64) -> bool {
        move |_, value| now >= value
    }

    #[test]
    fn a_list_again_before_its_fence_is_counted_flying() {
        let mut t = Tracker::<u32>::new();
        assert_eq!(t.submit(1, &[10, 11], reached(0)), Again::default());
        t.signal(1, 5, || 0);
        assert_eq!(t.submit(1, &[10], reached(4)), Again { flying: 1, ..Again::default() });
    }

    #[test]
    fn a_list_again_after_its_fence_is_harmless() {
        let mut t = Tracker::<u32>::new();
        t.submit(1, &[10], reached(0));
        t.signal(1, 5, || 0);
        assert_eq!(t.submit(1, &[10], reached(5)), Again { after: 1, ..Again::default() });
    }

    #[test]
    fn a_list_again_before_any_signal_is_unfenced() {
        let mut t = Tracker::<u32>::new();
        t.submit(1, &[10], reached(0));
        assert_eq!(t.submit(2, &[10], reached(9)), Again { unfenced: 1, ..Again::default() });
        // Its fence is the signal behind the later submission, not the
        // earlier queue's.
        t.signal(1, 7, || 0);
        t.signal(2, 8, || 0);
        assert_eq!(t.submit(1, &[10], reached(7)), Again { flying: 1, ..Again::default() });
    }

    #[test]
    fn a_reset_list_is_a_new_recording() {
        let mut t = Tracker::<u32>::new();
        t.submit(1, &[10], reached(0));
        t.signal(1, 5, || 0);
        t.forget(10);
        assert_eq!(t.submit(1, &[10], reached(0)), Again::default());
    }

    #[test]
    fn a_list_named_twice_in_one_submission_is_not_counted() {
        let mut t = Tracker::<u32>::new();
        assert_eq!(t.submit(1, &[10, 10], reached(0)), Again::default());
        t.signal(1, 5, || 0);
        assert_eq!(t.submit(1, &[10, 10], reached(0)), Again { flying: 1, ..Again::default() });
    }

    #[test]
    fn a_list_run_empty_adds_what_it_did_not_replay() {
        let before = std::array::from_fn::<u64, 3, _>(|i| EMPTY_TOTAL[i].load(Relaxed));
        ran_empty(&Summary { kind: 0, counts: [5, 2, 1, 4, 0, 0, 0], targets: true, parts: 2, pipelines: vec![] });
        let after = std::array::from_fn::<u64, 3, _>(|i| EMPTY_TOTAL[i].load(Relaxed));
        // (other tests may add too: at least this one's)
        assert!(after[0] > before[0] && after[1] >= before[1] + 8 && after[2] >= before[2] + 4);
    }
}
