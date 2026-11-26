//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence to reach the value (the title may
    /// release the fence meanwhile: ours holds it).
    Wait(crate::fence::FenceRef, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A signal of the queue's, due after what it holds before it.
    Signal(crate::fence::FenceRef, u64),
    /// With [`behind`]: the submission at the head of the line is being
    /// prepared and sent by our thread, with the line unlocked, so the
    /// title can add to its end meanwhile; whatever is added stays behind
    /// this (the line is not empty), signals too.
    Running,
    /// With [`async_present`]: a present, shown in its place in the line.
    Present(Box<dyn FnOnce() + Send>),
    /// With list calls written down (`crate::list::thin`): a submission
    /// whose lists are translated by threads of ours. It is free to run when
    /// they are, and then becomes what [`submit_cut`] makes of a submission.
    Lists(std::sync::Arc<[std::sync::Arc<crate::list::thin::Recording>]>),
}

/// A submission with list calls written down (`crate::list::thin`): put
/// behind what the queue holds by its recordings; [`release_held_now`] cuts
/// and sends it once they are translated.
#[cfg(windows)]
fn submit_thin(this: This, lists: &[This]) {
    let recordings = crate::list::thin::submitted(lists);
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        format!(
            "SUBMIT queue kind {} title lists {lists:?} to the line, translated by threads of ours",
            queue(this).kind
        )
    });
    if !recordings.is_empty() {
        lock(&queue(this).held).push(Held::Lists(recordings));
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    release_held(false);
}

/// A recording was translated: a submission in a line may be free to run.
#[cfg(windows)]
pub(crate) fn thin_moved() {
    if behind() {
        wake();
    }
}

/// This thread does work the line waits for (it translates lists): it never
/// runs the line itself ([`before_sync`]).
#[cfg(windows)]
pub(crate) fn no_drain_here() {
    RELEASING_HERE.set(true);
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    crate::timeline::waited("L:queue", || mutex.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not,
/// and every submission runs when it is made (as before).
#[cfg(windows)]
fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// Whether a submission is held list by list: the lists before the first
/// one whose wait for a memory value is not met run at once, and each list
/// that waits is held with the lists after it, behind its own waits.
/// Holding a submission whole made the title's queues wait for each other:
/// a graphics submission was held for a value a compute list writes, and
/// that compute list for a value an earlier list of the same graphics
/// submission writes. Nothing moved until the present forced it, and the
/// title reached its present only after waiting about 2.5 s, every frame
/// (runs 27-30: 2.4-2.9 s from the last submission to the present call,
/// 13-32 ms before waits were honoured; the frame's main graphics
/// submissions, 88 lists, ran forced, the compute lists before them).
/// On by default; `VICTORY_SPLIT_WAITS=0` holds whole submissions, as
/// before.
#[cfg(windows)]
fn split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a list is cut into host lists at its waits for memory values
/// (`crate::list::cut`), and a queue runs a submission part by part: it
/// stands at each wait until the value is in memory, with everything
/// recorded before the wait run and its bottom-of-pipe writes done, as on
/// the console. Held list by list ([`split`]) a wait still held its whole
/// list: the title's lists are chains of jobs, each behind a wait for a
/// value another queue's list writes in the middle of its own chain
/// (frame 858 of run 33: a compute list of 433 commands with 13 waits;
/// graphics lists of the form "341 dispatches, wait, draws"), so two lists
/// each waited for a write behind the other's wait and only the present
/// moved them, in the wrong order.
/// On by default where waits are honoured list by list;
/// `VICTORY_CUT_WAITS=0` holds whole lists, as before.
#[cfg(windows)]
pub(crate) fn cuts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    waits() && split() && *ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a present on the title's thread runs what queues still hold
/// (on by default: a wait is not kept across frames). `VICTORY_WAIT_FORCE=0`
/// keeps it held: for finding out whether the title's waits reach across
/// its presents; a wait nothing ever meets then stops the title for good.
/// A present in the line ([`async_present`]) never forces: it keeps the
/// waits in order whatever this says.
#[cfg(windows)]
fn forces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_FORCE").is_none_or(|v| v != "0"))
}

/// A submission to a queue whose lists are cut at their waits ([`cuts`]).
/// The lists before the first one that waits run at once if the queue
/// holds nothing; the rest goes behind what the queue holds, part by part
/// (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let first = if !behind() && lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 && crate::host::late() {
        // Sent to the card and not waited for: their host sides are taken
        // from the title's lists (which it may reset before they have run)
        // and come back, with their tables' slots, when they are settled.
        let (taken, order) = crate::list::detach(&lists[..first]);
        host.execute_late(taken, order);
    } else if first > 0 {
        let now: Vec<_> = lists[..first]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
        // They have run: their descriptor tables' slots are free again, as
        // on the path without cuts ([`execute`]). This path replaced that
        // one without the call, so the lists that run at once kept their
        // slots until the title's reset and the ring ran out as before
        // (run 45: 3,143 of 15,942 tables refused in one frame).
        crate::list::tables_done(&lists[..first]);
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            for (waits, taken, order) in parts {
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order));
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// A submission to a queue that honours waits, list by list ([`split`]).
/// shortcut: a wait still holds its whole list, not only the commands
/// after it; and lists named twice among those held stay in one piece.
#[cfg(windows)]
fn submit_split(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let by = crate::list::waits_by_list(lists);
    // Behind what the queue already holds, nothing runs now.
    let cut = if lock(&queue(this).held).is_empty() {
        (0..lists.len())
            .find(|&i| by[i].iter().any(|w| !w.met()))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {cut} run at once)",
            Some(queue(this).kind),
            cut < lists.len()
        )
    });
    if cut > 0 {
        let now: Vec<_> = lists[..cut]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
    }
    if cut < lists.len() {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = cut,
                waits = ?by[cut],
                "lists of a submission are held for a memory value"
            );
        }
        let rest = &lists[cut..];
        // A list's host side can be taken once.
        let twice = rest.iter().enumerate().any(|(i, l)| rest[..i].contains(l));
        let mut pieces: Vec<Held> = Vec::new();
        let mut from = cut;
        while from < lists.len() {
            // Up to the next list that waits for anything (met or not: it
            // is looked at again when its turn comes).
            let to = if twice {
                lists.len()
            } else {
                (from + 1..lists.len())
                    .find(|&i| !by[i].is_empty())
                    .unwrap_or(lists.len())
            };
            let waits: Vec<crate::list::ValueWait> = by[from..to].concat();
            pieces.extend(waits.into_iter().map(Held::Value));
            let (taken, order) = crate::list::detach(&lists[from..to]);
            pieces.push(Held::Run(taken, order));
            from = to;
        }
        lock(&queue(this).held).extend(pieces);
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just held
    // may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    held.push(what());
    true
}

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    if !all && behind() {
        // A thread of ours runs what is free to run ([`behind`]).
        wake();
        return;
    }
    release_held_now(all);
}

/// Whether a submission is only put in its queue's line by the thread
/// that submits it, and prepared and sent by a thread of ours
/// (`VICTORY_ASYNC_SUBMIT=1`; needs flights and lists cut at their waits,
/// both defaults).
///
/// Why: the profile of run 213 (292 ms a frame) has the title's render
/// thread inside `queue::execute` for 202 ms of every frame: 155 ms in
/// `Host::submit` (preparing: the copies of title memory, the write
/// watch, descriptors, textures; and waiting for flights), while its five
/// recording threads stand idle 78 % of the time and the card 43 %. The
/// title's thread does nothing with the result of that work: it only has
/// to be done before the lists run, in order.
///
/// With the switch a submission's lists are taken from the title's lists
/// (as they already are when the queue holds anything) and put behind
/// the queue's line; one thread of ours ([`worker`]) runs the line in
/// order. A signal of the queue goes behind them, as it does behind
/// anything held. Everything that must see earlier submissions done runs
/// the line itself first: a present, and every submission of ours that
/// is waited for ([`before_sync`]).
///
/// What it assumes: title memory a submission reads is not rewritten by
/// the title between its submit call and the moment our thread prepares
/// it, some milliseconds later. On the console the card reads that
/// memory when it runs the list, later still, so a title that rewrote it
/// earlier would be wrong there too.
#[cfg(windows)]
pub(crate) fn behind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1") && crate::host::late() && cuts();
        if on {
            tracing::info!("submissions are prepared and sent by a thread of ours, not by the thread that submits (VICTORY_ASYNC_SUBMIT=1)");
            let _ = std::thread::Builder::new().name("victory-submit".into()).spawn(worker);
        }
        on
    })
}

/// Whether a present is put behind what its queue holds and shown by the
/// thread that runs the line ([`behind`]), with the title's thread going
/// on at once (`VICTORY_ASYNC_PRESENT=1`, with `VICTORY_ASYNC_SUBMIT=1`;
/// off by default, and off while frames are captured, whose dumps are
/// made at the present call).
///
/// Still opt-in. Its first form passed every wait the queues held at the
/// present call, met or not, as a present on the title's thread does
/// ([`forces`]): in both runs with it (e20, e21) and again on 2026-11-25
/// the title faulted in its own job threads some seconds into the first
/// 3D scene (a null pointer, a read of reserved memory), and value waits
/// were passed unmet ("a wait for a memory value was not met at a
/// present", none without it). Run with the waits kept in order
/// (`VICTORY_WAIT_FORCE=0` then) it ran 175 s in the first room without a
/// fault and with the picture right. So a present in the line keeps them:
/// it stands in the line until every wait before it is met, and the title
/// stands at its next present until this one was shown
/// ([`previous_present_shown`]); only a wait that is still not met after
/// two seconds is passed there, from the title's thread, as before.
///
/// Why: a present ran the queues' lines on the title's render thread
/// first, that is, it stood until our thread had prepared and sent every
/// submission of the frame (the profile of run e16: 85 ms of a 131 ms
/// frame in `queue::lock` under `present`), and only then did the title
/// start to record the next frame, while our thread stood idle for want
/// of work (39 ms a frame). The two take turns where they could overlap.
/// With the present in the line, the title records frame N+1 while our
/// thread prepares and sends frame N: what the console does, where the
/// card runs a frame the title submitted while the title makes the next.
///
/// The title never gets more than one present ahead: its next present
/// waits until this one was shown.
///
/// What it assumes beyond [`behind`]: title memory a frame's submissions
/// read is not rewritten by the title while it records the next frame,
/// before its own fences and waits say that the frame has run. They are
/// signalled only once our thread has sent the frame and the card has run
/// it, as before.
#[cfg(windows)]
fn async_present() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_PRESENT").is_some_and(|v| v == "1")
            && behind()
            && crate::host::late_free_present()
            && !capturing();
        if on {
            tracing::info!(
                "presents are shown in their queue's line by a thread of ours, behind every wait before them; the title goes on \
                 (VICTORY_ASYNC_PRESENT=1: opt-in until a longer run has shown it stable)"
            );
        }
        on
    })
}

#[cfg(all(windows, feature = "diagnostics"))]
fn capturing() -> bool {
    crate::capture::folder().is_some()
}

#[cfg(all(windows, not(feature = "diagnostics")))]
fn capturing() -> bool {
    false
}

/// The presents in queues' lines that were not shown yet
/// ([`async_present`]), and what tells a waiting present that one was.
#[cfg(windows)]
static PRESENTS: (std::sync::Mutex<u64>, std::sync::Condvar) = (std::sync::Mutex::new(0), std::sync::Condvar::new());

/// Since the last look ([`present_line`]): presents put in a line,
/// presents shown at once (their queue held nothing), the microseconds
/// presents stood for the one before them to be shown, the longest such
/// stand, and the stands that reached two seconds and ran the lines from
/// the title's thread, waits passed.
#[cfg(windows)]
pub(crate) static PRESENT_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Before a present goes into its queue's line: the one before it has
/// been shown. If it has not after two seconds (our thread stands at a
/// wait nothing has met, or at something only this thread moves), the
/// lines are run from here, waits passed, as a present on this thread
/// does ([`forces`]); counted and said.
///
/// The waits a line holds at the present call are not passed there (the
/// first form of [`async_present`] passed them, and the title faulted): a
/// present in the line comes to be shown only behind them, in order, as
/// the title's queues meet them.
#[cfg(windows)]
fn previous_present_shown() {
    let started = std::time::Instant::now();
    let mut pending = lock(&PRESENTS.0);
    while *pending > 0 {
        let (next, _) = PRESENTS
            .1
            .wait_timeout(pending, std::time::Duration::from_millis(50))
            .unwrap_or_else(|e| e.into_inner());
        pending = next;
        if *pending > 0 && started.elapsed() > std::time::Duration::from_secs(2) {
            drop(pending);
            let forced = PRESENT_STATS[4].fetch_add(1, Ordering::Relaxed) + 1;
            if forced <= 8 || forced.is_power_of_two() {
                tracing::warn!(
                    forced,
                    "a present in the line was not shown within two seconds: the lines are run from the title's thread, waits passed"
                );
            }
            release_held_now(true);
            pending = lock(&PRESENTS.0);
            break;
        }
    }
    drop(pending);
    let us = started.elapsed().as_micros() as u64;
    PRESENT_STATS[2].fetch_add(us, Ordering::Relaxed);
    PRESENT_STATS[3].fetch_max(us, Ordering::Relaxed);
}

/// With the frame line, with [`async_present`]: what [`PRESENT_STATS`]
/// counted over `frames`.
#[cfg(windows)]
pub(crate) fn present_line(frames: f64) {
    if !async_present() {
        return;
    }
    let [queued, at_once, stood, longest] = [0, 1, 2, 3].map(|i| PRESENT_STATS[i].swap(0, Ordering::Relaxed));
    let forced = PRESENT_STATS[4].load(Ordering::Relaxed);
    tracing::info!(
        "presents in the line per frame (VICTORY_ASYNC_PRESENT=1): {:.2} put behind what their queue held, {:.2} shown at once \
         (it held nothing); the title stood {:.1} ms a frame at its present for the one before to be shown, {:.1} ms at most; \
         since the start {forced} stands of two seconds that passed the waits held (0 is what the present in the line needs)",
        queued as f64 / frames,
        at_once as f64 / frames,
        stood as f64 / 1000.0 / frames,
        longest as f64 / 1000.0,
    );
}

/// Whether there is something for [`worker`] to look at.
#[cfg(windows)]
static WORK: (std::sync::Mutex<bool>, std::sync::Condvar) = (std::sync::Mutex::new(false), std::sync::Condvar::new());

#[cfg(windows)]
fn wake() {
    *lock(&WORK.0) = true;
    WORK.1.notify_one();
}

/// The thread of [`behind`]: runs what queues hold as far as it is free
/// to run, each time something was added or a fence moved.
#[cfg(windows)]
fn worker() {
    crate::read_census::layer_thread(false);
    loop {
        {
            let mut work = lock(&WORK.0);
            let _span = (!*work).then(|| crate::timeline::span("idle>", "idle<", 0, 0));
            while !*work {
                work = WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
            }
            *work = false;
        }
        release_held_now(false);
    }
}

#[cfg(windows)]
thread_local! {
    /// This thread is inside [`release_held_now`].
    static RELEASING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Before a submission of ours that is waited for, with [`behind`]: what
/// the queues hold and is free to run runs first, on this thread, so the
/// submission keeps its place behind them.
#[cfg(windows)]
pub(crate) fn before_sync() {
    if behind() && !RELEASING_HERE.get() {
        // (Submissions whose lists are translated first: they are before.)
        crate::list::thin::settle();
        release_held_now(false);
    }
}

#[cfg(windows)]
fn release_held_now(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    let was_here = RELEASING_HERE.replace(true);
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            RELEASING_HERE.set(self.0);
        }
    }
    let _leave = Leave(was_here);
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            // SAFETY: a live queue: one released is taken off the list.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            loop {
                let passed = all;
                // A submission whose lists are not translated yet stands; one
                // passed (at a present) is waited for, with the line open.
                if let Some(Held::Lists(recordings)) = held.first()
                    && !crate::list::thin::translated(recordings)
                {
                    if !passed {
                        break;
                    }
                    let recordings = recordings.clone();
                    drop(held);
                    crate::list::thin::wait(&recordings);
                    held = lock(line);
                    continue;
                }
                match held.first() {
                    None | Some(Held::Running) => break,
                    Some(Held::Wait(fence, value))
                        if !passed && crate::fence::completed(fence.this()) < *value =>
                    {
                        break;
                    }
                    Some(Held::Value(wait)) if !passed && !value_met(wait) => break,
                    Some(_) => {}
                }
                match held.remove(0) {
                    Held::Wait(..) | Held::Running => {}
                    Held::Lists(recordings) => {
                        // Translated: cut at its waits and taken from the
                        // lists of ours as `submit_cut` does, in its place,
                        // with the line open meanwhile.
                        held.insert(0, Held::Running);
                        drop(held);
                        let backs = crate::list::thin::backs(&recordings);
                        let parts = crate::list::detach_parts(&backs);
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            let named: Vec<String> = recordings
                                .iter()
                                .zip(&backs)
                                .map(|(r, &b)| match crate::list::host_list(b) {
                                    Some(h) => format!("title {:#x} as {:#x}", r.title(), std::ptr::from_ref(h) as usize),
                                    None => format!("title {:#x}", r.title()),
                                })
                                .collect();
                            format!("TRANSLATED submission, {} parts: {named:?}", parts.len())
                        });
                        let mut pieces = Vec::new();
                        for (waits, taken, order) in parts {
                            pieces.extend(waits.into_iter().map(Held::Value));
                            pieces.push(Held::Run(taken, order));
                        }
                        held = lock(line);
                        if matches!(held.first(), Some(Held::Running)) {
                            held.remove(0);
                        }
                        held.splice(0..0, pieces);
                    }
                    Held::Present(show) => {
                        // Shown with the line open, as a submission is
                        // sent: the title adds behind it meanwhile.
                        held.insert(0, Held::Running);
                        drop(held);
                        show();
                        {
                            let mut pending = lock(&PRESENTS.0);
                            *pending = pending.saturating_sub(1);
                        }
                        PRESENTS.1.notify_all();
                        held = lock(line);
                        if matches!(held.first(), Some(Held::Running)) {
                            held.remove(0);
                        }
                    }
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if passed && !value_met(&wait) {
                            static SEEN: AtomicU64 = AtomicU64::new(0);
                            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                tracing::warn!(
                                    queue = format_args!("{q:#x}"),
                                    ?wait,
                                    "a wait for a memory value was not met at a present: the queue runs on"
                                );
                            }
                        }
                    }
                    Held::Run(lists, order) => {
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            format!(
                                "RUN held submission of {} lists (forced {all})",
                                order.len()
                            )
                        });
                        // Diagnostics: the part behind this one stands at
                        // waits that are met already, so the two could have
                        // been one host submission (one wait for the card,
                        // each range shaders write compared once).
                        #[cfg(feature = "diagnostics")]
                        {
                            let waits = held.iter().take_while(|h| matches!(h, Held::Value(_))).count();
                            if waits > 0
                                && matches!(held.get(waits), Some(Held::Run(..)))
                                && held[..waits].iter().all(|h| matches!(h, Held::Value(w) if w.met()))
                            {
                                crate::copystat::add(crate::copystat::C::PartsTogether, 1);
                            }
                        }
                        if crate::host::late()
                            && let Some(host) = crate::host::host()
                        {
                            // Not waited for: the host keeps the lists
                            // until they are settled.
                            if behind() && !all {
                                // The line is open while this part is
                                // prepared and sent: only this thread takes
                                // from its head (one releaser at a time).
                                held.insert(0, Held::Running);
                                drop(held);
                                host.execute_late(lists, order);
                                held = lock(line);
                                if matches!(held.first(), Some(Held::Running)) {
                                    held.remove(0);
                                }
                            } else {
                                host.execute_late(lists, order);
                            }
                        } else {
                            let ran = {
                                let runs: Vec<&crate::host::HostList> =
                                    order.iter().map(|&i| lists[i].host()).collect();
                                match crate::host::host() {
                                    Some(host) => {
                                        host.execute(&runs);
                                        true
                                    }
                                    None => false,
                                }
                            };
                            // They have run: their host lists are used again.
                            if ran {
                                crate::list::recycle(lists, &order);
                            }
                        }
                        if all {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Forced, 1);
                            static ONCE: AtomicBool = AtomicBool::new(false);
                            if !ONCE.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "a queue still held a submission at a present: its fence was not reached"
                                );
                            }
                        }
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if let Err(fence) = crate::host::late_signal(fence, value) {
                            crate::fence::set(fence.this(), value);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            break;
        }
    }
}

/// Whether a queue may pass `wait`: memory holds the value, or, with
/// submissions sent to the card without a wait (`crate::host::late`), it
/// will once those in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    wait.met() || (crate::host::late() && crate::host::late_met(wait))
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Queue>(this);
    #[cfg(windows)]
    if left == 0 {
        lock(&WAITING).retain(|&q| q != this as usize);
    }
    left
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let _span = crate::timeline::span("exec>", "exec<", this as u64, u64::from(count));
        crate::resubmit::submitted(this, lists);
        if crate::list::thin::on() && !this.is_null() {
            submit_thin(this, lists);
            return;
        }
        if cuts() && !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        if waits() && split() && !this.is_null() {
            submit_split(this, lists);
            return;
        }
        // A list that waits for a memory value not there yet: the queue
        // holds the submission, as it does behind a fence.
        if waits() && !this.is_null() {
            let unmet = crate::list::unmet_waits(lists);
            if !unmet.is_empty() {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(kind = queue(this).kind, ?unmet, "a submission is held for a memory value");
                }
                lock(&queue(this).held).extend(unmet.into_iter().map(Held::Value));
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
        }
        let kept = hold(this, || {
            let (taken, order) = crate::list::detach(lists);
            Held::Run(taken, order)
        });
        // Diagnostics: the order submissions are made in, next to the
        // commands of the frame's `ops` file (which are in recording order).
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || {
            let kind = (!this.is_null()).then(|| queue(this).kind);
            // The lists as the frame's other lines name them (their host sides).
            let named: Vec<String> = lists
                .iter()
                .map(|&l| match crate::list::host_list(l) {
                    Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                    None => format!("title {l:?}"),
                })
                .collect();
            format!("SUBMIT queue kind {kind:?} held {kept} lists {named:?}")
        });
        if kept {
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    crate::timeline::ev("signal", fence as u64, value);
    #[cfg(windows)]
    crate::resubmit::signalled(this, fence, value);
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(crate::fence::FenceRef::new(fence), value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late() && crate::host::late_signal(crate::fence::FenceRef::new(fence), value).is_ok() {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        crate::timeline::ev("qwait", fence as u64, value);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let behind = !held.is_empty() || now < value;
            if behind && waits() {
                held.push(Held::Wait(crate::fence::FenceRef::new(fence), value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    held = waits(),
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind && waits() {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        #[cfg(windows)]
        crate::timeline::ev("bop", at as u64, value);
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    #[cfg(windows)]
    crate::timeline::present(FRAMES.load(Ordering::Relaxed) + 1);
    #[cfg(windows)]
    let _span = crate::timeline::span("present.", "present<", 0, 0);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::profile::frame();
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    // (With list calls written down, every submission is translated first:
    // what is free to run is then what it was before. Not for a present
    // that goes into the line: it stands there behind the lists submitted
    // before it, which the line's thread sends only once they are
    // translated, and the title's thread need not wait for that.)
    #[cfg(windows)]
    let in_line = async_present();
    #[cfg(windows)]
    if !in_line {
        crate::list::thin::settle();
    }
    #[cfg(windows)]
    if in_line {
        // The present goes into the line below, behind the waits there;
        // at most one stands there.
        previous_present_shown();
    } else if crate::host::late() && !crate::host::late_free_present() {
        // (With `behind`: what the queues hold runs here first, and what
        // it sent is settled with the rest.)
        if behind() {
            release_held_now(false);
        }
        crate::host::settle_for_present();
        release_held_now(false);
        crate::host::settle_for_present();
    } else if crate::host::late() {
        // The present is sent ahead too: what waits for a value a flight
        // will write runs behind that flight, and nothing is settled here.
        release_held_now(false);
    }
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames. (With the present in
    // the line, the thread that runs the line does so when it comes to it.)
    #[cfg(windows)]
    if !in_line {
        release_held(forces());
    }
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::frame_drew(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
    }
    // With the Series X profile the title presents one plane with its
    // interface in it; a second plane (the interface apart, as with the
    // Series S profile: fxceless/victory-fork pull request #1081) is not
    // shown, and said once.
    if planes > 1 {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| tracing::warn!(planes, "the title presents more than one plane: only the first is shown"));
    }
    #[cfg(windows)]
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && crate::resource::host_resource(resource).is_some()
        && crate::host::host().is_some()
    {
        let shown_resource = resource as usize;
        let show = move || {
            let resource = shown_resource as This;
            let Some(host) = crate::host::host() else {
                return;
            };
            // The copy to the back buffer moves the texture to the copy
            // source state as any list does (its prelude, at submission,
            // from the state the submissions before it left): taking the
            // shared state here and putting it back after the present let a
            // submission prepared meanwhile on another thread start from the
            // wrong state, and the write-back undid the state it left
            // (Marvisak/VIctory pull request #3204).
            let shown = host.present(resource, frame);
            crate::timeline::ev("shown", frame, 0);
            #[cfg(feature = "diagnostics")]
            blinks(frame, resource, 0x800, shown);
            #[cfg(not(feature = "diagnostics"))]
            let _ = shown;
        };
        // In the line, behind the frame's submissions, if the queue holds
        // any (`async_present`); else now, as before.
        let mut show = Some(show);
        if in_line {
            let mut held = lock(&queue(this).held);
            if !held.is_empty()
                && let Some(show) = show.take()
            {
                *lock(&PRESENTS.0) += 1;
                held.push(Held::Present(Box::new(show)));
                PRESENT_STATS[0].fetch_add(1, Ordering::Relaxed);
                drop(held);
                wake();
            } else {
                PRESENT_STATS[1].fetch_add(1, Ordering::Relaxed);
            }
        }
        if let Some(show) = show {
            // Shown from here after all (its queue held nothing): what
            // other queues were given is translated first, as above.
            if in_line {
                crate::list::thin::settle();
            }
            show();
        }
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once, once the
/// flights sent ahead whose shaders may write the pages have landed
/// ([`crate::host::land_before_remap`]).
/// shortcut: a submission a queue still holds behind a wait (`Held`) runs
/// after the remap, where the console runs it before when it was sent to
/// the same queue first (fxceless/victory-fork #1131).
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    // The console maps the pages behind the work sent before: what that
    // work's shaders write must land in the pages mapped now, not in those
    // mapped next (flights sent ahead land after their submission returned).
    #[cfg(windows)]
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let len: u64 = ranges.iter().map(|r| u64::from(r.count) * PAGE).sum();
        crate::host::land_before_remap(b.destination, len);
    }
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// A wait and a signal a queue holds back hold their fences: the title
    /// may release a fence it signals before the signal is due, and the
    /// fence it waits for before the wait is met. Both are let go when the
    /// queue runs on (fxceless/victory-fork #404).
    #[cfg(windows)]
    #[test]
    fn held_waits_and_signals_hold_their_fences() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        let (mut waited, mut signalled) = (null, null);
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut waited), S_OK);
        assert_eq!(crate::fence::create(null, 0, 0, fence_iid, &mut signalled), S_OK);
        let add_ref = |f: This| slot::<extern "system" fn(This) -> u32>(f, 0x08)(f);
        let release = |f: This| slot::<extern "system" fn(This) -> u32>(f, 0x10)(f);
        assert_eq!(wait(q, waited, 1), S_OK);
        assert_eq!(signal(q, signalled, 7), S_OK);
        // The title lets go of the fence it signals: the held signal keeps
        // it alive.
        assert_eq!(release(signalled), 1, "the held signal's reference");
        assert_eq!(add_ref(waited), 3, "the title's, the held wait's and this one");
        assert_eq!(release(waited), 2);
        // The wait is met: the queue runs on, sets the signal and lets
        // both fences go.
        assert_eq!(crate::fence::signal(waited, 1), S_OK);
        assert_eq!(release(waited), 0, "the held wait's reference is gone");
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
