//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is. A
//! pipeline is waited for one frame's budget in all, not in every frame.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// Microseconds draws and dispatches have waited for it, over all
    /// frames ([`wait`]), and whether it is counted in [`PASSED_OVER`].
    waited_us: AtomicU64,
    passed_over: AtomicBool,
    /// The same pipeline made for other render target formats
    /// ([`for_targets`]); never removed while this one lives.
    variants: Mutex<Vec<Arc<Pending>>>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            waited_us: AtomicU64::new(0),
            passed_over: AtomicBool::new(false),
            variants: Mutex::new(Vec::new()),
        }
    }

    fn make(&self) {
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        if self.passed_over.swap(false, Ordering::SeqCst) {
            PASSED_OVER.fetch_sub(1, Ordering::SeqCst);
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or, for a draw, is made within the
/// frame's budget); queues it on first use. A compute pipeline is made
/// before this returns, whatever the budget.
///
/// A draw dropped for want of its pipeline is drawn again in the next
/// frame. A dispatch may not be: the title runs some compute passes once,
/// when a scene changes, and what they leave is read for as long as the
/// scene lasts. So a compute pipeline that is not made yet is made on the
/// calling thread, or waited for while a worker makes it: a wait of
/// seconds at most, the first time a program is met (the pipeline library
/// has it in the next run), against a picture that stays wrong
/// (fxceless/victory-fork pull request #1312, Exopus-0x65f7ad0: a program
/// that took 4.4 s to make on a machine short of memory had its one
/// dispatch dropped after the half second of budget, and the picture was
/// wrong for more than 80 s after it). Until now a dispatch waited within
/// the budget like a draw, and was dropped ("a dispatch without a host
/// pipeline dropped") and not repeated.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        if p.graphics.is_none() {
            // Made here, or waited for while a worker makes it.
            let started = Instant::now();
            p.make();
            COMPUTE_WAITS[0].fetch_add(1, Ordering::Relaxed);
            COMPUTE_WAITS[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        } else {
            wait(p, crate::frame::current());
        }
    }
    p.made.get()?.as_ref()
}

/// Since the start: compute pipelines a dispatch stood for until they were
/// made ([`ready`]), and the microseconds it stood.
static COMPUTE_WAITS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// With the frame line, when a dispatch has stood for its pipeline since
/// the line before.
pub(crate) fn compute_waits_line() {
    static SAID: AtomicU64 = AtomicU64::new(0);
    let n = COMPUTE_WAITS[0].load(Ordering::Relaxed);
    if SAID.swap(n, Ordering::Relaxed) == n {
        return;
    }
    tracing::info!(
        "compute pipelines a dispatch stood for until they were made, since the start: {n}, {:.1} ms in all (a dispatch is not \
         dropped for want of its pipeline; a draw still is, past VICTORY_PIPELINE_WAIT_MS a frame)",
        COMPUTE_WAITS[1].load(Ordering::Relaxed) as f64 / 1e3
    );
}

/// `p` made for render targets of `targets` instead of the formats the
/// title named, by the workers like any first use (and waited for within
/// the frame's budget); `None` while it is being made or if the host
/// refuses it. Each set of formats is made once per pipeline.
pub(crate) fn for_targets(
    p: &Pending,
    targets: windows::Win32::Graphics::Direct3D12::D3D12_RT_FORMAT_ARRAY,
) -> Option<&HostPipeline> {
    let g = p.graphics.as_deref()?;
    let variant = {
        let mut variants = p.variants.lock().unwrap_or_else(|e| e.into_inner());
        match variants.iter().find(|v| v.graphics.as_ref().is_some_and(|vg| vg.targets == targets)) {
            Some(v) => v.clone(),
            None => {
                let v = Arc::new(Pending::new(p.shaders.clone(), Some(Box::new(g.with_targets(targets)))));
                variants.push(v.clone());
                VARIANTS.fetch_add(1, Ordering::Relaxed);
                v
            }
        }
    };
    let made = ready(&variant)?;
    // SAFETY: the variant lives in `p.variants`, which only grows and lives
    // as long as `p`, so what it made outlives this borrow of `p`.
    Some(unsafe { &*std::ptr::from_ref(made) })
}

/// Pipelines made again for the formats of their bound targets, since the
/// start.
pub(crate) static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Pipelines still being made that draws no longer wait for: each has had
/// a whole frame's budget of waiting ([`wait`]).
static PASSED_OVER: AtomicU64 = AtomicU64::new(0);

/// [`PASSED_OVER`] now (for the frame line).
pub(crate) fn passed_over() -> u64 {
    PASSED_OVER.load(Ordering::Relaxed)
}

/// Waits for `p` while frame `frame` has budget left, unless `p` has had a
/// whole frame's budget of waiting already, over the frames before: then
/// its draws are dropped until it is made, without waiting (whether it is
/// made is looked at on every use all the same). A pipeline the driver
/// takes seconds for, or never finishes, cost the whole budget in every
/// frame that used it: fxceless/victory-fork issue #1181, four such
/// pipelines a whole session long, 500 ms waited in every frame of the
/// world (761 ms a frame against 252 with the budget at 50 ms).
fn wait(p: &Pending, frame: u64) {
    let budget = wait_budget();
    if budget.is_zero() || p.waited_us.load(Ordering::Relaxed) >= budget.as_micros() as u64 {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now).
    static SPENT: Mutex<(u64, Duration)> = Mutex::new((u64::MAX, Duration::ZERO));
    let shared = shared_budget();
    let left = {
        let mut spent = SPENT.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1);
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    let total = p.waited_us.fetch_add(waited.as_micros() as u64, Ordering::Relaxed) + waited.as_micros() as u64;
    if total >= budget.as_micros() as u64 && !p.passed_over.swap(true, Ordering::SeqCst) {
        PASSED_OVER.fetch_add(1, Ordering::SeqCst);
        // Made meanwhile: not passed over after all (whichever of this and
        // `Pending::make` takes the mark back counts it out).
        if p.made.get().is_some() && p.passed_over.swap(false, Ordering::SeqCst) {
            PASSED_OVER.fetch_sub(1, Ordering::SeqCst);
        }
    }
    {
        let mut spent = SPENT.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    /// Requested vertex/pixel graphics pipelines (including screen composition
    /// and tonemapping): compiled ahead of mesh/compute so the 3D image shows
    /// up quickly when the shader cache is cold.
    requested_graphics: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

/// Whether `p` is a traditional vertex graphics pipeline (as opposed to mesh
/// or compute).
fn is_vertex_graphics(p: &Pending) -> bool {
    p.shaders.iter().any(|s| s.stage == "vertex")
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested_graphics: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            if is_vertex_graphics(p) {
                self.requested_graphics.push_back(p.clone());
            } else {
                self.requested.push_back(p.clone());
            }
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    /// The next pipeline to make, and whether it is one started early only
    /// (no draw has asked for it yet): those only while fewer than
    /// `early_most` workers make such ones (counted in `early_making`).
    fn pop(&mut self, early_making: &AtomicU64, early_most: usize) -> Option<(Arc<Pending>, bool)> {
        if let Some(p) = self.requested_graphics.pop_front().or_else(|| self.requested.pop_front()) {
            p.queue_state.store(MAKING, Ordering::Relaxed);
            return Some((p, false));
        }
        if early_making.load(Ordering::Relaxed) >= early_most as u64 {
            return None;
        }
        // An early entry can have been promoted and claimed already.
        let p = std::iter::from_fn(|| self.early.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        early_making.fetch_add(1, Ordering::Relaxed);
        Some((p, true))
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        AVAILABLE.notify_one();
    }
}

fn next() -> (Arc<Pending>, bool) {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(&EARLY_MAKING, early_workers()) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Workers for `logical` logical processors: half of them, 2 to 8, unless
/// `setting` (`VICTORY_PIPELINE_WORKERS`) names a count; at most `logical`.
/// On 16 threads (empty cache, the intro): 8 made 7.5 pipelines/s while
/// some were waiting and the title spent 55 s below 1 fps; 14 made 17-41/s
/// and 6-12 s; 16 made 8.8/s and 66 s.
fn worker_count(setting: Option<&str>, logical: usize) -> usize {
    let default = (logical / 2).clamp(2, 8);
    setting
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map_or(default, |n| n.min(logical.max(1)))
}

/// The stack of a pipeline worker. The driver compiles a pipeline's shaders
/// on the thread that makes it, and NVIDIA's compiler (nvgpucomp64) recursed
/// past the default 2 MiB on the world's largest mesh programs: the stack
/// overflow ends the process, or the worker dies and the pipeline the title
/// waits for never appears (fxceless/victory-fork #28, which measured
/// 512 MiB enough; PainDe0Mie 59cb20ded3 and Marvisak/VIctory #1129 take
/// 64 MiB). Only reserved: pages are committed as the stack grows.
const WORKER_STACK: usize = 512 << 20;

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
        let setting = std::env::var("VICTORY_PIPELINE_WORKERS").ok();
        let workers = worker_count(setting.as_deref(), logical);
        tracing::info!(workers, logical, "host pipeline workers");
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .stack_size(WORKER_STACK)
                .spawn(|| {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        let (p, early) = next();
                        p.make();
                        if early {
                            // Another worker may take the next early one
                            // (under the queue's lock: no worker misses it).
                            let _queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            EARLY_MAKING.fetch_sub(1, Ordering::Relaxed);
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// Workers that may make pipelines started early at once
/// (`VICTORY_PIPELINE_EARLY_WORKERS`; by default an eighth of the logical
/// processors, at least one). The early list (the pipelines the run before
/// drew with: 3,327 of them) is made before anything asks for them; where
/// the library does not have them, each is a compile of the driver's, at
/// full tilt on the worker's thread and on threads of the driver's own.
/// With all eight workers on the early list, eleven of the sixteen logical
/// processors were in the driver's compiler for minutes, and the title's
/// frames and texture refreshes (which read, hash and detile hundreds of
/// MB) got what was left: a run 2026-11-26 went from 30 fps to 0.1-3.9 in
/// its loading screens, texture refreshes from 0.3-12 ms to 234 ms - 12.7 s
/// a frame, and never left them in four minutes. The workers hold no lock
/// of ours while the driver compiles (`library::pipeline` compiles outside
/// its locks; a save holds the library's own, which no frame takes): the
/// frame wanted processors. Pipelines a draw asks for take every worker.
fn early_workers() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
        let most = std::env::var("VICTORY_PIPELINE_EARLY_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or((logical / 8).max(1));
        tracing::info!(most, "host pipeline workers that may make pipelines started early");
        most
    })
}

/// Workers making a pipeline started early now ([`early_workers`]).
static EARLY_MAKING: AtomicU64 = AtomicU64::new(0);

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            from_other_translations = super::library::SEEDED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dispatch's pipeline is made before its first use returns, whoever
    /// has it: a worker that took it and has not begun does not make the
    /// dispatch go without (here: no host pipeline for no shaders, but the
    /// answer is there).
    #[test]
    fn a_compute_pipeline_is_made_before_its_first_use_returns() {
        let p = Pending::new(Vec::new(), None);
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);
        assert!(ready(&p).is_none());
        assert!(matches!(p.made.get(), Some(None)), "made on the calling thread");
    }

    /// A draw's first use queues the pipeline and, within the budget, gets
    /// the worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), Some(Box::default())));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline that is never made is waited for one frame's budget in
    /// all: once that is spent, later frames do not wait for it again, and
    /// it counts as passed over until it is made (fxceless #1181).
    #[test]
    fn a_pipeline_never_made_is_waited_for_once() {
        let p = Pending::new(Vec::new(), None);
        // A worker has taken it and does not finish.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let budget = wait_budget();
        if budget.is_zero() {
            return;
        }
        // Frames of the test's own (the budget is shared per frame).
        let mut frame = u64::MAX / 2;
        while !p.passed_over.load(Ordering::SeqCst) {
            assert!(frame < u64::MAX / 2 + 64, "the pipeline was never passed over");
            wait(&p, frame);
            frame += 1;
        }
        assert!(passed_over() >= 1);
        let started = Instant::now();
        wait(&p, frame);
        assert!(started.elapsed() < budget / 4, "a passed-over pipeline was waited for again");
        // Made at last: counted out.
        p.make();
        assert!(!p.passed_over.load(Ordering::SeqCst));
        assert!(p.made.get().is_some());
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        // (A draw's: a dispatch's would be made by its first use.)
        let mut p = Pending::new(Vec::new(), Some(Box::default()));
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    /// A pipeline is made again once for each set of target formats it is
    /// drawn into, with those formats; one without graphics state never.
    #[test]
    fn each_set_of_target_formats_is_made_once() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RT_FORMAT_ARRAY;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let p = Pending::new(Vec::new(), Some(Box::default()));
        let formats = |first: i32| {
            let mut t = D3D12_RT_FORMAT_ARRAY {
                NumRenderTargets: 1,
                ..Default::default()
            };
            t.RTFormats[0] = DXGI_FORMAT(first);
            t
        };
        // (No shaders: no host pipeline, whatever the formats.)
        assert!(for_targets(&p, formats(29)).is_none());
        assert!(for_targets(&p, formats(29)).is_none());
        assert!(for_targets(&p, formats(26)).is_none());
        let variants = p.variants.lock().unwrap();
        let made: Vec<i32> = variants.iter().map(|v| v.graphics.as_ref().unwrap().targets.RTFormats[0].0).collect();
        assert_eq!(made, [29, 26]);
        drop(variants);
        let compute = Pending::new(Vec::new(), None);
        assert!(for_targets(&compute, formats(29)).is_none());
        assert!(compute.variants.lock().unwrap().is_empty());
    }

    #[test]
    fn worker_count_is_half_the_processors_unless_set() {
        assert_eq!(worker_count(None, 16), 8);
        assert_eq!(worker_count(None, 32), 8);
        assert_eq!(worker_count(None, 2), 2);
        assert_eq!(worker_count(Some("14"), 16), 14);
        assert_eq!(worker_count(Some(" 12 "), 16), 12);
        assert_eq!(worker_count(Some("64"), 16), 16);
        assert_eq!(worker_count(Some("0"), 16), 8);
        assert_eq!(worker_count(Some("many"), 16), 8);
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        let making = AtomicU64::new(0);
        let (p, early) = queue.pop(&making, 8).unwrap();
        assert!(Arc::ptr_eq(&p, &requested) && !early);
        let (p, early) = queue.pop(&making, 8).unwrap();
        assert!(Arc::ptr_eq(&p, &promoted) && !early);
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        let (p, early) = queue.pop(&making, 8).unwrap();
        assert!(Arc::ptr_eq(&p, &earlier) && early);
        assert!(queue.pop(&making, 8).is_none());
    }

    /// Pipelines started early are given to at most so many workers at
    /// once; one a draw asks for is given at any time.
    #[test]
    fn early_pipelines_take_only_some_workers() {
        let mut queue = Queue::new();
        let early: Vec<Arc<Pending>> = (0..3).map(|_| Arc::new(Pending::new(Vec::new(), None))).collect();
        for p in &early {
            queue.push(p, EARLY);
        }
        let making = AtomicU64::new(0);
        assert!(queue.pop(&making, 2).is_some_and(|(_, e)| e));
        assert!(queue.pop(&making, 2).is_some_and(|(_, e)| e));
        assert!(queue.pop(&making, 2).is_none(), "two workers make early ones already");
        let asked = Arc::new(Pending::new(Vec::new(), None));
        queue.push(&asked, REQUESTED);
        let (p, e) = queue.pop(&making, 2).unwrap();
        assert!(Arc::ptr_eq(&p, &asked) && !e);
        making.fetch_sub(1, Ordering::Relaxed);
        assert!(queue.pop(&making, 2).is_some_and(|(_, e)| e), "one done: the next early one");
    }
}
