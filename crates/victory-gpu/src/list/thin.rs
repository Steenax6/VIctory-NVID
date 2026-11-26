//! List calls that only write down their arguments (`VICTORY_THIN=1`, off by
//! default); what they mean is worked out by threads of ours, beside the
//! title's.
//!
//! Why: the title makes some 41,000 list calls a frame in the first room, and
//! each was translated inside the call: 154 to 170 ms of processor time a
//! frame on the title's five recording threads, 144 of it in
//! `draw::State::bind`. The title records its frame as jobs, and after a
//! first phase of parallel jobs one job records a chain of 19 lists on one
//! thread for 36 to 50 ms; the render thread waits 22 to 42 ms for exactly
//! that job before it submits and presents (`VICTORY_TIMELINE`, 2026-11-26).
//! So the title's own scheduling of its jobs bounded the frame by our time
//! in its calls. On the console such a call writes a few words into the
//! list's memory and the card reads them when the list runs. So it is here
//! now: a call appends its arguments (and copies of what D3D12 reads at the
//! call) to a stream of words; threads of ours run the stream, call by call
//! and in order, through the list methods there were before, on a list of
//! ours ([`Recording::back`]) that the title never sees. They begin while
//! the title still records (every [`CHUNK`] words are handed over), several
//! lists at a time. A submission goes into its queue's line
//! (`crate::queue`, `Held::Lists`) by the recordings' names; when it comes
//! to the head of the line and they are translated, it is cut at its waits
//! as before (`super::detach_parts` on the lists of ours) and sent.
//!
//! Ported from avatarmanz/VIctory-7900XTX `crates/victory-gpu/src/list/thin.rs`
//! (the stream, the chunks handed over, the threads, the copies at the call)
//! and adapted to this tree: its line of submissions is ours (`queue::Held`),
//! upload memory stays the title list's and goes with the recording at its
//! close, a recording is begun at the list's creation, objects the title
//! releases are freed only once the recordings that may name them are
//! translated ([`bury`]), and the present and our waited submissions wait for
//! every submitted recording first ([`settle`]).
//!
//! What may be looked at later than before, call by call, and why that is
//! what the console's card sees (or what is copied at the call):
//!
//! - Plain values (topology, stencil reference, depth bounds, root constants
//!   one by one, draw, dispatch and copy arguments, addresses of root views,
//!   GPU handles of descriptor tables, fence and query arguments): passed by
//!   value, kept in the stream.
//! - Arrays and structures passed by pointer (barriers, viewports, scissors,
//!   blend factor, descriptor heaps, root constants, index and vertex buffer
//!   views, copy locations and the box, clear colours and rectangles): copied
//!   at the call, as D3D12 reads them there.
//! - Render target and depth views (`OMSetRenderTargets`, the clears): CPU
//!   descriptors, read at the call in D3D12 and written again right after by
//!   the title (it reuses its slots); their records are copied at the call,
//!   and the copy stands for the handle (`descriptor::rtv_record` reads the
//!   record by value).
//! - Root constant, shader resource and unordered access views by address:
//!   no path reads their bytes at the draw. `draw::root_view`, `root_uav`
//!   and `root_srv_table` bind the buffer's host copy and refresh it when the
//!   list is submitted (a constant buffer not on 256 bytes is filled into
//!   upload memory then); `crate::memory::ready` only names the address. So
//!   the bytes are the same as before whenever they change in between.
//! - Descriptor tables in shader-visible heaps (their records are read at
//!   the draw: the views `draw::State::bind` makes, the key of a fresh
//!   table), indirect argument and count buffers, upload memory the list
//!   handed out: the console's card reads all of it when the list runs,
//!   after the submission. The title may not change it before its fence
//!   behind the list says the list has run, and no such fence is set before
//!   the list is translated and sent: reading it at translation is between
//!   the call (where the old path read it) and the card's read. A title that
//!   changed it between call and submission would be wrong on the console
//!   and was read too early before. Diagnostics: `VICTORY_THIN_CHECK=1`
//!   compares, for every draw and dispatch, the first 32 bytes of each bound
//!   table at the call and at translation; a change is a draw the path in
//!   the call would have made otherwise (the line "thin check"; root views
//!   are counted apart, as no difference).
//! - Waits for memory values and bottom-of-pipe writes: they cut and order
//!   the list of ours as they did the title's (`super::cut`); the values are
//!   looked at when the queue comes to them, as before.
//! - Query resolves and read-backs: written when the list has run, as before.
//! - Texture uploads, host copies of title memory, the write watch: decided
//!   when the submission is prepared, after translation, as before.
//! - Objects named (pipelines, resources, signatures, heaps): must live until
//!   the list has run. A placed resource may be released by the title after a
//!   view of it went into a heap and before the list runs (the console's card
//!   reads the view, not the object), so a title object whose last reference
//!   goes while recordings are not translated is freed only once every
//!   recording begun before is ([`bury`]).
//!
//! What the title does with its lists: a recording reset or released without
//! being submitted is not run any further (only its reset and close are);
//! one submitted twice without a reset is translated once and the second
//! submission finds its list of ours taken, as a title list's host side was
//! (`crate::resubmit::ran_empty`); a list closed and submitted while it is
//! still translated stands in its queue's line until it is, and the queues'
//! other lines go on.

use super::*;
use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::descriptor::RtvRecord;
use crate::draw::{IndexView, VertexView};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

/// Words after which a stream is handed to the translating threads.
const CHUNK: usize = 1024;

/// Whether list calls are only written down (`VICTORY_THIN=1`; needs
/// submissions sent by our thread, `VICTORY_ASYNC_SUBMIT=1`, whose line the
/// recordings wait in).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !wanted() || crate::host::host().is_none() {
            return false;
        }
        if !crate::queue::behind() {
            tracing::warn!("VICTORY_THIN=1 needs VICTORY_ASYNC_SUBMIT=1 (and its flights and cut lists): list calls are translated in the call");
            return false;
        }
        start();
        true
    })
}

/// Whether it is asked for.
pub(crate) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THIN").is_some_and(|v| v == "1"))
}

/// What a title list keeps: the recording being written and the words of it
/// not handed over yet.
#[derive(Default)]
pub(super) struct Front {
    recording: Option<Arc<Recording>>,
    words: Vec<u64>,
    closed: bool,
    submitted: bool,
    /// Diagnostics (`VICTORY_THIN_CHECK=1`): the tables and root views set,
    /// by compute and slot.
    #[cfg(feature = "diagnostics")]
    roots: [[u64; ROOTS]; 2],
}

impl Drop for Front {
    fn drop(&mut self) {
        self.leave();
    }
}

impl Front {
    /// Lets go of the recording there is: one never submitted is not run any
    /// further.
    fn leave(&mut self) {
        if let Some(recording) = self.recording.take()
            && !self.submitted
        {
            recording.abandoned.store(true, Release);
        }
        self.words.clear();
    }

    /// Begins a recording of title list `title`, with `pipeline` bound.
    pub(super) fn begin(&mut self, kind: u32, pipeline: This, title: usize) {
        sweep();
        self.leave();
        self.recording = Some(Arc::new(Recording::new(kind, title)));
        self.closed = false;
        self.submitted = false;
        #[cfg(feature = "diagnostics")]
        {
            self.roots = [[0; ROOTS]; 2];
        }
        self.put(RESET, &[pipeline as u64]);
    }

    #[inline]
    fn put(&mut self, op: u64, args: &[u64]) {
        if self.closed || self.recording.is_none() {
            return;
        }
        self.words.push(op | (args.len() as u64) << 16);
        self.words.extend_from_slice(args);
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    /// [`Self::put`] of some words and then the bytes of some things.
    fn put_with(&mut self, op: u64, args: &[u64], blobs: &[&[u8]]) {
        if self.closed || self.recording.is_none() {
            return;
        }
        let at = self.words.len();
        self.words.push(0);
        self.words.extend_from_slice(args);
        for blob in blobs {
            let from = self.words.len();
            self.words.resize(from + blob.len().div_ceil(8), 0);
            // SAFETY: the words just made hold the bytes; plain data.
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), self.words.as_mut_ptr().add(from).cast(), blob.len())
            };
        }
        let len = (self.words.len() - at - 1) as u64;
        self.words[at] = op | len << 16;
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    fn hand_over(&mut self) {
        let Some(recording) = &self.recording else { return };
        if self.words.is_empty() {
            return;
        }
        let fresh = held(&WORDS).pop().unwrap_or_else(|| Vec::with_capacity(CHUNK + 256));
        let words = std::mem::replace(&mut self.words, fresh);
        CHUNKS.fetch_add(1, Relaxed);
        let schedule = {
            let mut inner = held(&recording.inner);
            inner.chunks.push_back(words);
            !std::mem::replace(&mut inner.scheduled, true)
        };
        if schedule {
            held(&READY).push_back(recording.clone());
            WORK.notify_one();
        }
    }

    /// Ends the recording: `uploads` (what the title list handed out of its
    /// upload memory) go with it, to its list of ours.
    fn close(&mut self, uploads: &Mutex<Uploads>) {
        if self.closed {
            return;
        }
        if let Some(recording) = &self.recording {
            let taken = std::mem::take(&mut *held(uploads));
            held(&recording.inner).uploads = Some(taken);
        }
        self.put(CLOSE, &[]);
        self.closed = true;
        self.hand_over();
    }

    /// Diagnostics: a draw or dispatch is written down; with
    /// `VICTORY_THIN_CHECK=1` the first bytes of what it reads by table and
    /// root view are hashed now, to compare at translation.
    #[inline]
    fn work(&mut self, compute: bool) {
        #[cfg(feature = "diagnostics")]
        if check() {
            let mut w = Vec::new();
            for &address in self.roots[usize::from(compute)].iter().filter(|&&a| a != 0) {
                w.push(address);
                w.extend_from_slice(&words_at(address & !TABLE));
            }
            if !w.is_empty() {
                self.put(CHECK, &w);
            }
        }
        let _ = compute;
    }

    /// Diagnostics: a table (`table`) or root view is set (the check's
    /// addresses).
    #[inline]
    fn root(&mut self, compute: bool, slot: u32, address: u64, table: bool) {
        #[cfg(feature = "diagnostics")]
        if check()
            && let Some(at) = self.roots[usize::from(compute)].get_mut(slot as usize)
        {
            *at = if address == 0 { 0 } else if table { address | TABLE } else { address };
        }
        let _ = (compute, slot, address, table);
    }
}

/// Streams' memory to use again.
static WORDS: Mutex<Vec<Vec<u64>>> = Mutex::new(Vec::new());
/// Recordings with words to translate.
static READY: Mutex<VecDeque<Arc<Recording>>> = Mutex::new(VecDeque::new());
static WORK: Condvar = Condvar::new();
/// Lists of ours nobody uses, closed.
static BACKS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Since the last line of the log: streams handed over.
static CHUNKS: AtomicU64 = AtomicU64::new(0);
/// Since the last line: calls translated and the microseconds that took (all
/// threads), recordings translated, recordings let go of in the middle,
/// objects whose freeing waited for translation and those freed past a
/// recording still open, the microseconds a present or a waited submission
/// of ours stood for translation; diagnostics: draws checked, those whose
/// memory changed between call and translation, and the tables and root
/// views that changed.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
const S_CALLS: usize = 0;
const S_MICROS: usize = 1;
const S_LISTS: usize = 2;
const S_LEFT: usize = 3;
const S_BURIED: usize = 4;
const S_PAST_OPEN: usize = 5;
const S_SETTLE: usize = 6;
#[cfg(feature = "diagnostics")]
const S_CHECKED: usize = 7;
#[cfg(feature = "diagnostics")]
const S_CHANGED: usize = 8;
#[cfg(feature = "diagnostics")]
const S_CHANGED_TABLES: usize = 9;
#[cfg(feature = "diagnostics")]
const S_CHANGED_VIEWS: usize = 10;

fn held<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The recordings begun and not translated to their end (nor let go of), by
/// number: what a title object released now may still be named by.
static ACTIVE: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
/// The number of the next recording.
static NEXT: AtomicU64 = AtomicU64::new(0);

/// Recordings submitted and not translated yet, and what tells that one was.
static PENDING: (Mutex<u64>, Condvar) = (Mutex::new(0), Condvar::new());

/// One recording of a title list: what was written of it and not translated
/// yet, and the list of ours it goes to.
pub(crate) struct Recording {
    kind: u32,
    /// The title's list (diagnostics: named in the frame's ops).
    title: usize,
    number: u64,
    inner: Mutex<Inner>,
    /// Translated to its end.
    done: AtomicBool,
    moved: Condvar,
    abandoned: AtomicBool,
}

#[derive(Default)]
struct Inner {
    chunks: VecDeque<Vec<u64>>,
    /// Streams translated (commands may name places in them until the end).
    kept: Vec<Vec<u64>>,
    /// It is among [`READY`] or being translated.
    scheduled: bool,
    /// The list of ours (0: none yet).
    back: usize,
    /// That list was closed by the stream.
    closed: bool,
    /// The title list's upload memory, given at the close.
    uploads: Option<Uploads>,
    /// Counted in [`PENDING`].
    pending: bool,
}

impl Recording {
    fn new(kind: u32, title: usize) -> Self {
        let number = NEXT.fetch_add(1, Relaxed);
        held(&ACTIVE).insert(number);
        Recording {
            kind,
            title,
            number,
            inner: Mutex::default(),
            done: AtomicBool::new(false),
            moved: Condvar::new(),
            abandoned: AtomicBool::new(false),
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.done.load(Acquire)
    }

    /// Waits until it is translated.
    pub(crate) fn wait(&self) {
        let mut inner = held(&self.inner);
        while !self.done() {
            inner = self.moved.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The list of ours it went to (null: it was never begun).
    pub(crate) fn back(&self) -> This {
        held(&self.inner).back as This
    }

    /// The title's list it is a recording of.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn title(&self) -> usize {
        self.title
    }

    /// Translates what there is.
    fn run(&self) {
        loop {
            let (chunk, back) = {
                let mut inner = held(&self.inner);
                let Some(chunk) = inner.chunks.pop_front() else {
                    inner.scheduled = false;
                    return;
                };
                if inner.back == 0 {
                    inner.back = back_list(self.kind) as usize;
                }
                (chunk, inner.back as This)
            };
            let began = Instant::now();
            let (calls, closed) = {
                let _span = crate::timeline::span("thin>", "thin<", self.title as u64, self.number);
                translate(back, &chunk, || self.abandoned.load(Acquire))
            };
            STATS[S_CALLS].fetch_add(calls, Relaxed);
            STATS[S_MICROS].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
            let mut inner = held(&self.inner);
            inner.kept.push(chunk);
            if closed {
                inner.closed = true;
                // The upload memory goes with the list of ours, which a
                // submission takes it from (`super::detach_parts`).
                if let Some(uploads) = inner.uploads.take() {
                    // SAFETY: a list of ours, which only this thread uses now.
                    *held(unsafe { &(*back.cast::<List>()).uploads }) = uploads;
                }
                let kept = std::mem::take(&mut inner.kept);
                self.done.store(true, Release);
                if std::mem::take(&mut inner.pending) {
                    let mut pending = held(&PENDING.0);
                    *pending = pending.saturating_sub(1);
                    PENDING.1.notify_all();
                }
                self.moved.notify_all();
                drop(inner);
                held(&ACTIVE).remove(&self.number);
                give_back(kept);
                STATS[S_LISTS].fetch_add(1, Relaxed);
                crate::queue::thin_moved();
            }
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        held(&ACTIVE).remove(&self.number);
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        give_back(inner.chunks.drain(..).chain(std::mem::take(&mut inner.kept)).collect());
        if inner.back != 0 {
            if !inner.closed {
                // Left in the middle.
                STATS[S_LEFT].fetch_add(1, Relaxed);
                super::close(inner.back as This);
            }
            held(&BACKS).push(inner.back);
        }
    }
}

fn give_back(streams: Vec<Vec<u64>>) {
    let mut words = held(&WORDS);
    for mut s in streams {
        if words.len() < 1024 && s.capacity() <= 16 * CHUNK {
            s.clear();
            words.push(s);
        }
    }
}

/// A list of ours, closed.
fn back_list(kind: u32) -> This {
    if let Some(back) = held(&BACKS).pop() {
        // SAFETY: a list of ours nobody uses.
        unsafe { (*(back as *mut List)).kind = kind };
        return back as This;
    }
    let list: This = Box::into_raw(super::boxed(kind, std::ptr::null(), true)).cast_const().cast();
    super::close(list);
    list
}

thread_local! {
    /// This thread translates lists.
    static TRANSLATES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// How many threads translate (`VICTORY_THIN_THREADS`, 6 unless told).
///
/// The processor has 8 cores and 16 threads. Busy beside them: the title's
/// render thread and five job threads (with this switch they mostly write
/// words), our submitting thread, the landing thread and four helpers, at
/// times. The translation is the 150 ms or so a frame that the job threads
/// spent in the calls before, so at least four threads are needed to keep
/// up with a frame of 40 ms; six leave room for the others and translate
/// six lists at once, as many as the title's jobs record at the start of a
/// frame. They run at the normal priority, as the title's job threads do:
/// above it they would stand before the threads that write the streams
/// they translate, below it behind the title's streaming.
fn threads() -> usize {
    std::env::var("VICTORY_THIN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(6usize).clamp(1, 16)
}

/// The translating threads' priority (`VICTORY_THIN_PRIORITY`, a Windows
/// thread priority from -2 to 2; 0, normal, unless told).
fn priority() -> i32 {
    std::env::var("VICTORY_THIN_PRIORITY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0i32).clamp(-2, 2)
}

fn start() {
    let threads = threads();
    let priority = priority();
    for i in 0..threads {
        let worker = move || {
            if priority != 0 {
                use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
                // SAFETY: this thread's own priority.
                unsafe { SetThreadPriority(GetCurrentThread(), priority) };
            }
            TRANSLATES.set(true);
            crate::queue::no_drain_here();
            loop {
                let recording = {
                    let mut ready = held(&READY);
                    loop {
                        match ready.pop_front() {
                            Some(r) => break r,
                            None => ready = WORK.wait(ready).unwrap_or_else(|e| e.into_inner()),
                        }
                    }
                };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recording.run())).is_err() {
                    // Its submission would stand in its line for good.
                    crate::host::fail(format_args!("a thread that translates lists panicked"));
                }
            }
        };
        if let Err(e) = std::thread::Builder::new().name(format!("victory-thin-{i}")).spawn(worker) {
            tracing::warn!(error = %e, "a thread less to translate lists");
        }
    }
    BURIES.store(true, Release);
    tracing::info!(threads, priority, "list calls are written down and translated by threads of ours (VICTORY_THIN=1)");
}

/// Closes the recordings of submitted `lists` and gives them, in order (a
/// list never begun is left out); each counts as pending until translated.
pub(crate) fn submitted(lists: &[This]) -> Arc<[Arc<Recording>]> {
    sweep();
    lists
        .iter()
        .filter(|l| !l.is_null())
        .filter_map(|&l| {
            // SAFETY: the title passes its lists, which nobody records into now.
            let list = unsafe { &mut *l.cast_mut().cast::<List>() };
            list.thin.close(&list.uploads);
            list.thin.submitted = true;
            let recording = list.thin.recording.clone()?;
            {
                let mut inner = held(&recording.inner);
                if !recording.done() && !inner.pending {
                    inner.pending = true;
                    *held(&PENDING.0) += 1;
                }
            }
            Some(recording)
        })
        .collect()
}

/// Whether all of `recordings` are translated.
pub(crate) fn translated(recordings: &[Arc<Recording>]) -> bool {
    recordings.iter().all(|r| r.done())
}

/// Waits until all of `recordings` are translated.
pub(crate) fn wait(recordings: &[Arc<Recording>]) {
    let began = Instant::now();
    recordings.iter().for_each(|r| r.wait());
    STATS[S_SETTLE].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
}

/// The lists of ours `recordings` went to, in order.
pub(crate) fn backs(recordings: &[Arc<Recording>]) -> Vec<This> {
    recordings.iter().map(|r| r.back()).filter(|b| !b.is_null()).collect()
}

/// Before something that has to see every submission made so far in its
/// line, translated (a present, a submission of ours that is waited for):
/// waits until every submitted recording is. Not on a thread that
/// translates.
pub(crate) fn settle() {
    if !wanted() || !on() || TRANSLATES.get() {
        return;
    }
    let began = Instant::now();
    let mut pending = held(&PENDING.0);
    if *pending == 0 {
        return;
    }
    let _span = crate::timeline::span("thinwait>", "thinwait<", *pending, 0);
    while *pending > 0 {
        pending = PENDING.1.wait(pending).unwrap_or_else(|e| e.into_inner());
    }
    drop(pending);
    STATS[S_SETTLE].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
}

/// Whether title objects released now are freed later ([`bury`]).
static BURIES: AtomicBool = AtomicBool::new(false);

/// A title object whose last reference went, to be freed later.
struct Buried(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: freed by a title thread, as it would have been at its release;
// nothing else touches it meanwhile (the title let go of it).
unsafe impl Send for Buried {}

/// Released title objects, with the number of the next recording at their
/// release and when.
static BURIED: Mutex<VecDeque<(u64, Instant, Buried)>> = Mutex::new(VecDeque::new());

/// How long a released object waits at most for a recording still open.
const BURIED_MOST: Duration = Duration::from_secs(2);

/// Whether a released title object is to be given to [`bury`].
#[inline]
pub(crate) fn buries() -> bool {
    BURIES.load(Relaxed)
}

/// A title object whose last reference went (`object::release_boxed`):
/// freed once every recording begun before is translated or let go of, at
/// the latest after [`BURIED_MOST`] (counted: a recording held open that
/// long). The title may release an object whose view went into a heap, or
/// that a command names, before the list runs: on the console the card
/// reads the view, here a translation reads the object.
pub(crate) fn bury<T: 'static>(object: Box<T>) {
    STATS[S_BURIED].fetch_add(1, Relaxed);
    held(&BURIED).push_back((NEXT.load(Relaxed), Instant::now(), Buried(object)));
    if !TRANSLATES.get() {
        sweep();
    }
}

/// Frees the released objects no recording may name any more. On the
/// title's threads only: freeing a resource hands its host copy to
/// `crate::host::retire`.
fn sweep() {
    if !buries() {
        return;
    }
    let first = held(&ACTIVE).first().copied().unwrap_or(u64::MAX);
    let mut free = Vec::new();
    {
        let mut buried = held(&BURIED);
        while let Some((before, at, _)) = buried.front() {
            let past = at.elapsed() >= BURIED_MOST;
            if *before > first && !past {
                break;
            }
            if past && *before > first {
                STATS[S_PAST_OPEN].fetch_add(1, Relaxed);
            }
            if let Some(entry) = buried.pop_front() {
                free.push(entry);
            }
        }
    }
    // (Freed with the lock let go: a drop may release more.)
    drop(free);
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !wanted() || !on() {
        return;
    }
    sweep();
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64 / frames).collect();
    tracing::info!(
        "list calls translated by threads of ours per frame (VICTORY_THIN): {:.0} in {:.1} ms (all threads), {:.1} lists, {:.1} pieces; \
         left in the middle {:.2}; title objects freed after translation {:.1}, past a recording held open {:.2}; presents and waited \
         submissions stood {:.2} ms for translation",
        s[S_CALLS],
        s[S_MICROS] / 1000.0,
        s[S_LISTS],
        CHUNKS.swap(0, Relaxed) as f64 / frames,
        s[S_LEFT],
        s[S_BURIED],
        s[S_PAST_OPEN],
        s[S_SETTLE] / 1000.0,
    );
    #[cfg(feature = "diagnostics")]
    if check() {
        tracing::info!(
            "thin check (VICTORY_THIN_CHECK=1), per frame: {:.0} draws and dispatches checked, {:.2} of them would have drawn otherwise \
             in the call (a table's first descriptor changed between the call and the translation: {:.2} tables); root views changed \
             {:.2} (read by both paths at the submission: no difference)",
            s[S_CHECKED],
            s[S_CHANGED],
            s[S_CHANGED_TABLES],
            s[S_CHANGED_VIEWS]
        );
    }
}

/// Diagnostics: whether draws compare what they read at call and
/// translation (`VICTORY_THIN_CHECK=1`).
#[cfg(feature = "diagnostics")]
fn check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THIN_CHECK").is_some_and(|v| v == "1"))
}

/// Root slots looked at by the check.
#[cfg(feature = "diagnostics")]
const ROOTS: usize = 16;

/// The check's mark of a table among its addresses (a root view has none).
#[cfg(feature = "diagnostics")]
const TABLE: u64 = 1 << 63;

/// The first 32 bytes at `address` (zeros if they cannot be read): the
/// first descriptor of a table, the first two vectors of constants of a
/// view; what the draw reads for certain (the extent of a table is its
/// root signature's, and the bytes after it are the next draw's in a ring).
#[cfg(feature = "diagnostics")]
fn words_at(address: u64) -> [u64; 4] {
    let mut words = [0u64; 4];
    // SAFETY: the words are 32 plain bytes of ours.
    let bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), 32) };
    if !crate::host::read_into(address as *const u8, bytes) {
        return [0; 4];
    }
    words
}

#[inline]
fn front<'a>(this: This) -> &'a mut Front {
    // SAFETY: the title passes one of our lists; a list is recorded by one
    // thread at a time.
    unsafe { &mut (*this.cast_mut().cast::<List>()).thin }
}

/// An argument as a word.
trait Word: Copy {
    fn word(self) -> u64;
    fn unword(w: u64) -> Self;
}
macro_rules! words {
    ($($t:ty),*) => { $(impl Word for $t {
        #[inline]
        fn word(self) -> u64 { self as u64 }
        #[inline]
        fn unword(w: u64) -> Self { w as $t }
    })* };
}
words!(u8, u32, i32, u64, This);
impl Word for f32 {
    #[inline]
    fn word(self) -> u64 {
        u64::from(self.to_bits())
    }
    #[inline]
    fn unword(w: u64) -> Self {
        f32::from_bits(w as u32)
    }
}

/// The methods whose arguments are all plain values: the one the title calls,
/// and the call made of it later. What follows `=>` is done on the title
/// list's front first (draws and dispatches, tables and root views: the
/// diagnostics check).
macro_rules! plain {
    ($($op:ident $name:ident = $was:path, ($($a:ident: $t:ty),*) $(=> $hook:expr)?;)*) => {
        #[allow(clippy::upper_case_acronyms, non_camel_case_types)]
        #[repr(u64)]
        enum Plain { $($op),* , END }
        $(extern "system" fn $name(this: This $(, $a: $t)*) {
            let front = front(this);
            $(($hook)(&mut *front);)?
            front.put(Plain::$op as u64, &[$(Word::word($a)),*]);
        })*
        /// Whether `op` was one of them; done on `list`.
        #[inline]
        fn run_plain(op: u64, list: This, w: &[u64]) -> bool {
            match op {
                $(n if n == Plain::$op as u64 => {
                    let mut words = w.iter();
                    $(let $a: $t = Word::unword(words.next().copied().unwrap_or(0));)*
                    let _ = words;
                    $was(list $(, $a)*);
                })*
                _ => return false,
            }
            true
        }
    };
}

plain! {
    TOPOLOGY topology = state::topology, (v: u32);
    STENCIL_REF stencil_ref = state::stencil_ref, (v: u32);
    DEPTH_BOUNDS depth_bounds = state::depth_bounds, (min: f32, max: f32);
    COMPUTE_TABLE compute_table = state::compute_table, (slot: u32, handle: u64) => |f: &mut Front| f.root(true, slot, handle, true);
    GRAPHICS_TABLE graphics_table = state::graphics_table, (slot: u32, handle: u64) => |f: &mut Front| f.root(false, slot, handle, true);
    COMPUTE_CONSTANT compute_constant = state::compute_constant, (slot: u32, value: u32, offset: u32);
    GRAPHICS_CONSTANT graphics_constant = state::graphics_constant, (slot: u32, value: u32, offset: u32);
    COMPUTE_VIEW compute_view = state::compute_view, (slot: u32, address: u64) => |f: &mut Front| f.root(true, slot, address, false);
    GRAPHICS_VIEW graphics_view = state::graphics_view, (slot: u32, address: u64) => |f: &mut Front| f.root(false, slot, address, false);
    SET_PIPELINE set_pipeline = super::set_pipeline, (pipeline: This);
    DRAW draw = state::draw, (vertices: u32, instances: u32, first: u32, first_instance: u32) => |f: &mut Front| f.work(false);
    DRAW_INDEXED draw_indexed = state::draw_indexed, (indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) => |f: &mut Front| f.work(false);
    DISPATCH dispatch = state::dispatch, (x: u32, y: u32, z: u32) => |f: &mut Front| f.work(true);
    DISPATCH_X dispatch_x = super::dispatch_x, (x: u32, y: u32, z: u32, flags: u32) => |f: &mut Front| f.work(true);
    DISPATCH_MESH dispatch_mesh = state::dispatch_mesh, (x: u32, y: u32, z: u32) => |f: &mut Front| f.work(false);
    DISPATCH_MESH_X dispatch_mesh_x = state::dispatch_mesh_x, (x: u32, flags: u64, indirect: u64, payload: u64) => |f: &mut Front| f.work(false);
    COPY_BUFFER copy_buffer_region = super::copy_buffer_region, (dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64);
    COPY_RESOURCE copy_resource = super::copy_resource, (dst: This, src: This);
    COPY_RESOURCE_X copy_resource_x = super::copy_resource_x, (dst: This, src: This, flags: u32);
    COPY_MEMORY copy_memory = super::copy_memory, (dst: u64, src: u64, len: u64, flags: u32);
    RESOLVE resolve = super::resolve, (dst: This, dst_sub: u32, src: This, src_sub: u32, format: u32);
    RESOLVE_QUERIES resolve_queries = super::resolve_queries, (heap: This, kind: u32, first: u32, count: u32, buffer: This, offset: u64);
    EXECUTE_INDIRECT execute_indirect = super::execute_indirect, (signature: This, max: u32, arguments: This, offset: u64, count: This, count_offset: u64) => |f: &mut Front| { f.work(false); f.work(true) };
    WRITE32 write32 = super::write32, (address: u64, value: u32, flush: u32, flags: u32);
    WRITE64 write64 = super::write64, (address: u64, value: u64, flush: u32, flags: u32);
    WAIT_VALUE wait_value = super::wait_value, (address: u64, comparison: u32, reference: u64);
    WAIT_VALUE64 wait_value64 = super::wait_value64, (address: u64, comparison: u32, reference: u64);
    SYNC sync = super::sync, (flags: u32, control: u32, address: u64, size: u64);
}

const fn after(n: u64) -> u64 {
    Plain::END as u64 + n
}
const RESET: u64 = after(0);
const CLOSE: u64 = after(1);
const BARRIER: u64 = after(3);
const BARRIER_GROUPS: u64 = after(4);
const VIEWPORTS: u64 = after(5);
const SCISSORS: u64 = after(6);
const BLEND_FACTOR: u64 = after(7);
const HEAPS: u64 = after(8);
const COMPUTE_CONSTANTS: u64 = after(9);
const GRAPHICS_CONSTANTS: u64 = after(10);
const INDEX_BUFFER: u64 = after(11);
const VERTEX_BUFFERS: u64 = after(12);
const TARGETS: u64 = after(13);
const CLEAR_RTV: u64 = after(14);
const CLEAR_DSV: u64 = after(15);
const COPY_TEXTURE: u64 = after(16);
/// Diagnostics: (address, hash) pairs of what the next draw reads.
#[cfg(feature = "diagnostics")]
const CHECK: u64 = after(17);

/// The bytes of `count` things at `items` (none if null).
fn bytes_of<'a, T>(items: *const T, count: usize) -> &'a [u8] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: the title passes `count` items.
    unsafe { std::slice::from_raw_parts(items.cast(), count * size_of::<T>()) }
}

/// Where the things are in a stream (null for none).
fn at<T>(w: &[u64], from: usize, count: usize) -> *const T {
    if count == 0 || from >= w.len() { std::ptr::null() } else { w[from..].as_ptr().cast() }
}

/// [`at`] where an empty array is not null.
fn at_or<T>(w: &[u64], from: usize) -> *const T {
    // SAFETY: at most one past the end.
    unsafe { w.as_ptr().add(from.min(w.len())).cast() }
}

/// A view's record as it is now (zeros for no view), so that its slot may be
/// written again: the copy stands for the handle.
fn record(handle: u64) -> [u64; RECORD] {
    let mut words = [0u64; RECORD];
    if handle != 0 {
        // SAFETY: a descriptor slot the title passes (they are larger).
        unsafe {
            std::ptr::copy_nonoverlapping(handle as *const u8, words.as_mut_ptr().cast(), size_of::<RtvRecord>())
        };
    }
    words
}
const RECORD: usize = size_of::<RtvRecord>().div_ceil(8);

extern "system" fn list_close(this: This) -> HRESULT {
    crate::timeline::ev("close", this as u64, 0);
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &mut *this.cast_mut().cast::<List>() };
    list.thin.close(&list.uploads);
    S_OK
}
extern "system" fn list_reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    crate::timeline::ev("reset", this as u64, 0);
    // What is recorded next is not the recording submitted before.
    crate::resubmit::forget(this);
    // SAFETY: the title passes one of our lists, and neither records into
    // it nor submits it during its reset.
    let list = unsafe { &mut *this.cast_mut().cast::<List>() };
    // Upload memory handed out and never closed into a recording goes with
    // the recording let go of (a closed one has it).
    *held(&list.uploads) = Uploads::default();
    list.thin.begin(list.kind, pipeline, this as usize);
    S_OK
}
extern "system" fn list_reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    list_reset(this, allocator, pipeline)
}
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    if !barriers.is_null() {
        front(this).put_with(BARRIER, &[u64::from(count)], &[bytes_of(barriers, count as usize)]);
    }
    0
}
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    front(this).put(BARRIER_GROUPS, &[u64::from(count)]);
}
extern "system" fn viewports(this: This, count: u32, items: *const D3D12_VIEWPORT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(VIEWPORTS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn scissors(this: This, count: u32, items: *const RECT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(SCISSORS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    let given = !factor.is_null();
    front(this).put_with(BLEND_FACTOR, &[u64::from(given)], &[bytes_of(factor, usize::from(given))]);
}
extern "system" fn heaps(this: This, count: u32, items: *const This) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(HEAPS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
fn constants(this: This, op: u64, slot: u32, count: u32, values: *const u32, offset: u32) {
    let count = if values.is_null() { 0 } else { count };
    front(this).put_with(
        op,
        &[u64::from(slot) | u64::from(offset) << 32, u64::from(count)],
        &[bytes_of(values, count as usize)],
    );
}
extern "system" fn compute_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, COMPUTE_CONSTANTS, slot, count, values, offset);
}
extern "system" fn graphics_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, GRAPHICS_CONSTANTS, slot, count, values, offset);
}
extern "system" fn index_buffer(this: This, view: *const IndexView) {
    let given = !view.is_null();
    front(this).put_with(INDEX_BUFFER, &[u64::from(given)], &[bytes_of(view, usize::from(given))]);
}
extern "system" fn vertex_buffers(this: This, start: u32, count: u32, views: *const VertexView) {
    let given = !views.is_null();
    front(this).put_with(
        VERTEX_BUFFERS,
        &[u64::from(start) | u64::from(count) << 32, u64::from(given)],
        &[bytes_of(views, if given { count as usize } else { 0 })],
    );
}
extern "system" fn targets(this: This, count: u32, handles: *const u64, one_range: i32, depth: *const u64) {
    let count = if handles.is_null() { 0 } else { (count as usize).min(8) };
    // SAFETY: the title passes a handle or null.
    let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
    let mut records = [0u64; RECORD * 9];
    let mut present = 0u64;
    for i in 0..count {
        // SAFETY: the title passes `count` handles, or the first of a range
        // of views of 128 bytes.
        let handle = unsafe { if one_range != 0 { *handles + 128 * i as u64 } else { *handles.add(i) } };
        present |= u64::from(handle != 0) << i;
        records[i * RECORD..(i + 1) * RECORD].copy_from_slice(&record(handle));
    }
    records[count * RECORD..(count + 1) * RECORD].copy_from_slice(&record(depth));
    let records = &records[..(count + 1) * RECORD];
    front(this).put_with(
        TARGETS,
        &[count as u64, present, u64::from(depth != 0)],
        &[bytes_of(records.as_ptr(), records.len())],
    );
}
extern "system" fn clear_rtv(this: This, handle: u64, color: *const [f32; 4], count: u32, rects: *const Rect) {
    if handle == 0 || color.is_null() {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_RTV,
        &[u64::from(count)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(color, 1), bytes_of(rects, count as usize)],
    );
}
extern "system" fn clear_dsv(this: This, handle: u64, flags: u32, depth: f32, stencil: u8, count: u32, rects: *const Rect) {
    if handle == 0 {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_DSV,
        &[u64::from(count), u64::from(flags) | u64::from(stencil) << 32, Word::word(depth)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(rects, count as usize)],
    );
}
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    if dst.is_null() || src.is_null() {
        return;
    }
    let boxed = !source_box.is_null();
    front(this).put_with(
        COPY_TEXTURE,
        &[u64::from(x) | u64::from(y) << 32, u64::from(z) | u64::from(boxed) << 32],
        &[bytes_of(dst, 1), bytes_of(src, 1), bytes_of(source_box, usize::from(boxed))],
    );
}
#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

/// Makes the calls of a stream on `list`: how many, and whether the last one
/// closed it. Once `left` says so only its reset and close are done.
fn translate(list: This, stream: &[u64], left: impl Fn() -> bool) -> (u64, bool) {
    let (mut i, mut calls, mut closed) = (0, 0, false);
    let mut skip = left();
    while i < stream.len() {
        let (op, len) = (stream[i] & 0xffff, (stream[i] >> 16) as usize);
        let w = &stream[i + 1..i + 1 + len];
        i += 1 + len;
        calls += 1;
        if calls % 64 == 0 {
            skip = left();
        }
        if skip && op != CLOSE && op != RESET {
            continue;
        }
        if run_plain(op, list, w) {
            continue;
        }
        let n = w.first().map_or(0, |&n| n as usize);
        match op {
            RESET => {
                // (Only whether there is an allocator is looked at.)
                let allocator: This = std::ptr::NonNull::<u64>::dangling().as_ptr().cast_const().cast();
                super::reset(list, allocator, w[0] as This);
            }
            CLOSE => {
                super::close(list);
                closed = true;
            }
            BARRIER => {
                super::barrier(list, n as u32, at_or(w, 1));
            }
            BARRIER_GROUPS => super::barrier_groups(list, n as u32, std::ptr::null()),
            VIEWPORTS => state::viewports(list, n as u32, at(w, 1, n)),
            SCISSORS => state::scissors(list, n as u32, at(w, 1, n)),
            BLEND_FACTOR => state::blend_factor(list, at(w, 1, n)),
            HEAPS => state::heaps(list, n as u32, at(w, 1, n)),
            COMPUTE_CONSTANTS => {
                state::compute_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            GRAPHICS_CONSTANTS => {
                state::graphics_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            INDEX_BUFFER => state::index_buffer(list, at(w, 1, n)),
            VERTEX_BUFFERS => {
                let views = if w[1] != 0 { at_or(w, 2) } else { std::ptr::null() };
                state::vertex_buffers(list, w[0] as u32, (w[0] >> 32) as u32, views);
            }
            TARGETS => {
                let mut handles = [0u64; 8];
                let place = |k: usize| w[3 + k * RECORD..].as_ptr() as u64;
                for (k, h) in handles.iter_mut().enumerate().take(n) {
                    if w[1] >> k & 1 != 0 {
                        *h = place(k);
                    }
                }
                let depth = if w[2] != 0 { place(n) } else { 0 };
                state::targets(list, n as u32, handles.as_ptr(), 0, &depth);
            }
            CLEAR_RTV => {
                super::clear_rtv(list, w[1..].as_ptr() as u64, at_or(w, 1 + RECORD), n as u32, at(w, 1 + RECORD + 2, n));
            }
            CLEAR_DSV => {
                super::clear_dsv(
                    list,
                    w[3..].as_ptr() as u64,
                    w[1] as u32,
                    Word::unword(w[2]),
                    (w[1] >> 32) as u8,
                    n as u32,
                    at(w, 3 + RECORD, n),
                );
            }
            COPY_TEXTURE => {
                let step = size_of::<CopyLocation>().div_ceil(8);
                let boxed = (w[1] >> 32) as usize;
                super::copy_texture_region(
                    list,
                    at_or(w, 2),
                    w[0] as u32,
                    (w[0] >> 32) as u32,
                    w[1] as u32,
                    at_or(w, 2 + step),
                    at(w, 2 + 2 * step, boxed),
                );
            }
            #[cfg(feature = "diagnostics")]
            CHECK => {
                STATS[S_CHECKED].fetch_add(1, Relaxed);
                // A table's descriptors are read at the draw by the path in
                // the call (`draw::State::bind`: the views it makes, the key
                // of a fresh table): a change there is a difference. A root
                // view's bytes are read by both paths only when the list is
                // submitted (`draw::root_view`, `root_uav`,
                // `root_srv_table`: a refresh of the host copy or a fill at
                // submission): a change between call and translation is
                // none, and is counted apart.
                let mut changed = false;
                for entry in w.chunks_exact(5) {
                    let table = entry[0] & TABLE != 0;
                    let address = entry[0] & !TABLE;
                    let now = words_at(address);
                    if now[..] == entry[1..] {
                        continue;
                    }
                    changed |= table;
                    STATS[if table { S_CHANGED_TABLES } else { S_CHANGED_VIEWS }].fetch_add(1, Relaxed);
                    // The first few of each, to find out what they are.
                    static NAMED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
                    if NAMED[usize::from(table)].fetch_add(1, Relaxed) < if table { 16 } else { 4 } {
                        tracing::info!(
                            address = format_args!("{address:#x}"),
                            table,
                            buffer = crate::shadow::find_any(address, 32).is_some(),
                            at_call = format_args!("{:x?}", &entry[1..]),
                            at_translation = format_args!("{now:x?}"),
                            "thin check: what a draw reads changed between its call and its translation"
                        );
                    }
                }
                if changed {
                    STATS[S_CHANGED].fetch_add(1, Relaxed);
                }
            }
            _ => {}
        }
    }
    (calls, closed)
}

/// The table of a list whose calls are written down: that of the others with
/// the methods that record replaced (slots as in [`super::METHODS`]).
const fn table(newer: bool) -> [*const (); 160] {
    let mut f = super::list_table(newer);
    let mut i = 0;
    while i < MINE.len() {
        let (slot, method) = MINE[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    if newer {
        f[146] = barrier_groups as *const ();
    }
    f
}

pub(super) static FUNCTIONS: Table<160> = Table(table(false));
pub(super) static NEWER_FUNCTIONS: Table<160> = Table(table(true));

/// The methods that record, written down. Upload memory (0x4a0 of newer
/// lists) is the title list's own, as before (`super::allocate_upload`).
const MINE: [(usize, *const ()); 50] = [
    (9, list_close as *const ()),
    (10, list_reset as *const ()),
    (12, draw as *const ()),
    (13, draw_indexed as *const ()),
    (14, dispatch as *const ()),
    (15, copy_buffer_region as *const ()),
    (16, copy_texture_region as *const ()),
    (17, copy_resource as *const ()),
    (19, resolve as *const ()),
    (20, topology as *const ()),
    (21, viewports as *const ()),
    (22, scissors as *const ()),
    (23, blend_factor as *const ()),
    (24, stencil_ref as *const ()),
    (25, set_pipeline as *const ()),
    (26, barrier as *const ()),
    (30, resolve_queries as *const ()),
    (32, heaps as *const ()),
    (35, compute_table as *const ()),
    (36, graphics_table as *const ()),
    (37, compute_constant as *const ()),
    (38, graphics_constant as *const ()),
    (39, compute_constants as *const ()),
    (40, graphics_constants as *const ()),
    (41, compute_view as *const ()),
    (42, graphics_view as *const ()),
    (43, compute_view as *const ()),
    (44, graphics_view as *const ()),
    (45, compute_view as *const ()),
    (46, graphics_view as *const ()),
    (47, index_buffer as *const ()),
    (48, vertex_buffers as *const ()),
    (50, targets as *const ()),
    (51, clear_dsv as *const ()),
    (52, clear_rtv as *const ()),
    (59, execute_indirect as *const ()),
    (60, copy_memory as *const ()),
    (68, depth_bounds as *const ()),
    (69, write32 as *const ()),
    (70, write64 as *const ()),
    (71, wait_value as *const ()),
    (72, wait_value64 as *const ()),
    (78, dispatch_x as *const ()),
    (83, copy_texture_region_x as *const ()),
    (84, copy_resource_x as *const ()),
    (98, dispatch_mesh_x as *const ()),
    (127, dispatch_mesh as *const ()),
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, list_reset_x as *const ()),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method of the tables that records is written down too: what is
    /// left alike in both records nothing (the object's own methods, query
    /// marks, driver hints, state the host has no use for).
    #[test]
    fn every_recording_method_is_written_down() {
        const NOTHING: [usize; 11] = [0, 2, 7, 8, 28, 29, 33, 34, 73, 99, 124];
        let mine: Vec<usize> = MINE.iter().map(|m| m.0).collect();
        for (slot, _) in METHODS.iter().chain(MORE_METHODS.iter()) {
            assert!(mine.contains(slot) != NOTHING.contains(slot), "slot {slot}");
        }
        for &slot in &mine {
            assert!(
                METHODS.iter().chain(MORE_METHODS.iter()).any(|m| m.0 == slot),
                "slot {slot} is written down and not there"
            );
        }
        // Newer lists: the enhanced barrier is written down; upload memory
        // and the unknown 0x470 stay as they are.
        let newer = table(true);
        assert_eq!(newer[146], barrier_groups as *const ());
        assert_eq!(newer[148], super::super::allocate_upload as *const ());
        assert_eq!(newer[newer_slot(59)], execute_indirect as *const ());
    }

    /// Arguments come out of a stream as they went in.
    #[test]
    #[allow(clippy::field_reassign_with_default)] // (a type with its own drop)
    fn a_stream_gives_back_what_was_written() {
        let mut f = Front::default();
        f.recording = Some(Arc::new(Recording::new(0, 0)));
        f.put(Plain::DRAW_INDEXED as u64, &[3, 1, 0, Word::word(-7i32), 5]);
        let viewport = [1.0f32, 2.0, 3.0, 4.0, 0.0, 1.0];
        f.put_with(VIEWPORTS, &[1], &[bytes_of(viewport.as_ptr(), 6)]);
        f.put_with(COMPUTE_CONSTANTS, &[4 | 2 << 32, 3], &[bytes_of([7u32, 8, 9].as_ptr(), 3)]);
        let s = std::mem::take(&mut f.words);
        assert_eq!((s[0] & 0xffff, s[0] >> 16), (Plain::DRAW_INDEXED as u64, 5));
        assert_eq!(<i32 as Word>::unword(s[4]), -7);
        let w = &s[6..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16, w[1]), (VIEWPORTS, 4, 1));
        // SAFETY: what was written just now.
        assert_eq!(unsafe { *at::<[f32; 6]>(&w[1..], 1, 1) }, viewport);
        let w = &w[5..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16), (COMPUTE_CONSTANTS, 4));
        // SAFETY: as above.
        assert_eq!(unsafe { *at::<[u32; 3]>(&w[1..], 2, 3) }, [7, 8, 9]);
        assert!(at::<u32>(&w[1..], 2, 0).is_null());
        // Nothing is written once the recording is closed.
        f.closed = true;
        f.put(Plain::DRAW as u64, &[1, 1, 0, 0]);
        assert!(f.words.is_empty());
    }

    /// A view's record is the one at the call, also if its slot is written
    /// again right after.
    #[test]
    fn a_target_is_the_one_at_the_call() {
        let mut slot = [0x5256_5476u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let kept = record(slot.as_ptr() as u64);
        slot[1] = 99;
        assert_eq!(slot[1], 99);
        assert_eq!(kept[1], 2);
        assert_eq!(record(0), [0; RECORD]);
    }

    /// A stream longer than a piece is handed over in pieces, in order, and
    /// a recording is translated (here: let go of) only once.
    #[test]
    #[allow(clippy::field_reassign_with_default)] // (a type with its own drop)
    fn long_streams_go_in_pieces() {
        let recording = Arc::new(Recording::new(0, 0));
        let mut f = Front::default();
        f.recording = Some(recording.clone());
        // Not translated by anyone here: taken out of the ready ones below.
        for k in 0..CHUNK as u64 {
            f.put(Plain::TOPOLOGY as u64, &[k]);
        }
        held(&READY).retain(|r| !Arc::ptr_eq(r, &recording));
        let inner = held(&recording.inner);
        assert!(inner.scheduled);
        let first = inner.chunks.front().expect("a piece handed over");
        assert_eq!(first[0] & 0xffff, Plain::TOPOLOGY as u64);
        assert_eq!(first[1], 0);
        drop(inner);
        // (Not submitted: let go of, as a reset does.)
        f.leave();
        assert!(recording.abandoned.load(Acquire));
    }
}
