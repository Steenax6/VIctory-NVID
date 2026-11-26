//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (the title asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: the title's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Uploads>,
    #[cfg(windows)]
    host: Option<crate::host::HostList>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
    /// The parts of the recording before `host`, in order, each closed: the
    /// list is cut at its waits for memory values ([`cut`]).
    #[cfg(windows)]
    earlier: Vec<Part>,
    /// The waits recorded before the first work of `host`: the queue stands
    /// at them before it runs `host`.
    #[cfg(windows)]
    gate: Vec<ValueWait>,
    /// What the recording holds, counted as it is made, and what it held
    /// when its host side was taken at a submission that does not wait
    /// ([`note_detached`]); both go at the title's reset.
    #[cfg(windows)]
    tally: Tally,
    #[cfg(windows)]
    after: std::sync::Mutex<Option<crate::resubmit::Summary>>,
    /// With calls only written down (`VICTORY_THIN=1`, [`thin`]): the
    /// recording being written and its words not handed over yet.
    #[cfg(windows)]
    thin: thin::Front,
}

/// List calls written down and translated by threads of ours
/// (`VICTORY_THIN=1`).
#[cfg(windows)]
pub(crate) mod thin;

/// Indices into [`Tally::counts`].
#[cfg(windows)]
const KIND_DRAW: usize = 0;
#[cfg(windows)]
const KIND_INDEXED: usize = 1;
#[cfg(windows)]
const KIND_INDIRECT: usize = 2;
#[cfg(windows)]
const KIND_DISPATCH: usize = 3;
#[cfg(windows)]
const KIND_CLEAR: usize = 4;
#[cfg(windows)]
const KIND_COPY: usize = 5;
#[cfg(windows)]
const KIND_RESOLVE: usize = 6;

/// The commands a list's recording holds, by kind (the draws count mesh
/// dispatches too).
#[cfg(windows)]
#[derive(Default)]
struct Tally {
    counts: [std::sync::atomic::AtomicU32; 7],
    targets: std::sync::atomic::AtomicBool,
}

#[cfg(windows)]
fn tally(this: This, what: usize) {
    if !this.is_null() {
        // SAFETY: the title passes one of our lists.
        unsafe { &*this.cast::<List>() }.tally.counts[what].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(windows)]
fn tally_targets(this: This) {
    if !this.is_null() {
        // SAFETY: as above.
        unsafe { &*this.cast::<List>() }.tally.targets.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The host side of `list` is taken at a submission that is not waited
/// for, and the list is left a new, empty one. A list taken from again
/// before its reset runs that empty one: counted
/// ([`crate::resubmit::ran_empty`]); the summary kept is that of the
/// recording.
#[cfg(windows)]
fn note_detached(list: &List, host: &crate::host::HostList) {
    let mut after = list.after.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(recorded) = after.as_ref() {
        crate::resubmit::ran_empty(recorded);
        return;
    }
    let mut pipelines = Vec::new();
    for part in list.earlier.iter().map(|p| &p.host).chain([host]) {
        pipelines.extend(part.first_bound());
        if pipelines.len() >= 3 {
            break;
        }
    }
    pipelines.truncate(3);
    let counts = std::array::from_fn(|i| list.tally.counts[i].load(std::sync::atomic::Ordering::Relaxed));
    *after = Some(crate::resubmit::Summary {
        kind: list.kind,
        counts,
        targets: list.tally.targets.load(std::sync::atomic::Ordering::Relaxed),
        parts: list.earlier.len() as u32 + 1,
        pipelines,
    });
}

/// One part of a title list cut at its waits: the waits the queue stands
/// at, then the host list that runs once they are met.
#[cfg(windows)]
struct Part {
    waits: Vec<ValueWait>,
    host: crate::host::HostList,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(feature = "diagnostics")]
    crate::copystat::recorded(started);
    #[cfg(feature = "diagnostics")]
    crate::profile::seen(crate::profile::Seen::Record);
}

/// The submission of `lists` has run: each gives its descriptor tables'
/// slots back ([`crate::draw::State::tables_done`]). A list cut at waits
/// keeps them: its state goes with its last part.
#[cfg(windows)]
pub(crate) fn tables_done(lists: &[This]) {
    for &this in lists {
        if this.is_null() {
            continue;
        }
        // SAFETY: the title passes our lists.
        let list = unsafe { &*this.cast::<List>() };
        if list.earlier.is_empty() {
            list.state.lock().unwrap_or_else(|e| e.into_inner()).tables_done();
        }
    }
}

/// The host list behind a title list, if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    unsafe { (*this.cast::<List>()).host.as_ref() }
}

/// The host side of a submitted list, taken from it: a queue that waits
/// for a fence runs it later, and the title may reset the list meanwhile.
/// With it go the upload memory and the dynamic descriptor slots its
/// commands name, kept until it has run.
#[cfg(windows)]
pub(crate) struct Detached {
    host: crate::host::HostList,
    _uploads: Uploads,
    _state: crate::draw::State,
}

#[cfg(windows)]
impl Detached {
    pub(crate) fn host(&self) -> &crate::host::HostList {
        &self.host
    }
}

/// A host list that has run and waits to be used again, and the bytes of
/// upload memory it keeps.
#[cfg(windows)]
struct Idle(crate::host::HostList, u64);

// SAFETY: D3D12 objects are free-threaded, and an idle list is touched by
// nobody until it is taken out again.
#[cfg(windows)]
unsafe impl Send for Idle {}

/// Host lists that have run, reset and ready to record: the host side of
/// a list a queue held, or of a part of a list cut at a wait, is taken
/// from the title's list at its submission, and the title's list needs
/// another. Each was made new and thrown away after its run: a hundred a
/// frame in gameplay, each two command allocators and two lists, and for
/// four in ten a budget and an argument buffer (run 61: "104 host lists
/// made 21.2 ms" a frame, and the graphics kernel's time to free them).
#[cfg(windows)]
static IDLE: std::sync::Mutex<(Vec<Idle>, u64)> = std::sync::Mutex::new((Vec::new(), 0));

/// The most host lists kept idle, and the most upload memory they may keep
/// between them (a list keeps scratch chunks over a reset only while lists
/// do not share them, `VICTORY_SCRATCH_POOL=0`).
#[cfg(windows)]
// (512 since run 221: with submissions prepared by a thread of ours the
// title has its lists back at once and resets them while the flights of
// the frame before are still in the air, so more than 128 are out at a
// time: 80 host lists were made and as many thrown away every frame,
// 8 ms to make and more to free, on the thread that submits.)
const IDLE_MOST: (usize, u64) = (512, 256 << 20);

/// Whether host lists that have run are used again (on by default;
/// `VICTORY_LIST_POOL=0`: each is made new and thrown away, as before).
#[cfg(windows)]
fn pools() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIST_POOL").is_none_or(|v| v != "0"))
}

/// A host list open for recording: one that has run, as a title's list is
/// after its reset, or a new one.
#[cfg(windows)]
fn fresh_list() -> Option<crate::host::HostList> {
    let idle = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = idle.0.pop();
        if let Some(Idle(_, bytes)) = &taken {
            idle.1 -= bytes;
        }
        taken
    };
    match idle {
        Some(Idle(list, _)) => Some(list),
        None => crate::host::host().and_then(|h| h.new_list().ok()),
    }
}

/// The host sides of lists whose submission has run (submissions are
/// waited for; `ran` names the ones it ran, by their place in `lists`):
/// what they held is let go, as before, and the host lists are reset and
/// kept for [`fresh_list`].
#[cfg(windows)]
pub(crate) fn recycle(lists: Vec<Detached>, ran: &[usize]) {
    if !pools() {
        return;
    }
    for (i, Detached { host, _uploads, _state }) in lists.into_iter().enumerate() {
        // The descriptor slots and the upload memory its commands named.
        drop(_state);
        drop(_uploads);
        if !ran.contains(&i) || IDLE.lock().unwrap_or_else(|e| e.into_inner()).0.len() >= IDLE_MOST.0 {
            continue;
        }
        // As the title's reset does: what the list kept for its commands
        // goes, its allocators stay.
        host.reset();
        let bytes = host.scratch_kept();
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        if idle.0.len() < IDLE_MOST.0 && idle.1 + bytes <= IDLE_MOST.1 {
            idle.0.push(Idle(host, bytes));
            idle.1 += bytes;
        }
    }
}

/// A wait for a memory value recorded into list `this`, with waits cut
/// into lists (`crate::queue::cuts`). On the console the queue runs the
/// list up to the wait, stands there until the value is in memory, and
/// runs on; the list's bottom-of-pipe writes before the wait are done by
/// then, and other queues may be waiting for exactly those. So the
/// recording is cut here: the host list so far is closed and kept as a
/// part of its own ([`Part`]), the title's calls record into a new host
/// list from here on, and the wait stands before that one. A submission
/// then gives the queue the parts in order ([`detach_parts`]).
///
/// A wait before any work (the first call after a reset, or a wait right
/// after another) cuts nothing: it joins the waits before the current
/// host list.
/// shortcut: a bottom-of-pipe write is done when its part has run, not at
/// its place in the part: a queue waiting for it starts later than on the
/// console, never earlier, and nothing can wait for itself that way (a
/// part has no wait after its first work).
#[cfg(windows)]
fn cut(this: This, wait: ValueWait) {
    if this.is_null() {
        return;
    }
    // SAFETY: the title passes one of our lists; a list is recorded by one
    // thread at a time and is not submitted while it records.
    let list = unsafe { &mut *this.cast_mut().cast::<List>() };
    let Some(current) = list.host.as_ref() else {
        return;
    };
    if current.untouched() {
        list.gate.push(wait);
        return;
    }
    let Some(fresh) = fresh_list() else {
        // shortcut: without a new host list the wait stands before the part
        // it was recorded in.
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no host list for the part after a wait: the wait moves to the part's start"));
        list.gate.push(wait);
        return;
    };
    current.close();
    static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        tracing::info!(parts = list.earlier.len() + 2, "a list is cut at a wait for a memory value");
    }
    if let Some(done) = list.host.replace(fresh) {
        let waits = std::mem::replace(&mut list.gate, vec![wait]);
        list.earlier.push(Part { waits, host: done });
    }
    // What the title set so far, on the new host list.
    if let Some(host) = list.host.as_ref() {
        list.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .replay(host);
    }
}

/// Whether list `this` recorded a wait for a memory value (with waits cut
/// into lists).
#[cfg(windows)]
pub(crate) fn cut_list(this: This) -> bool {
    if this.is_null() {
        return false;
    }
    // SAFETY: one of our lists, closed and submitted: nothing records
    // into it.
    let list = unsafe { &*this.cast::<List>() };
    !list.earlier.is_empty() || !list.gate.is_empty()
}

/// What a queue is given for submitted lists that were cut at their waits:
/// the waits it stands at, the host lists it runs once they are met, and
/// the order those run in.
#[cfg(windows)]
pub(crate) type Parts = Vec<(Vec<ValueWait>, Vec<Detached>, Vec<usize>)>;

/// Takes the host sides of the submitted `lists`, part by part ([`cut`]),
/// and leaves each list a new, closed host list for its next reset, as
/// [`detach`] does. What comes back is in the order the queue runs it: a
/// new entry starts at every part that has waits; host lists without
/// waits join the entry before them. The list's upload memory and draw
/// state (its dynamic descriptor slots) go with its last part, which runs
/// last.
/// shortcut: a list the submission names a second time runs again only if
/// it has one part and that part is in the entry being filled; otherwise
/// the second naming is dropped (logged once).
#[cfg(windows)]
pub(crate) fn detach_parts(lists: &[This]) -> Parts {
    let mut parts: Parts = vec![(Vec::new(), Vec::new(), Vec::new())];
    // The lists taken: where its one host list is (entry, place), or none
    // for a list of several parts.
    let mut seen: Vec<(This, Option<(usize, usize)>)> = Vec::new();
    for &this in lists {
        if this.is_null() {
            continue;
        }
        if let Some(&(_, at)) = seen.iter().find(|(t, _)| *t == this) {
            match at {
                Some((entry, place)) if entry + 1 == parts.len() => parts[entry].2.push(place),
                _ => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!("a list named twice by a submission that waits for memory values runs once")
                    });
                }
            }
            continue;
        }
        let Some(fresh) = fresh_list() else {
            continue;
        };
        // A new host list is open; the title's next call on its list is a
        // reset, which wants it closed.
        fresh.close();
        // SAFETY: one of our lists, closed and submitted: nothing records
        // into it until the title resets it.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        let Some(host) = list.host.replace(fresh) else {
            continue;
        };
        note_detached(list, &host);
        let earlier = std::mem::take(&mut list.earlier);
        let gate = std::mem::take(&mut list.gate);
        let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
        let state = std::mem::take(&mut *list.state.lock().unwrap_or_else(|e| e.into_inner()));
        let single = earlier.is_empty() && gate.is_empty();
        let mut pieces: Vec<(Vec<ValueWait>, Detached)> = earlier
            .into_iter()
            .map(|part| {
                (
                    part.waits,
                    Detached {
                        host: part.host,
                        _uploads: Uploads::default(),
                        _state: Default::default(),
                    },
                )
            })
            .collect();
        pieces.push((
            gate,
            Detached {
                host,
                _uploads: uploads,
                _state: state,
            },
        ));
        let mut last = None;
        for (waits, detached) in pieces {
            if !waits.is_empty() {
                parts.push((waits, Vec::new(), Vec::new()));
            }
            let entry = parts.len() - 1;
            let (_, taken, order) = &mut parts[entry];
            let place = taken.len();
            order.push(place);
            taken.push(detached);
            last = Some((entry, place));
        }
        seen.push((this, if single { last } else { None }));
    }
    // (The first entry is empty if the first list begins with a wait.)
    parts.retain(|(_, taken, _)| !taken.is_empty());
    parts
}

/// Takes the host sides of submitted lists (each once, however often the
/// submission names it) and leaves each list a new, closed host list for
/// its next reset: what was taken, and which of them run in what order.
#[cfg(windows)]
pub(crate) fn detach(lists: &[This]) -> (Vec<Detached>, Vec<usize>) {
    let mut taken: Vec<(This, Detached)> = Vec::new();
    let mut order = Vec::new();
    for &this in lists {
        if this.is_null() {
            continue;
        }
        if let Some(i) = taken.iter().position(|(t, _)| *t == this) {
            order.push(i);
            continue;
        }
        let Some(fresh) = fresh_list() else {
            continue;
        };
        // A new host list is open; the title's next call on its list is a
        // reset, which wants it closed.
        fresh.close();
        let list = this.cast_mut().cast::<List>();
        // SAFETY: one of our lists, closed and submitted: nothing records
        // into it until the title resets it.
        let (host, uploads, state) = unsafe {
            let Some(host) = (*list).host.replace(fresh) else {
                continue;
            };
            note_detached(&*list, &host);
            let uploads =
                std::mem::take(&mut *(*list).uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let state =
                std::mem::take(&mut *(*list).state.lock().unwrap_or_else(|e| e.into_inner()));
            (host, uploads, state)
        };
        order.push(taken.len());
        taken.push((
            this,
            Detached {
                host,
                _uploads: uploads,
                _state: state,
            },
        ));
    }
    (taken.into_iter().map(|(_, d)| d).collect(), order)
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

/// `pipeline` is the list's initial pipeline (null: none), bound until the
/// title sets another, as after a reset.
fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    #[cfg(windows)]
    if thin::on() {
        // The title's list only writes its calls down; a list of ours does
        // them ([`thin`]), and it needs no host list of its own.
        let mut list = boxed(kind, pipeline, false);
        list.object = if crate::device::newer() {
            Object::new(&thin::NEWER_FUNCTIONS)
        } else {
            Object::new(&thin::FUNCTIONS)
        };
        // Lists start open: what the title records before its first reset
        // is a recording too.
        let title = std::ptr::from_ref::<List>(&*list) as usize;
        list.thin.begin(kind, pipeline, title);
        return object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    }
    object::hand_out(LIST, boxed(kind, pipeline, true), iid, &LIST_IIDS, out)
}

/// A list of `kind` with `pipeline` bound, open, with a host list if
/// `host` and there is a host GPU.
fn boxed(kind: u32, pipeline: This, host: bool) -> Box<List> {
    #[cfg(not(windows))]
    let _ = host;
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        host: crate::host::host().filter(|_| host).and_then(|h| match h.new_list() {
            Ok(list) => Some(list),
            Err(e) => {
                tracing::warn!(error = %e, "no host command list");
                None
            }
        }),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
        #[cfg(windows)]
        earlier: Vec::new(),
        #[cfg(windows)]
        gate: Vec::new(),
        #[cfg(windows)]
        tally: Tally::default(),
        #[cfg(windows)]
        after: Default::default(),
        #[cfg(windows)]
        thin: Default::default(),
    });
    #[cfg(not(windows))]
    let _ = pipeline;
    list
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    let left = object::release_boxed::<List>(this);
    #[cfg(windows)]
    if left == 0 {
        crate::resubmit::forget(this);
    }
    left
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    // With what the recording holds: draws, indexed draws, indirect
    // executions and dispatches, sixteen bits each.
    #[cfg(all(windows, feature = "diagnostics"))]
    if !this.is_null() {
        // SAFETY: the title passes one of our lists.
        let counts = unsafe { &(*this.cast::<List>()).tally.counts };
        let held = [KIND_DRAW, KIND_INDEXED, KIND_INDIRECT, KIND_DISPATCH].iter().enumerate().fold(0u64, |all, (at, &kind)| {
            all | u64::from(counts[kind].load(std::sync::atomic::Ordering::Relaxed).min(0xffff)) << (16 * at)
        });
        crate::timeline::ev("close", this as u64, held);
    }
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    crate::timeline::ev("reset", this as u64, 0);
    // What is recorded next is not the recording submitted before.
    #[cfg(windows)]
    crate::resubmit::forget(this);
    // A wait recorded and never submitted goes with the recording.
    VALUE_WAITS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|&(list, _)| list != this as usize);
    // Every submission of the list has run (submissions are waited for).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    *uploads.lock().unwrap_or_else(|e| e.into_inner()) = Uploads::default();
    // The parts of a recording that was cut and never submitted go with it
    // (a submission takes the parts it runs).
    #[cfg(windows)]
    {
        // SAFETY: the title passes one of our lists, and neither records
        // into it nor submits it during its reset.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        list.earlier.clear();
        list.gate.clear();
    }
    #[cfg(windows)]
    {
        // SAFETY: the title passes one of our lists.
        let list = unsafe { &*this.cast::<List>() };
        list.tally.counts.iter().for_each(|c| c.store(0, std::sync::atomic::Ordering::Relaxed));
        list.tally.targets.store(false, std::sync::atomic::Ordering::Relaxed);
        *list.after.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
    #[cfg(windows)]
    with_state(this, |host, state| {
        host.reset();
        state.reset(pipeline);
    });
    #[cfg(not(windows))]
    let _ = (this, pipeline);
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    // The barrier names the host resource by a borrowed pointer: the list
    // holds it until it is reset, as it does the textures it moves, since
    // the title may release its resource (and the sweep its host copy)
    // before the list has run.
    let held = |texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource| -> This {
        list.keep(texture);
        texture.as_raw().cast_const().cast()
    };
    let host = |r: This| crate::resource::host_resource(r).map(held);
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                // A depth texture's unordered access is its colour twin's
                // (the host gives the depth texture none): the barrier is
                // on the texture shaders write.
                match crate::resource::twin_resource(b.first) {
                    Some(twin) => held(twin),
                    None => host(b.first)?,
                }
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WaitsForBuffers, 1);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    tally(this, KIND_COPY);
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

#[cfg(all(test, windows))]
mod whole_tests {
    /// A copy writes all of a subresource when it starts at its corner and
    /// brings as many elements as it has: texel for texel, block for
    /// block, or a texel for a block between a block-compressed format and
    /// one whose texel is as wide (the bake's copies).
    #[test]
    fn a_copy_of_every_element_writes_the_whole_subresource() {
        use super::writes_whole;
        // RGBA8 4x4 into RGBA8 4x4.
        assert!(writes_whole((28, [4, 4, 1]), [0; 3], (28, [4, 4, 1])));
        // A part of it, or not at its corner.
        assert!(!writes_whole((28, [4, 4, 1]), [0; 3], (28, [2, 4, 1])));
        assert!(!writes_whole((28, [4, 4, 1]), [1, 0, 0], (28, [4, 4, 1])));
        // 512x512 texels of 16 bytes into a 2048x2048 BC3 level, and the
        // last levels: one texel for a level of 2x2 and of 1x1 texels.
        assert!(writes_whole((78, [2048, 2048, 1]), [0; 3], (3, [512, 512, 1])));
        assert!(writes_whole((78, [2, 2, 1]), [0; 3], (3, [1, 1, 1])));
        assert!(writes_whole((78, [1, 1, 1]), [0; 3], (3, [1, 1, 1])));
        assert!(!writes_whole((78, [2048, 2048, 1]), [0; 3], (3, [64, 64, 1])));
        // Block for block, and a volume's depth.
        assert!(writes_whole((71, [256, 256, 1]), [0; 3], (71, [256, 256, 1])));
        assert!(!writes_whole((28, [4, 4, 4]), [0; 3], (28, [4, 4, 1])));
        // A format with no known element, nothing copied, the top of the
        // range: a part, never a panic.
        assert!(!writes_whole((0, [4, 4, 1]), [0; 3], (28, [4, 4, 1])));
        assert!(!writes_whole((28, [4, 4, 1]), [0; 3], (28, [0, 0, 0])));
        assert!(!writes_whole((28, [u32::MAX; 3]), [u32::MAX; 3], (78, [u32::MAX; 3])));
    }
}

/// Whether one of two resources is a target in a stand-in format and the
/// other is not ([`crate::resource::target_format`]): their texels differ
/// in size, so the host cannot copy between them (logged once).
#[cfg(windows)]
fn unlike(dst: This, src: This) -> bool {
    let unlike = crate::resource::stands_in(dst) != crate::resource::stands_in(src);
    if unlike {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(?dst, ?src, "copy between a stand-in target and title-format texels skipped")
        });
    }
    unlike
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    tally(this, KIND_COPY);
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if unlike(dst, src) {
                return;
            }
            // A source the title fills itself is uploaded first (upstream
            // PR #671): a copy read whatever the host texture held.
            if crate::texture::title_written(from) {
                crate::texture::upload(list, src, from);
            }
            // Depth textures shaders also write: what they wrote, first.
            list.twin_use(dst, false, true);
            list.twin_use(src, false, false);
            // A destination made level by level gets its levels' memory
            // ahead of this list (`crate::tiles`).
            crate::texture::card_writes(list, dst, to, None);
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_written(list, dst, to, None);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(to, from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// The texels of subresource `sub` of a host texture described by `d`:
/// width, height, depth.
#[cfg(windows)]
fn level_extent(d: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC, sub: u32) -> [u32; 3] {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let level = sub % u32::from(d.MipLevels).max(1);
    let down = |texels: u64| (texels >> level.min(63)).max(1).min(u64::from(u32::MAX)) as u32;
    let depth = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { down(u64::from(d.DepthOrArraySize)) } else { 1 };
    [down(d.Width), down(u64::from(d.Height)), depth]
}

/// Whether a copy to the corner `at` of a destination subresource (its
/// format and texels) from a source of `source` (its format and the
/// texels copied) writes all of that subresource. Counted in the formats'
/// elements: the host copies between a block-compressed format and one
/// whose texel is as wide as a block, a texel for a block (the title's
/// bake copies its compressed result so). A format without a known element
/// is taken to write a part.
#[cfg(windows)]
fn writes_whole(destination: (u32, [u32; 3]), at: [u32; 3], source: (u32, [u32; 3])) -> bool {
    let elements = |(format, texels): (u32, [u32; 3])| {
        let block = crate::resource::element(format)?.1.max(1) as u32;
        Some([texels[0].div_ceil(block), texels[1].div_ceil(block), texels[2]])
    };
    let (Some(to), Some(from)) = (elements(destination), elements(source)) else {
        return false;
    };
    at == [0; 3] && (0..3).all(|k| from[k] >= to[k])
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission. Texture to buffer (read back) is not done
/// yet.
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    tally(this, KIND_COPY);
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            // Texture to buffer: a read back. The list copies the texels
            // into a host buffer of ours; once it has run they go to the
            // title's buffer (`HostList::defer_read_back`).
            // The title reads two tiny float targets back every frame (5x1
            // and 1x1 RGBA32F, 80 and 16 bytes: run 90), and read a buffer
            // nobody wrote. `VICTORY_READ_BACK=0` leaves them unmade, as
            // before.
            static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let made = read_back(list, dst, x, y, z, src, source_box);
            if n <= 16 || n.is_multiple_of(4096) || made.is_err() {
                static REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if made.is_err() && REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 16 {
                    return;
                }
                let source = crate::resource::host_resource(src.resource).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{}x{} f{} mips {}", d.Width, d.Height, d.Format.0, d.MipLevels)
                });
                tracing::info!(
                    copies = n,
                    made = ?made,
                    source = ?source,
                    source_kind = src.kind,
                    destination_kind = dst.kind,
                    destination_offset = dst.footprint.offset,
                    destination_bytes = ?dst.footprint.bytes(),
                    "a texture to buffer copy (a read back)"
                );
            }
            return;
        };
        if unlike(dst.resource, src.resource) {
            return;
        }
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let (Some(host), Some(bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let Some((buffer, staging)) = list.upload(host, bytes) else {
                return;
            };
            // The list owns this mapped D3D12 upload buffer. Wine can report
            // its address as MEM_FREE, so use the host-upload path.
            list.defer_fill(staging, data, bytes as usize);
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = 0;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() }, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            list.twin_use(src.resource, false, false);
            if crate::texture::title_written(texture) {
                crate::texture::upload(list, src.resource, texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        // A depth texture shaders also write: what they wrote, first (the
        // source above too).
        list.twin_use(dst.resource, false, true);
        // A copy that leaves some of the destination's subresource as it
        // was writes over what that held: for a texture the title fills
        // that is title memory unless the card wrote it last, so its upload
        // is asked for first (decided, as any, when the list is submitted;
        // Marvisak/VIctory pull request #1934). A copy of all of a
        // subresource needs nothing of it.
        // SAFETY: the title's box, six dwords (left, top, front, right,
        // bottom, back), or null.
        let boxed = (!source_box.is_null()).then(|| unsafe { *source_box });
        if dst.kind != FOOTPRINT && crate::texture::title_written(to) {
            let source = (src.kind != FOOTPRINT).then(|| crate::resource::host_resource(src.resource)).flatten();
            // SAFETY: live resources.
            let of_source = match source {
                Some(texture) => {
                    let d = unsafe { texture.GetDesc() };
                    Some((d.Format.0 as u32, level_extent(&d, src.footprint.offset as u32)))
                }
                None => (src.kind == FOOTPRINT).then(|| src.footprint.extent()),
            };
            // SAFETY: a live resource.
            let d = unsafe { to.GetDesc() };
            let whole = of_source.is_some_and(|(format, extent)| {
                let extent = boxed.map_or(extent, |b| {
                    [b[3].saturating_sub(b[0]), b[4].saturating_sub(b[1]), b[5].saturating_sub(b[2])]
                });
                writes_whole(
                    (d.Format.0 as u32, level_extent(&d, dst.footprint.offset as u32)),
                    [x, y, z],
                    (format, extent),
                )
            });
            if !whole {
                crate::texture::upload(list, dst.resource, to);
            }
        }
        // A destination made level by level gets the level's memory ahead
        // of this list (`crate::tiles`): the title's bake copies each level
        // of its result into a block-compressed texture of that kind, and
        // a copy into a level without memory is dropped.
        crate::texture::card_writes(list, dst.resource, to, Some(dst.footprint.offset as u32));
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_written(list, dst.resource, to, (dst.kind != FOOTPRINT).then_some(dst.footprint.offset as u32));
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

/// Records a texture to buffer copy (a read back) of [`copy_texture_region`]
/// into `list`, or says why not.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)] // the method's own arguments
fn read_back(
    list: &crate::host::HostList,
    dst: CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: CopyLocation,
    source_box: *const [u32; 6],
) -> Result<(), &'static str> {
    use windows::Win32::Graphics::Direct3D12::D3D12_BOX;
    use windows::core::Interface;
    const FOOTPRINT: u32 = 1;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_READ_BACK").is_none_or(|v| v != "0")) {
        return Err("switched off");
    }
    if dst.kind != FOOTPRINT || src.kind == FOOTPRINT {
        return Err("not a texture subresource into a placed footprint");
    }
    if (x, y, z) != (0, 0, 0) {
        return Err("a destination corner other than the footprint's");
    }
    let host = crate::host::host().ok_or("no host")?;
    let (footprint, [size, row, rows, pitch]) = dst.footprint.host_rows().ok_or("an unknown format")?;
    // The title's buffer holds every row at its own pitch.
    let bytes = pitch * (rows - 1) + row;
    let data = buffer_range(dst.resource, dst.footprint.offset, bytes).ok_or("a destination outside its buffer")?;
    let texture = crate::resource::host_resource(src.resource).ok_or("no host texture")?;
    let buffer = host.read_back_buffer(size).ok_or("no read-back buffer")?;
    // As a texture source of any copy: what shaders wrote to a depth twin
    // and what the title wrote itself first, then the state.
    list.twin_use(src.resource, false, false);
    if crate::texture::title_written(texture) {
        crate::texture::upload(list, src.resource, texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let mut from = src;
    from.resource = texture.as_raw().cast_const().cast();
    let mut to = dst;
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint = footprint;
    let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "read back of {:?} sub {} into buffer {:?}+{:#x}: {rows} rows of {row} bytes",
            src.resource, src.footprint.offset as u32, dst.resource, dst.footprint.offset
        )
    });
    // SAFETY: an open list; both locations in the host's layout (asserted
    // in `copy_texture_region`), naming a live texture and our buffer, which
    // the list keeps until it is reset; the box is the title's.
    unsafe {
        list.list
            .CopyTextureRegion((&raw const to).cast(), 0, 0, 0, (&raw const from).cast(), host_box)
    };
    list.defer_read_back(buffer, data, [row as usize, rows as usize, pitch as usize]);
    Ok(())
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

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    tally(this, KIND_RESOLVE);
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        if unlike(dst, src) {
            return;
        }
        // Depth textures shaders also write: what they wrote, first.
        list.twin_use(dst, false, true);
        list.twin_use(src, false, false);
        // A source smaller than the destination resolves into a part of
        // it: what the rest held stays, so the destination's upload is asked
        // for first, as for a copy of a part.
        // SAFETY: live resources.
        if crate::texture::title_written(to) && unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            (a.Width, a.Height) != (b.Width, b.Height)
        } {
            crate::texture::upload(list, dst, to);
        }
        // As for a copy (`crate::tiles`).
        crate::texture::card_writes(list, dst, to, Some(dst_sub));
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_written(list, dst, to, Some(dst_sub));
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = if crate::resource::stands_in(dst) { a.Format } else { DXGI_FORMAT(format as i32) };
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    tally(this, KIND_CLEAR);
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        #[cfg(feature = "diagnostics")]
        crate::capture::op(std::ptr::from_ref(list) as usize, || {
            format!(
                "CLEAR target {:#x} to {color:?}, {} rectangles",
                record.resource as usize,
                rects.map_or(0, <[_]>::len)
            )
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(Some(texture), record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    tally(this, KIND_CLEAR);
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        // A depth texture shaders also write: what they wrote, first (the
        // clear may be of one slice or of some rectangles).
        list.twin_use(record.resource, false, true);
        ensure_state(list, record.resource, DEPTH_WRITE);
        if crate::resource::depth_and_unordered(record.resource) {
            crate::resource::DEPTH_UNORDERED[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    tally(this, KIND_COPY);
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
                // Numbered for the texture pass once the bytes are there.
                crate::descriptor::host::title_copied(dst as usize, len);
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with the console's
/// `COMPUTE_DISPATCH_INITIATOR` flags. Bit 5 ([`USE_THREAD_DIMENSIONS`])
/// says `x`, `y` and `z` count threads, not groups ([`dispatch_groups`]);
/// the ray-tracing structure builder dispatches this way (0x20, 0x8020 for
/// its 32-lane programs). The other bits ask nothing of the host.
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, flags: u32) {
    #[cfg(windows)]
    state::dispatch_x(this, x, y, z, flags);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z, flags);
        count_dispatch();
    }
}

/// `COMPUTE_DISPATCH_INITIATOR` bit 5, `USE_THREAD_DIMENSIONS`: a
/// dispatch's three numbers are thread counts, and the console runs as
/// many groups as hold them, the last along an axis cut to the count.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) const USE_THREAD_DIMENSIONS: u32 = 0x20;

/// The groups a console dispatch of `dims` with `flags` runs, for a
/// program of `group` threads a group (`numthreads`), and whether its last
/// group along some axis is partial (the console starts only the threads
/// up to the count there). None: a dispatch that counts threads and has
/// none along an axis, which runs nothing.
///
/// Taking the thread counts for group counts ran the structure builder up
/// to 64 times over: its box stages write 8 KiB a group, and in the first
/// frame of a run that failed they ran 272,948 and 47,026 groups where
/// 5,983 and 4,217 hold the threads, 2.2 GiB of empty nodes written past
/// their structures into the title's own objects (fxceless/victory-fork
/// #646 and #994, as DexNom bc3cd4b and uiliannn/VItoriaBr 633ba6e found
/// too).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn dispatch_groups(dims: [u32; 3], group: [u32; 3], flags: u32) -> Option<([u32; 3], bool)> {
    if flags & USE_THREAD_DIMENSIONS == 0 {
        return Some((dims, false));
    }
    if dims.contains(&0) {
        return None;
    }
    let mut groups = [0; 3];
    let mut partial = false;
    for i in 0..3 {
        let g = group[i].max(1);
        groups[i] = dims[i].div_ceil(g);
        partial |= !dims[i].is_multiple_of(g);
    }
    Some((groups, partial))
}

/// `BeginQuery`/`EndQuery(heap, type, index)`: queries are answered when
/// resolved.
extern "system" fn query_mark(_: This, _heap: This, _kind: u32, _index: u32) {}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// shortcut: no query runs on the host GPU; results are plausible values,
/// not measured ones (time spans come out as the submission's length).
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    if offset + u64::from(count) * bytes > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. The host orders
/// nothing between draws and dispatches of a list by itself: a sync waits
/// for every shader write before it, as the title's barriers do (one
/// unordered access barrier on all resources). Dispatches that read what
/// the one before wrote, with a sync and no barrier between them (the
/// ray-tracing structure builder has no other wait between its stages),
/// ran overlapped before (avatarmanz/VIctory-7900XTX 0b5f191b8,
/// Marvisak/VIctory #1228). `VICTORY_SYNC_BARRIER=0`: nothing, as before,
/// to measure what the barriers cost on the card.
extern "system" fn sync(this: This, flags: u32, control: u32, address: u64, size: u64) {
    #[cfg(windows)]
    if sync_barrier()
        && let Some(list) = host_list(this)
    {
        // Whether and how the title syncs (the first calls).
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            tracing::info!(
                flags = format_args!("{flags:#x}"),
                control = format_args!("{control:#x}"),
                address = format_args!("{address:#x}"),
                size,
                "a pipeline sync: waits for earlier shader writes"
            );
        }
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    let _ = (this, flags, control, address, size);
}

/// Whether a pipeline sync waits for earlier shader writes ([`sync`]; on
/// by default, `VICTORY_SYNC_BARRIER=0`: not).
#[cfg(windows)]
fn sync_barrier() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_BARRIER").is_none_or(|v| v != "0"))
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. The title's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some((size, len)) = usize::try_from(size).ok().and_then(|s| Some((s, s.checked_add(align)?))) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    let mut uploads = uploads.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(windows)]
    if upload::reachable()
        && let Some(at) = uploads.carve(size, align)
    {
        // Zeros, as the heap blocks below are (a chunk comes back with what
        // its last list left in it).
        // SAFETY: bytes just carved out of a committed chunk the list holds.
        unsafe { at.write_bytes(0, size) };
        return at;
    }
    let _ = size;
    let mut block = vec![0u8; len];
    let start = block.as_mut_ptr();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // Moving the vector keeps its heap block where it is.
    uploads.heap.push(block);
    at
}

/// What [`allocate_upload`] handed out to a list, kept until the list is
/// reset or, once submitted, until the submission has run.
#[derive(Default)]
pub(crate) struct Uploads {
    /// Chunks of upload memory shaders reach by address ([`upload`]), the
    /// last one being carved, and the bytes of it handed out.
    #[cfg(windows)]
    chunks: Vec<upload::Chunk>,
    #[cfg(windows)]
    used: usize,
    /// Blocks of the process heap: requests larger than a chunk, or all of
    /// them when no chunk can be had.
    heap: Vec<Vec<u8>>,
}

impl Uploads {
    /// `size` bytes at a multiple of `align` (a power of two) from the
    /// list's last chunk, or from a new one.
    #[cfg(windows)]
    fn carve(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        if size > upload::CHUNK || align > upload::CHUNK {
            return None;
        }
        let fits = |used: usize| used.next_multiple_of(align) + size <= upload::CHUNK;
        if self.chunks.is_empty() || !fits(self.used) {
            self.chunks.push(upload::Chunk::take()?);
            self.used = 0;
        }
        let offset = self.used.next_multiple_of(align);
        self.used = offset + size;
        // (Chunks start at multiples of their size, so of `align`.)
        Some((self.chunks.last()?.at() + offset) as *mut u8)
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        #[cfg(windows)]
        if !self.chunks.is_empty() {
            return false;
        }
        self.heap.is_empty()
    }
}

/// List upload memory ([`allocate_upload`]) that shaders reach by address.
///
/// The structure builder writes what its stages read into list upload
/// memory and hands the stages its addresses: its header stage gets the
/// address of its 40-byte records as a root view and reads them through a
/// buffer descriptor it makes of that address, that is by address. Blocks
/// of the process heap lie wherever the heap puts them (at 1.5 to 1.8 TiB
/// in the source's runs), beyond the console's 40 bits and the directory of
/// title memory by address (`crate::memory`, below 128 GiB): the records
/// read as zeros, so the stage stored its headers at address 0, which drops
/// them, and no structure got a header. With the builder asked for, the
/// memory comes from 4 MiB chunks of one reservation where the title's GPU
/// memory lies (from 4 GiB, below 128 GiB), write-watched as XMem makes
/// the title's: each chunk is one window of memory outside buffers
/// (`crate::shadow`), whose pages the CPU wrote are copied when a list that
/// names it is submitted, and none is on the process heap, which the guards
/// must not touch (`crate::watch::protect_here`).
#[cfg(windows)]
pub(crate) mod upload {
    use std::sync::Mutex;

    use windows_sys::Win32::System::Memory::{
        MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
        MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
        VirtualAlloc2,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// A chunk: one window of memory outside buffers (`crate::shadow`).
    pub(crate) const CHUNK: usize = 4 << 20;
    /// Chunks reserved (512 MiB).
    const CHUNKS: usize = 128;
    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Whether list upload memory comes from the chunks: with the
    /// structure builder asked for (`VICTORY_BVH_BUILDER=1`), or
    /// `VICTORY_UPLOAD_CHUNKS=1`; `=0` never. Otherwise from the process
    /// heap, as before.
    pub(crate) fn reachable() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| match std::env::var("VICTORY_UPLOAD_CHUNKS").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => crate::pipeline::builder_allowed(),
        })
    }

    struct Region {
        /// The reservation's start; 0 if it could not be made.
        base: usize,
        /// Chunks committed so far, from the start.
        next: usize,
        /// Committed chunks no list holds.
        free: Vec<usize>,
    }

    static REGION: Mutex<Option<Region>> = Mutex::new(None);

    /// A chunk a list holds; it goes back to the free ones when dropped.
    pub(crate) struct Chunk(usize);

    impl Chunk {
        /// A free chunk, or a new one; none when all are held (the heap
        /// serves then, out of shaders' reach).
        pub(crate) fn take() -> Option<Chunk> {
            let mut region = REGION.lock().unwrap_or_else(|e| e.into_inner());
            let r = region.get_or_insert_with(|| Region {
                base: reserve().unwrap_or(0),
                next: 0,
                free: Vec::new(),
            });
            if r.base == 0 {
                return None;
            }
            if let Some(at) = r.free.pop() {
                return Some(Chunk(at));
            }
            if r.next == CHUNKS {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        chunks = CHUNKS,
                        "list upload memory: every chunk is held; more comes from the heap, which shaders cannot reach by address"
                    );
                });
                return None;
            }
            let at = r.base + r.next * CHUNK;
            // SAFETY: a chunk of our reservation.
            if unsafe { VirtualAlloc(at as *const _, CHUNK, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
                return None;
            }
            r.next += 1;
            Some(Chunk(at))
        }

        pub(crate) fn at(&self) -> usize {
            self.0
        }
    }

    impl Drop for Chunk {
        fn drop(&mut self) {
            if let Some(r) = REGION.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                r.free.push(self.0);
            }
        }
    }

    fn reserve() -> Option<usize> {
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: (((victory_shader::emit::MEMORY_PAGES as usize) << 16) - 1) as *mut _,
            Alignment: CHUNK,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: a reservation of our own; the parameter and the
        // requirements it points at outlive the call.
        let at = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                CHUNKS * CHUNK,
                MEM_RESERVE | MEM_WRITE_WATCH,
                PAGE_NOACCESS,
                &mut parameter,
                1,
            )
        };
        if at.is_null() {
            tracing::warn!("list upload memory: no room below 128 GiB; it comes from the heap, which shaders cannot reach by address");
            return None;
        }
        tracing::info!(
            at = format_args!("{:#x}", at as usize),
            "list upload memory comes from 4 MiB chunks shaders reach by address (VICTORY_UPLOAD_CHUNKS)"
        );
        Some(at as usize)
    }
}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth order and shading rate hints): any arguments,
/// nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        // (Through the draw state, which keeps them for the host list
        // that continues this one after a cut: `State::replay`.)
        with_state(this, |list, s| s.set_viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.set_scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        with_state(this, |list, s| {
            // SAFETY: the title passes four floats or null.
            s.set_blend_factor(list, unsafe { factor.as_ref() })
        });
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.set_stencil_ref(list, reference));
    }
    pub(super) extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
        with_state(this, |list, s| s.set_depth_bounds(list, min, max));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        if !handles.is_empty() || depth != 0 {
            super::tally_targets(this);
        }
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        super::tally(this, super::KIND_DRAW);
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        dispatch_x(this, x, y, z, 0);
    }
    pub(super) fn dispatch_x(this: This, x: u32, y: u32, z: u32, flags: u32) {
        super::count_dispatch();
        super::tally(this, super::KIND_DISPATCH);
        with_state(this, |list, s| s.dispatch(list, [x, y, z], flags));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        super::tally(this, super::KIND_DRAW);
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`. One whose
    /// group count is in memory (`indirect`, a title address) runs as an
    /// indirect execution of one mesh record (`D3D12_DISPATCH_MESH_ARGUMENTS`,
    /// three group counts, 12 bytes) through `State::execute_indirect`, as
    /// an `ExecuteIndirect` with a mesh signature does: its count is
    /// clamped there like any record's. The layout is inferred from the
    /// direct form and the PC's method, not seen in the title's code
    /// (Marvisak/VIctory#1546, which this follows; before, such dispatches
    /// were dropped). The flags and payload are logged the first time one
    /// is not 0; the host has no use for them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        flags: u64,
        indirect: u64,
        payload: u64,
    ) {
        super::count_draw();
        super::tally(this, super::KIND_DRAW);
        if flags != 0 || payload != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::info!(
                    flags = format_args!("{flags:#x}"),
                    payload = format_args!("{payload:#x}"),
                    "a mesh dispatch with flags or a payload, which the host does not use (logged once)"
                );
            });
        }
        if indirect != 0 {
            const MESH: u32 = 10;
            const STRIDE: u32 = 12;
            let mut ran = false;
            with_state(this, |list, s| {
                ran = s.execute_indirect(list, (MESH, STRIDE, 1), indirect, None);
            });
            if !ran {
                super::INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        indirect = format_args!("{indirect:#x}"),
                        "an indirect mesh dispatch did not run on the host (first time; counted with the dropped indirect executions)"
                    );
                });
            }
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        super::tally(this, super::KIND_INDEXED);
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as depth_bounds;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. The title's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    #[cfg(windows)]
    tally(this, KIND_INDIRECT);
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

/// A GPU wait for a memory value: the list's commands after it run when
/// the 32 or 64 bits at `address` compare as asked with `reference`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueWait {
    address: u64,
    comparison: u32,
    reference: u64,
    wide: bool,
}

impl ValueWait {
    /// Whether the memory holds what is waited for now.
    /// shortcut: the comparison codes are taken to be the comparison
    /// functions' (1 never .. 8 always, value against reference); one not
    /// known counts as met, and so does memory that is not mapped.
    ///
    /// The value is read through the kernel, which reports memory that is
    /// not there instead of faulting (as the copies of title memory are
    /// read, `crate::host::copy_mapped`). Asking Windows first whether the
    /// memory is committed (`VirtualQuery`) walks the page tables to the
    /// end of the region: the title keeps these values in a 1.1 GB
    /// allocation (0xd00000000), where one question took about 8 ms, and a
    /// queue asks at every submission and every signal, for each queue that
    /// stands at a wait (the title has five queues). That was
    /// 1.0-2.2 s of every cutscene frame on the thread that submits (runs
    /// 36-49: "between its submissions ... kernel"; 47 of 110 looks at
    /// that thread in run 49 found it in this question).
    /// `VICTORY_WAIT_ASK=1` asks first, as before.
    pub(crate) fn met(&self) -> bool {
        #[cfg(all(windows, feature = "diagnostics"))]
        let started = std::time::Instant::now();
        let met = self.holds();
        #[cfg(all(windows, feature = "diagnostics"))]
        {
            use crate::copystat::{C, add};
            add(C::WaitChecks, 1);
            add(C::WaitCheckNs, started.elapsed().as_nanos() as u64);
        }
        met
    }

    fn holds(&self) -> bool {
        let bytes = if self.wide { 8 } else { 4 };
        if self.address == 0 {
            return true;
        }
        static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ASK.get_or_init(|| std::env::var_os("VICTORY_WAIT_ASK").is_some_and(|v| v == "1"))
            && !crate::texture::committed(self.address as *const u8, bytes)
        {
            return true;
        }
        // The low `bytes` bytes; the rest stays zero.
        let mut raw = [0u8; 8];
        if !crate::host::read_into(self.address as *const u8, &mut raw[..bytes]) {
            return true;
        }
        self.holds_for(u64::from_le_bytes(raw))
    }

    /// Where the value waited for is: address and bytes.
    #[cfg(windows)]
    pub(crate) fn place(&self) -> (usize, usize) {
        (self.address as usize, if self.wide { 8 } else { 4 })
    }

    /// Whether the wait is met with `value` at its address (its low half
    /// for a 32-bit wait).
    pub(crate) fn holds_for(&self, value: u64) -> bool {
        let (value, reference) = if self.wide {
            (value, self.reference)
        } else {
            (value & 0xFFFF_FFFF, self.reference & 0xFFFF_FFFF)
        };
        match self.comparison {
            1 => false,
            2 => value < reference,
            3 => value == reference,
            4 => value <= reference,
            5 => value > reference,
            6 => value != reference,
            7 => value >= reference,
            _ => true,
        }
    }
}

/// The waits lists recorded since their last reset, by list.
static VALUE_WAITS: std::sync::Mutex<Vec<(usize, ValueWait)>> = std::sync::Mutex::new(Vec::new());

/// The waits of the submitted `lists` that are not met yet (taken from
/// them): what the console's GPU would stop at. The title orders work
/// between its queues this way (the title: the compute queue's
/// lighting passes wait for a value the graphics queue's depth pass writes
/// at the bottom of the pipe; run at once, they read a cleared depth
/// buffer and nothing was lit).
/// shortcut: a wait holds its whole list (and submission) back, not only
/// the commands after it.
pub(crate) fn unmet_waits(lists: &[This]) -> Vec<ValueWait> {
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut unmet = Vec::new();
    waits.retain(|&(list, wait)| {
        if !lists.iter().any(|&l| l as usize == list) {
            return true;
        }
        if !wait.met() {
            unmet.push(wait);
        }
        false
    });
    unmet
}

/// The waits each of the submitted `lists` recorded (taken from them),
/// met or not, at the list's first place in the submission.
#[cfg(windows)]
pub(crate) fn waits_by_list(lists: &[This]) -> Vec<Vec<ValueWait>> {
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut by = vec![Vec::new(); lists.len()];
    waits.retain(|&(list, wait)| {
        match lists.iter().position(|&l| l as usize == list) {
            Some(i) => {
                by[i].push(wait);
                false
            }
            None => true,
        }
    });
    by
}

fn wait_for(this: This, address: u64, comparison: u32, reference: u64, wide: bool) {
    let wait = ValueWait {
        address,
        comparison,
        reference,
        wide,
    };
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
        tracing::info!(
            address = format_args!("{address:#x}"),
            comparison,
            reference,
            wide,
            met = wait.met(),
            "a list waits for a memory value"
        );
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some(host) = host_list(this) {
        crate::capture::op(std::ptr::from_ref(host) as usize, || {
            format!(
                "WAIT VALUE at {address:#x} comparison {comparison} reference {reference:#x} wide {wide} met {}",
                wait.met()
            )
        });
    }
    // The list is cut here, and the wait stands before what follows.
    #[cfg(windows)]
    if crate::queue::cuts() {
        cut(this, wait);
        return;
    }
    VALUE_WAITS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((this as usize, wait));
}
extern "system" fn wait_value(this: This, address: u64, comparison: u32, reference: u64) {
    wait_for(this, address, comparison, reference, false);
}
extern "system" fn wait_value64(this: This, address: u64, comparison: u32, reference: u64) {
    wait_for(this, address, comparison, reference, true);
}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    // Diagnostics: counted, and left out under `VICTORY_BARRIER_TEST=groups`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if count > 0 {
        crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
        if crate::copystat::barrier_test() & 1 != 0 {
            return;
        }
    }
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // SAFETY: the same layout; no resource named.
        let barrier = unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
    value_written(this, address, u64::from(value));
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
    if let Written::Value(value) = what {
        value_written(this, address, value);
    }
}

/// The title wrote a value at the bottom of the pipe (not a timestamp, not
/// a query result): listed (diagnostics), and with `VICTORY_CUT_WRITES=1`
/// the host list is cut right behind it, so that the value is stored when
/// the card has run the list up to there, as on the console, and not when
/// the whole list has run. Lists of other queues wait for these values
/// (the title: the compute list that builds the depth pyramid
/// waits for one, and the meshlet culling of the same frame reads that
/// pyramid). Off by default: every such write then ends a host list, and
/// each part is a submission of its own.
fn value_written(this: This, address: u64, value: u64) {
    #[cfg(windows)]
    {
        if address == 0 || this.is_null() {
            return;
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = value;
        #[cfg(feature = "diagnostics")]
        if let Some(host) = host_list(this) {
            crate::capture::op(std::ptr::from_ref(host) as usize, || {
                format!("WRITE VALUE at {address:#x} value {value:#x}")
            });
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WRITES").is_some_and(|v| v == "1"))
            || !crate::queue::cuts()
        {
            return;
        }
        // SAFETY: the title passes one of our lists; a list is recorded by
        // one thread at a time and is not submitted while it records.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        let Some(current) = list.host.as_ref() else {
            return;
        };
        let Some(fresh) = fresh_list() else {
            return;
        };
        current.close();
        if let Some(done) = list.host.replace(fresh) {
            // The part so far keeps its waits; nothing stands before the
            // next one but this part itself.
            let waits = std::mem::take(&mut list.gate);
            list.earlier.push(Part { waits, host: done });
        }
        if let Some(host) = list.host.as_ref() {
            list.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .replay(host);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, address, value);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, state::depth_bounds as *const ()),       // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait_value as *const ()),                // 0x238, 32 bits
    (72, wait_value64 as *const ()),              // 0x240, 64 bits
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 9] = [
    (28, query_mark as *const ()),             // 0xe0 BeginQuery
    (29, query_mark as *const ()),             // 0xe8 EndQuery
    (30, resolve_queries as *const ()),        // 0xf0
    (78, dispatch_x as *const ()),             // 0x270
    (98, state::dispatch_mesh_x as *const ()), // 0x310
    (99, driver_hint as *const ()),            // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// The title's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from the title's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as the title asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        // Where shaders reach title memory by address; more than a chunk
        // comes from the heap. (Carved from chunks only with the builder
        // asked for: the chunks themselves, here.)
        #[cfg(windows)]
        {
            let mut chunks = Uploads::default();
            if let Some(p) = chunks.carve(40, 0x100) {
                assert!((1 << 32..(victory_shader::emit::MEMORY_PAGES as usize) << 16).contains(&(p as usize)));
                let q = chunks.carve(24, 16);
                assert_eq!(q.map(|q| q as usize), Some(p as usize + 48), "carved from the same chunk");
                assert!(chunks.carve(8 << 20, 16).is_none(), "more than a chunk comes from the heap");
                // SAFETY: committed bytes of the chunk.
                unsafe { p.write_bytes(0xab, 40) };
            }
            assert!(!allocate_upload(list, 8 << 20, 16).is_null());
        }
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// Group counts pass as they are; thread counts (bit 5, also with the
    /// 32-lane bit 0x8000) become the groups that hold them, and a count
    /// that is not a multiple of the group says its last group is partial
    /// (after fxceless/victory-fork#646's test).
    #[test]
    fn a_dispatch_that_counts_threads_runs_the_groups_that_hold_them() {
        assert_eq!(dispatch_groups([48, 1, 1], [64, 1, 1], 0), Some(([48, 1, 1], false)));
        assert_eq!(dispatch_groups([48, 1, 1], [64, 1, 1], 0x8000), Some(([48, 1, 1], false)));
        assert_eq!(dispatch_groups([0, 1, 1], [64, 1, 1], 0), Some(([0, 1, 1], false)));
        assert_eq!(dispatch_groups([48, 1, 1], [64, 1, 1], 0x20), Some(([1, 1, 1], true)));
        assert_eq!(dispatch_groups([1, 1, 1], [32, 1, 1], 0x8020), Some(([1, 1, 1], true)));
        assert_eq!(dispatch_groups([65, 1, 1], [64, 1, 1], 0x20), Some(([2, 1, 1], true)));
        assert_eq!(dispatch_groups([128, 1, 1], [64, 1, 1], 0x20), Some(([2, 1, 1], false)));
        assert_eq!(dispatch_groups([10, 16, 5], [8, 8, 4], 0x20), Some(([2, 2, 2], true)));
        assert_eq!(dispatch_groups([16, 16, 4], [8, 8, 4], 0x20), Some(([2, 2, 1], false)));
        // The builder's box stage, run 64 times over before.
        assert_eq!(dispatch_groups([382_912, 1, 1], [64, 1, 1], 0x20), Some(([5_983, 1, 1], false)));
        // A program whose group size is not known divides by 1.
        assert_eq!(dispatch_groups([7, 1, 1], [0; 3], 0x20), Some(([7, 1, 1], false)));
        assert_eq!(dispatch_groups([0, 1, 1], [64, 1, 1], 0x20), None);
    }

    /// An indirect `DispatchMeshX` goes to the indirect execution, which
    /// without a host GPU or a bound pipeline does not run it: it is
    /// counted with the dropped executions, not lost unseen.
    #[cfg(windows)]
    #[test]
    fn an_indirect_mesh_dispatch_goes_to_the_indirect_execution() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut list = null;
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        let _ = take_indirect();
        state::dispatch_mesh_x(list, 0, 0, 0x1_0000_0000, 0);
        assert!(take_indirect() >= 1);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }
}
