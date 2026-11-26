//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Library calls overlap: the host documents the library as safe to use
//! from several threads ("The pipeline library is thread-safe to use, and
//! will internally synchronize as necessary", `CreatePipelineLibrary`,
//! Thread Safety), with one exception kept here: two loads of the same
//! pipeline are made one after the other ([`same_name`]). Under one lock of
//! ours for every call, the eight workers made pipelines one at a time:
//! 87 to 96 of 110 looks at each worker found it waiting for that lock
//! (run 49), and an earlier run's 2,300 pipelines, all in the library, were
//! still being loaded 46 s into the opening cutscene.
//! `VICTORY_LIBRARY_LOCK=1`: one call at a time, as before.
//!
//! The library's file is mapped, not read into memory ([`Blob`]): the host
//! keeps pointing into the bytes it was made from for as long as it lives,
//! and read into a buffer they were 1,352 MB of the process's own memory
//! for the whole run (run 60), all of it in physical memory. Mapped, the
//! pages are the file's: only those the host reads are in memory, and the
//! system can drop them without writing them anywhere. A save writes the
//! new library through a mapping of the new file in the same way (a buffer
//! of the library's size was made, filled and written out at each save).
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.
//!
//! The mapped file's handle stays open for as long as the mapping lives
//! ([`holds`]). A save renames the new library over the file this run has
//! mapped, and with the handle closed Windows then gives up the old file's
//! contents: every page of the mapping that is not in memory at that
//! moment, or leaves memory later (the trim below, or Windows taking pages
//! back), reads as zeros from then on. The host's library reads such pages
//! as stored pipelines; the driver, handed a pipeline of zeros, removes the
//! device with DXGI_ERROR_DRIVER_INTERNAL_ERROR (runs 72, 73 and 82, each
//! within 5 ms of the first library call after a save and a trim; and the
//! library run 77 saved from such pages was refused by run 78). Measured
//! with a 64 MB file written past the system's cache, mapped and replaced:
//! 16,352 of 16,384 pages read zeros with the handle closed, none with it
//! open (the test below). `VICTORY_LIBRARY_HANDLE=0`: closed, as before.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on: the
/// early list in `dir` itself, the library in a directory of its
/// translation's there ([`library_dir`]).
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The library's directory: `DIR/<translation>`, the first 16 hex digits of
/// a hash of the translation's fingerprint
/// (`victory_shader::emit::translation_fingerprint`, which the shader cache
/// keys on too), and the library of another translation it is seeded from
/// ([`keyed`]). Worked out at the first use, after the device has told the
/// translator its facts (wave size, view heap limit).
///
/// A pipeline's name ([`name`]) is a hash of what its host description
/// holds: our root signature, the translated shaders' code and the state.
/// A translator change changes the names of the pipelines whose shaders it
/// translates differently, and of no other: a pipeline of an unchanged
/// shader keeps its name in every translation. So a library of another
/// translation holds the right pipeline for every name it still has. What
/// a translation's own library holds is what this translation used:
/// pipelines made, and pipelines found in the seed, which are stored into
/// it as they are found ([`pipeline`]); pipelines of shaders that
/// translate differently now are never asked for, so they stay behind in
/// the older libraries, which go when they are no longer among the [`KEPT`]
/// newest. The own library therefore does not grow past what the title
/// uses (Marvisak/VIctory #2706 wanted a library per translation because a
/// first run after a change wrote a small library over the big one; one of
/// its own each, seeded, keeps both).
fn library_dir() -> Option<&'static PathBuf> {
    keyed_dirs().map(|(dir, _)| dir)
}

/// The library files the own library is seeded from, in the order they
/// are asked ([`library_dir`], [`keyed`]).
fn seed_files() -> &'static [PathBuf] {
    keyed_dirs().map_or(&[], |(_, seeds)| seeds.as_slice())
}

fn keyed_dirs() -> Option<&'static (PathBuf, Vec<PathBuf>)> {
    static KEYED: OnceLock<Option<(PathBuf, Vec<PathBuf>)>> = OnceLock::new();
    KEYED
        .get_or_init(|| {
            let dir = DIR.get()?;
            let print = blake3::hash(&victory_shader::emit::translation_fingerprint());
            let name: String = print.as_bytes()[..8].iter().map(|b| format!("{b:02x}")).collect();
            Some(keyed(dir, &name))
        })
        .as_ref()
}

/// Translations that keep a library: the one in use and the two before (a
/// run cut short leaves a small library, and the one before it is still
/// there to seed from).
const KEPT: usize = 3;

/// `dir/<name>`, and the libraries to seed it from, newest first: the
/// other translations' libraries, then the one file earlier builds kept in
/// `dir` itself (left as it is: it no longer changes). The other
/// translations' libraries are pruned to the newest [`KEPT`]` - 1` (by
/// when they were last written). Only what this wrote goes: directories
/// named by 16 lowercase hex digits that hold a library.
fn keyed(dir: &std::path::Path, name: &str) -> (PathBuf, Vec<PathBuf>) {
    let path = dir.join(name);
    let ours = |other: &std::path::Path| {
        other.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
            n.len() == 16 && n.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }) && other.join(LIBRARY_FILE).is_file()
    };
    let written = |file: &std::path::Path| file.metadata().ok()?.modified().ok();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (path, Vec::new());
    };
    let mut others: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|other| *other != path && ours(other))
        .filter_map(|other| Some((written(&other.join(LIBRARY_FILE))?, other)))
        .collect();
    others.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
    let mut seeds = Vec::new();
    for (k, (_, other)) in others.into_iter().enumerate() {
        if k < KEPT - 1 {
            seeds.push(other.join(LIBRARY_FILE));
        } else {
            tracing::info!(dir = %other.display(), "pipeline library of an older translation removed");
            let _ = std::fs::remove_dir_all(&other);
        }
    }
    let shared = dir.join(LIBRARY_FILE);
    if written(&shared).is_some() {
        seeds.push(shared);
    }
    (path, seeds)
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// For the memory lines: the bytes of the library's file, kept in memory
/// while the library lives, and the bytes the last save wrote (a buffer of
/// that size is made, filled and written every time pipelines were added,
/// at most every 20 s).
pub(crate) static BLOB: AtomicU64 = AtomicU64::new(0);
pub(crate) static SAVED: AtomicU64 = AtomicU64::new(0);

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// Whether the mapped library file's handle is kept open while it is
/// mapped (see the module's notes: without it a save in the same run turns
/// the mapping's pages into zeros as they leave memory). On by default;
/// `VICTORY_LIBRARY_HANDLE=0` closes it after mapping, as before.
fn holds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_HANDLE").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`] ([`holds`]): open
    /// until after the view is unmapped, so that the file a rename took
    /// the name from keeps its contents for the view.
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the file's pages
        // after the mapping object and the file are closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(mapping, if write { FILE_MAP_WRITE } else { FILE_MAP_READ }, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped { at, len, _file: None })
    }

    /// The file at `path`, mapped for reading; `None` if it is not there,
    /// is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        Mapped::open_holding(path, holds())
    }

    /// As [`Mapped::open`]; `hold` says whether the file stays open with
    /// the mapping.
    fn open_holding(path: &std::path::Path, hold: bool) -> Option<Mapped> {
        // Opened as the standard library opens a file to read: others may
        // read, write, rename and delete it meanwhile, so a save still
        // renames the new library over this one.
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        if hold {
            mapped._file = Some(file);
        }
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, tests, or a file
    /// that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    fn of(dir: &std::path::Path) -> Blob {
        let path = dir.join(LIBRARY_FILE);
        let next = dir.join(NEXT_FILE);
        if next.exists() && std::fs::rename(&next, &path).is_err() {
            tracing::warn!("the host pipeline library saved last could not take the old one's place");
        }
        if mapped()
            && let Some(file) = Mapped::open(&path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(&path) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

/// The mapped library file's view (address, bytes; 0: not mapped), when a
/// pipeline was last loaded from the library (`crate::host::now_ms`), and
/// whether the view's pages were let go since ([`trim`]).
static VIEW: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];
static LOADED_AT: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// The bytes of the view let go so far, for the memory line.
pub(crate) static TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the view's pages were last let go (`crate::host::now_ms`), and how
/// often and when a save of this run last renamed a new library over the
/// mapped file: for [`removal_note`].
static TRIMMED_AT: AtomicU64 = AtomicU64::new(0);
static REPLACED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Calls into the host driver for pipelines, for [`driver_line`]: how many
/// run now (all kinds, and the creating ones), the most at once since the
/// last line (the same two), and per kind (create, load, store) the calls
/// and their microseconds since the last line. The NVIDIA driver was
/// reported to build about two pipelines at a time per process, however
/// many threads call it; the line shows whether ours is serialised.
static IN_DRIVER: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static MOST_IN_DRIVER: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static DRIVER_CALLS: [[AtomicU64; 2]; 3] = [const { [const { AtomicU64::new(0) }; 2] }; 3];
static DRIVER_LINE_AT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
enum Driver {
    Create,
    Load,
    Store,
}

/// A call into the driver of this thread, counted until dropped.
struct InDriver(Driver, std::time::Instant);

fn in_driver(kind: Driver) -> InDriver {
    let all = IN_DRIVER[0].fetch_add(1, Ordering::Relaxed) + 1;
    MOST_IN_DRIVER[0].fetch_max(all, Ordering::Relaxed);
    if matches!(kind, Driver::Create) {
        let creating = IN_DRIVER[1].fetch_add(1, Ordering::Relaxed) + 1;
        MOST_IN_DRIVER[1].fetch_max(creating, Ordering::Relaxed);
    }
    InDriver(kind, std::time::Instant::now())
}

impl Drop for InDriver {
    fn drop(&mut self) {
        let us = self.1.elapsed().as_micros() as u64;
        let slot = &DRIVER_CALLS[self.0 as usize];
        slot[0].fetch_add(1, Ordering::Relaxed);
        slot[1].fetch_add(us, Ordering::Relaxed);
        IN_DRIVER[0].fetch_sub(1, Ordering::Relaxed);
        if matches!(self.0, Driver::Create) {
            IN_DRIVER[1].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// With the periodic lines: the driver calls since the last such line, if
/// there were any (the library's loads and stores count with the creates:
/// all of them are the driver's work on a pipeline).
pub(crate) fn driver_line() {
    let now = crate::host::now_ms();
    let wall = now.saturating_sub(DRIVER_LINE_AT.swap(now, Ordering::Relaxed)).max(1);
    let [[creates, create_us], [loads, load_us], [stores, store_us]] =
        std::array::from_fn(|k| [0, 1].map(|i| DRIVER_CALLS[k][i].swap(0, Ordering::Relaxed)));
    // What runs now counts for the next interval's most.
    let most = [0, 1].map(|i| MOST_IN_DRIVER[i].swap(IN_DRIVER[i].load(Ordering::Relaxed), Ordering::Relaxed));
    let calls = creates + loads + stores;
    if calls == 0 {
        return;
    }
    let sum_ms = (create_us + load_us + store_us) as f64 / 1e3;
    tracing::info!(
        "pipelines made by the driver: {calls} in the interval, {sum_ms:.0} ms of calls in sum over {wall} ms of wall time          ({:.2} at once on average), the most in flight at once: {} (creating: {} of them); by kind: {creates} created in          {:.0} ms, {loads} loaded from the library in {:.0} ms, {stores} stored in {:.0} ms",
        sum_ms / wall as f64,
        most[0],
        most[1],
        create_us as f64 / 1e3,
        load_us as f64 / 1e3,
        store_us as f64 / 1e3,
    );
}

/// The library calls running now: thread, what (load, make, store,
/// serialise), the pipeline, and since when (`crate::host::now_us`).
static CALLS: Mutex<Vec<(u32, &'static str, String, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// The pipeline this thread is making (its shaders), for [`CALLS`].
    static LABEL: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Names the pipeline this thread makes next.
pub(crate) fn label(text: String) {
    LABEL.with(|l| *l.borrow_mut() = text);
}

/// A library call of this thread, listed in [`CALLS`] until dropped.
struct Call(u32, u64);

fn call(what: &'static str, name: &str) -> Call {
    let (thread, since) = (crate::host::thread_id(), crate::host::now_us());
    let label = LABEL.with(|l| l.borrow().clone());
    CALLS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((thread, what, format!("{name} {label}"), since));
    Call(thread, since)
}

impl Drop for Call {
    fn drop(&mut self) {
        CALLS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|c| (c.0, c.3) != (self.0, self.1));
    }
}

/// For the report when the host device is removed: the library calls
/// running at that moment, whether this run replaced the mapped file and
/// let its pages go, and how many pages of the mapping read as zeros now.
/// A library holds no page of zeros (the 259 MB one of run 83 has no run of
/// 2,048 zero bytes): each such page is one whose contents were lost (see
/// the module's notes), and a pipeline stored there is what the driver was
/// handed.
pub(crate) fn removal_note() -> String {
    use std::fmt::Write;
    let now = crate::host::now_us();
    let mut text = String::from("pipeline library: ");
    {
        let calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        if calls.is_empty() {
            text.push_str("no call running");
        }
        for (i, (thread, what, name, since)) in calls.iter().enumerate() {
            let _ = write!(
                text,
                "{}{what} of {name} running for {:.1} ms on thread {thread}",
                if i == 0 { "" } else { ", " },
                now.saturating_sub(*since) as f64 / 1000.0
            );
        }
    }
    let (at, len) = (VIEW[0].load(Ordering::Relaxed), VIEW[1].load(Ordering::Relaxed));
    if at == 0 {
        text.push_str("; its file is not mapped");
        return text;
    }
    let ago = |ms: u64| crate::host::now_ms().saturating_sub(ms) as f64 / 1000.0;
    let _ = write!(
        text,
        "; its file is mapped ({len} bytes), the handle {}",
        if holds() { "kept open" } else { "closed (VICTORY_LIBRARY_HANDLE=0)" }
    );
    match REPLACED[0].load(Ordering::Relaxed) {
        0 => text.push_str("; this run has not replaced the file"),
        n => {
            let _ = write!(
                text,
                "; this run replaced the file under the mapping {n} times, last {:.1} s ago",
                ago(REPLACED[1].load(Ordering::Relaxed))
            );
        }
    }
    if TRIMMED_BYTES.load(Ordering::Relaxed) > 0 {
        let _ = write!(text, "; its pages were last let go {:.1} s ago", ago(TRIMMED_AT.load(Ordering::Relaxed)));
    }
    // Within two seconds: pages not in memory come from the disk.
    let started = std::time::Instant::now();
    // SAFETY: the view lives as long as the library, which is never dropped.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, len) };
    let (mut zero, mut looked, mut first) = (0u64, 0u64, None);
    for (n, page) in bytes.chunks(4096).enumerate() {
        if n % 4096 == 0 && started.elapsed() > std::time::Duration::from_secs(2) {
            break;
        }
        looked += 1;
        if page.iter().all(|&b| b == 0) {
            zero += 1;
            if first.is_none() {
                first = Some(n);
            }
        }
    }
    let _ = write!(text, "; {zero} of the {looked} pages looked at read as zeros now");
    if let Some(first) = first {
        let _ = write!(text, " (the first is page {first}): the library's contents were lost there");
    }
    text
}

/// Lets the mapped library file's pages go from the process's memory once
/// no pipeline has been loaded for ten seconds. The host reads nearly the
/// whole file while the early pipelines load (run 65: all 1,490 MB of it
/// were in the process's memory a minute in, and 1,300 MB still at the
/// end, on a PC with 2 GB free); after that it reads the file again only
/// when a new pipeline loads. Pages let go stay in Windows' file cache
/// while nothing else wants the memory, and come back on the next read.
/// Called once a second by the sweep; does its work on a thread of its own
/// (hundreds of thousands of pages), once after each burst of loads.
/// `VICTORY_LIBRARY_TRIM=0`: the pages stay until Windows takes them.
pub(crate) fn trim() {
    static ON: OnceLock<bool> = OnceLock::new();
    let (at, len) = (VIEW[0].load(Ordering::Relaxed), VIEW[1].load(Ordering::Relaxed));
    if at == 0
        || TRIMMED.load(Ordering::Relaxed)
        || !*ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_TRIM").is_none_or(|v| v != "0"))
    {
        return;
    }
    let quiet_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    if crate::host::now_ms() < LOADED_AT.load(Ordering::Relaxed) + quiet_ms
        || TRIMMED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let work = move || {
        // SAFETY: the view lives as long as the library, which is never
        // dropped. Unlocking pages that are not locked fails, as expected,
        // and takes them out of the working set: the documented way to let
        // pages of a mapped file go.
        unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len) };
        TRIMMED_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        TRIMMED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::host::trail(|| format!("pipeline library: the mapped file's pages let go from memory ({len} bytes)"));
    };
    if std::thread::Builder::new().name("victory-library-trim".into()).spawn(work).is_err() {
        TRIMMED.store(false, Ordering::Relaxed);
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// Stores share it and a save takes it alone: the size the library
    /// asks for must still be its size when it is written out.
    saving: RwLock<()>,
    /// Libraries of other translations, asked for a pipeline this one does
    /// not have ([`seed_files`]); read only.
    seeds: Vec<Seed>,
}

/// A library of another translation and the bytes it is made from (in
/// this order: the library goes before its bytes).
struct Seed {
    library: ID3D12PipelineLibrary1,
    _blob: Blob,
}

/// Pipelines found in a seed and stored into the own library since the
/// start ([`pipeline`]).
pub(crate) static SEEDED: AtomicU64 = AtomicU64::new(0);

// SAFETY: D3D12 objects are free-threaded, and the library synchronises
// its own calls (see the module's notes); what it does not, two loads of
// one name, is serialised by `same_name`.
unsafe impl Send for Library {}
// SAFETY: as above; the other fields are an atomic and a lock.
unsafe impl Sync for Library {}

/// Held by a load of the pipeline named `name`: loads of the same name
/// wait for each other (the host's one exception to the library's thread
/// safety: such a load may change that pipeline's entry). Names share 64
/// locks, so two different names seldom wait.
fn same_name(name: &str) -> MutexGuard<'static, ()> {
    static LOADING: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];
    let at = name.bytes().fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(usize::from(b)));
    LOADING[at % LOADING.len()].lock().unwrap_or_else(|e| e.into_inner())
}

/// With `VICTORY_LIBRARY_LOCK=1`: held by every library call, as before.
fn one_call() -> Option<MutexGuard<'static, ()>> {
    static ON: OnceLock<bool> = OnceLock::new();
    static CALLING: Mutex<()> = Mutex::new(());
    ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_LOCK").is_some_and(|v| v == "1"))
        .then(|| CALLING.lock().unwrap_or_else(|e| e.into_inner()))
}

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = library_dir().map_or(Blob::None, |d| Blob::of(d));
            let mut library = open(&device, blob);
            match &mut library {
                Some(library) => library.seeds = seed_files().iter().filter_map(|f| open_seed(&device, f)).collect(),
                None => tracing::warn!("no host pipeline library: pipelines are not kept"),
            }
            library
        })
        .as_ref()
}

/// The library in the file `path` of another translation, to seed this
/// one's from; `None` if it is not there or the host rejects it.
fn open_seed(device: &ID3D12Device1, path: &std::path::Path) -> Option<Seed> {
    let blob = match mapped().then(|| Mapped::open(path)).flatten() {
        Some(file) => Blob::Mapped(file),
        None => Blob::Held(std::fs::read(path).ok().filter(|b| !b.is_empty())?),
    };
    // SAFETY: the bytes outlive the library (kept beside it).
    let made = unsafe { device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(blob.bytes()) }
        .and_then(|l| l.cast::<ID3D12PipelineLibrary1>());
    match made {
        Ok(library) => {
            tracing::info!(file = %path.display(), bytes = blob.bytes().len(), "host pipeline library of another translation opened to seed this one's");
            Some(Seed { library, _blob: blob })
        }
        Err(e) => {
            tracing::info!(file = %path.display(), error = %e, "host pipeline library of another translation rejected");
            None
        }
    }
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.bytes().is_empty() {
        match make(blob.bytes()) {
            Ok(library) => {
                let bytes = blob.bytes().len();
                BLOB.store(bytes as u64, Ordering::Relaxed);
                tracing::info!(bytes, mapped = matches!(blob, Blob::Mapped(_)), "host pipeline library loaded");
                if let Blob::Mapped(file) = &blob {
                    VIEW[1].store(file.len, Ordering::Relaxed);
                    VIEW[0].store(file.at.Value as usize, Ordering::Relaxed);
                    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
                }
                return Some(Library {
                    library,
                    _blob: blob,
                    added: AtomicU64::new(0),
                    saving: RwLock::new(()),
                    seeds: Vec::new(),
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        added: AtomicU64::new(0),
        saving: RwLock::new(()),
                    seeds: Vec::new(),
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else from a library of another
/// translation (and stored into this one), else made by `create` and
/// stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        let _driver = in_driver(Driver::Create);
        return create();
    };
    pipeline_in(library, name, load, create)
}

/// [`pipeline`] with `library`.
fn pipeline_in(
    library: &Library,
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let began = std::time::Instant::now();
    let loaded = {
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load", name);
        let _driver = in_driver(Driver::Load);
        load(&library.library, PCWSTR(wide.as_ptr()))
    };
    // The load read the library's file (see `trim`).
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
    let refused = match loaded {
        Ok(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            crate::host::trail(|| {
                let label = LABEL.with(|l| l.borrow().clone());
                format!("pipeline library: loaded {label} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3)
            });
            return Ok(pso);
        }
        Err(e) => e,
    };
    // Another translation's library: a pipeline of a shader this
    // translation translates as it did is there under the same name.
    let seeded = library.seeds.iter().find_map(|seed| {
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load from a seed", name);
        let _driver = in_driver(Driver::Load);
        load(&seed.library, PCWSTR(wide.as_ptr())).ok()
    });
    let from_seed = seeded.is_some();
    let pso = match seeded {
        Some(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            SEEDED.fetch_add(1, Ordering::Relaxed);
            pso
        }
        None => {
            // Compiled outside any lock: compiles take up to seconds.
            let pso = {
                let _call = call("make", name);
                let _driver = in_driver(Driver::Create);
                create()?
            };
            MADE.fetch_add(1, Ordering::Relaxed);
            pso
        }
    };
    crate::host::trail(|| {
        let label = LABEL.with(|l| l.borrow().clone());
        format!(
            "pipeline library: {} {label} in {:.1} ms (the library did not load it: {refused})",
            if from_seed { "loaded from another translation's" } else { "made" },
            began.elapsed().as_secs_f64() * 1e3
        )
    });
    let _all = one_call();
    let _stores = library.saving.read().unwrap_or_else(|e| e.into_inner());
    let _call = call("store", name);
    let _driver = in_driver(Driver::Store);
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { library.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        library.added.fetch_add(1, Ordering::Relaxed);
    }
    Ok(pso)
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    // (The library first: its directory is worked out after the device.)
    let Some(library) = library() else { return };
    let (Some(early_dir), Some(dir)) = (DIR.get(), library_dir()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    let tmp = dir.join(LIBRARY_FILE).with_extension("tmp");
    // The bytes of the library written to `tmp`; 0 when no pipeline was
    // added: the library on disk is then what it was, and only the early
    // list is written (a title pipeline first drawn with wrote all 1,352 MB
    // of the library again: run 60's second save).
    let bytes = {
        let _all = one_call();
        // No store between asking the size and writing; loads go on.
        let _alone = library.saving.write().unwrap_or_else(|e| e.into_inner());
        let added = library.added.load(Ordering::Relaxed);
        if !USED_CHANGED.swap(false, Ordering::Relaxed) && added == 0 {
            return;
        }
        if added == 0 {
            0
        } else {
            // SAFETY: a plain query.
            let size = unsafe { library.library.GetSerializedSize() };
            let _call = call("serialise", "the whole library");
            if let Err(e) = serialise(&library.library, &tmp, size) {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            library.added.store(0, Ordering::Relaxed);
            SAVED.store(size as u64, Ordering::Relaxed);
            size
        }
    };
    let used: Vec<String> = USED.lock().unwrap_or_else(|e| e.into_inner()).iter().flatten().cloned().collect();
    let library_written = if bytes > 0 { replace_library(dir, &tmp) } else { Ok(()) };
    if bytes > 0 && library_written.is_ok() {
        // In place unless the system refused and the new one waits beside it.
        let in_place = !dir.join(NEXT_FILE).exists();
        if in_place && VIEW[0].load(Ordering::Relaxed) != 0 {
            REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
            REPLACED[0].fetch_add(1, Ordering::Relaxed);
        }
        crate::host::trail(|| {
            format!(
                "pipeline library: saved {bytes} bytes, {}",
                if in_place { "renamed over the library's file" } else { "left beside the library's file for the next start" }
            )
        });
    }
    let written = library_written
        .and_then(|()| write_atomically(&early_dir.join(EARLY_FILE), used.join("\n").as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            from_other_translations = SEEDED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes the library's `size` serialised bytes to the file `tmp`: through
/// a mapping of that file, so that the bytes are the file's pages and not
/// 1.3 GB more of the process's own memory for the seconds a save takes;
/// through a buffer where the file cannot be mapped (and with
/// `VICTORY_LIBRARY_MAP=0`).
fn serialise(library: &ID3D12PipelineLibrary1, tmp: &std::path::Path, size: usize) -> std::io::Result<()> {
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    if mapped()
        && let Some(mut file) = Mapped::create(tmp, size)
    {
        // SAFETY: a buffer of the size the library asks for.
        return unsafe { library.Serialize(file.bytes_mut()) }.map_err(failed);
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: a buffer of the size the library asks for.
    unsafe { library.Serialize(&mut bytes) }.map_err(failed)?;
    std::fs::write(tmp, &bytes)
}

/// Renames the written library `tmp` over the library's file in `dir`. This
/// run may have that file mapped: Windows 11 replaces a mapped file, and
/// the mapping goes on reading the old one for as long as a handle on it
/// is open ([`holds`]; with none, only the pages in memory at that moment
/// stay: the rest read as zeros). Where a system refuses, the new library
/// is left beside it as [`NEXT_FILE`] for the next start.
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(tmp, dir.join(LIBRARY_FILE)).or_else(|_| std::fs::rename(tmp, dir.join(NEXT_FILE)))
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys requested by draws or dispatches (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was requested in this or an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// A draw or dispatch requested the title pipeline with `key`.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16))).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Blob::None));
    }

    /// A library of another translation gives a pipeline this one does not
    /// have under the same name, and the pipeline is stored into this one:
    /// nothing is compiled.
    #[test]
    fn a_pipeline_comes_from_another_translations_library() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let source = "RWByteAddressBuffer b : register(u0); [numthreads(1, 1, 1)] void main() { b.Store(0, 1); }";
        let blob_bytes = |b: &ID3DBlob| {
            // SAFETY: a live blob's own bytes.
            unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }.to_vec()
        };
        // SAFETY: plain compiles and object creation with live inputs.
        let (root, code) = unsafe {
            let mut code: Option<ID3DBlob> = None;
            D3DCompile(source.as_ptr().cast(), source.len(), None, None, None, windows::core::s!("main"), windows::core::s!("cs_5_1"), 0, 0, &mut code, None).unwrap();
            let range = D3D12_DESCRIPTOR_RANGE { RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV, NumDescriptors: 1, ..Default::default() };
            let parameter = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &parameter, ..Default::default() };
            let mut serialized: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut serialized, None).unwrap();
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &blob_bytes(serialized.as_ref().unwrap())).unwrap();
            (root, blob_bytes(code.as_ref().unwrap()))
        };
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
            ..Default::default()
        };
        // SAFETY: the description points at live data.
        let load = |library: &ID3D12PipelineLibrary1, name: PCWSTR| unsafe { library.LoadComputePipeline(name, &desc) };
        // SAFETY: as above.
        let create = || unsafe { device.CreateComputePipelineState(&desc) };
        // The other translation's library, holding the pipeline.
        let other = open(&device, Blob::None).unwrap();
        pipeline_in(&other, "seeded test", load, create).unwrap();
        // SAFETY: plain queries and a buffer of the asked size.
        let bytes = unsafe {
            let mut bytes = vec![0u8; other.library.GetSerializedSize()];
            other.library.Serialize(&mut bytes).unwrap();
            bytes
        };
        // SAFETY: the bytes live beside the library.
        let seed = unsafe { device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(&bytes) }.unwrap().cast().unwrap();
        let mut own = open(&device, Blob::None).unwrap();
        own.seeds.push(Seed { library: seed, _blob: Blob::Held(bytes) });
        let seeded = SEEDED.load(Ordering::Relaxed);
        pipeline_in(&own, "seeded test", load, || panic!("compiled although another translation's library had it")).unwrap();
        assert!(SEEDED.load(Ordering::Relaxed) > seeded);
        assert_eq!(own.added.load(Ordering::Relaxed), 1, "stored into the own library");
        // Next time from the own library itself.
        own.seeds.clear();
        assert!(load(&own.library, PCWSTR(HSTRING::from("seeded test").as_ptr())).is_ok());
    }

    /// Each translation's library has a directory of its own; the newest
    /// two of the other translations' stay and are the seeds, newest first,
    /// then the file of earlier builds; what this did not write (a
    /// directory of another name, one without a library) is left alone.
    #[test]
    fn a_library_per_translation_seeded_from_the_ones_before() {
        let dir = tempfile::tempdir().unwrap();
        let library = |name: &str, age: u64| {
            let at = dir.path().join(name);
            std::fs::create_dir_all(&at).unwrap();
            let file = std::fs::File::create(at.join(LIBRARY_FILE)).unwrap();
            let when = std::time::SystemTime::now() - std::time::Duration::from_secs(age);
            file.set_modified(when).unwrap();
        };
        library("00000000000000aa", 300);
        library("00000000000000bb", 100);
        library("00000000000000cc", 200);
        library("00000000000000ee", 400);
        std::fs::write(dir.path().join(LIBRARY_FILE), b"earlier builds").unwrap();
        std::fs::create_dir_all(dir.path().join("backup")).unwrap();
        std::fs::create_dir_all(dir.path().join("00000000000000dd")).unwrap(); // no library
        let (path, seeds) = keyed(dir.path(), "0123456789abcdef");
        assert_eq!(path, dir.path().join("0123456789abcdef"));
        let of = |name: &str| dir.path().join(name).join(LIBRARY_FILE);
        assert_eq!(seeds, [of("00000000000000bb"), of("00000000000000cc"), dir.path().join(LIBRARY_FILE)]);
        let left = |name: &str| dir.path().join(name).exists();
        assert!(!left("00000000000000aa") && !left("00000000000000ee"), "older ones go");
        assert!(left("backup") && left("00000000000000dd"), "nothing this did not write goes");
    }

    /// A mapped file reads as the file's bytes, a file written through a
    /// mapping holds what was written, and the library's file can be
    /// replaced while it is mapped (or the new one waits as the next).
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let tmp = path.with_extension("tmp");
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        {
            let mut file = Mapped::create(&tmp, new.len()).expect("a new file, mapped");
            file.bytes_mut().copy_from_slice(&new);
        }
        assert_eq!(std::fs::read(&tmp).unwrap(), new);
        replace_library(dir.path(), &tmp).unwrap();
        assert!(!tmp.exists());
        // The mapping still reads the old library.
        assert_eq!(blob.bytes(), &old[..]);
        drop(blob);
        // The next start reads the new one, wherever the save left it.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.chunks_exact_mut(4).enumerate() {
            let v = ((at / 4 + i) as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
            word.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The device loss of runs 72, 73 and 82: the library's file is
    /// replaced while it is mapped, and pages of the mapping that were not
    /// in memory then are read afterwards (in the runs: pages the trim let
    /// go and Windows gave to something else). With the file's handle kept
    /// open they still read as the old file. (With it closed, as before,
    /// all but the first few read as zeros on the PC this was found on.)
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        if !write_uncached(&path, LEN) {
            return; // no uncached writes here: nothing to show
        }
        let mapped = Mapped::open_holding(&path, true).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        let tmp = path.with_extension("tmp");
        {
            let mut file = Mapped::create(&tmp, LEN).expect("a new file, mapped");
            file.bytes_mut().fill(0xEE);
        }
        replace_library(dir.path(), &tmp).unwrap();
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(bytes == &expected[..], "page {page} of the replaced library no longer reads as the old file");
        }
    }
}
