//! Texture slots of the view heap and the first upload of the host texture
//! they show.
//!
//! A host slot is a descriptor in the heap the card reads while it runs a
//! list: a shader that picks the slot by index reads whatever the slot holds
//! at that moment, not what it held when the list was recorded. The texture
//! pass, a draw that reads the slot through a table, and the slots that
//! waited for a texture's host copy made a slot from a texture's view at
//! once, and the upload of the texture's texels went into the list being
//! recorded, which is sent later. Every submission sent before that one (in
//! the air on the card, being prepared, or sent meanwhile from another
//! thread's lists) could read the slot before the texture held anything,
//! and, its memory being newer than such a submission, before the card had
//! that memory mapped for it. In motion in the open world 1 to 364 slots a
//! second were made so, nearly all by the texture pass, every one before
//! its texture's upload was in any submission and with its texture made
//! after a submission still in the air had begun; 230 to 610 submissions a
//! second were sent while such a slot named such a texture (5 to 8 in the
//! air on average). The host device of a run that enters the open world is
//! lost now and then with page faults of the card seconds before; another
//! tree of the same family lost its device to a page fault inside a texture
//! made a moment before, and no more once its slots showed a new texture
//! only after its upload had run.
//!
//! So such a slot is held: it goes on showing what it showed if that can
//! stand for the new texture (`super::host::keep_shown`), or nothing, and
//! is made from the new view by the next texture pass after the submission
//! with the texture's first upload has finished on the card
//! ([`show_after`]) and every submission begun before the texture was made
//! has ([`show_new_later`]). The slot holds no reference to the new texture
//! meanwhile (the list with the upload does, until it has run), and the old
//! one stays held by the slot as it is while it shows it. A texture whose
//! upload has not come in two seconds is shown then, as before, once the
//! submissions begun before it was made have finished. With both switches
//! in motion: 0 slots made so but for three seconds with 47, 19 and 2 holds
//! given up; 2.3 frames held on average; the frame time and the pictures as
//! without.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

use super::View;

/// Whether a slot shows a texture that is new to it only once the
/// submission with the texture's first upload has finished on the card
/// (`VICTORY_SHOW_AFTER`; on, `=0`: at once, as before).
pub(crate) fn show_after() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHOW_AFTER").is_none_or(|v| v != "0"))
}

/// Whether a slot shows a texture made while submissions begun before it
/// may still run only once those have finished (`VICTORY_SHOW_NEW_LATER`;
/// on, `=0`: at once, as before). With [`show_after`] that is so of every
/// texture whose upload runs; this also holds the slots of one whose upload
/// is not decided within a hold's two seconds.
pub(crate) fn show_new_later() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHOW_NEW_LATER").is_none_or(|v| v != "0"))
}

/// Whether the bookkeeping runs.
fn on() -> bool {
    cfg!(feature = "diagnostics") || show_after() || show_new_later()
}

/// A host texture the title fills whose first upload has not run on the
/// card yet.
#[derive(Clone, Copy)]
struct Fresh {
    /// The submission its first upload is decided for (0: none yet).
    upload: u64,
    /// The submissions begun when the texture was made: those up to this
    /// one ran on the card before its memory was there.
    begun: u64,
    /// When it was made (`crate::host::now_ms`): one never uploaded is
    /// forgotten after a minute (its texture is gone, or never shown).
    at: u64,
}

/// [`Fresh`] textures, by the address of their host resource.
static FRESH: Mutex<Option<HashMap<usize, Fresh>>> = Mutex::new(None);

/// How many there are, read without the lock: none is the common case.
static FRESH_LEN: AtomicUsize = AtomicUsize::new(0);

/// The host texture of a title texture the title fills was made now.
pub(crate) fn made(texture: &ID3D12Resource) {
    if !on() {
        return;
    }
    let [begun, _] = crate::host::submissions_numbered();
    let at = crate::host::now_ms();
    let mut fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    let fresh = fresh.get_or_insert_with(HashMap::new);
    // (A texture made at the address of one that went is another texture.)
    fresh.insert(texture.as_raw() as usize, Fresh { upload: 0, begun, at });
    FRESH_LEN.store(fresh.len(), Relaxed);
}

/// An upload into `texture` was decided for the submission being prepared
/// (`crate::texture::decide_uploads`), whether it copies anything or not:
/// what the texture holds is settled when that submission has run.
pub(crate) fn uploaded(texture: &ID3D12Resource) {
    if FRESH_LEN.load(Relaxed) == 0 {
        return;
    }
    let [started, _] = crate::host::submissions_numbered();
    let mut fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(f) = fresh.as_mut().and_then(|m| m.get_mut(&(texture.as_raw() as usize)))
        && f.upload == 0
    {
        f.upload = started;
    }
}

/// What is known of `texture` if its first upload has not run on the card
/// or it was made while a submission still in the air had begun: the
/// submission the upload is decided for (0: none yet) and the submissions
/// begun when it was made.
fn unsettled(texture: &ID3D12Resource) -> Option<(u64, u64)> {
    if FRESH_LEN.load(Relaxed) == 0 {
        return None;
    }
    let [_, finished] = crate::host::submissions_numbered();
    let fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    let f = fresh.as_ref()?.get(&(texture.as_raw() as usize))?;
    (f.upload == 0 || f.upload > finished || f.begun > finished).then_some((f.upload, f.begun))
}

/// Forgets the textures whose first upload has run, and those made a minute
/// ago and never uploaded.
fn forget_settled(finished: u64) {
    if FRESH_LEN.load(Relaxed) == 0 {
        return;
    }
    let now = crate::host::now_ms();
    let mut fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = fresh.as_mut() {
        m.retain(|_, f| {
            if f.upload == 0 { now.saturating_sub(f.at) < 60_000 } else { f.upload > finished || f.begun > finished }
        });
        FRESH_LEN.store(m.len(), Relaxed);
    }
}

/// Who makes a slot on this thread now ([`by`]): the texture pass, a draw
/// that reads it through a table bound in the heap, a draw that makes its
/// table afresh and the title's slot in place, the slots that waited for a
/// texture's host copy; anything else (the title's own write). And a maker
/// whose slots are never held ([`AT_ONCE`]).
pub(crate) const PASS: u8 = 0;
pub(crate) const TABLE: u8 = 1;
pub(crate) const IN_PLACE: u8 = 2;
pub(crate) const WAITING: u8 = 3;
const OTHER: u8 = 4;
/// The texture pass before a bake draw, which reads its textures in its
/// own list once, and a hold's end after its two seconds.
pub(crate) const AT_ONCE: u8 = 5;

thread_local! {
    static BY: std::cell::Cell<u8> = const { std::cell::Cell::new(OTHER) };
}

/// While the guard lives, slots made on this thread are made by `who`.
pub(crate) fn by(who: u8) -> By {
    By(BY.with(|by| by.replace(who)))
}

/// What [`by`] returns: the maker before it comes back when it goes.
pub(crate) struct By(u8);

impl Drop for By {
    fn drop(&mut self) {
        BY.with(|by| by.set(self.0));
    }
}

/// A slot held ([`hold`]): the record it is to be made from, when it was
/// held (`crate::host::now_ms`, `crate::frame::current`).
struct Hold {
    key: u64,
    ms: u64,
    frame: u64,
}

/// The held slots, by heap (its address) and slot.
static HOLDS: Mutex<Option<HashMap<(usize, u32), Hold>>> = Mutex::new(None);

/// How many there are, read without the lock.
static HOLDS_LEN: AtomicUsize = AtomicUsize::new(0);

/// The longest a slot is held.
const HOLD_MS: u64 = 2000;

/// Since the start: slots held; made when their texture was settled; made
/// after [`HOLD_MS`]; the frames the made ones waited in all, and the most;
/// those that showed nothing meanwhile (the others showed what they showed).
static HELD: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Whether slot `index` of the heap at `heap`, about to be made from the
/// record `key` (whose view is `view`, a texture view), is held instead
/// ([`show_after`], [`show_new_later`]): `Some(true)` if held now,
/// `Some(false)` if held already. The caller holds the slot's lock; when
/// held now it leaves the slot as it is, or makes it show nothing
/// ([`held_null`]).
pub(crate) fn hold(heap: usize, index: u32, key: u64, view: &View) -> Option<bool> {
    if !show_after() && !show_new_later() {
        return None;
    }
    // (A draw that reads a held slot through a table bound in the heap has
    // the table made afresh for it: `crate::draw` `tables`.)
    if BY.with(std::cell::Cell::get) == AT_ONCE {
        return None;
    }
    let texture = match super::host::placed_texture(view).unwrap_or_else(|| view.clone()) {
        View::Shader(resource, _) if crate::resource::buffer_bytes(resource as crate::object::This).is_none() => {
            resource as crate::object::This
        }
        _ => return None,
    };
    let at = (heap, index);
    // Held already for this record: asked again by each draw and pass.
    if HOLDS_LEN.load(Relaxed) > 0
        && HOLDS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|m| m.get(&at))
            .is_some_and(|h| h.key == key)
    {
        return Some(false);
    }
    // A texture without a host copy is made with it right after this (the
    // texture pass uploads what it makes ready) or, if not, shown when the
    // hold ends.
    let held = match crate::resource::host_made(texture) {
        None => crate::resource::title_fills(texture),
        Some(host) => unsettled(host).is_some_and(|(upload, begun)| {
            let [_, finished] = crate::host::submissions_numbered();
            show_after() && (upload == 0 || upload > finished) || show_new_later() && begun > finished
        }),
    };
    if !held {
        return None;
    }
    let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    let holds = holds.get_or_insert_with(HashMap::new);
    holds.insert(at, Hold { key, ms: crate::host::now_ms(), frame: crate::frame::current() });
    HOLDS_LEN.store(holds.len(), Relaxed);
    HELD[0].fetch_add(1, Relaxed);
    Some(true)
}

/// Whether slot `index` of the heap at `heap` is held ([`hold`]).
pub(crate) fn held(heap: usize, index: u32) -> bool {
    HOLDS_LEN.load(Relaxed) > 0
        && HOLDS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|m| m.contains_key(&(heap, index)))
}

/// A slot [`hold`] held shows nothing meanwhile: what it showed cannot
/// stand for the new texture.
pub(crate) fn held_null() {
    HELD[5].fetch_add(1, Relaxed);
}

/// The texture pass of the heap at `heap`: makes the held slots whose
/// texture is settled now, and those held for [`HOLD_MS`], by `make` (the
/// slot, made from whatever the title's record holds now). Returns how
/// many.
pub(crate) fn flip(heap: usize, mut make: impl FnMut(u32)) -> usize {
    if HOLDS_LEN.load(Relaxed) == 0 {
        return 0;
    }
    let now = crate::host::now_ms();
    let frame = crate::frame::current();
    let [_, finished] = crate::host::submissions_numbered();
    let mine: Vec<(u32, Hold)> = {
        let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = holds.as_mut() else { return 0 };
        let keys: Vec<(usize, u32)> = m.keys().filter(|k| k.0 == heap).copied().collect();
        let taken = keys.into_iter().filter_map(|k| m.remove(&k).map(|h| (k.1, h))).collect();
        HOLDS_LEN.store(m.len(), Relaxed);
        taken
    };
    let mut made = 0;
    let mut back = Vec::new();
    for (index, h) in mine {
        // Past the longest hold, what [`show_after`] waits for is given up
        // (an upload that does not come), not what [`show_new_later`] does:
        // the submissions it waits for finish.
        let late = now.saturating_sub(h.ms) >= HOLD_MS;
        if !settled(h.key, finished, !late) {
            back.push((index, h));
            continue;
        }
        let waited = frame.saturating_sub(h.frame);
        HELD[if late { 2 } else { 1 }].fetch_add(1, Relaxed);
        HELD[3].fetch_add(waited, Relaxed);
        HELD[4].fetch_max(waited, Relaxed);
        let _by = by(if late { AT_ONCE } else { PASS });
        make(index);
        made += 1;
    }
    if !back.is_empty() {
        let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
        let m = holds.get_or_insert_with(HashMap::new);
        for (index, h) in back {
            // (Held anew meanwhile: that hold is the newer.)
            m.entry((heap, index)).or_insert(h);
        }
        HOLDS_LEN.store(m.len(), Relaxed);
    }
    made
}

/// Whether the texture the record `key` views is settled: its host copy
/// made, its first upload run (if `upload`) and every submission begun
/// before it was made finished (`finished` is the last finished).
fn settled(key: u64, finished: u64, upload: bool) -> bool {
    let after = upload && show_after();
    let Some(view) = super::view_of(key) else {
        // The record names nothing any more: made (null) at once.
        return true;
    };
    let texture = match super::host::placed_texture(&view).unwrap_or(view) {
        View::Shader(resource, _) => resource as crate::object::This,
        _ => return true,
    };
    let Some(host) = crate::resource::host_made(texture) else {
        // Never made: made with the slot once the hold is given up.
        return !upload;
    };
    unsettled(host).is_none_or(|(sent, begun)| {
        !(after && (sent == 0 || sent > finished) || show_new_later() && begun > finished)
    })
}

/// Diagnostics, since the last line: texture slots made from a texture
/// whose first upload had not run on the card while a submission was in the
/// air or being prepared; of them, those whose upload was in no submission
/// yet, and those whose texture was made after a submission still in the
/// air had begun; the same by maker ([`by`]: the pass, a table in the heap,
/// a table made afresh, the slots that waited, other, a hold given up or a
/// bake's pass); submissions sent while such a slot named such a texture, and
/// the slots named so at those sendings.
#[cfg(feature = "diagnostics")]
static COUNTS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// Diagnostics: slots made from a texture whose first upload had not run,
/// for [`sending`]: the heap (its address), the slot, the texture.
#[cfg(feature = "diagnostics")]
static NAMING: Mutex<Vec<(usize, u32, usize)>> = Mutex::new(Vec::new());

/// Slot `index` of the heap at `heap` was made now from a view of `texture`
/// (a host texture; diagnostics: counted if its first upload has not run).
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
pub(crate) fn named(heap: usize, index: u32, texture: &ID3D12Resource) {
    #[cfg(feature = "diagnostics")]
    if let Some((upload, begun)) = unsettled(texture) {
        let [started, finished] = crate::host::submissions_numbered();
        if upload == 0 || upload > finished {
            if started > finished {
                COUNTS[0].fetch_add(1, Relaxed);
                COUNTS[1].fetch_add(u64::from(upload == 0), Relaxed);
                COUNTS[2].fetch_add(u64::from(begun > finished), Relaxed);
                COUNTS[3 + usize::from(BY.with(std::cell::Cell::get).min(AT_ONCE))].fetch_add(1, Relaxed);
            }
            let mut naming = NAMING.lock().unwrap_or_else(|e| e.into_inner());
            if naming.len() < 1 << 16 {
                naming.push((heap, index, texture.as_raw() as usize));
            }
        }
    }
}

/// Submission `number` goes to the card now, its uploads decided: counts the
/// slots that name a texture whose first upload is in none before or in
/// this one (diagnostics), and forgets the textures that are settled.
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
pub(crate) fn sending(number: u64) {
    #[cfg(feature = "diagnostics")]
    {
        let fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
        let mut naming = NAMING.lock().unwrap_or_else(|e| e.into_inner());
        if !naming.is_empty() {
            // A slot whose texture's upload is in this submission or one
            // before is read after it from here on; one whose texture is
            // forgotten (gone, or uploaded and run) is not counted again.
            naming.retain(|&(_, _, texture)| {
                fresh.as_ref().and_then(|m| m.get(&texture)).is_some_and(|f| f.upload == 0 || f.upload > number)
            });
            if !naming.is_empty() {
                COUNTS[9].fetch_add(1, Relaxed);
                COUNTS[10].fetch_add(naming.len() as u64, Relaxed);
            }
        }
    }
    let [_, finished] = crate::host::submissions_numbered();
    forget_settled(finished);
}

/// Diagnostics: the line of [`COUNTS`] and [`HELD`].
#[cfg(feature = "diagnostics")]
pub(crate) fn line() {
    let c = std::array::from_fn::<u64, 11, _>(|i| COUNTS[i].swap(0, Relaxed));
    let h = std::array::from_fn::<u64, 6, _>(|i| HELD[i].load(Relaxed));
    let naming = NAMING.lock().unwrap_or_else(|e| e.into_inner()).len();
    tracing::info!(
        "texture slots made from a texture whose first upload had not run on the card, since the last line: {} while a \
         submission was in the air or being prepared, {} of them with the upload in no submission yet, {} with the \
         texture made after a submission still in the air had begun; by the texture pass {}, a draw's table in the heap \
         {}, a draw's table made afresh {}, the slots that waited for a host copy {}, other {}, a hold given up {}; {} \
         submissions sent while such a slot named such a texture, {} slots named so at those sendings; {} slots name one \
         now, {} textures wait \
         for their first upload to run; since the start, slots held until it had (VICTORY_SHOW_AFTER={}, \
         VICTORY_SHOW_NEW_LATER={}): {}, made when it had {}, after {} ms {}, {:.2} frames after they were held, the \
         longest {}; {} showed nothing meanwhile; {} held now",
        c[0],
        c[1],
        c[2],
        c[3],
        c[4],
        c[5],
        c[6],
        c[7],
        c[8],
        c[9],
        c[10],
        naming,
        FRESH_LEN.load(Relaxed),
        u8::from(show_after()),
        u8::from(show_new_later()),
        h[0],
        h[1],
        HOLD_MS,
        h[2],
        h[3] as f64 / (h[1] + h[2]).max(1) as f64,
        h[4],
        h[5],
        HOLDS_LEN.load(Relaxed)
    );
}
