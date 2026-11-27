//! Texture slots of the view heap and the first upload of the host texture
//! they show.
//!
//! A host slot is a descriptor in the heap the card reads while it runs a
//! list: a shader that picks the slot by index reads whatever the slot holds
//! at that moment, not what it held when the list was recorded. The texture
//! pass, a draw that reads the slot through a table, and the slots that
//! waited for a texture's host copy make a slot from a texture's view at
//! once, and the upload of the texture's texels goes into the list being
//! recorded, which is sent later. Every submission sent before that one (in
//! the air on the card, being prepared, or sent meanwhile from another
//! thread's lists) can read the slot before the texture holds anything, and
//! if the texture's memory is newer than such a submission, before the card
//! has that memory mapped for it.
//!
//! Here: which host textures the title fills have not had their first upload
//! run on the card, and (diagnostics) how often a slot names one while a
//! submission that could read it is sent.
// (Diagnostics only so far: without them the bookkeeping does nothing.)
#![cfg_attr(not(feature = "diagnostics"), allow(dead_code, unused_imports))]

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

/// A host texture the title fills whose first upload has not run on the
/// card yet.
#[derive(Clone, Copy)]
struct Fresh {
    /// The submission its first upload is in (0: in none yet).
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

/// Whether the bookkeeping runs.
fn on() -> bool {
    cfg!(feature = "diagnostics")
}

/// The number of the last submission begun, and of the last finished.
fn numbers() -> [u64; 2] {
    #[cfg(feature = "diagnostics")]
    return crate::host::submissions_numbered();
    #[cfg(not(feature = "diagnostics"))]
    [0, 0]
}

/// The host texture of a title texture the title fills was made now.
pub(crate) fn made(texture: &ID3D12Resource) {
    if !on() {
        return;
    }
    let [begun, _] = numbers();
    let at = crate::host::now_ms();
    let mut fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    let fresh = fresh.get_or_insert_with(HashMap::new);
    // (A texture made at the address of one that went is another texture.)
    fresh.insert(texture.as_raw() as usize, Fresh { upload: 0, begun, at });
    FRESH_LEN.store(fresh.len(), Relaxed);
}

/// An upload into `texture` was decided for the submission being prepared
/// (`crate::texture::decide_uploads`).
pub(crate) fn uploaded(texture: &ID3D12Resource) {
    if FRESH_LEN.load(Relaxed) == 0 {
        return;
    }
    let [started, _] = numbers();
    let mut fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(f) = fresh.as_mut().and_then(|m| m.get_mut(&(texture.as_raw() as usize)))
        && f.upload == 0
    {
        f.upload = started;
    }
}

/// What is known of `texture` if its first upload has not run on the card:
/// the submission it is in (0: none yet) and the submissions begun when it
/// was made.
fn unsettled(texture: &ID3D12Resource) -> Option<(u64, u64)> {
    if FRESH_LEN.load(Relaxed) == 0 {
        return None;
    }
    let [_, finished] = numbers();
    let fresh = FRESH.lock().unwrap_or_else(|e| e.into_inner());
    let f = fresh.as_ref()?.get(&(texture.as_raw() as usize))?;
    (f.upload == 0 || f.upload > finished).then_some((f.upload, f.begun))
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
        m.retain(|_, f| if f.upload == 0 { now.saturating_sub(f.at) < 60_000 } else { f.upload > finished });
        FRESH_LEN.store(m.len(), Relaxed);
    }
}

/// Who makes a slot on this thread now ([`by`]): the texture pass, a draw
/// that reads it through a table bound in the heap, a draw that makes its
/// table afresh and the title's slot in place, the slots that waited for a
/// texture's host copy; anything else (the title's own write).
pub(crate) const PASS: u8 = 0;
pub(crate) const TABLE: u8 = 1;
pub(crate) const IN_PLACE: u8 = 2;
pub(crate) const WAITING: u8 = 3;
const OTHER: u8 = 4;

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

/// Diagnostics, since the last line: texture slots made from a texture
/// whose first upload had not run on the card while a submission was in the
/// air or being prepared; of them, those whose upload was in no submission
/// yet, and those whose texture was made after a submission still in the
/// air had begun; the same by maker ([`by`]); submissions sent while such a
/// slot named such a texture, and the slots named so at those sendings.
#[cfg(feature = "diagnostics")]
static COUNTS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

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
        let [started, finished] = numbers();
        if started > finished {
            COUNTS[0].fetch_add(1, Relaxed);
            COUNTS[1].fetch_add(u64::from(upload == 0), Relaxed);
            COUNTS[2].fetch_add(u64::from(begun > finished), Relaxed);
            COUNTS[3 + usize::from(BY.with(std::cell::Cell::get).min(OTHER))].fetch_add(1, Relaxed);
        }
        let mut naming = NAMING.lock().unwrap_or_else(|e| e.into_inner());
        if naming.len() < 1 << 16 {
            naming.push((heap, index, texture.as_raw() as usize));
        }
    }
}

/// Submission `number` goes to the card now, its uploads decided: counts the
/// slots that name a texture whose first upload is in none before or in
/// this one (diagnostics), and forgets the textures whose upload has run.
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
                COUNTS[8].fetch_add(1, Relaxed);
                COUNTS[9].fetch_add(naming.len() as u64, Relaxed);
            }
        }
    }
    let [_, finished] = numbers();
    forget_settled(finished);
}

/// Diagnostics: the line of [`COUNTS`].
#[cfg(feature = "diagnostics")]
pub(crate) fn line() {
    let c = std::array::from_fn::<u64, 10, _>(|i| COUNTS[i].swap(0, Relaxed));
    let naming = NAMING.lock().unwrap_or_else(|e| e.into_inner()).len();
    tracing::info!(
        "texture slots made from a texture whose first upload had not run on the card, since the last line: {} while a \
         submission was in the air or being prepared, {} of them with the upload in no submission yet, {} with the \
         texture made after a submission still in the air had begun; by the texture pass {}, a draw's table in the heap \
         {}, a draw's table made afresh {}, the slots that waited for a host copy {}, other {}; {} submissions sent while \
         such a slot named such a texture, {} slots named so at those sendings; {} slots name one now, {} textures wait \
         for their first upload to run",
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
        naming,
        FRESH_LEN.load(Relaxed)
    );
}
