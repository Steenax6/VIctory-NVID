//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::slots::{SlotList, SlotLocks, SlotMap, SlotSet, Striped, null_ranges_into, present_into, print_is_new};
use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count`: [`DYNAMIC`] in view heaps, none in
    /// sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there: 2^16 and 2^17 both ran out in this title.)
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    /// Read and written only under `locks`: the slot's own lock for work on
    /// that slot, every slot's for work that reaches others (stale views
    /// made null). Atomics only so that the slots need no cell each.
    made: Vec<std::sync::atomic::AtomicU64>,
    /// The locks of the slots' work ([`SlotLocks`]): what one mutex for the
    /// whole of `made` did, striped, so that draws recording on several
    /// threads, and the submit thread's pass over records copied in, do
    /// not wait for each other's slots (the recording threads stood some
    /// 28 ms a frame each in this lock and in the one of the heaps' list).
    locks: SlotLocks,
    /// A fingerprint of the record each slot held when the pass over
    /// records copied in as bytes last looked at it
    /// ([`HostHeap::mirror_copied`]; 0: never looked at). Made on first use.
    copied: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// Locked by stripes of slots, not as a whole.
    kept: Striped<Option<ID3D12Resource>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`; entries: [`stride_entry`]):
    /// an upload buffer of `SLOT_STRIDES_COUNT` dwords, its GPU address
    /// and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// View heaps, with the structure builder asked for: the title address
    /// of the buffer or ray-tracing structure view at each title slot, for
    /// shaders that take it out of a descriptor they pick by an index they
    /// compute (`victory_shader::emit::SLOT_VIEW_ADDRESSES`): an upload
    /// buffer of two dwords a slot, its GPU address and its mapping.
    addresses: Option<(ID3D12Resource, u64, usize)>,
    /// Title slots holding buffer views. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots`] for shaders that pick
    /// buffers from the whole heap.
    /// (A bitmap: a tree here was most of what noting a write cost.)
    buffers: Mutex<SlotSet>,
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Mutex<SlotList>,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: Mutex<TextureQueue>,
    /// A bit per slot set while the slot is in `textures`' queue
    /// ([`HostHeap::maybe_queued`]).
    queued: std::sync::Arc<[std::sync::atomic::AtomicU64]>,
    /// Per title slot, the frame (plus one; 0: never) in which a draw last
    /// read it through a table, or made a table it lies in (with
    /// [`TABLED_READ`] set if the draw read it). The draw that
    /// reads such a slot brings its texture up to date itself and the slot
    /// then leaves `textures`; a write of a slot that was a table's within
    /// the last [`TABLED_FRAMES`] frames is due only [`TABLED_GRACE`]
    /// frames later, by when its draw has read it. A title that builds a
    /// table per draw writes thousands of texture views a frame into slots
    /// only tables read (the title: some 2,500 a frame in its
    /// opening cutscene, a few hundred textures over and over), and every
    /// one of them stood in the queue of textures picked by index, ahead
    /// of or behind the ones shaders do pick by index.
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// Per title slot the texture pass uploaded, the number its upload's
    /// read began at (`crate::writes::stamp`; `crate::writes::NOT_READ`
    /// until the list that holds the upload is submitted): a write behind
    /// the guards numbered higher under the slot's texture came after the
    /// read ([`HostHeap::slots_written_under`]).
    stamps: Mutex<std::collections::HashMap<u32, std::sync::Arc<std::sync::atomic::AtomicU64>>>,
    /// The slots whose last upload read before a write that came after it
    /// had its bytes there (by those numbers), until uploaded again.
    /// With what wrote the memory (`crate::writes::Cause`).
    stale: Mutex<std::collections::BTreeMap<u32, u8>>,
    /// The slots uploaded before a tell that lay in their texture's span
    /// and was taken for another texture's (one whose view starts between
    /// the span's start and the tell: [`Spans::slots_over`] names them,
    /// [`Spans::slots_holding`] and [`Spans::slots_in`] do not), with who
    /// told, until uploaded again. Counted, and asked when a look finds
    /// such a slot's memory changed (`crate::census`); nothing is made due
    /// by it.
    hidden: Mutex<std::collections::BTreeMap<u32, u8>>,
    /// Per title slot the texture pass uploaded, the number of the write
    /// journal's asking its upload read after (`crate::watch::asked_now`;
    /// 0 until the read), for [`HostHeap::texture_slot_found`].
    asked: Mutex<std::collections::HashMap<u32, std::sync::Arc<std::sync::atomic::AtomicU64>>>,
}

/// The spans of every view heap made ([`HostHeap::placed`]), for the
/// texture code, which has no heap at hand ([`next_start`],
/// [`first_viewed`]).
static PLACED_SPANS: Mutex<Vec<std::sync::Weak<Mutex<Spans>>>> = Mutex::new(Vec::new());

/// Where the next texture placed inside `[start, end)` starts, past the
/// first bytes of the one at `start` ([`HEAD`]): what is there of the
/// texture at `start` ends there, whatever its full size. `None`: no view
/// in a slot shaders pick by index describes memory in that range.
pub(crate) fn next_start(start: u64, end: u64) -> Option<u64> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).next_start(start, end))
        .min()
}

/// The lowest level that a view of the placed texture at `start` shows,
/// over the slots shaders pick by index that hold one now. The title
/// views a texture from the first level it holds. `None`: no such slot.
pub(crate) fn first_viewed(start: u64) -> Option<u32> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).first_viewed(start))
        .min()
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose first level with memory changed since a texture pass last took
/// them ([`HostHeap::texture_views_remade`]).
static RESHOWN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
}

/// The starts [`levels_changed`] noted, taken (each once).
pub(crate) fn take_levels_changed() -> Vec<u64> {
    let mut starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
    starts.sort_unstable();
    starts.dedup();
    starts
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose last upload left a held level without tiles (the card had no
/// room, or no heap could be had), and when (`crate::host::moment`). A
/// texture pass makes their slots due again once the card has room
/// ([`HostHeap::texture_slots_waiting`]).
static WAITING: Mutex<std::collections::BTreeMap<u64, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// The last upload of the placed texture at `start` left a held level
/// without tiles. A texture picked by index is uploaded once and then
/// only when its memory is found changed, and its staging buffer goes
/// with the upload: without this the level stayed empty (the texture
/// blurred, or black if no level had tiles) for as long as the title left
/// its memory alone, where a texture refused for want of room is tried
/// again.
pub(crate) fn levels_waiting(start: u64) {
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).insert(start, crate::host::moment());
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default, Clone)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: SlotMap<u64>,
    /// The level each slot's view starts at ([`Spans::shows`]).
    first: SlotMap<u32>,
    /// Every slot `by_slot` or `first` may hold (no fewer): most slots
    /// [`Spans::forget`] is asked about hold no placed texture view, and are
    /// told by a bit instead of two lookups.
    known: SlotSet,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
    /// The starts of `by_start` by the length of their span ([`CLASSES`]):
    /// a span that reaches into a range starts at most its class's length
    /// before it, so every span over a range is found without a walk over
    /// the short ones that lie before it ([`Spans::slots_over`]).
    classed: std::collections::BTreeSet<(u8, u64)>,
}

/// The longest span of each class of [`Spans::classed`].
const CLASSES: [u64; 4] = [64 << 10, 1 << 20, 16 << 20, u64::MAX];

/// The class of a span of `len` bytes.
fn class_of(len: u64) -> u8 {
    CLASSES.iter().position(|&most| len <= most).unwrap_or(CLASSES.len() - 1) as u8
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        if let Some(&known) = self.sizes.get(desc) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(desc);
        self.sizes.insert(*desc, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        // (A range whose end lies below its start has nothing inside; the
        // map's own lookup panics on one.)
        let inside = (start < end).then(|| self.by_start.range(start..end).map(|(at, _)| *at));
        before.into_iter().chain(inside.into_iter().flatten()).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                self.classed.remove(&(class_of(old_end - at), at));
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        if !self.by_start.contains_key(&start) {
            self.classed.insert((class_of(end - start), start));
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        self.known.insert(index);
        stale
    }

    /// Whether [`Spans::note`] of the same arguments would name stale slots
    /// (other slots' views it makes stale), changing nothing: a write that
    /// would not is done under its own slot's lock alone
    /// ([`HostHeap::mirror`]).
    fn would_stale(&self, index: u32, start: u64, end: u64, newer: bool) -> bool {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return false;
        }
        if end <= start || !newer || self.rule == Rule::Never {
            return false;
        }
        // What `note` sees after it forgot `index`: a span that held only
        // `index` is gone.
        let old = self.by_slot.get(&index).copied();
        let left = |at: u64| {
            self.by_start
                .get(&at)
                .filter(|span| !(old == Some(at) && span.1.len() == 1 && span.1.contains(&index)))
        };
        let clashes = |at: u64| left(at).is_some() && !(at == start && same(left(at)));
        match self.rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .any(|(&at, _)| clashes(at)),
            Rule::Overlap | Rule::Never => {
                // `overlapping` as `note` would see it.
                let before = self
                    .by_start
                    .range(..start)
                    .rev()
                    .find(|&(&at, _)| left(at).is_some())
                    .filter(|(_, span)| span.0 > start)
                    .map(|(&at, _)| at);
                before.into_iter().chain(self.by_start.range(start..end).map(|(&at, _)| at)).any(clashes)
            }
        }
    }

    /// The view in slot `index` starts at level `first` of its texture.
    fn shows(&mut self, index: u32, first: u32) {
        self.first.insert(index, first);
        self.known.insert(index);
    }

    /// The lowest level a view of the texture at `start` starts at.
    fn first_viewed(&self, start: u64) -> Option<u32> {
        self.by_start.get(&start)?.1.iter().filter_map(|slot| self.first.get(slot)).min().copied()
    }

    /// The first span that starts inside `[start, end)` past the first
    /// [`HEAD`] bytes of it.
    fn next_start(&self, start: u64, end: u64) -> Option<u64> {
        // A span no longer than its first block has nothing past that
        // block: a texture of a few texels (a 1x1 one takes 16 bytes), a
        // view of no bytes, a start at the top of the address space, an end
        // below the start. The map's lookup panics when the range's start
        // lies above its end, and the texture pass asks this of every
        // texture it uploads (`HostHeap::texture_span_others`): on a thread
        // that records the title's commands that ends the process.
        let from = start.saturating_add(HEAD);
        if from >= end {
            return None;
        }
        self.by_start.range(from..end).next().map(|(at, _)| *at)
    }

    /// The slots that hold a view of the texture at `start`.
    fn slots_at(&self, start: u64) -> Vec<u32> {
        self.by_start.get(&start).map(|span| span.1.iter().copied().collect()).unwrap_or_default()
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        if !self.known.remove(index) {
            return;
        }
        self.first.remove(&index);
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                let end = span.0;
                self.by_start.remove(&start);
                self.classed.remove(&(class_of(end - start), start));
            }
        }
    }

    /// The slots of every span that reaches into `[start, end)` at its
    /// full size, whatever starts inside it: more than
    /// [`Spans::slots_holding`] and [`Spans::slots_in`] name, which take a
    /// texture to end where the next one starts. A texture all of whose
    /// levels the title holds is cut short by them wherever an older view
    /// of other memory still stands inside its span.
    fn slots_over(&self, start: u64, end: u64) -> Vec<u32> {
        let mut slots = Vec::new();
        if end <= start {
            return slots;
        }
        for (class, most) in CLASSES.iter().enumerate() {
            let class = class as u8;
            let from = start.saturating_sub(most.saturating_sub(1));
            for &(_, at) in self.classed.range((class, from)..(class, end)) {
                if let Some(span) = self.by_start.get(&at)
                    && span.0 > start
                {
                    slots.extend(span.1.iter().copied());
                }
            }
        }
        slots
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }

    /// The slots of the textures that hold bytes of `[start, end)`, for a
    /// write there: as [`Spans::slots_in`], but when a texture starts at
    /// `start` itself the one before it is left out (it ends where that
    /// one starts, and a request that fills a texture begins at its first
    /// byte: `slots_in` would name its neighbour with every such write).
    fn slots_holding(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start {
            return Vec::new();
        }
        let before = if self.by_start.contains_key(&start) {
            None
        } else {
            self.by_start.range(..start).next_back().filter(|(_, span)| span.0 > start).map(|(_, span)| span)
        };
        before
            .into_iter()
            .chain(self.by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }
}

/// Title memory written behind the guards' back whose bytes are there
/// (DirectStorage's reads into physical pages, at the addresses the title
/// has them mapped, and its reads to plain addresses;
/// `crate::watch::title_memory_written`), since a texture pass last took
/// the ranges ([`page_written`]): start, end, and the write's number
/// (`crate::writes::next`, given under this lock, so a pass that takes
/// them takes every number up to the last). Kept apart from [`REMAPPED`]
/// so that what such writes do is counted by itself. Also the layer's own
/// copies into title memory ([`title_copied`]), told apart by the last
/// field (a `crate::writes::Cause`).
static PAGE_WRITTEN: Mutex<Vec<(u64, u64, u64, u8)>> = Mutex::new(Vec::new());

use crate::writes::Cause;

/// Since the start: the layer's copies into title memory under texture
/// views ([`title_copied`]); their bytes; the slots stale by them (of
/// `PAGE_WRITES[3]`).
pub(crate) static COPY_WRITES: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots whose last upload read before the
/// write's bytes were there (`crate::writes`): stale; due slots the pass
/// left for after a read the title has submitted into their memory
/// (`crate::writes::pending_over`).
pub(crate) static PAGE_WRITES: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Slots noted stale now ([`HostHeap::slots_written_under`]), over the
/// heaps.
pub(crate) static STALE_NOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The bytes of a write the guards do not see are in title memory `[at,
/// at + len)`: the placed textures there changed with no descriptor
/// written and nothing mapped. Kept for the next texture pass
/// ([`HostHeap::texture_slots_page_written`]), as [`remapped`] keeps its
/// ranges. Returns the write's number (`crate::writes`); 0 for memory no
/// texture view describes, where nothing waits for a pass. Locks one mutex
/// and calls nothing.
pub(crate) fn page_written(at: usize, len: usize, pages: bool) -> u64 {
    written(at, len, if pages { Cause::Pages } else { Cause::Plain })
}

/// The layer copied the title's bytes into title memory `[at, at + len)`
/// (a buffer copy of the title's done at its submission, one that went
/// late and is done when its submission lands, `CopyMemoryToMemoryX`):
/// as [`page_written`], after the bytes are there. A texture uploaded
/// before holds what was there before (a title that reads a texture's
/// bytes into an upload buffer and copies them into place fills its
/// textures this way). One compare for memory no texture view describes.
pub(crate) fn title_copied(at: usize, len: usize) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    let number = written(at, len, Cause::Copy);
    if number != 0 {
        COPY_WRITES[0].fetch_add(1, Relaxed);
        COPY_WRITES[1].fetch_add(len as u64, Relaxed);
    }
    number
}

/// [`page_written`] and [`title_copied`], for a write `by` them.
fn written(at: usize, len: usize, by: Cause) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    let by = by as u8;
    /// Ranges kept until a pass takes them: a world's loading writes
    /// thousands before a pass runs.
    const MOST: usize = 1 << 20;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if len == 0 || !spanned(start, end) {
        return 0;
    }
    let mut ranges = PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    PAGE_WRITES[0].fetch_add(1, Relaxed);
    crate::census::tell(by, crate::census::T_RANGES, 1);
    let number = crate::writes::next();
    // Joined with the one before of the same kind: the whole range carries
    // the newer number (an upload that read between the two is uploaded
    // again either way).
    if let Some(last) = ranges.last_mut()
        && last.1 == start
        && last.3 == by
    {
        (last.1, last.2) = (end, number);
        return number;
    }
    if ranges.len() >= MOST {
        // No pass took them (none runs): what read before them is not
        // uploaded again. Counted.
        PAGE_WRITES[1].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end, number, by));
    number
}

/// Whether title memory `[start, end)` reaches into the spans of the
/// texture views noted so far ([`SPANNED`]); outside them nothing the
/// runtime does is the texture pass's business.
pub(crate) fn spanned(start: u64, end: u64) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    end > SPANNED[0].load(Relaxed) && start < SPANNED[1].load(Relaxed)
}

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]): start, end, a number
/// of the writes' ([`crate::writes::next`], given when the range is told:
/// an upload that began its read after it read what was mapped by then)
/// and who told (a `crate::writes::Cause`).
static REMAPPED: Mutex<Vec<(u64, u64, u64, u8)>> = Mutex::new(Vec::new());

/// More ranges came than [`REMAPPED`] keeps: the next texture pass looks
/// at every placed slot once instead ([`HostHeap::texture_slots_remapped`]).
static OVERFLOWED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). `cause`: who tells. Called with
/// the runtime's own locks held: this locks one mutex of its own and calls
/// nothing.
pub(crate) fn remapped(at: usize, len: usize, cause: Cause) {
    use std::sync::atomic::Ordering::Relaxed;
    let cause = cause as u8;
    /// Ranges kept until a pass takes them: a world's loading maps more
    /// than 2^15 apart before a pass runs.
    const MOST: usize = 1 << 20;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    crate::census::tell(cause, crate::census::T_RANGES, 1);
    let number = crate::writes::next();
    // Pages are mapped one after the other.
    if let Some(last) = ranges.last_mut()
        && last.1 == start
        && last.3 == cause
    {
        (last.1, last.2) = (end, number);
        return;
    }
    if ranges.len() >= MOST {
        // No pass runs (no shader that picks textures by index is bound):
        // the ranges go, and the next pass looks at every placed slot. Lost,
        // which textures streamed in there were looked at again hung on
        // timing: walls, a window frame and an outfit stayed wrong for a
        // whole run, at random (PastiiJay/klbf-lab 74315e765).
        STALE[2].fetch_add(ranges.len() as u64, Relaxed);
        OVERFLOWED.store(true, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end, number, cause));
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 240;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 8;

/// In [`HostHeap::tabled`]: a draw read the slot in that frame.
const TABLED_READ: u32 = 1 << 31;

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// `VICTORY_TABLED_SKIP_FRAMES=<n>`: a slot is left to its draws only if a
/// draw read it through a table within the last `n` frames (0: such slots
/// are mirrored again, as with `VICTORY_HEAP_SYNC_TABLES=1`). `off`: the
/// window is [`TABLED_FRAMES`] and the read bit is not asked for.
///
/// steenax: unset is 2, not `off`. With the window of [`TABLED_FRAMES`] the
/// title died at world entry, reproducibly (a write to 0 at
/// `victory-image+0x23cf683`, pull request #827 upstream), and 2 gave clean
/// runs through the world there.
fn skip_frames() -> Option<u64> {
    static N: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *N.get_or_init(|| tabled_skip(std::env::var("VICTORY_TABLED_SKIP_FRAMES").ok().as_deref()))
}

/// [`skip_frames`] for the variable's value.
fn tabled_skip(value: Option<&str>) -> Option<u64> {
    const DEFAULT: u64 = 2;
    match value.map(str::trim) {
        Some("off") => None,
        Some(v) => Some(v.parse().unwrap_or(DEFAULT)),
        None => Some(DEFAULT),
    }
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    checked: (u64, u64, u64, u8, u64),
    /// What made it due (`crate::census`: the view written, or who told of
    /// its memory), for the count of uploads by reason.
    why: u8,
}

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    writes: u64,
    /// Slots held back because the card had no room for the host copy
    /// they need ([`TextureQueue::hold`]), each with the write it was
    /// held as.
    held: std::collections::BTreeMap<u32, u64>,
    /// A bit per slot, set while the slot is in `slots`, for those who ask
    /// without the queue's lock ([`HostHeap::maybe_queued`]); none in tests.
    queued: Option<std::sync::Arc<[std::sync::atomic::AtomicU64]>>,
    /// The slots by the frame they are due from: a pass asks for the due
    /// ones and finds them without a walk over the others. (Every pass
    /// walked all of the queue, twice: 2.3 ms a frame for some 28,000 slots
    /// of which a few hundred were due, and the same with none due.) Kept
    /// with `slots` by [`TextureQueue::put`], [`TextureQueue::take`] and
    /// [`TextureQueue::move_due`].
    by_due: std::collections::BTreeSet<(u64, u32)>,
    /// What [`TextureQueue::census`] last counted, and when (milliseconds).
    counted: ([u64; 5], u64),
}

impl TextureQueue {
    /// Puts `index` in the queue as `w`.
    fn put(&mut self, index: u32, w: Waiting) {
        if let Some(old) = self.slots.insert(index, w) {
            self.by_due.remove(&(old.due, index));
        }
        self.by_due.insert((w.due, index));
        self.mark(index, true);
    }

    /// Takes `index` out of the queue.
    fn take(&mut self, index: u32) -> Option<Waiting> {
        let w = self.slots.remove(&index);
        if let Some(w) = &w {
            self.by_due.remove(&(w.due, index));
            self.mark(index, false);
        }
        w
    }

    /// Changes the entry of `index` if it is still the write the pass was
    /// handed, whatever of it `change` changes, the frame it is due from
    /// among it. True if it was.
    fn move_due(&mut self, index: u32, write: u64, change: impl FnOnce(&mut Waiting)) -> bool {
        let Some(w) = self.slots.get_mut(&index).filter(|w| w.write == write) else {
            return false;
        };
        let before = w.due;
        change(w);
        let after = w.due;
        if after != before {
            self.by_due.remove(&(before, index));
            self.by_due.insert((after, index));
        }
        true
    }

    fn mark(&self, index: u32, on: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(word) = self.queued.as_ref().and_then(|q| q.get(index as usize / 64)) {
            let bit = 1u64 << (index % 64);
            if on {
                word.fetch_or(bit, Relaxed);
            } else {
                word.fetch_and(!bit, Relaxed);
            }
        }
    }

    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        self.writes += 1;
        let write = self.writes;
        let why = crate::census::VIEW;
        self.put(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0), why });
    }

    fn remove(&mut self, index: u32) {
        self.take(index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in
    /// the order to serve them:
    /// - the ones looked at before (their memory was not mapped, or only
    ///   part of it), newest write first: the look is a query, and memory
    ///   that has arrived since is what the title streamed in last;
    /// - the others, the newest write first and the oldest next, in turns:
    ///   slot order served the low slots of a backlog of 25,000-40,000
    ///   while the textures of the shot on screen waited (run 52: 98% of
    ///   the drawn pixels had no colour texture in the first 3D frames);
    /// - the two kinds one after the other, one looked at before first.
    ///   With all those looked at before first, a backlog of thousands
    ///   streaming in a part at a time filled every pass, which the pass's
    ///   time then cut, and slots never looked at waited 300 to 1,500
    ///   frames: faces whose textures are picked by index read an empty
    ///   slot (black) or an older view (red, noise) in the opening
    ///   cutscene (fxceless/victory-fork pull request #1246, RTX 5060 Ti).
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        let mut again: Vec<(u64, u32, u64)> = Vec::new();
        let mut fresh: Vec<(u64, u32, u64)> = Vec::new();
        for &(_, index) in self.by_due.range(..=(frame, u32::MAX)) {
            let Some(w) = self.slots.get(&index) else {
                continue;
            };
            let class = if w.looks > 0 || w.mapped > 0 { &mut again } else { &mut fresh };
            class.push((w.written, index, w.write));
        }
        // By the number of the write, counted over the run.
        let newest_first = |a: &(u64, u32, u64), b: &(u64, u32, u64)| b.2.cmp(&a.2);
        again.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);
        let mut fresh_order: Vec<(u32, u64)> = Vec::with_capacity(fresh.len().min(max));
        let (mut new, mut old) = (0, fresh.len());
        while new < old && fresh_order.len() < max {
            fresh_order.push((fresh[new].1, fresh[new].2));
            new += 1;
            if new < old && fresh_order.len() < max {
                old -= 1;
                fresh_order.push((fresh[old].1, fresh[old].2));
            }
        }
        let mut again_order = again.iter().map(|&(_, index, write)| (index, write));
        let mut fresh_order = fresh_order.into_iter();
        let mut order = Vec::with_capacity(max.min(fresh.len() + again.len()));
        while order.len() < max {
            let (a, b) = (again_order.next(), fresh_order.next());
            if a.is_none() && b.is_none() {
                break;
            }
            order.extend(a);
            order.extend(b);
        }
        order.truncate(max);
        order
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.slots.get(&index).is_some_and(|w| w.write == write) {
            self.take(index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        self.move_due(index, write, |w| {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
        });
    }

    /// The card has no room for the host copy slot `index` needs: it is
    /// not handed out again before frame `due`, or before the card has
    /// room again ([`TextureQueue::room`]). Nothing else of it changes: it
    /// is not a slot "waiting for its memory", and when it is due it has
    /// its old place among the slots never looked at. (It stayed due, so
    /// every pass on a full card looked at the same few hundred slots
    /// again and stopped there, and the slots behind them were never
    /// reached: the ones whose host copy exists and only wants bringing up
    /// to date, and the ones whose memory went, which is what lets host
    /// copies go and makes room.)
    fn hold(&mut self, index: u32, write: u64, due: u64) {
        if self.move_due(index, write, |w| w.due = w.due.max(due)) {
            self.held.insert(index, write);
        }
    }

    /// Slot `index` is left for after a write the title has submitted
    /// into its memory: due from frame `due`, nothing else of it changed
    /// (it is no slot "waiting for its memory").
    fn after(&mut self, index: u32, write: u64, due: u64) {
        self.move_due(index, write, |w| w.due = w.due.max(due));
    }

    /// The card has room again at `frame`: the slots held back for lack of
    /// it are due now, as if they had never been held. Returns how many.
    fn room(&mut self, frame: u64) -> usize {
        let held = std::mem::take(&mut self.held);
        let mut back = 0;
        for (index, write) in held {
            back += usize::from(self.move_due(index, write, |w| w.due = w.due.min(frame)));
        }
        back
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        self.move_due(index, write, |w| (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1)));
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print`: looked at again from frame `due`, for a
    /// title that fills a texture's memory after it wrote the view (the
    /// `looks`-th such look; `uploaded` is the upload's frame).
    #[allow(clippy::too_many_arguments)]
    fn recheck(&mut self, index: u32, write: u64, due: u64, at: (u64, u64), print: u64, looks: u8, uploaded: u64) {
        self.move_due(index, write, |w| {
            (w.due, w.looks, w.mapped) = (due, 1, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
        });
    }

    /// What [`TextureQueue::recheck`] noted of slot `index`, if it is
    /// waiting for such a look.
    fn checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.slots.get(&index).map(|w| w.checked).filter(|c| c.2 != 0)
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out). `why`: what says so (`crate::census`).
    fn changed(&mut self, index: u32, frame: u64, due: u64, why: u8) {
        self.writes += 1;
        let write = self.writes;
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        self.put(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0), why });
    }

    /// What made slot `index` due, and whether it was uploaded in part
    /// before.
    fn why(&self, index: u32) -> Option<(u8, bool)> {
        self.slots.get(&index).map(|w| (w.why, w.mapped > 0))
    }

    /// The queue by what its slots wait for at `frame`
    /// (`crate::census::QUEUE`), but for the two kinds of memory not
    /// mapped, which are one here; and the slots of that kind.
    fn kinds(&self, frame: u64) -> ([u64; crate::census::KINDS], Vec<u32>) {
        let mut kinds = [0u64; crate::census::KINDS];
        let mut away = Vec::new();
        for (&index, w) in &self.slots {
            let kind = if w.due > frame && self.held.contains_key(&index) {
                2
            } else if w.checked.2 != 0 {
                6 + usize::from(w.checked.3).min(4)
            } else if w.mapped > 0 {
                5
            } else if w.looks > 0 {
                away.push(index);
                4
            } else {
                usize::from(w.due > frame)
            };
            kinds[kind] += 1;
        }
        (kinds, away)
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }

    /// For the frame line: slots in the queue; of them due at `frame`;
    /// waiting for their memory (looked at, never uploaded); uploaded in
    /// part; and the frames the oldest due one has waited since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [self.slots.len() as u64, 0, 0, 0, 0];
        for w in self.slots.values() {
            if w.mapped > 0 {
                c[3] += 1;
            } else if w.looks > 0 {
                c[2] += 1;
            }
            if w.due <= frame {
                c[1] += 1;
                c[4] = c[4].max(frame.saturating_sub(w.written));
            }
        }
        c
    }

    /// [`TextureQueue::census`] as counted at most a quarter of a second
    /// ago (`now`: milliseconds), with the slots in the queue as they are:
    /// the count walks every slot, and a pass asks for it each time.
    fn census_lately(&mut self, frame: u64, now: u64) -> [u64; 5] {
        if now >= self.counted.1.saturating_add(250) || now < self.counted.1 {
            self.counted = (self.census(frame), now);
        }
        self.counted.0[0] = self.slots.len() as u64;
        self.counted.0
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element of a buffer view, as the console's descriptor holds
/// them for indexed reads (structured: the stride; raw: 1, the index is the
/// byte; typed: the format's); `None` if `view` is not a buffer view.
fn view_stride(view: &View) -> Option<u32> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format)
        }
    };
    Some(if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes as u32)
    })
}

/// The table-of-strides entry of a slot that holds `view`, made or a null
/// view of its kind: a buffer view's stride, 0 for a constant buffer view
/// (shaders read it as one) and for a buffer's own view (made without a
/// description: its element size unknown), `SLOT_STRIDES_SKIP` for anything
/// else, which shaders that pick a buffer there read as nothing.
fn stride_entry(view: &View) -> u32 {
    match view {
        View::Constant(..) => 0,
        View::Shader(r, None) | View::Unordered(r, _, None)
            if crate::resource::buffer_bytes(*r as crate::object::This).is_some() =>
        {
            0
        }
        _ => view_stride(view).unwrap_or(victory_shader::emit::SLOT_STRIDES_SKIP),
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The size of every host view heap on a host that takes more than
/// [`PORTABLE_VIEWS`] (AMD): the title's heap (a little over 1,000,000
/// views in the retail game) plus [`DYNAMIC`] fits with room to spare, and
/// the last slot is the null slot every view index at or past it reads.
const UNCAPPED_VIEWS: u32 = 1 << 21;

/// Picks, once, the null slot at the end of every host view heap and tells
/// the translator now (`set_view_heap_limit`) instead of when the title's
/// heap is made: a shader keeps the limit it was translated with, and the
/// title makes 25 pipelines (their shaders translated at once) before it
/// makes its view heap. Called before a shader is translated and before a
/// view heap is made.
///
/// Every host clamps. A host that caps view heaps at [`PORTABLE_VIEWS`]
/// (NVIDIA) hung on a view index past the heap; one that does not (AMD
/// RDNA3, RX 7900 XT) took a page fault and removed the device on entering
/// the world: a compute program read view slot 16,777,088 (0xffff80) of a
/// heap of 1,507,328, past its end. Unclamped, that index is undefined on
/// any host.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        let fits = |count: u32| {
            // SAFETY: plain heap creation; the heap is released at once.
            let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            };
            made.is_ok()
        };
        let views = if fits(UNCAPPED_VIEWS) {
            UNCAPPED_VIEWS
        } else {
            PORTABLE_VIEWS
        };
        tracing::info!(
            views,
            "host view heaps hold {views} views; view indices are clamped to the null slot at the end from the first shader on"
        );
        victory_shader::emit::set_view_heap_limit(views - 1);
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

/// How many of the title's `count` view slots a host view heap whose null
/// slot is `null` holds: all of them while they and the `extra` dynamic
/// slots fit below the null views of each class kept under the null slot
/// (`victory_shader::emit::NULL_BUFFER_SRV_BELOW`).
fn view_heap_fit(count: u32, extra: u32, null: u32) -> u32 {
    let room = null - victory_shader::emit::NULL_CBV_BELOW;
    if count + extra <= room {
        count
    } else {
        room - extra
    }
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let extra = if sampler { 0 } else { DYNAMIC };
        // A view heap ends at the null slot shaders clamp view indices to
        // (`probe_view_limit`), whatever the title's own count: the title's
        // slots, then the dynamic ones, then null up to and with that slot,
        // the three just below it the null views of each class a buffer
        // picked where the stride table says no buffer view reads.
        if !sampler {
            probe_view_limit();
        }
        let null = if sampler {
            0
        } else {
            victory_shader::emit::view_heap_limit()
        };
        let (heap, count, size) = if null == 0 {
            match make(count + extra) {
                Ok(heap) => (heap, count, count + extra),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            let size = null + 1;
            let fitted = view_heap_fit(count, extra, null);
            if fitted < count {
                // shortcut: descriptors past the host's limit are left out; a title that reaches
                // them needs remapping. They read the null slot instead.
                tracing::warn!(
                    count,
                    "host view heap capped at {size}; views past it read a null slot"
                );
            }
            match make(size) {
                Ok(heap) => (heap, fitted, size),
                Err(e) => {
                    tracing::warn!(error = %e, count, size, "no host view heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(size) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 4;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // Every slot holds no buffer view yet (`SLOT_STRIDES_SKIP`).
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0xFF, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        // (16 MiB of upload memory a view heap: only for the builder's runs,
        // whose structures the programs that read it walk.)
        let addresses = (!sampler && crate::pipeline::builder_allowed())
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "view addresses by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        let queued: std::sync::Arc<[std::sync::atomic::AtomicU64]> =
            (0..count.div_ceil(64)).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: (0..count).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            locks: SlotLocks::new(striped()),
            copied: Mutex::new(Vec::new()),
            kept: Striped::new((count + extra) as usize, || None),
            next: Mutex::new((0, vec![false; DYNAMIC as usize])),
            free: std::sync::atomic::AtomicU32::new(DYNAMIC),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            addresses,
            buffers: Mutex::default(),
            changed: Mutex::default(),
            textures: Mutex::new(TextureQueue { queued: Some(queued.clone()), ..TextureQueue::default() }),
            queued,
            tabled: (0..if sampler { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            placed: {
                let spans = std::sync::Arc::<Mutex<Spans>>::default();
                if !sampler {
                    PLACED_SPANS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(std::sync::Arc::downgrade(&spans));
                }
                spans
            },
            stamps: Mutex::default(),
            stale: Mutex::default(),
            hidden: Mutex::default(),
            asked: Mutex::default(),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        if null != 0 {
            null_buffer_view(h.cpu(null - victory_shader::emit::NULL_BUFFER_SRV_BELOW));
            null_buffer_uav(h.cpu(null - victory_shader::emit::NULL_BUFFER_UAV_BELOW));
            null_constant_view(h.cpu(null - victory_shader::emit::NULL_CBV_BELOW));
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for a list that sets one of our root
/// signatures while the title binds no sampler heap: every one says its
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`), and
/// D3D12 requires such a heap to be bound before such a signature is set
/// (the debug layer's id 1419); NVIDIA removes the device for it
/// (Marvisak/VIctory #1754, a compute list that reads views and no
/// samplers). Slot 0 holds a point sampler, so an indexed read of it gives
/// something defined.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    fallback(true)
}

/// A view heap of our own, for a list that sets one of our root signatures
/// while the title binds no view heap: they index the view heap directly
/// too (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`), with the same rule as
/// [`fallback_samplers`] (Marvisak/VIctory #2894). Made as a title heap of
/// no slots: every slot up to the one shaders clamp view indices to is a
/// null view, so whatever a shader indexes reads zeros.
pub(crate) fn fallback_views() -> Option<&'static HostHeap> {
    fallback(false)
}

/// [`fallback_samplers`] and [`fallback_views`], each made on first use.
fn fallback(sampler: bool) -> Option<&'static HostHeap> {
    static HEAPS: [std::sync::OnceLock<Option<HostHeap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    HEAPS[usize::from(sampler)]
        .get_or_init(|| {
            if sampler {
                let heap = HostHeap::new(true, 1)?;
                heap.null_sampler(0);
                Some(heap)
            } else {
                HostHeap::new(false, 0)
            }
        })
        .as_ref()
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let enough = self.free.load(std::sync::atomic::Ordering::Relaxed) >= n
            && n < self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..self.dynamic as usize).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts.take(if enough { DYNAMIC as usize } else { 0 }) {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            if enough {
                self.too_long.fetch_min(n as u32, std::sync::atomic::Ordering::Relaxed);
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        taken[first..first + n].fill(true);
        self.free.fetch_sub(n as u32, std::sync::atomic::Ordering::Relaxed);
        *next = ((first + n) % DYNAMIC as usize) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// The table of view addresses by title slot (view heaps, with the
    /// builder asked for): its GPU address and its buffer, for a list to
    /// hold.
    pub(crate) fn addresses(&self) -> Option<(u64, &ID3D12Resource)> {
        self.addresses.as_ref().map(|a| (a.1, &a.0))
    }

    /// Notes the title address of the view now at title slot `index` in
    /// [`HostHeap::addresses`].
    fn note_address(&self, index: u32, address: u64) {
        if let Some((_, _, data)) = &self.addresses
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: inside the mapping (two dwords a slot); a shader that
            // reads while we write may see half of the new address, as a
            // console shader may read a descriptor the CPU is writing.
            unsafe {
                let at = (*data as *mut u32).add(2 * index as usize);
                at.write_volatile(address as u32);
                at.add(1).write_volatile((address >> 32) as u32);
            }
        }
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in the order to serve them
    /// ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        let census = textures.census_lately(frame, crate::host::now_ms());
        (if max == 0 { Vec::new() } else { textures.due(frame, max) }, census)
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => textures.done(index, write),
        }
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.recheck(index, write, due, at, print, looks, uploaded);
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).checked(index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.later(index, write, frame, most);
    }

    /// See [`TextureQueue::hold`].
    pub(crate) fn texture_slot_held(&self, index: u32, write: u64, due: u64) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).hold(index, write, due);
    }

    /// See [`TextureQueue::room`].
    pub(crate) fn texture_slots_room(&self, frame: u64) -> usize {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).room(frame)
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws. After ranges were dropped ([`OVERFLOWED`]) it
    /// makes every placed slot due, once. Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        // Taken once: only another overflow sets it again.
        let all = OVERFLOWED.swap(false, Relaxed);
        if ranges.is_empty() && !all {
            return 0;
        }
        // Per slot whose texture's memory a range lies in: the number of
        // the last such tell and who told; and the same for the slots whose
        // span a range lies in though another texture starts between.
        let (mut slots, mut over): (Vec<(u32, u64, u8)>, Vec<(u32, u64, u8)>) = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            if all {
                (placed.by_slot.keys().map(|&slot| (slot, 0, Cause::Mapped as u8)).collect(), Vec::new())
            } else {
                let (mut slots, mut over) = (Vec::new(), Vec::new());
                for &(start, end, number, cause) in &ranges {
                    let mut held = placed.slots_in(start, end);
                    held.sort_unstable();
                    over.extend(
                        placed
                            .slots_over(start, end)
                            .into_iter()
                            .filter(|s| held.binary_search(s).is_err())
                            .map(|s| (s, number, cause)),
                    );
                    slots.extend(held.into_iter().map(|s| (s, number, cause)));
                }
                (slots, over)
            }
        };
        if all {
            tracing::info!(slots = slots.len(), "remapped ranges overflowed: every placed slot is looked at again");
        }
        let newest = |list: &mut Vec<(u32, u64, u8)>| {
            list.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            list.dedup_by_key(|s| s.0);
        };
        newest(&mut slots);
        newest(&mut over);
        for &(_, _, cause) in &slots {
            crate::census::tell(cause, crate::census::T_HELD, 1);
        }
        slots.retain(|&(slot, ..)| !self.tabled_lately(slot, frame));
        over.retain(|&(slot, ..)| !self.tabled_lately(slot, frame) && slots.binary_search_by_key(&slot, |s| s.0).is_err());
        self.note_hidden(&over);
        {
            let stamps = self.stamps.lock().unwrap_or_else(|e| e.into_inner());
            for &(slot, number, cause) in &slots {
                crate::census::tell(cause, crate::census::T_DUE, 1);
                let read_after = stamps
                    .get(&slot)
                    .is_some_and(|stamp| crate::writes::holds(stamp.load(std::sync::atomic::Ordering::SeqCst), number));
                crate::census::tell(cause, crate::census::T_READ_AFTER, u64::from(read_after));
            }
        }
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &(slot, _, cause) in &slots {
            textures.changed(slot, frame, frame + 1, crate::census::told(Cause::of(cause)));
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// `over`: slots (each once, with the number of a tell and who told)
    /// whose texture's span the tell lay in while it was taken for another
    /// texture's. Those the pass uploaded before the tell are noted
    /// (`hidden`) until uploaded again, and counted when first noted.
    fn note_hidden(&self, over: &[(u32, u64, u8)]) {
        use std::sync::atomic::Ordering::SeqCst;
        if over.is_empty() {
            return;
        }
        let before: Vec<(u32, u8)> = {
            let stamps = self.stamps.lock().unwrap_or_else(|e| e.into_inner());
            over.iter()
                .filter(|&&(slot, number, _)| {
                    stamps.get(&slot).is_some_and(|stamp| !crate::writes::holds(stamp.load(SeqCst), number))
                })
                .map(|&(slot, _, cause)| (slot, cause))
                .collect()
        };
        let mut hidden = self.hidden.lock().unwrap_or_else(|e| e.into_inner());
        for (slot, cause) in before {
            if hidden.insert(slot, cause).is_none() {
                crate::census::tell(cause, crate::census::T_HIDDEN, 1);
            }
        }
    }

    /// Takes the ranges of title memory written behind the guards' back
    /// since the last call ([`page_written`]) and notes stale the slots
    /// whose last upload read before a write's bytes were there
    /// ([`HostHeap::slots_written_under`]). With `VICTORY_PAGE_WRITES` bit
    /// 2 (`crate::watch::page_writes`) every stale slot is due at once as
    /// well; without, a stale slot waits for whatever else makes it due.
    /// Returns how many slots it noted stale.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_page_written(&self, frame: u64) -> usize {
        let ranges = std::mem::take(&mut *PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        let stale = self.slots_written_under(&ranges, frame);
        if crate::watch::page_writes() & 2 != 0 && !stale.is_empty() {
            let by = self.stale.lock().unwrap_or_else(|e| e.into_inner());
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &slot in &stale {
                let cause = Cause::of(by.get(&slot).copied().unwrap_or(0));
                textures.changed(slot, frame, frame, crate::census::told(cause));
            }
        }
        stale.len()
    }

    /// [`HostHeap::texture_slots_page_written`] for `ranges` (start, end,
    /// the write's number, what wrote it): the slots whose placed texture holds bytes of
    /// them ([`Spans::slots_holding`]) and whose last upload by the texture
    /// pass read before the write's number was given
    /// ([`HostHeap::texture_slot_stamp`], `crate::writes::holds`) show the
    /// bytes from before: noted stale (`stale`) until uploaded again. A
    /// slot never uploaded, or whose upload reads later (its list is not
    /// submitted yet), is left as it is: that upload takes the bytes. So is
    /// a slot tables read: its draws bring it up to date. Returns the slots
    /// found stale now.
    fn slots_written_under(&self, ranges: &[(u64, u64, u64, u8)], frame: u64) -> Vec<u32> {
        use std::sync::atomic::Ordering::{Relaxed, SeqCst};
        // Per slot, the highest number of a write it holds bytes of, and
        // what wrote that one.
        let (mut needs, mut over): (Vec<(u32, u64, u8)>, Vec<(u32, u64, u8)>) = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let (mut needs, mut over) = (Vec::new(), Vec::new());
            for &(start, end, number, by) in ranges {
                let mut held = placed.slots_holding(start, end);
                held.sort_unstable();
                over.extend(
                    placed.slots_over(start, end).into_iter().filter(|s| held.binary_search(s).is_err()).map(|s| (s, number, by)),
                );
                needs.extend(held.into_iter().map(|s| (s, number, by)));
            }
            (needs, over)
        };
        let newest = |list: &mut Vec<(u32, u64, u8)>| {
            list.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
            list.dedup_by_key(|n| n.0);
        };
        newest(&mut needs);
        newest(&mut over);
        needs.retain(|&(slot, ..)| !self.tabled_lately(slot, frame));
        over.retain(|&(slot, ..)| !self.tabled_lately(slot, frame) && needs.binary_search_by_key(&slot, |n| n.0).is_err());
        self.note_hidden(&over);
        PAGE_WRITES[2].fetch_add(needs.len() as u64, Relaxed);
        for &(_, _, by) in &needs {
            crate::census::tell(by, crate::census::T_HELD, 1);
        }
        // Read before the write: the stamp of its last upload is older.
        let stale: Vec<(u32, u8)> = {
            let stamps = self.stamps.lock().unwrap_or_else(|e| e.into_inner());
            needs
                .into_iter()
                .filter(|&(slot, number, _)| {
                    stamps.get(&slot).is_some_and(|stamp| !crate::writes::holds(stamp.load(SeqCst), number))
                })
                .map(|(slot, _, by)| (slot, by))
                .collect()
        };
        PAGE_WRITES[3].fetch_add(stale.len() as u64, Relaxed);
        COPY_WRITES[2].fetch_add(stale.iter().filter(|s| s.1 == Cause::Copy as u8).count() as u64, Relaxed);
        for &(_, by) in &stale {
            crate::census::tell(by, crate::census::T_DUE, 1);
        }
        let mut noted = self.stale.lock().unwrap_or_else(|e| e.into_inner());
        noted.extend(stale.iter().copied());
        STALE_NOW.store(noted.len() as u64, Relaxed);
        drop(noted);
        stale.into_iter().map(|s| s.0).collect()
    }

    /// The slots noted stale now, by what made them so: DirectStorage,
    /// the layer's copies.
    pub(crate) fn texture_slots_stale_by(&self) -> [u64; 2] {
        let stale = self.stale.lock().unwrap_or_else(|e| e.into_inner());
        let copies = stale.values().filter(|&&by| by == Cause::Copy as u8).count() as u64;
        [stale.len() as u64 - copies, copies]
    }

    /// What made title slot `index` due (`crate::census`), and whether it
    /// was uploaded in part before; `None` if the queue does not hold it.
    pub(crate) fn texture_slot_why(&self, index: u32) -> Option<(u8, bool)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).why(index)
    }

    /// The queue by what its slots wait for at `frame`
    /// (`crate::census::QUEUE`). The queue's lock, then the spans': 28,000
    /// slots are a fraction of a millisecond, once a second.
    pub(crate) fn texture_queue_kinds(&self, frame: u64) -> [u64; crate::census::KINDS] {
        let (mut kinds, away) = self.textures.lock().unwrap_or_else(|e| e.into_inner()).kinds(frame);
        let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
        let null = away.iter().filter(|slot| placed.by_slot.get(slot) == Some(&0)).count() as u64;
        kinds[3] += null;
        kinds[4] = kinds[4].saturating_sub(null);
        kinds
    }

    /// For an upload of the texture of span `[start, end)` whose view
    /// holds its first `own` bytes: where the first other texture's view
    /// starts inside the span (past its first block), whether that is
    /// inside the part the view holds, and how many slots show the span.
    /// A span that is none (no bytes, or an end below its start) is
    /// counted (`crate::census::FLAGS`) and has nothing inside.
    pub(crate) fn texture_span_others(&self, start: u64, end: u64, own: u64) -> (Option<u64>, bool, usize) {
        if end <= start {
            crate::census::FLAGS[crate::census::NO_SPAN].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
        let inner = placed.next_start(start, end);
        (inner, inner.is_some_and(|at| at < start.saturating_add(own)), placed.slots_at(start).len())
    }

    /// A look found the memory `[at, at + len)` of title slot `index`'s
    /// texture changed since its upload: what is known of it, as a place
    /// of `crate::census::FOUND`. In this order: its upload has not read
    /// yet; a numbered write made it stale; a tell lay in its span and was
    /// taken for another texture's; the guards or the write watch saw a
    /// store since the upload's read; blocks of it are not guarded;
    /// nothing was seen.
    pub(crate) fn texture_slot_found(&self, index: u32, at: u64, len: u64) -> usize {
        use crate::census::{F_HIDDEN, F_OPEN, F_STALE, F_STORE, F_UNREAD, F_UNSEEN};
        use std::sync::atomic::Ordering::SeqCst;
        let stamp = self.stamps.lock().unwrap_or_else(|e| e.into_inner()).get(&index).map(|s| s.load(SeqCst));
        if stamp.is_none_or(|s| s == crate::writes::NOT_READ) {
            return F_UNREAD;
        }
        if let Some(&by) = self.stale.lock().unwrap_or_else(|e| e.into_inner()).get(&index) {
            return F_STALE + usize::from(by);
        }
        if let Some(&by) = self.hidden.lock().unwrap_or_else(|e| e.into_inner()).get(&index) {
            return F_HIDDEN + usize::from(by);
        }
        let asked = self.asked.lock().unwrap_or_else(|e| e.into_inner()).get(&index).map_or(0, |a| a.load(SeqCst));
        let (noted, written, open) = crate::watch::seen_since(at as usize, at.saturating_add(len) as usize, asked);
        if noted || written > 0 {
            F_STORE
        } else if open > 0 {
            F_OPEN
        } else {
            F_UNSEEN
        }
    }

    /// The texture pass uploaded title slot `index`, and the upload's read
    /// will begin (or began) at the number in `stamp` (`crate::writes`;
    /// `NOT_READ` until it does): kept for the writes that come after it
    /// ([`HostHeap::slots_written_under`]); the slot is stale no more.
    /// `asked`: where the read notes the journal's asking it came after
    /// ([`HostHeap::texture_slot_found`]).
    pub(crate) fn texture_slot_stamp(
        &self,
        index: u32,
        stamp: std::sync::Arc<std::sync::atomic::AtomicU64>,
        asked: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) {
        self.stamps.lock().unwrap_or_else(|e| e.into_inner()).insert(index, stamp);
        self.asked.lock().unwrap_or_else(|e| e.into_inner()).insert(index, asked);
        self.hidden.lock().unwrap_or_else(|e| e.into_inner()).remove(&index);
        let mut stale = self.stale.lock().unwrap_or_else(|e| e.into_inner());
        if stale.remove(&index).is_some() {
            STALE_NOW.store(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Title slot `index` was left for after a write the title has
    /// submitted: due again from frame `due`, nothing else of it changed.
    pub(crate) fn texture_slot_after(&self, index: u32, write: u64, due: u64) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).after(index, write, due);
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full. `made` first, then `kept`,
        // as `mirror` takes them.
        let _slot = self.locks.slot(index);
        self.made_key(index).is_some_and(|made| made != 0 && made == key)
            && self.kept.with(index, |kept| kept.is_some()).unwrap_or(false)
    }

    /// Whether title slot `index` may be in the queue of textures picked
    /// by index, asked without its lock: false only if it is not (a slot
    /// goes in under that lock, and its bit is set with it: a write that
    /// finds it clear comes before any write that puts the slot in).
    fn maybe_queued(&self, index: u32) -> bool {
        self.queued
            .get(index as usize / 64)
            .is_none_or(|w| w.load(std::sync::atomic::Ordering::Relaxed) & (1 << (index % 64)) != 0)
    }

    /// What slot `index` was made from (under its lock); `None` past the
    /// title's slots.
    fn made_key(&self, index: u32) -> Option<u64> {
        self.made.get(index as usize).map(|m| m.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Notes slot `index` made from `key` (under its lock).
    fn set_made(&self, index: u32, key: u64) {
        if let Some(m) = self.made.get(index as usize) {
            m.store(key, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Levels on demand (`crate::tiles`): makes again the host slots of the
    /// placed textures whose first level with memory changed since the
    /// last call ([`levels_changed`]), each from the view it was made
    /// from, so that it shows the texture from that level ([`make`]). Only
    /// slots that are made; a slot draws read through a table is made by
    /// its draw. Returns how many it made.
    /// shortcut: the starts are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    /// Levels on demand (`crate::tiles`): the card has room, so the slots
    /// of up to [`Self::WAITING_A_PASS`] placed textures whose upload left
    /// a held level without tiles at least a second ago
    /// ([`levels_waiting`]) are due again from the next frame, as memory
    /// mapped under them would make them
    /// ([`HostHeap::texture_slots_remapped`]): the upload then gives the
    /// level its tiles. To be called only while the card has room: without
    /// it the upload would give nothing again. Slots tables read are left
    /// to their draws (`texture.rs` `upload` asks again for those). Returns
    /// how many slots it made due.
    /// shortcut: kept in one place for all heaps, as the starts below.
    pub(crate) fn texture_slots_waiting(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts: Vec<u64> = {
            let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.is_empty() {
                return 0;
            }
            let starts: Vec<u64> = waiting
                .iter()
                .filter(|&(_, &since)| crate::host::long_ago(since, 1000, 2))
                .map(|(&start, _)| start)
                .take(Self::WAITING_A_PASS)
                .collect();
            for start in &starts {
                waiting.remove(start);
            }
            starts
        };
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1, crate::census::LEVELS);
        }
        crate::tiles::STATS[10].fetch_add(slots.len() as u64, std::sync::atomic::Ordering::Relaxed);
        slots.len()
    }

    /// Textures a pass asks again for their levels
    /// ([`HostHeap::texture_slots_waiting`]): the first ones given tiles
    /// may take the room the others were asked for, and an upload that
    /// gives nothing is a whole texture staged and copied for nothing.
    const WAITING_A_PASS: usize = 128;

    pub(crate) fn texture_views_remade(&self) -> usize {
        // With the views published at landings (`crate::tiles::publishing`)
        // the slots are made again only at a moment with nothing in flight
        // ([`remake_at_quiet_point`]), not here, while submissions run.
        if crate::tiles::publishing() {
            return 0;
        }
        let starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        let flying = crate::host::submissions_in_flight() > 0;
        let remade = self.texture_views_remade_at(&starts);
        if flying {
            crate::tiles::ORDER[5].fetch_add(remade as u64, std::sync::atomic::Ordering::Relaxed);
        }
        remade
    }

    /// [`HostHeap::texture_views_remade`] for the placed textures that
    /// start at `starts`.
    pub(crate) fn texture_views_remade_at(&self, starts: &[u64]) -> usize {
        if self.sampler || starts.is_empty() {
            return 0;
        }
        let slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        let mut remade = 0;
        for slot in slots {
            // Each slot under its own lock: making one view reaches no other.
            let _slot = self.locks.slot(slot);
            let Some(key) = self.made_key(slot) else {
                continue;
            };
            if key == 0 {
                continue;
            }
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if let Some(kept) = make(&view, self.cpu(slot)) {
                self.keep(slot, kept);
                remade += 1;
            }
        }
        remade
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        let _slot = self.locks.slot(index);
        if self.made_key(index).is_none_or(|key| key == 0) {
            return false;
        }
        self.unmake(index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Makes host slot `index` null again, as it is before its view is
    /// made: a null view of the kind it held. The caller holds the slot's
    /// lock, or every slot's.
    fn unmake(&self, index: u32) {
        let Some(key) = self
            .made
            .get(index as usize)
            .map(|m| m.swap(0, std::sync::atomic::Ordering::Relaxed))
        else {
            return;
        };
        if key == 0 {
            return;
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.cpu(index)),
            None => null_view(self.cpu(index)),
        }
        self.keep(index, None);
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) {
        self.tabled_in(index, read, crate::frame::current());
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(stamp) = self.tabled.get(index as usize) else {
            return;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let old = stamp.load(Relaxed);
        if !read {
            if old & !TABLED_READ != now {
                stamp.store(now, Relaxed);
            }
            return;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            stamp.store(now | TABLED_READ, Relaxed);
            if tabled_out() && self.maybe_queued(index) {
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                if textures.take(index).is_some() {
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other).
    ///
    /// `VICTORY_TABLED_SKIP_FRAMES=<n>` asks the stricter question the
    /// mirror pass needs: a slot may be left to its draws only if a draw
    /// read it **through a table** within the last `n` frames, because only
    /// then did that draw make its own view of it. Without the read bit the
    /// window of [`TABLED_FRAMES`] (240) also lets a slot the title stopped
    /// using through a table be written directly while the mirror skips it,
    /// and a draw then reads a record no pass made.
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index);
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        if let Some(frames) = skip_frames() {
            return stamp & TABLED_READ != 0
                && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= frames;
        }
        tabled_out() && stamp & !TABLED_READ != 0
            && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u32 {
        self.tabled
            .get(index as usize)
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`].
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, frame: u64) {
        let stamp = if tabled_out() { self.tabled_stamp(index) } else { 0 };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw.
        let lately = stamp != 0 && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        // Not a texture view, in a slot the queue does not hold: nothing to
        // take out, and no lock (the recording threads stood 2.4 ms a frame
        // in this one, mostly to take out what was not there).
        if !texture && !self.maybe_queued(index) {
            return;
        }
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        if texture && !read_now {
            let due = if lately && !seen {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                0
            };
            textures.written(index, frame, due, seen);
        } else {
            textures.remove(index);
        }
    }

    /// The title slots that hold buffer views.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        self.buffers.lock().unwrap_or_else(|e| e.into_inner()).to_vec()
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call (in the order first written; the set this was gave them
    /// in no order).
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        self.changed.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: its element stride in the table, and whether
    /// it is a buffer view at all.
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held.
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]): out of the queue of textures already, their host
    /// slots are the caller's to make null (it holds `made`).
    fn note_buffer(&self, index: u32, view: Option<&View>, seen: bool) -> Vec<u32> {
        self.note_buffer_in(index, view, seen, true).unwrap_or_default()
    }

    /// [`HostHeap::note_buffer`]; with `may_stale` false (the caller holds
    /// only its own slot's lock), `Err` and nothing done if the write would
    /// make other slots' views stale: the caller then takes every slot's
    /// lock and writes again.
    fn note_buffer_in(&self, index: u32, view: Option<&View>, seen: bool, may_stale: bool) -> Result<Vec<u32>, NeedsAll> {
        if self.sampler {
            return Ok(Vec::new());
        }
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && !self.tabled_lately(index, crate::frame::current()) =>
            {
                // (And the level the view starts at: word 4 of a texture
                // view's description, after format, dimension, mapping
                // and a pad.)
                Some((resource, u64::from(desc[10]) | u64::from(desc[11]) << 32, desc[4]))
            }
            _ => None,
        };
        let stale = {
            use std::sync::atomic::Ordering::Relaxed;
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let span = placed_view.and_then(|(resource, start, first)| {
                Some((start, start.saturating_add(placed.bytes(resource)?), first))
            });
            if let Some((start, end, _)) = span
                && !may_stale
            {
                placed.rule = stale_rule();
                if placed.would_stale(index, start, end, !seen) {
                    return Err(NeedsAll);
                }
            }
            match span {
                Some((start, end, first)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    placed.rule = stale_rule();
                    let stale = placed.note(index, start, end, !seen);
                    placed.shows(index, first);
                    stale
                }
                None => {
                    placed.forget(index);
                    Vec::new()
                }
            }
        };
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object. (After the spans, which may send the
        // write back for every slot's lock; the queue is the slot's own.)
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(View::Shader(r, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        self.queue_texture(index, texture, seen, crate::frame::current());
        if !stale.is_empty() {
            STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &slot in &stale {
                textures.remove(slot);
            }
        }
        let stride = view.and_then(view_stride);
        // Until `mirror` has written the slot (and then the entry), the
        // entry says it holds no buffer view: the new stride written here
        // sat beside the slot's old descriptor, a texture's read as a
        // buffer's.
        self.set_stride(index, victory_shader::emit::SLOT_STRIDES_SKIP);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        self.note_address(index, view.map_or(0, crate::draw::view_address));
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let was = if stride.is_some() {
            buffers.insert(index);
            true
        } else {
            buffers.remove(index)
        };
        if was {
            self.changed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(index);
        }
        Ok(stale)
    }

    /// Of a write of title slot `index` with a buffer view (`Quick`): what
    /// [`HostHeap::note_buffer`] does for it, which is the same for every
    /// such view whatever slot it is in.
    fn quick(&self, index: u32, view: &View) -> Option<Quick> {
        let buffer = match view {
            View::Constant(..) | View::Placed(true, ..) => true,
            // A placed texture's view notes its memory, which may make
            // other slots stale: not quick.
            View::Placed(false, resource, _) => resource[0] == 1,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as crate::object::This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        };
        buffer.then(|| Quick {
            index,
            stride: if self.sampler { None } else { view_stride(view) },
            entry: stride_entry(view),
            address: if self.sampler { 0 } else { crate::draw::view_address(view) },
            null: (!self.sampler).then(|| null_kind(view)),
            viewed: true,
        })
    }

    /// The rest of the writes `run` noted by [`HostHeap::quick`], in the
    /// order [`HostHeap::mirror`] does them for each: [`HostHeap::note_buffer`]
    /// (the queue of textures, the placed spans, the stride table, the sets
    /// of buffer slots), the null view, then [`HostHeap::keep`]. Each lock
    /// once for the run, the null views in one copy, nothing allocated
    /// once `scratch` has grown. The caller holds the slots' locks and has
    /// noted them unmade.
    fn finish_quick(&self, run: &mut Vec<Quick>, scratch: &mut PickScratch) {
        if run.is_empty() {
            return;
        }
        if !self.sampler {
            scratch.indices.clear();
            scratch.indices.extend(run.iter().map(|q| q.index));
            {
                // Not a texture view: out of the queue
                // (`HostHeap::queue_texture`).
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                present_into(&textures.slots, &scratch.indices, &mut scratch.present);
                for &k in &scratch.present {
                    textures.remove(k);
                }
            }
            {
                // No placed texture view: out of the spans.
                let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
                for &index in &scratch.indices {
                    placed.forget(index);
                }
            }
            // Until the slots are written (`write_nulls` below), their entries
            // say they hold no buffer view, as in `note_buffer`.
            for q in run.iter() {
                self.set_stride(q.index, victory_shader::emit::SLOT_STRIDES_SKIP);
            }
            if self.addresses.is_some() {
                for q in run.iter() {
                    self.note_address(q.index, q.address);
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            {
                let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
                let mut changed = None;
                for q in run.iter() {
                    let was = if q.stride.is_some() {
                        buffers.insert(q.index);
                        true
                    } else {
                        buffers.remove(q.index)
                    };
                    if was {
                        changed
                            .get_or_insert_with(|| self.changed.lock().unwrap_or_else(|e| e.into_inner()))
                            .insert(q.index);
                    }
                }
            }
            self.write_nulls(run, scratch);
            // The slots hold null views of their kind: now their entries
            // may say so, behind the slots' own bytes.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            for q in run.iter() {
                self.set_stride(q.index, q.entry);
            }
        }
        // `keep(index, None)` of each, the resources they let go of retired
        // together (one lock and one allocation for the run, not each).
        let mut old: Vec<ID3D12Resource> = Vec::new();
        for q in run.drain(..) {
            if let Some(resource) = self.kept.with(q.index, Option::take).flatten() {
                old.push(resource);
            }
        }
        if !old.is_empty() {
            crate::host::retire(old);
        }
    }

    /// The null views of `run`'s slots: copied from [`NullTemplates`] in one
    /// call, or made one by one as [`null_for`] makes them if there are none.
    fn write_nulls(&self, run: &[Quick], scratch: &mut PickScratch) {
        let Some(host) = host() else { return };
        scratch.nulls.clear();
        scratch.nulls.extend(run.iter().filter_map(|q| q.null.map(|k| (q.index, k.slot()))));
        if scratch.nulls.is_empty() {
            return;
        }
        let Some(templates) = null_templates() else {
            for q in run {
                if let Some(kind) = q.null {
                    make_null_kind(kind, self.cpu(q.index));
                }
            }
            return;
        };
        for part in scratch.nulls.chunks(NULL_RUN as usize) {
            null_ranges_into(part, NULL_RUN, &mut scratch.dst, &mut scratch.src);
            scratch.dst_starts.clear();
            scratch.dst_starts.extend(scratch.dst.iter().map(|&(s, _)| self.cpu(s)));
            scratch.dst_sizes.clear();
            scratch.dst_sizes.extend(scratch.dst.iter().map(|&(_, n)| n));
            scratch.src_starts.clear();
            scratch.src_starts.extend(scratch.src.iter().map(|&(s, _)| templates.at(s)));
            scratch.src_sizes.clear();
            scratch.src_sizes.extend(scratch.src.iter().map(|&(_, n)| n));
            // SAFETY: slots of our heap and of the template heap (CPU only,
            // the same type), as many on each side.
            unsafe {
                host.device.CopyDescriptors(
                    scratch.dst_starts.len() as u32,
                    scratch.dst_starts.as_ptr(),
                    Some(scratch.dst_sizes.as_ptr()),
                    scratch.src_starts.len() as u32,
                    scratch.src_starts.as_ptr(),
                    Some(scratch.src_sizes.as_ptr()),
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                )
            };
        }
    }

    /// For the pass over records copied in as bytes (`super::sync_heaps`):
    /// [`HostHeap::mirror`] of each of `picked` (title slots, at the
    /// handles `handle_of` gives). `view_of` reads the view of a record's
    /// words, as [`view_at`] does. Returns how many were done in runs.
    ///
    /// Each record is first looked at without a lock: one whose slot is
    /// made from it already, or that names nothing in a slot that shows
    /// nothing, is left as `mirror` would leave it; a buffer view, or
    /// nothing where the slot showed something, goes into a run
    /// ([`HostHeap::quick`]); any other record is mirrored by `mirror`
    /// itself, after the run before it. A run takes its slots' locks, keeps
    /// the records that neither the title nor another thread changed
    /// meanwhile (the others are mirrored one by one after it), and does
    /// them together ([`HostHeap::finish_quick`]). Every record ends as
    /// `mirror` alone would leave it.
    /// `VICTORY_HEAP_SYNC_FAST=0`: not used, each record is mirrored by
    /// itself, as before.
    pub(crate) fn mirror_picked(
        &self,
        picked: &[u32],
        handle_of: impl Fn(u32) -> u64,
        view_of: &mut impl FnMut(u32, u32) -> Option<View>,
        scratch: &mut PickScratch,
    ) -> u64 {
        let mut quick = 0;
        let mut kinds = [0u64; PICKED_KINDS];
        let mut run = std::mem::take(&mut scratch.run);
        let mut at = 0;
        while at < picked.len() {
            run.clear();
            scratch.seen.clear();
            let mut alone = None;
            while at < picked.len() && run.len() < NULL_RUN as usize {
                let index = picked[at];
                at += 1;
                if index >= self.count {
                    // (`mirror` would only say so.)
                    continue;
                }
                let handle = handle_of(index);
                // SAFETY: the handle is a descriptor slot in title memory.
                let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
                let key = u64::from(generation) << 32 | u64::from(i);
                let view = if tag == VIEW_TAG { view_of(i, generation) } else { None };
                let made = self.made_key(index).unwrap_or(0);
                let q = match &view {
                    Some(_) if made == key => {
                        kinds[PICKED_SAME] += 1;
                        continue;
                    }
                    None if made == 0 => {
                        kinds[PICKED_NOTHING] += 1;
                        continue;
                    }
                    Some(view) => self.quick(index, view).inspect(|_| kinds[PICKED_BUFFER] += 1),
                    // Made null, as `mirror` makes it: `note_buffer` of
                    // nothing, then `make_null` (no buffer view there).
                    None if !self.sampler => {
                        kinds[PICKED_GONE] += 1;
                        Some(Quick {
                            index,
                            stride: None,
                            entry: victory_shader::emit::SLOT_STRIDES_SKIP,
                            address: 0,
                            null: Some(NullKind::Plain),
                            viewed: false,
                        })
                    }
                    None => None,
                };
                match q {
                    Some(mut q) => {
                        q.viewed = tag == VIEW_TAG;
                        run.push(q);
                        scratch.seen.push((key, made));
                    }
                    None => {
                        let placed = matches!(&view, Some(View::Placed(false, r, _)) if r[0] != 1);
                        kinds[if placed { PICKED_PLACED } else { PICKED_OTHER }] += 1;
                        alone = Some(index);
                        break;
                    }
                }
            }
            if !run.is_empty() {
                scratch.again.clear();
                {
                    scratch.indices.clear();
                    scratch.indices.extend(run.iter().map(|q| q.index));
                    let _slots = self.locks.stripes(&scratch.indices);
                    // Under the locks: only the records as they were
                    // looked at, in slots nobody made meanwhile.
                    let mut k = 0;
                    run.retain(|q| {
                        let (key, made) = scratch.seen[k];
                        k += 1;
                        // SAFETY: as above.
                        let [tag, i, generation] =
                            unsafe { (handle_of(q.index) as *const [u32; 3]).read_unaligned() };
                        let now = u64::from(generation) << 32 | u64::from(i);
                        let viewed = tag == VIEW_TAG;
                        let same = now == key && viewed == q.viewed && self.made_key(q.index) == Some(made);
                        if same {
                            // Left for a draw or a pass, as `mirror` leaves it.
                            self.set_made(q.index, 0);
                        } else {
                            scratch.again.push(q.index);
                        }
                        same
                    });
                    quick += run.len() as u64;
                    self.finish_quick(&mut run, scratch);
                }
                for k in 0..scratch.again.len() {
                    let index = scratch.again[k];
                    let _ = self.mirror(index, handle_of(index), false);
                }
            }
            // The record that was not for a run: by `mirror`, which reads it
            // again under its own lock, after every record before it.
            if let Some(index) = alone {
                let _ = self.mirror(index, handle_of(index), false);
            }
        }
        scratch.run = run;
        for (count, n) in PICKED.iter().zip(kinds) {
            count.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        }
        quick
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let given = slots.iter().filter(|taken| **taken).count() as u32;
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        // A running submission may still read the old resource through
        // this slot.
        let Some((old, same)) = self.kept.with(index, |slot| {
            let same = matches!((&*slot, &resource), (Some(a), Some(b)) if windows::core::Interface::as_raw(a) == windows::core::Interface::as_raw(b));
            (std::mem::replace(slot, resource), same)
        }) else {
            return;
        };
        if let Some(old) = old {
            if same {
                // The slot holds the same resource again: the reference it
                // had goes now, as the slot's new one keeps the resource as
                // it was (a draw making a table slot with the buffer the
                // slot showed before retired it every time, under the
                // retire list's lock, which the recording threads stood in).
                drop(old);
            } else {
                crate::host::retire(old);
            }
        }
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds); a null one for no bytes.
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = match constant_view_bytes(size) {
            Some(bytes) if gpu != 0 => D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: bytes,
            },
            // A zero location and size: the null constant view.
            _ => D3D12_CONSTANT_BUFFER_VIEW_DESC::default(),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// For a slot on a page of the title's heap that changed with no call
    /// (`super::sync_heaps`: the title copies records into its heap as
    /// bytes): [`HostHeap::mirror`], if the `bytes` bytes of the record at
    /// `handle` are not the ones this last saw there. True if it did.
    ///
    /// The page says only that something on it was written, and `mirror`
    /// by itself would take every buffer or placed-texture record on it for
    /// newly written each time (it leaves such a slot unmade until a draw
    /// or a pass uses it): the texture queue's entry for the slot would
    /// start again, and the buffer pass would look at 2,048 slots for one
    /// that changed.
    /// `record`: the record's bytes as the caller read them (through the
    /// kernel, which fails on a page the title unmapped meanwhile where a
    /// read in place would fault: the title unmaps heap pages too).
    pub(crate) fn mirror_copied(&self, index: u32, handle: u64, record: &[u8]) -> bool {
        let new = {
            let mut copied = self.copied_prints();
            self.copied_new(&mut copied, index, record)
        };
        if new {
            self.mirror(index, handle, false);
        }
        new
    }

    /// The fingerprints [`HostHeap::copied_new`] keeps, locked (for a page
    /// of slots at once: only the submit thread's pass takes it).
    pub(crate) fn copied_prints(&self) -> std::sync::MutexGuard<'_, Vec<u64>> {
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        if copied.is_empty() {
            copied.resize(self.count as usize, 0);
        }
        copied
    }

    /// [`HostHeap::copied_new`] of the records `records` (`step` bytes each)
    /// of title slots `first` on, pushing the new ones to `picked`; returns
    /// how many it looked at. The same records come out, but a record's
    /// print ([`quick_print`]) is compared before anything else, so that the
    /// many unchanged ones cost a few instructions each, and what every
    /// record would ask the same of (the frame, the switches) is asked once.
    /// (`COPIED_TABLED` then counts only new records left to their draws.)
    pub(crate) fn scan_copied(&self, copied: &mut [u64], first: u32, records: &[u8], step: usize, picked: &mut Vec<u32>) -> u64 {
        let frame = crate::frame::current();
        let tables = !sync_tables();
        let (mut looked, mut tabled) = (0u64, 0u64);
        for (k, record) in records.chunks_exact(step).enumerate() {
            let index = first + k as u32;
            if index >= self.count {
                break;
            }
            let Some(last) = copied.get_mut(index as usize) else {
                break;
            };
            looked += 1;
            // Never 0, which is "never looked at".
            let print = quick_print(record) | 1;
            if *last == print {
                continue;
            }
            if tables && self.tabled_lately(index, frame) {
                tabled += 1;
                continue;
            }
            if print_is_new(last, print, record) {
                picked.push(index);
            }
        }
        COPIED_TABLED.fetch_add(tabled, std::sync::atomic::Ordering::Relaxed);
        looked
    }

    /// Of [`HostHeap::mirror_copied`]: whether the record `record` of title
    /// slot `index` is to be mirrored (one this has not seen there, in a
    /// slot tables do not read), noting it seen.
    pub(crate) fn copied_new(&self, copied: &mut [u64], index: u32, record: &[u8]) -> bool {
        if index >= self.count {
            return false;
        }
        // A slot draws read through a table lately is left to its draws,
        // as the passes over slots picked by index leave it
        // (`texture_slots_remapped`, `slots_written_under`): the title
        // writes its per-draw tables into the same heap every frame, and
        // each draw makes what its table shows in a slot of its own. Those
        // were most of the 26,000 to 43,000 records a frame this pass
        // mirrored in run 200 and in the play session after it, each then
        // made and brought up to date once more by the pass over buffer
        // views (some 50 ms a frame). `VICTORY_HEAP_SYNC_TABLES=1`: they
        // are mirrored too, as in run 200.
        if !sync_tables() && self.tabled_lately(index, crate::frame::current()) {
            COPIED_TABLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        // Never 0, which is "never looked at". (A multiply-and-fold over
        // the record's words: BLAKE3 here was most of the 35 to 42 ms a
        // frame this pass took over some 250,000 to 300,000 records, run
        // 200.)
        let print = record_print(record) | 1;
        let Some(last) = copied.get_mut(index as usize) else {
            return false;
        };
        // A slot never looked at that holds nothing: as it was.
        print_is_new(last, print, record)
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // The slot's own lock, unless the write of a placed texture's view
        // makes other slots' views stale ([`HostHeap::note_buffer`]): then
        // every slot's. Known only when the write is noted, which says so
        // before it changes anything ([`Spans::would_stale`]): the record is
        // then read again under every slot's lock. (Taking every slot's lock
        // for any placed texture's view that changed stopped all recording
        // threads behind each such write and the views being made before
        // it, some 10 ms a frame of theirs.)
        let mut all = false;
        loop {
            let slot = if all { self.locks.all() } else { self.locks.slot(index) };
            // SAFETY: the handle is a descriptor slot in title memory.
            let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            // Generations start at 1, so a made slot's key is never 0.
            let key = u64::from(generation) << 32 | u64::from(i);
            let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
            match self.mirror_locked(index, handle, key, view, for_draw, slot.exclusive()) {
                Ok(view) => return view,
                Err(NeedsAll) => {
                    drop(slot);
                    all = true;
                }
            }
        }
    }

    /// [`HostHeap::mirror`] of the record `key` (whose view is `view`) at
    /// `handle`, under the lock of slot `index` (`all`: of every slot).
    /// `Err` with nothing done if the write makes other slots' views
    /// stale and `all` is false.
    fn mirror_locked(
        &self,
        index: u32,
        handle: u64,
        key: u64,
        view: Option<View>,
        for_draw: bool,
        all: bool,
    ) -> Result<Option<View>, NeedsAll> {
        let made = |i: u32| self.made_key(i).unwrap_or(0);
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made(index) != 0 {
                self.set_made(index, 0);
                // Out of the stride table before the slot is a null 2D view.
                let _ = self.note_buffer(index, None, false);
                self.make_null(index);
            } else if for_draw {
                // Left for later and stale since: out of the sets and the stride table.
                let _ = self.note_buffer(index, None, false);
            }
            return Ok(None);
        };
        if made(index) != key {
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held. (Only under every slot's
            // lock: without it, the write comes back first.)
            for stale in self.note_buffer_in(index, Some(&view), for_draw, all)? {
                if stale != index {
                    self.unmake(stale);
                }
            }
            // A view of a texture object that has no host copy yet waits
            // for it, as buffer views wait for a draw: the slot is made
            // when the texture is first used (`super::make_waiting`), or
            // here when a draw reads the slot through a table.
            let waiting = if for_draw { None } else { view.waiting_texture() };
            if let Some(texture) = waiting {
                super::wait_for(texture, handle);
            }
            let later = !for_draw && (view.is_buffer() || waiting.is_some());
            // Another view of the placed texture the slot shows already (a
            // title that streams a texture in writes its view again with
            // more mips each time some arrive): the slot goes on showing
            // the old view until a draw or the texture pass makes the new
            // one and uploads what it shows. Made null here, the texture
            // read as nothing for as long as it waited in the queue, every
            // time it got sharper.
            if later
                && keep_views()
                && view_at(made(index) as u32, (made(index) >> 32) as u32)
                    .is_some_and(|old| same_placed_texture(&old, &view))
            {
                KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Some(view));
            }
            self.set_made(index, if later { 0 } else { key });
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index))
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            // The slot holds the view or a null view of its kind: now its
            // entry may say so, behind the slot's own bytes.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            self.set_stride(index, stride_entry(&view));
            self.keep(index, kept.flatten());
        }
        Ok(Some(view))
    }

    /// Writes slot `index`'s entry in the table of strides, which shaders
    /// read while we write (a dword; the caller orders it against the
    /// slot's own bytes).
    fn set_stride(&self, index: u32, entry: u32) {
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: inside the mapping.
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(entry) };
        }
    }

    /// Slot `index`'s entry in the table of strides (tests); `None`
    /// without one.
    #[cfg(test)]
    pub(crate) fn stride(&self, index: u32) -> Option<u32> {
        let (_, _, data) = self.strides.as_ref()?;
        (index < victory_shader::emit::SLOT_STRIDES_COUNT)
            // SAFETY: inside the mapping.
            .then(|| unsafe { (*data as *const u32).add(index as usize).read_volatile() })
    }

    /// Whether slot `index` is backed by a made host descriptor (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        let _slot = self.locks.slot(index);
        self.made_key(index).is_some_and(|key| key != 0)
    }

    /// Whether host slot `index` shows `buffer`; `None` if it shows no
    /// resource (diagnostics: a buffer view's slot must show the host copy
    /// that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        self.kept
            .with(index, |kept| {
                let shown = kept.as_ref()?;
                Some(windows::core::Interface::as_raw(shown) == windows::core::Interface::as_raw(buffer))
            })
            .flatten()
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = {
            let _slot = self.locks.slot(index);
            self.made_key(index)?
        };
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// The words of a shader view of a 1D texture or 1D array as the 2D view
/// of the host's copy (`resource::host_dimension`): same mips and slices,
/// plane 0. The 2D forms have `PlaneSlice` before the clamp (word 6 of a
/// 2D view, word 8 of a 2D array).
fn srv_words_2d(words: &mut [u32; 10]) {
    const TEXTURE1D: u32 = 2;
    const TEXTURE1DARRAY: u32 = 3;
    match words[1] {
        TEXTURE1D => {
            words[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
            words[7] = words[6];
            words[6] = 0;
        }
        TEXTURE1DARRAY => {
            words[1] = D3D12_SRV_DIMENSION_TEXTURE2DARRAY.0 as u32;
            words[9] = words[8];
            words[8] = 0;
        }
        _ => {}
    }
}

/// [`srv_words_2d`] for unordered access views: the 2D forms add
/// `PlaneSlice` at their end.
fn uav_words_2d(words: &mut [u32; 10]) {
    const TEXTURE1D: u32 = 2;
    const TEXTURE1DARRAY: u32 = 3;
    match words[1] {
        TEXTURE1D => {
            words[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
            words[3] = 0;
        }
        TEXTURE1DARRAY => {
            words[1] = D3D12_UAV_DIMENSION_TEXTURE2DARRAY.0 as u32;
            words[5] = 0;
        }
        _ => {}
    }
}

/// A view description from its words, with the format as the host has it
/// and a 1D view as the 2D view of the host's copy ([`srv_words_2d`]).
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    srv_words_2d(&mut words);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The unsigned integer format with the layout of a 16-bit float or
/// signed integer format (one, two or four channels), if `format` is one.
pub(crate) fn half_format(format: u32) -> Option<u32> {
    match format {
        // R16G16B16A16_FLOAT and _SINT: R16G16B16A16_UINT.
        10 | 14 => Some(12),
        // R16G16_FLOAT and _SINT: R16G16_UINT.
        34 | 38 => Some(36),
        // R16_FLOAT and _SINT: R16_UINT.
        54 | 59 => Some(57),
        _ => None,
    }
}

/// `view` as a shader is shown it where it moves a buffer's 16-bit
/// channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`): a buffer view in a
/// 16-bit float or signed format, in the unsigned format of the same
/// layout, through which every 16-bit pattern is read and stored as it
/// is. `None` for every other view: the title's own serves (an unsigned
/// one already carries the bits; a texture can only be viewed in the
/// formats it was made for).
pub(crate) fn half_view(view: &View) -> Option<View> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Shader(*resource, Some(desc)))
        }
        View::Unordered(resource, counter, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Unordered(*resource, *counter, Some(desc)))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Placed(*unordered, *resource, desc))
        }
        _ => None,
    }
}

/// The format of `view`, a buffer view through which a shader moves
/// 16-bit channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`), if those integers
/// are not what the console gives: a format of wider channels, a
/// normalised one or a signed 8-bit one, whose values the console converts
/// to and from halves. `None` for the formats the integers serve exactly
/// (16 bits a channel of any kind, through [`half_view`] where needed, and
/// unsigned channels of fewer bits; the title's are 16-bit unsigned) and
/// for any other view.
pub(crate) fn half_inexact(view: &View) -> Option<u32> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    // R16G16B16A16, R16G16 and R16 (float, unsigned, signed), then
    // R10G10B10A2, R8G8B8A8, R8G8 and R8 unsigned.
    const EXACT: [u32; 13] = [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62];
    let format = match view {
        View::Shader(_, Some(desc)) | View::Unordered(_, _, Some(desc)) if desc[1] == BUFFER => {
            desc[0]
        }
        View::Placed(_, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => desc[0],
        _ => return None,
    };
    (!EXACT.contains(&format)).then_some(format)
}

/// [`srv_desc`] for unordered access views ([`uav_words_2d`]).
fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    uav_words_2d(&mut words);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    // A view off its buffer's elements is made on a copy of its own bytes
    // (`own_bytes`): the same question `buffer_srv` and `buffer_uav` ask,
    // of the copy they find, so that the copy brought up to date is the
    // one the slot shows.
    if let Some((base, width)) = buffer_extent(view)
        && let Some((first, count, element, stride, unknown, raw)) = buffer_fields(view)
        && let Some(whole) = crate::shadow::find(base, width)
        && let Some((start, bytes)) = own_bytes(whole.offset(base), base, first, count, element, stride, unknown, raw)
        && let Some(own) = crate::shadow::exact(start, bytes)
    {
        return Some((own, start, bytes));
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = crate::shadow::find(base, width)
    {
        return Some((shadow, address, len));
    }
    crate::shadow::find(address, len).map(|shadow| (shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Whether the pass over records copied in as bytes also mirrors slots
/// that draws read through tables lately (`VICTORY_HEAP_SYNC_TABLES=1`;
/// off by default: [`HostHeap::mirror_copied`]).
fn sync_tables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_TABLES").is_some_and(|v| v == "1"))
}

/// Since the last frame line: slots on changed heap pages that
/// [`HostHeap::mirror_copied`] left to their draws (tables' slots).
pub(crate) static COPIED_TABLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A 64-bit fingerprint of a descriptor record's bytes (16 or 32 of them):
/// equal records give equal prints, and two different records the same
/// print about once in 2^64. Not a hash anyone can attack: the records
/// are our own.
fn record_print(record: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = K ^ record.len() as u64;
    let mut words = record.chunks_exact(8);
    for word in &mut words {
        let w = u64::from_le_bytes(word.try_into().unwrap_or([0; 8]));
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in words.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// [`record_print`]'s job for the batched pass ([`HostHeap::scan_copied`]):
/// each 64-bit word times its own odd constant, summed, so the multiplies
/// do not wait for each other (`record_print` chains them: most of the
/// 14 ms a frame the scan took over 2.9 million records, run e13). A
/// change of any one word always changes the sum (multiplying by an odd
/// number is one-to-one), and the final fold is one-to-one too.
fn quick_print(record: &[u8]) -> u64 {
    const K: [u64; 4] = [0x9e37_79b9_7f4a_7c15, 0xc2b2_ae3d_27d4_eb4f, 0x1656_67b1_9e37_79f9, 0x27d4_eb2f_1656_67c5];
    let word = |i: usize| u64::from_le_bytes(record[i * 8..i * 8 + 8].try_into().unwrap_or([0; 8]));
    let h = match record.len() {
        32 => word(0)
            .wrapping_mul(K[0])
            .wrapping_add(word(1).wrapping_mul(K[1]))
            .wrapping_add(word(2).wrapping_mul(K[2]))
            .wrapping_add(word(3).wrapping_mul(K[3])),
        16 => word(0).wrapping_mul(K[0]).wrapping_add(word(1).wrapping_mul(K[1])),
        _ => return record_print(record),
    };
    h ^ (h >> 32)
}

/// Whether a buffer view off its buffer's elements gets a copy of its own
/// bytes (`VICTORY_VIEW_OWN_COPY=0`: it is refused and reads zeros, as
/// before).
fn own_copy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_OWN_COPY").is_none_or(|v| v != "0"))
}

/// The title memory (start, bytes) of a buffer view that cannot be made on
/// the host copy of its buffer: `count` elements of `element` bytes from
/// element `first` of the buffer at `base`, which lies `offset` bytes into
/// that copy, where `offset` is not a whole number of elements and the view
/// cannot be a raw view of the same bytes either ([`raw_window`]:
/// `unknown`, the view has no format, and `stride` is its structure's).
/// The host counts a view's first element from its buffer's start, so such
/// a view has no host view there; it is made on a copy of its own bytes
/// (`crate::shadow::exact`), where it starts at element 0. `None`: the
/// view is made on its buffer's copy, as every other.
///
/// A raw view (`raw`: counted in dwords) is also made on a copy of its own
/// bytes when they do not start on 16 in its buffer's copy: the host takes
/// raw views only from a 16-byte boundary, so it was left null and read
/// zeros ("buffer view the host would refuse left null", `first=2 ...
/// raw=true`: 8 bytes in). Its own copy starts at the view's first byte
/// (`VICTORY_VIEW_RAW_OWN_COPY=0`: left null, as before).
#[allow(clippy::too_many_arguments)]
fn own_bytes(
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
    raw: bool,
) -> Option<(u64, u64)> {
    own_bytes_if(own_copy_on(), raw_own_copy_on(), offset, base, first, count, element, stride, unknown, raw)
}

/// Whether a raw buffer view whose bytes do not start on 16 in its
/// buffer's copy gets a copy of its own bytes ([`own_bytes`]).
fn raw_own_copy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_RAW_OWN_COPY").is_none_or(|v| v != "0"))
}

#[allow(clippy::too_many_arguments)]
fn own_bytes_if(
    on: bool,
    raw_on: bool,
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
    raw: bool,
) -> Option<(u64, u64)> {
    if !on || element == 0 || count == 0 {
        return None;
    }
    // A raw view on its copy's dwords, not on 16 bytes there.
    if raw_on
        && raw
        && element == 4
        && offset.is_multiple_of(4)
        && !first.checked_mul(4)?.checked_add(offset)?.is_multiple_of(16)
    {
        let start = base.checked_add(first.checked_mul(4)?)?;
        return Some((start, u64::from(count) * 4));
    }
    if offset.is_multiple_of(element) {
        return None;
    }
    if unknown && raw_window(offset, first, count, stride).is_some() {
        return None;
    }
    let start = base.checked_add(first.checked_mul(element)?)?;
    Some((start, u64::from(count) * element))
}

/// Of a buffer view: its first element, its elements, the bytes of one (a
/// raw view's: 4), its structure's stride, whether it has no format, and
/// whether it is raw. As [`buffer_srv`] and [`buffer_uav`] read them from
/// the same words.
fn buffer_fields(view: &View) -> Option<(u64, u32, u64, u32, bool, bool)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => (d[..10].try_into().ok()?, *unordered),
        _ => return None,
    };
    if unordered {
        let d = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(d.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        let raw = b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        Some((b.FirstElement, b.NumElements, element, b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN, raw))
    } else {
        let d = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        let raw = b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        Some((b.FirstElement, b.NumElements, element_size(&d), b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN, raw))
    }
}

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// Why [`buffer_view_ok`] refuses a view with these fields, in a few words.
fn refusal(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> &'static str {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        if format != DXGI_FORMAT_R32_TYPELESS {
            return "raw, not R32 typeless";
        }
        if stride != 0 {
            return "raw with a stride";
        }
        if !first.is_multiple_of(4) {
            return "raw, not on 16 bytes";
        }
        4
    } else if stride > 0 {
        if format != DXGI_FORMAT_UNKNOWN {
            return "structured with a format";
        }
        if stride > 2048 || !stride.is_multiple_of(4) {
            return "a stride the host has no structured buffers for";
        }
        u64::from(stride)
    } else {
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return "typed without a usable format (typeless, unknown or block)",
        }
    };
    if count == 0 {
        return "no elements";
    }
    match first.checked_add(u64::from(count)).and_then(|end| end.checked_mul(element)) {
        Some(end) if end <= width => "accepted",
        _ => "past the end of its buffer's copy",
    }
}

/// A buffer view left null ([`buffer_view_ok`]): logged with its reason the
/// first four times for each reason and at every power of two after (each
/// site said it once, so the reasons behind the frame line's count of
/// views left null were not known).
fn refused_view(unordered: bool, format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) {
    use std::sync::atomic::{AtomicU64, Ordering};
    const REASONS: [&str; 10] = [
        "raw, not R32 typeless",
        "raw with a stride",
        "raw, not on 16 bytes",
        "structured with a format",
        "a stride the host has no structured buffers for",
        "typed without a usable format (typeless, unknown or block)",
        "no elements",
        "past the end of its buffer's copy",
        "accepted",
        "",
    ];
    static SEEN: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];
    let why = refusal(format, first, count, stride, raw, width);
    let at = REASONS.iter().position(|r| *r == why).unwrap_or(REASONS.len() - 1);
    let n = SEEN[at].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 4 || n.is_power_of_two() {
        tracing::warn!(
            unordered,
            why,
            times = n,
            format = format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, offset) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        let raw = b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        match own_bytes(offset, address, b.FirstElement, b.NumElements, element_size(&desc), b.StructureByteStride, unknown, raw)
            .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, 0)
            }
            None => (shadow, offset),
        }
    };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        refused_view(false, desc.Format, first, count, stride, raw, width);
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, own) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        let raw = b.StructureByteStride == 0 && b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        match own_bytes(shadow.offset(address), address, b.FirstElement, b.NumElements, element, b.StructureByteStride, unknown, raw)
            .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, true)
            }
            None => (shadow, false),
        }
    };
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = if own { 0 } else { shadow.offset(address) };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(offset, element, "unordered buffer view not on an element of its host copy");
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        refused_view(true, desc.Format, first, count, stride, raw, width);
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// A shader view whose format has the same bytes as its texture's but
/// the channels in other places: in which format of the texture's family
/// the host makes it, and which channel of that view each channel of the
/// title's comes from (0 to 3: red to alpha; 4: zero; 5: one). The console
/// reads the bytes as the view's format says; the host allows views only
/// in the texture's own family. So an alpha-only (A8) view of an R8
/// texture reads (0, 0, 0, r), an R8 view of an A8 texture (a, 0, 0, 1),
/// a B8G8R8X8 view of a B8G8R8A8 texture (r, g, b, 1), and B8G8R8A8 views
/// of R8G8B8A8 textures (and the other way round) have red and blue
/// swapped. Only UNORM (and sRGB) views of UNORM, sRGB or typeless
/// textures: the others would read the bytes as another kind of number.
/// `None`: not such a pair (the view is made, or refused, as it names).
/// Unordered access views have no component mapping: such a view of
/// another family is refused (`fits`).
pub(crate) fn other_channels(view: u32, own: u32) -> Option<(u32, [u32; 4])> {
    const R8_TYPELESS: u32 = 60;
    const R8_UNORM: u32 = 61;
    const A8_UNORM: u32 = 65;
    // Typeless, UNORM and sRGB.
    const RGBA: [u32; 3] = [27, 28, 29];
    const BGRA: [u32; 3] = [90, 87, 91];
    const SRGB: [u32; 3] = [29, 91, 93];
    // In the UNORM or sRGB format of the texture's family, as the title's
    // view is one or the other (`fits` keeps it on a typed texture where
    // the host casts, else takes the texture's).
    let host = |family: [u32; 3]| if SRGB.contains(&view) { family[2] } else { family[1] };
    match view {
        A8_UNORM if matches!(own, R8_TYPELESS | R8_UNORM) => Some((R8_UNORM, [4, 4, 4, 0])),
        R8_UNORM if own == A8_UNORM => Some((A8_UNORM, [3, 4, 4, 5])),
        // B8G8R8A8 and B8G8R8X8 views of R8G8B8A8 textures.
        87 | 91 if RGBA.contains(&own) => Some((host(RGBA), [2, 1, 0, 3])),
        88 | 93 if RGBA.contains(&own) => Some((host(RGBA), [2, 1, 0, 5])),
        // B8G8R8X8 and R8G8B8A8 views of B8G8R8A8 textures.
        88 | 93 if BGRA.contains(&own) => Some((host(BGRA), [0, 1, 2, 5])),
        28 | 29 if BGRA.contains(&own) => Some((host(BGRA), [2, 1, 0, 3])),
        _ => None,
    }
}

/// The title's component mapping `title` for a view whose channels come
/// from the host view's as `from` says ([`other_channels`]): each channel
/// the title's mapping takes from the view takes what the view has there.
pub(crate) fn mapped_through(title: u32, from: [u32; 4]) -> u32 {
    // Set in every mapping the host takes.
    const ALWAYS: u32 = 1 << 12;
    let mut mapping = title & !0xfff | ALWAYS;
    for channel in 0..4 {
        let source = title >> (3 * channel) & 7;
        let source = from.get(source as usize).copied().unwrap_or(source);
        mapping |= source << (3 * channel);
    }
    mapping
}

/// Shader views read through other channels ([`other_channels`]), since
/// the start.
pub(crate) static OTHER_CHANNELS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The title's shader view `desc` of `texture` as the host makes it when
/// its format has the texture's bytes in other channels
/// ([`other_channels`]). Each distinct case is logged once (at most 32),
/// and so is each distinct alpha-only view of an alpha-only texture, which
/// needs no change: they say how the title's A8 tables are read.
/// `title_dimension` is the title's view dimension and the dimension the
/// title made the texture in (a 1D view and texture are made 2D:
/// `resource::host_dimension`).
fn read_through(texture: &ID3D12Resource, desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, title_dimension: (u32, u32)) {
    const A8_UNORM: u32 = 65;
    // SAFETY: a live resource.
    let own = unsafe { texture.GetDesc() }.Format.0 as u32;
    let view = desc.Format.0 as u32;
    let other = other_channels(view, own);
    if let Some((format, from)) = other {
        OTHER_CHANNELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        desc.Format = DXGI_FORMAT(format as i32);
        desc.Shader4ComponentMapping = mapped_through(desc.Shader4ComponentMapping, from);
    }
    if other.is_some() || view == A8_UNORM || own == A8_UNORM {
        static SEEN: Mutex<Vec<(u32, u32, i32, u32)>> = Mutex::new(Vec::new());
        let case = (view, own, (title_dimension.0 | title_dimension.1 << 8) as i32, desc.Shader4ComponentMapping);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.len() < 32 && !seen.contains(&case) {
            seen.push(case);
            tracing::info!(
                view_format = view,
                texture_format = own,
                host_format = desc.Format.0,
                dimension = title_dimension.0,
                texture_dimension = title_dimension.1,
                host_dimension = desc.ViewDimension.0,
                mapping = desc.Shader4ComponentMapping,
                through_other_channels = other.is_some(),
                "a shader view of an alpha-only format, or read through other channels"
            );
        }
    }
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view and reads the bytes as the view's format says; the host refuses
/// (and removes the device for) views without the resource's permission,
/// of typed depth, and in another family than the resource's. A view of a
/// typed texture in another format of its family keeps that format where
/// the host casts between the two ([`casts`]: an sRGB view of a UNORM
/// texture converts, a UINT view of a UNORM texture reads integers), else
/// takes the texture's format. A shader view in a format with the
/// texture's bytes in other channels is one of the texture's family by
/// now ([`read_through`]).
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !(casts(texture, own, view, unordered))
    {
        *format = d.Format;
    }
    true
}

/// Whether the host makes a view in format `view` of a typed texture in
/// format `own` of the same family (D3D12's relaxed format casting, where
/// the host has it: `CastingFullyTypedFormatSupported`). Never between a
/// float and another kind, nor between SNORM and UNORM; not for depth
/// formats; no unordered access in B8G8R8A8 or B8G8R8X8 (optional on
/// the host). (fxceless PR #644 kept sRGB and linear RGBA8 views so; the
/// casting rules allow every such pair.)
fn casts(texture: &ID3D12Resource, own: u32, view: u32, unordered: bool) -> bool {
    const FLOAT: [u32; 9] = [2, 6, 10, 16, 34, 41, 54, 95, 96];
    const SNORM: [u32; 8] = [13, 31, 37, 51, 58, 63, 81, 84];
    const UNORM: [u32; 22] = [11, 24, 28, 29, 35, 49, 56, 61, 71, 72, 74, 75, 77, 78, 80, 83, 87, 88, 91, 93, 98, 99];
    const DEPTH: [u32; 10] = [19, 20, 21, 22, 40, 44, 45, 46, 47, 55];
    if own == view {
        return true;
    }
    let float = |f: u32| FLOAT.contains(&f);
    let normalized = |a: u32, b: u32| SNORM.contains(&a) && UNORM.contains(&b);
    let pair = float(own) == float(view)
        && !normalized(own, view)
        && !normalized(view, own)
        && !DEPTH.contains(&own)
        && !DEPTH.contains(&view)
        && !(unordered && (87..=93).contains(&view));
    pair && typed_casting(texture)
}

/// Whether the host has relaxed format casting (asked of the device of
/// the first texture viewed; logged once).
fn typed_casting(texture: &ID3D12Resource) -> bool {
    static CASTING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CASTING.get_or_init(|| {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: a live resource's device; the query writes `options` of
        // the size passed.
        let asked = unsafe {
            texture.GetDevice(&mut device).is_ok()
                && device.is_some_and(|d| {
                    d.CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                })
        };
        let supported = asked && options.CastingFullyTypedFormatSupported.as_bool();
        tracing::info!(supported, "views of typed textures in other formats of their family (relaxed format casting)");
        supported
    })
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// The bytes of a host constant buffer view of a title's of `size`: whole
/// 256s (the host's step), and at most the 64 KiB a shader reads (a larger
/// view is an invalid call, for which NVIDIA removes the device). `None`
/// for a view of no bytes, which reads zeros: a null view, not one of 256
/// bytes of whatever lies at its address.
fn constant_view_bytes(size: u32) -> Option<u32> {
    (size != 0).then(|| size.min(crate::descriptor::CONSTANT_VIEW_MOST).next_multiple_of(256))
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let bytes = constant_view_bytes(size)?;
            if address == 0 {
                return None;
            }
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: bytes,
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // The title's views of textures with a stencil plane, as it
                // describes them (format, dimension, component mapping, a
                // pad, then first level, levels, plane): each distinct one
                // once. They say how its shaders read the stencil plane
                // (the lighting takes each pixel's material class and its
                // "inside a room" bit from it).
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let title_dimension =
                    desc.map(|d| (d[1], crate::resource::title_dimension(resource).unwrap_or(0)));
                let mut desc = desc.as_ref().map(srv_desc);
                if let (Some(desc), Some(dimension)) = (desc.as_mut(), title_dimension) {
                    read_through(texture, desc, dimension);
                }
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // Levels on demand (`crate::tiles`): a texture made level by
                // level is shown from the first level that is the title's
                // and has memory, if the title's view starts before it (a
                // level without memory reads zeros: a black texture). The
                // title's own view mostly starts there already. One made
                // without its most detailed levels keeps its first level and
                // is clamped above them (`crate::tiles::shape_view`).
                if let Some(levels) = crate::resource::view_levels(resource)
                    && levels.first_tiled > 0
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    // The texture's own view (the title gave none): all of it.
                    let whole = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: own.Format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: 5768,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: u32::MAX,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    };
                    crate::tiles::shape_view(desc.get_or_insert(whole), levels, u32::from(own.MipLevels));
                } else if let Some(first) = crate::texture::levels_shown_from(resource) {
                    // Uploaded whole while it streams in: shown from its first
                    // level whose pages are there, its larger ones zeros
                    // until then (`crate::texture` `note_levels_there`). A
                    // view the title gave is clamped (2D and 2D arrays; a
                    // clamp of its own is kept if higher); the default view
                    // of a one-layer 2D texture too. Cubes, volumes and the
                    // default view of an array are not: they show the zeros.
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    if desc.is_none() && own.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && own.DepthOrArraySize == 1
                    {
                        desc = Some(D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: own.Format,
                            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                            Shader4ComponentMapping: 5768,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Texture2D: D3D12_TEX2D_SRV {
                                    MostDetailedMip: 0,
                                    MipLevels: u32::MAX,
                                    PlaneSlice: 0,
                                    ResourceMinLODClamp: 0.0,
                                },
                            },
                        });
                    }
                    if let Some(desc) = desc.as_mut() {
                        crate::tiles::clamp_from(desc, first, u32::from(own.MipLevels));
                    }
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            // The first such views of depth textures are logged as the
            // title describes them (format, dimension, then mip, slices and
            // plane): they say which planes shaders write, and as what.
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            // An empty title buffer: nothing to view, and no host copy to
            // look for (#70: such views were in flight at device removals).
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether both are shader views of one texture the title placed in
/// memory: the same address and the same resource description, whatever
/// part of it each shows.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            resource[0] != 1 && resource == other && desc[10..12] == other_desc[10..12]
        }
        _ => false,
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if host().is_none() {
        return;
    }
    make_null_kind(null_kind(view), at);
}

/// The kinds of null view [`null_for`] makes. Each is the same descriptor
/// wherever it is made, so one made once can be copied to any slot
/// ([`NullTemplates`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NullKind {
    /// A null constant buffer view.
    Constant,
    /// [`null_buffer_view`].
    BufferSrv,
    /// [`null_view`].
    Plain,
    /// A null unordered access view of a buffer.
    BufferUav,
    /// [`null_uav_desc`] of this dimension (one it tells apart: 2D arrays,
    /// 3D; 2D for every other).
    TextureUav(u32),
    /// [`null_like`] of a description of this dimension (4 to 10).
    Like(u32),
}

/// The dimensions [`NullKind::TextureUav`] holds, in its template order.
/// The title's 1D and 1D-array views are the 2D and 2D-array ones, as the
/// translation declares them (`resource::host_dimension`).
const UAV_DIMENSIONS: [D3D12_UAV_DIMENSION; 3] = [
    D3D12_UAV_DIMENSION_TEXTURE2D,
    D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
    D3D12_UAV_DIMENSION_TEXTURE3D,
];

/// The dimensions [`NullKind::Like`] holds: 2D on, a 1D (array) taking
/// the 2D (array) one.
const LIKE_DIMENSIONS: std::ops::RangeInclusive<u32> = 4..=10;

/// How many kinds of [`NullKind`] there are.
const NULL_KINDS: usize = 4 + UAV_DIMENSIONS.len() + 7;

impl NullKind {
    /// A null texture UAV of the title's view dimension, as
    /// [`null_uav_desc`] tells them apart.
    fn texture_uav(dimension: Option<u32>) -> NullKind {
        let d = dimension.map(|d| match D3D12_UAV_DIMENSION(d as i32) {
            D3D12_UAV_DIMENSION_TEXTURE1D => D3D12_UAV_DIMENSION_TEXTURE2D,
            D3D12_UAV_DIMENSION_TEXTURE1DARRAY => D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            d => d,
        });
        let told = d.filter(|d| UAV_DIMENSIONS.contains(d)).unwrap_or(D3D12_UAV_DIMENSION_TEXTURE2D);
        NullKind::TextureUav(told.0 as u32)
    }

    /// A null view like the description of dimension `dimension`
    /// ([`null_like`] makes [`null_view`] for the others).
    fn like(dimension: u32) -> NullKind {
        const TEXTURE1D: u32 = 2;
        const TEXTURE1DARRAY: u32 = 3;
        let dimension = match dimension {
            TEXTURE1D => D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32,
            TEXTURE1DARRAY => D3D12_SRV_DIMENSION_TEXTURE2DARRAY.0 as u32,
            d => d,
        };
        if LIKE_DIMENSIONS.contains(&dimension) {
            NullKind::Like(dimension)
        } else {
            NullKind::Plain
        }
    }

    /// Its place among the kinds (below [`NULL_KINDS`]).
    pub(crate) fn slot(self) -> usize {
        match self {
            NullKind::Constant => 0,
            NullKind::BufferSrv => 1,
            NullKind::Plain => 2,
            NullKind::BufferUav => 3,
            NullKind::TextureUav(d) => {
                4 + UAV_DIMENSIONS.iter().position(|u| u.0 as u32 == d).unwrap_or(0)
            }
            NullKind::Like(d) => 4 + UAV_DIMENSIONS.len() + (d.clamp(4, 10) - 4) as usize,
        }
    }

    /// Every kind, in the order of [`NullKind::slot`].
    fn all() -> Vec<NullKind> {
        let mut all = vec![NullKind::Constant, NullKind::BufferSrv, NullKind::Plain, NullKind::BufferUav];
        all.extend(UAV_DIMENSIONS.iter().map(|d| NullKind::TextureUav(d.0 as u32)));
        all.extend(LIKE_DIMENSIONS.map(NullKind::Like));
        all
    }
}

/// The kind of null view [`null_for`] makes for `view`.
fn null_kind(view: &View) -> NullKind {
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let uav = |buffer: bool, dimension: Option<u32>| {
        if buffer { NullKind::BufferUav } else { NullKind::texture_uav(dimension) }
    };
    match view {
        View::Unordered(r, _, desc) => uav(is_buffer(r) || desc.is_some_and(|d| d[1] == 1), desc.map(|d| d[1])),
        View::Placed(true, resource, desc) => uav(resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => NullKind::BufferSrv,
        View::Placed(false, resource, _) if resource[0] == 1 => NullKind::BufferSrv,
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => NullKind::like(desc[1]),
        View::Placed(false, _, desc) => NullKind::like(desc[1]),
        View::Constant(..) => NullKind::Constant,
        _ => NullKind::Plain,
    }
}

/// Makes the null view of kind `kind` at `at`.
fn make_null_kind(kind: NullKind, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let uav = |desc: D3D12_UNORDERED_ACCESS_VIEW_DESC| {
        // SAFETY: a slot of a heap of ours; a null resource is allowed with
        // a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                at,
            )
        };
    };
    match kind {
        NullKind::Constant => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of a heap of ours; a zero location makes a null view.
            unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        NullKind::BufferSrv => null_buffer_view(at),
        NullKind::Plain => null_view(at),
        NullKind::BufferUav => uav(D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }),
        NullKind::TextureUav(d) => uav(null_uav_desc(Some(d))),
        // (`null_like` reads the dimension word alone.)
        NullKind::Like(d) => null_like(&[0, d, 0, 0, 0, 0, 0, 0, 0, 0], at),
    }
}

/// Copies of each null view ([`NullKind`]) in a heap of the CPU's, from
/// which the pass over records copied in fills a run of slots with one
/// `CopyDescriptors` instead of making a view in each: kind `k` at
/// descriptors `k * NULL_RUN` to `(k + 1) * NULL_RUN`.
pub(crate) struct NullTemplates {
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
}

// SAFETY: written once when made, only copied from after (descriptor
// copies are free-threaded).
unsafe impl Send for NullTemplates {}
// SAFETY: as above.
unsafe impl Sync for NullTemplates {}

impl NullTemplates {
    fn at(&self, i: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.cpu + i as usize * self.step as usize }
    }
}

/// Slots a run of the pass over records copied in holds at once, and
/// copies of each null view in [`NullTemplates`].
pub(crate) const NULL_RUN: u32 = 64;

/// The null views to copy from, made on first use; `None` if the host
/// has no heap for them (null views are then made one by one).
fn null_templates() -> Option<&'static NullTemplates> {
    static TEMPLATES: std::sync::OnceLock<Option<NullTemplates>> = std::sync::OnceLock::new();
    TEMPLATES
        .get_or_init(|| {
            let host = host()?;
            let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            // SAFETY: plain heap creation and copies inside it.
            unsafe {
                let heap: ID3D12DescriptorHeap = host
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: NULL_KINDS as u32 * NULL_RUN,
                        ..Default::default()
                    })
                    .ok()?;
                let t = NullTemplates {
                    cpu: heap.GetCPUDescriptorHandleForHeapStart().ptr,
                    step: host.device.GetDescriptorHandleIncrementSize(kind),
                    _heap: heap,
                };
                for null in NullKind::all() {
                    let first = null.slot() as u32 * NULL_RUN;
                    make_null_kind(null, t.at(first));
                    let mut filled = 1;
                    while filled < NULL_RUN {
                        let n = filled.min(NULL_RUN - filled);
                        host.device.CopyDescriptorsSimple(n, t.at(first + filled), t.at(first), kind);
                        filled += n;
                    }
                }
                Some(t)
            }
        })
        .as_ref()
}

/// A write of a title slot with a buffer view, by the pass over records
/// copied in ([`HostHeap::quick`]): what is left to do for it once the slot
/// is noted unmade.
pub(crate) struct Quick {
    index: u32,
    /// Its element stride ([`view_stride`]); `None` in sampler heaps.
    stride: Option<u32>,
    /// Its entry in the table of strides once its slot is written
    /// ([`stride_entry`]).
    entry: u32,
    /// The title address it starts at (`crate::draw::view_address`).
    address: u64,
    /// The null view its host slot gets; `None` in sampler heaps.
    null: Option<NullKind>,
    /// Whether the record had the tag of a view (for the look again under
    /// the lock: a record that names nothing may yet have it).
    viewed: bool,
}

/// [`HostHeap::mirror_picked`]'s buffers, kept from call to call.
#[derive(Default)]
pub(crate) struct PickScratch {
    run: Vec<Quick>,
    seen: Vec<(u64, u64)>,
    again: Vec<u32>,
    indices: Vec<u32>,
    present: Vec<u32>,
    nulls: Vec<(u32, usize)>,
    dst: Vec<(u32, u32)>,
    src: Vec<(u32, u32)>,
    dst_starts: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    dst_sizes: Vec<u32>,
    src_starts: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    src_sizes: Vec<u32>,
}

// SAFETY: plain numbers and descriptor handles (addresses), no references.
unsafe impl Send for PickScratch {}

/// A write that makes other slots' views stale, attempted under its own
/// slot's lock alone: to be done again under every slot's.
pub(crate) struct NeedsAll;

/// What the records [`HostHeap::mirror_picked`] was handed were, since the
/// last frame line (indices into [`PICKED`]).
const PICKED_SAME: usize = 0;
const PICKED_NOTHING: usize = 1;
const PICKED_BUFFER: usize = 2;
const PICKED_GONE: usize = 3;
const PICKED_PLACED: usize = 4;
const PICKED_OTHER: usize = 5;
const PICKED_KINDS: usize = 6;

/// Records the pass over records copied in found new by their bytes, by
/// what they were to the slot: the view the slot is made from already
/// (bytes past the record's words changed, or the same view written
/// again); nothing, in a slot that shows nothing; a buffer view (done in
/// a run); nothing where the slot showed something (in a run); a placed
/// texture's view; any other view (both by `mirror` alone).
pub(crate) static PICKED: [std::sync::atomic::AtomicU64; PICKED_KINDS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; PICKED_KINDS];

/// Whether slots have locks of their own ([`SlotLocks`]);
/// `VICTORY_DESCRIPTOR_STRIPES=0`: one lock for all slots of a heap, as
/// before.
fn striped() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DESCRIPTOR_STRIPES").is_none_or(|v| v != "0"))
}

/// A null texture UAV of the title's view dimension (2D arrays, 3D), 2D if
/// it names none: a 2D null view where the shader writes a 3D texture is a
/// mismatch GPU-based validation reports (the world's wind field,
/// CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24). 1D and 1D arrays
/// are 2D and 2D arrays, as the translation declares them
/// (`resource::host_dimension`).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY | D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A raw buffer UAV that drops writes and reads zeros.
fn null_buffer_uav(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A constant buffer view that reads zeros.
fn null_constant_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
    // SAFETY: a slot of our heap; a zero location makes a null view.
    unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    let desc = null_like_desc(words);
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// [`null_like`]'s description: one mip, one slice or cube, plane 0
/// (fields a null view still checks); a 1D (array) as the 2D (array) view
/// [`srv_desc`] makes of it.
fn null_like_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// Views read through other channels, made as [`make`] makes them and
/// read by a shader on the software adapter.
#[cfg(test)]
mod channel_tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::core::Interface;

    const DEFAULT: u32 = 5768;
    const A8_UNORM: u32 = 65;
    const R8_TYPELESS: u32 = 60;
    const R8_UNORM: u32 = 61;
    const TEXTURE_2D: u32 = 4;

    /// Each texel of a 4x1 texture, as a float4.
    const LOAD: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
Texture2D<float4> t : register(t0);
RWStructuredBuffer<float4> o : register(u0);
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { o[id.x] = t.Load(int3(id.x, 0, 0)); }
"#;

    fn warp() -> Option<ID3D12Device> {
        // SAFETY: plain DXGI queries and device creation on the software
        // adapter; a card is never opened.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                return None;
            }
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
            device
        }
    }

    /// The title's 2D view of mip 0 in `format` with `mapping`.
    fn view(format: u32, mapping: u32) -> [u32; 10] {
        [format, TEXTURE_2D, mapping, 0, 0, 1, 0, 0, 0, 0]
    }

    /// What a shader reads from a 4x1 texture of `format` holding `bytes`
    /// through the title's view `words`, made as [`make`] makes it.
    fn read(device: &ID3D12Device, format: u32, bytes: &[u8], words: [u32; 10]) -> [[f32; 4]; 4] {
        let cpu_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let one = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
        let code = crate::host::compile(LOAD, windows::core::s!("main"), windows::core::s!("cs_5_1"))
            .expect("the test's shader");
        // SAFETY: plain object creation on the software adapter; the
        // mappings hold what is written and read; the list is waited for.
        unsafe {
            let mut texture: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &cpu_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 4,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT(format as i32),
                        SampleDesc: one,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut texture,
                )
                .expect("a texture");
            let texture = texture.unwrap();
            texture.Map(0, None, None).expect("a mapped texture");
            let len = bytes.len() as u32;
            texture
                .WriteToSubresource(0, None, bytes.as_ptr().cast(), len, len)
                .expect("the texels");
            texture.Unmap(0, None);
            let mut desc = srv_desc(&words);
            read_through(&texture, &mut desc, (words[1], 3));
            assert!(fits(&texture, Some(&mut desc.Format), false), "a view the host makes");
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("a heap");
            device.CreateShaderResourceView(&texture, Some(&desc), heap.GetCPUDescriptorHandleForHeapStart());
            let mut out: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &cpu_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 64,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: one,
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut out,
                )
                .expect("an output buffer");
            let out = out.unwrap();
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).expect("a root signature");
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("a queue");
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("an allocator");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso)
                .expect("a list");
            list.SetComputeRootSignature(&root);
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.Close().expect("a closed list");
            queue.ExecuteCommandLists(&[Some(list.cast().expect("a command list"))]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("a fence");
            queue.Signal(&fence, 1).expect("a signal");
            // No event: the call returns when the fence has the value.
            fence
                .SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default())
                .expect("the list ran");
            let mut data = std::ptr::null_mut();
            out.Map(0, None, Some(&mut data)).expect("a mapped output");
            let texels = data.cast::<[[f32; 4]; 4]>().read();            out.Unmap(0, None);
            texels
        }
    }

    fn near(got: [[f32; 4]; 4], want: [[f32; 4]; 4]) -> bool {
        got.iter().flatten().zip(want.iter().flatten()).all(|(g, w)| (g - w).abs() < 1e-6)
    }

    /// The bytes of each texel over 255.
    fn unorm(bytes: [u8; 4]) -> [f32; 4] {
        bytes.map(|b| f32::from(b) / 255.0)
    }

    #[test]
    fn mappings_take_each_channel_from_where_the_view_has_it() {
        // (0, 0, 0, r): the A8 view of an R8 texture.
        assert_eq!(mapped_through(DEFAULT, [4, 4, 4, 0]), 4388);
        // The title's own (a, a, a, a) of an A8 view: (r, r, r, r).
        let alpha = 3 | 3 << 3 | 3 << 6 | 3 << 9 | 1 << 12;
        assert_eq!(mapped_through(alpha, [4, 4, 4, 0]), 1 << 12);
        // The title's constants stay.
        let constants = 4 | 5 << 3 | 4 << 6 | 5 << 9 | 1 << 12;
        assert_eq!(mapped_through(constants, [3, 4, 4, 5]), constants);
        assert_eq!(other_channels(A8_UNORM, R8_UNORM), Some((R8_UNORM, [4, 4, 4, 0])));
        assert_eq!(other_channels(A8_UNORM, R8_TYPELESS), Some((R8_UNORM, [4, 4, 4, 0])));
        assert_eq!(other_channels(R8_UNORM, A8_UNORM), Some((A8_UNORM, [3, 4, 4, 5])));
        // Integer and signed views read other numbers: not mapped.
        assert_eq!(other_channels(62, A8_UNORM), None);
        assert_eq!(other_channels(A8_UNORM, 62), None);
        // Typeless textures: in the UNORM or sRGB format the view names.
        assert_eq!(other_channels(91, 27), Some((29, [2, 1, 0, 3])));
        assert_eq!(other_channels(88, 90), Some((87, [0, 1, 2, 5])));
        assert_eq!(other_channels(28, 91), Some((87, [2, 1, 0, 3])));
        // Views of their own family are left to `fits`.
        assert_eq!(other_channels(A8_UNORM, A8_UNORM), None);
        assert_eq!(other_channels(R8_UNORM, R8_TYPELESS), None);
        assert_eq!(other_channels(28, 27), None);
    }

    /// The case of the title's blur strength table: its alpha is read
    /// (it read 1 everywhere when the A8 view became an R8 view).
    #[test]
    fn warp_an_alpha_only_view_of_an_r8_texture_reads_alpha() {
        let Some(device) = warp() else {
            return;
        };
        let bytes = [0x00, 0x40, 0x80, 0xff];
        let alpha = bytes.map(|b| [0.0, 0.0, 0.0, f32::from(b) / 255.0]);
        for texture in [R8_UNORM, R8_TYPELESS, A8_UNORM] {
            let got = read(&device, texture, &bytes, view(A8_UNORM, DEFAULT));
            assert!(near(got, alpha), "A8 view of a texture of format {texture}: {got:?}");
        }
        // The other way round: an R8 view of an A8 texture.
        let red = bytes.map(|b| [f32::from(b) / 255.0, 0.0, 0.0, 1.0]);
        let got = read(&device, A8_UNORM, &bytes, view(R8_UNORM, DEFAULT));
        assert!(near(got, red), "R8 view of an A8 texture: {got:?}");
        // The title's mapping (a, a, a, a) of the A8 view.
        let all = 3 | 3 << 3 | 3 << 6 | 3 << 9 | 1 << 12;
        let wide = bytes.map(|b| [f32::from(b) / 255.0; 4]);
        let got = read(&device, R8_UNORM, &bytes, view(A8_UNORM, all));
        assert!(near(got, wide), "A8 view with (a, a, a, a): {got:?}");
    }

    /// An sRGB view of a UNORM texture converts, a UNORM view of an sRGB
    /// texture does not (where the host casts); an SNORM view of a UNORM
    /// texture, which no host casts to, reads as the texture.
    #[test]
    fn warp_views_in_another_format_of_the_family_read_as_that_format() {
        let Some(device) = warp() else {
            return;
        };
        let bytes = [0x80; 16];
        let linear = [[128.0 / 255.0; 4]; 4];
        let mut srgb = [[0.215_860_5; 4]; 4];
        for texel in &mut srgb {
            texel[3] = 128.0 / 255.0;
        }
        let close = |got: [[f32; 4]; 4], want: [[f32; 4]; 4]| {
            got.iter().flatten().zip(want.iter().flatten()).all(|(g, w)| (g - w).abs() < 0.002)
        };
        // SAFETY: a plain query into our own structure.
        let casting = unsafe {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            device
                .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, (&raw mut options).cast(), size_of_val(&options) as u32)
                .is_ok()
                && options.CastingFullyTypedFormatSupported.as_bool()
        };
        let got = read(&device, 28, &bytes, view(29, DEFAULT));        assert!(close(got, if casting { srgb } else { linear }), "sRGB of UNORM: {got:?}");
        let got = read(&device, 29, &bytes, view(28, DEFAULT));
        assert!(close(got, if casting { linear } else { srgb }), "UNORM of sRGB: {got:?}");
        let got = read(&device, 28, &bytes, view(31, DEFAULT));
        assert!(close(got, linear), "SNORM of UNORM: {got:?}");
    }

    #[test]
    fn warp_views_of_four_bytes_in_another_order_swap_and_fill_channels() {
        let Some(device) = warp() else {
            return;
        };
        let texel = [10, 20, 30, 40];
        let bytes: Vec<u8> = texel.repeat(4);
        // B8G8R8A8 view of an R8G8B8A8 texture: byte 0 is blue.
        let got = read(&device, 28, &bytes, view(87, DEFAULT));
        assert!(near(got, [unorm([30, 20, 10, 40]); 4]), "BGRA of RGBA: {got:?}");
        // B8G8R8X8 view of a B8G8R8A8 texture: alpha 1.
        let mut opaque = unorm([30, 20, 10, 40]);
        opaque[3] = 1.0;
        let got = read(&device, 87, &bytes, view(88, DEFAULT));
        assert!(near(got, [opaque; 4]), "BGRX of BGRA: {got:?}");
        // R8G8B8A8 view of a typeless B8G8R8A8 texture: byte 0 is red.
        let got = read(&device, 90, &bytes, view(28, DEFAULT));
        assert!(near(got, [unorm(texel); 4]), "RGBA of BGRA: {got:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_skip_window_is_two_frames_unless_set() {
        assert_eq!(tabled_skip(None), Some(2));
        assert_eq!(tabled_skip(Some("5")), Some(5));
        assert_eq!(tabled_skip(Some(" 0 ")), Some(0));
        assert_eq!(tabled_skip(Some("off")), None);
        assert_eq!(tabled_skip(Some("many")), Some(2));
    }

    /// The unsigned twin of a 16-bit float or signed format has its layout,
    /// and only buffer views in such a format are shown as another view.
    #[test]
    fn only_16_bit_float_and_signed_buffer_views_get_an_unsigned_twin() {
        for (format, unsigned) in [(10, 12), (14, 12), (34, 36), (38, 36), (54, 57), (59, 57)] {
            assert_eq!(half_format(format), Some(unsigned));
            assert_eq!(
                crate::resource::element(format),
                crate::resource::element(unsigned),
                "format {format}"
            );
        }
        // The unsigned formats themselves, normalised 16-bit formats, and
        // wider and narrower ones.
        for other in [0, 12, 36, 57, 11, 13, 35, 37, 56, 58, 42, 17, 30, 61] {
            assert_eq!(half_format(other), None, "format {other}");
        }
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        // R16G16B16A16_FLOAT: a buffer of halves a program moves as they are.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(10, BUFFER)))),
            Some(View::Unordered(7, 0, Some(words(12, BUFFER))))
        );
        assert_eq!(
            half_view(&View::Shader(7, Some(words(54, BUFFER)))),
            Some(View::Shader(7, Some(words(57, BUFFER))))
        );
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(38, BUFFER));
        placed[10] = 0x1234_0000;
        let mut twin = placed;
        twin[0] = 36;
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(
            half_view(&View::Placed(true, memory, placed)),
            Some(View::Placed(true, memory, twin))
        );
        // The title's kind (R16G16B16A16_UINT and R16_UINT buffers), a
        // texture view (dimension 4: 2D), a view without a description, a
        // constant buffer: the title's own.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(12, BUFFER)))),
            None
        );
        assert_eq!(half_view(&View::Shader(7, Some(words(57, BUFFER)))), None);
        assert_eq!(half_view(&View::Unordered(7, 0, Some(words(10, 4)))), None);
        assert_eq!(half_view(&View::Shader(7, None)), None);
        assert_eq!(half_view(&View::Constant(0x1000, 256)), None);
    }

    /// The buffer views through which the integers a shader moves halves
    /// as are not what the console gives are named; the title's are not.
    #[test]
    fn views_whose_values_the_console_would_convert_are_named() {
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        for exact in [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62] {
            let view = View::Unordered(7, 0, Some(words(exact, BUFFER)));
            assert_eq!(half_inexact(&view), None, "format {exact}");
        }
        // R32_FLOAT, R32_UINT, R16G16B16A16_UNORM, R8G8B8A8_UNORM, R8_SINT,
        // and a structured or raw view (no format).
        for inexact in [41, 42, 11, 28, 64, 0] {
            let view = View::Shader(7, Some(words(inexact, BUFFER)));
            assert_eq!(half_inexact(&view), Some(inexact));
        }
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(41, BUFFER));
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(half_inexact(&View::Placed(true, memory, placed)), Some(41));
        // Not a buffer view.
        assert_eq!(half_inexact(&View::Shader(7, Some(words(41, 4)))), None);
        assert_eq!(half_inexact(&View::Constant(0x1000, 256)), None);
    }

    /// The two ways a translated `buffer_load_format_d16_xyzw` and
    /// `buffer_store_format_d16_xyzw` have moved a typed buffer's 16-bit
    /// channels, on the host device, over a buffer that holds every 16-bit
    /// pattern. As unsigned integers through an unsigned view (the title's
    /// own R16G16B16A16_UINT, or the twin `half_view` gives a float one)
    /// every pattern comes back as it was. By way of floats, as before
    /// (load as floats, each to its half's bits; each half widened and
    /// stored as a float): through the title's integer view that is
    /// outside what D3D12 defines, and what the device gives is printed,
    /// not checked (1 pattern of 65,536 on the cards tried: the network
    /// that moves the title's faces computed zeros).
    #[test]
    fn halves_cross_an_unsigned_buffer_view_unchanged() {
        const MOVES: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
#ifdef BY_FLOATS
RWBuffer<float4> from : register(u0);
RWBuffer<float4> to : register(u1);
#else
RWBuffer<uint4> from : register(u0);
RWBuffer<uint4> to : register(u1);
#endif
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void BY_FLOATS_OR_NOT(uint3 id : SV_DispatchThreadID) {
#ifdef BY_FLOATS
    uint4 h = f32tof16(from[id.x]);
#else
    uint4 h = from[id.x] & 0xffff;
#endif
    uint low = h.x | h.y << 16;
    uint high = h.z | h.w << 16;
#ifdef BY_FLOATS
    to[id.x] = float4(f16tof32(low), f16tof32(low >> 16), f16tof32(high), f16tof32(high >> 16));
#else
    to[id.x] = uint4(low & 0xffff, low >> 16, high & 0xffff, high >> 16);
#endif
}
"#;
        const ELEMENTS: usize = 1 << 14;
        const BYTES: u64 = ELEMENTS as u64 * 8;
        const TITLES: u32 = 12; // R16G16B16A16_UINT
        let Some(host) = host() else {
            return; // no host GPU
        };
        let heap: ID3D12DescriptorHeap = {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain heap creation.
            unsafe { host.device.CreateDescriptorHeap(&desc) }.expect("a view heap")
        };
        // How many of the 65,536 patterns come back as they were, moved by
        // the program `by_floats` or not, through views in `format`.
        let moved = |by_floats: bool, format: u32| -> usize {
            let source = if by_floats {
                MOVES
                    .replace("BY_FLOATS_OR_NOT", "main")
                    .replace("#ifdef BY_FLOATS", "#if 1")
            } else {
                MOVES
                    .replace("BY_FLOATS_OR_NOT", "main")
                    .replace("#ifdef BY_FLOATS", "#if 0")
            };
            let code = crate::host::compile(
                &source,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
            )
            .expect("the test's shader");
            let (from, filled) = host.shared_buffer(BYTES).expect("a buffer");
            let (to, result) = host.shared_buffer(BYTES).expect("a buffer");
            // SAFETY: the shader's own root signature and plain creation
            // calls; both mappings hold `BYTES` bytes; the views are of
            // live buffers, in slots of the heap made above; an open list.
            unsafe {
                let root: ID3D12RootSignature = host
                    .device
                    .CreateRootSignature(0, &code)
                    .expect("a root signature");
                let pso: ID3D12PipelineState = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .expect("a pipeline");
                let filled = std::slice::from_raw_parts_mut(filled.cast::<u16>(), ELEMENTS * 4);
                for (pattern, half) in filled.iter_mut().enumerate() {
                    *half = pattern as u16;
                }
                std::ptr::write_bytes(result, 0xee, BYTES as usize);
                let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT(format as i32),
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            NumElements: ELEMENTS as u32,
                            ..Default::default()
                        },
                    },
                };
                let start = heap.GetCPUDescriptorHandleForHeapStart();
                let step = host
                    .device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize;
                host.device.CreateUnorderedAccessView(
                    &from,
                    None::<&ID3D12Resource>,
                    Some(&view),
                    start,
                );
                let second = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + step,
                };
                host.device.CreateUnorderedAccessView(
                    &to,
                    None::<&ID3D12Resource>,
                    Some(&view),
                    second,
                );
                let list = host.new_list().expect("a host list");
                list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list
                    .SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
                list.list.Dispatch(ELEMENTS as u32 / 64, 1, 1);
                list.close();
                host.execute(&[&list]);
                assert!(
                    host.device.GetDeviceRemovedReason().is_ok(),
                    "the device, after views in format {format}"
                );
                let result = std::slice::from_raw_parts(result.cast::<u16>(), ELEMENTS * 4);
                (0..=u16::MAX)
                    .filter(|&p| result[usize::from(p)] == p)
                    .count()
            }
        };
        // A float view of halves is shown as the title's kind of view.
        assert_eq!(half_format(10), Some(TITLES));
        let unsigned = moved(false, TITLES);
        let by_floats = moved(true, TITLES);
        eprintln!(
            "halves through an R16G16B16A16_UINT buffer view: {unsigned} of 65536 patterns came back unchanged moved as unsigned integers, {by_floats} moved by way of floats"
        );
        assert_eq!(
            unsigned, 65_536,
            "patterns unchanged, moved as unsigned integers"
        );
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        // 1D and 1D arrays as the host's 2D copies of them.
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(Some(3)), D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
        // The null templates fold 1D into 2D as well.
        assert_eq!(NullKind::texture_uav(Some(2)), NullKind::texture_uav(Some(4)));
        assert_eq!(NullKind::texture_uav(Some(3)), NullKind::texture_uav(Some(5)));
        assert_eq!(NullKind::like(2), NullKind::Like(4));
        assert_eq!(NullKind::like(3), NullKind::Like(5));
        assert_eq!(null_like_desc(&[0, 3, 0, 0, 0, 0, 0, 0, 0, 0]).ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
    }

    /// The title's 1D shader views become 2D views of the same mips and
    /// slices, the clamp moved past `PlaneSlice` (0); 2D and buffer views
    /// are left as they are (Marvisak/VIctory PR #1727).
    #[test]
    fn one_d_shader_views_become_2d_views() {
        let clamp = 0.5f32.to_bits();
        // TEXTURE1D: mip 1, 3 levels, clamp.
        let d = srv_desc(&[28, 2, 0x1688, 0, 1, 3, clamp, 0, 0, 0]);
        assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
        // SAFETY: the member the dimension names.
        let t = unsafe { d.Anonymous.Texture2D };
        assert_eq!((t.MostDetailedMip, t.MipLevels, t.PlaneSlice), (1, 3, 0));
        assert_eq!(t.ResourceMinLODClamp, 0.5);
        assert_eq!(d.Shader4ComponentMapping, 0x1688);
        // TEXTURE1DARRAY: mip 0, 1 level, slices 2..6, clamp.
        let d = srv_desc(&[28, 3, 0x1688, 0, 0, 1, 2, 4, clamp, 0]);
        assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        let t = unsafe { d.Anonymous.Texture2DArray };
        assert_eq!((t.MipLevels, t.FirstArraySlice, t.ArraySize, t.PlaneSlice), (1, 2, 4, 0));
        assert_eq!(t.ResourceMinLODClamp, 0.5);
        // A 2D view and a buffer view keep their words.
        let d = srv_desc(&[28, 4, 0x1688, 0, 1, 3, 0, clamp, 0, 0]);
        assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2D.ResourceMinLODClamp }, 0.5);
        let d = srv_desc(&[28, 1, 0x1688, 0, 7, 0, 9, 0, 0, 0]);
        assert_eq!(d.ViewDimension, D3D12_SRV_DIMENSION_BUFFER);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { (d.Anonymous.Buffer.FirstElement, d.Anonymous.Buffer.NumElements) }, (7, 9));
    }

    /// The title's 1D unordered access views become 2D ones, plane 0.
    #[test]
    fn one_d_unordered_views_become_2d_views() {
        let d = uav_desc(&[28, 2, 2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2D);
        // SAFETY: the member the dimension names.
        let t = unsafe { d.Anonymous.Texture2D };
        assert_eq!((t.MipSlice, t.PlaneSlice), (2, 0));
        let d = uav_desc(&[28, 3, 1, 3, 5, 0, 0, 0, 0, 0]);
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        let t = unsafe { d.Anonymous.Texture2DArray };
        assert_eq!((t.MipSlice, t.FirstArraySlice, t.ArraySize, t.PlaneSlice), (1, 3, 5, 0));
    }

    /// A 1D texture's host copy is a 2D texture one texel high, and the
    /// views the title makes of it as 1D (shader and unordered, plain and
    /// array) are accepted on it by the real device, as are the 1D null
    /// views.
    #[test]
    fn one_d_texture_is_a_2d_host_texture_its_1d_views_fit() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        assert_eq!(crate::resource::host_dimension(2), 3);
        assert_eq!(crate::resource::host_dimension(3), 3);
        assert_eq!(crate::resource::host_dimension(4), 4);
        let made = |slices: u16| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION(crate::resource::host_dimension(2) as i32),
                Width: 128,
                Height: 1,
                DepthOrArraySize: slices,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: plain object creation into our out parameter.
            unsafe {
                host.device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut texture,
                )
            }
            .expect("a 128x1 texture");
            texture.expect("a 128x1 texture")
        };
        let (one, array) = (made(1), made(4));
        // SAFETY: plain heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })
        }
        .expect("a view heap");
        // SAFETY: a live device.
        let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        // SAFETY: a live heap.
        let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let at = |k: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + k * step as usize };
        // The title's TEXTURE1D views (format 61, R8_UNORM, all levels),
        // its TEXTURE1DARRAY views (slices 1..3), and the 1D null views.
        let srv = srv_desc(&[61, 2, 0x1688, 0, 0, u32::MAX, 0, 0, 0, 0]);
        let srv_array = srv_desc(&[61, 3, 0x1688, 0, 0, u32::MAX, 1, 2, 0, 0]);
        let uav = uav_desc(&[61, 2, 0, 0, 0, 0, 0, 0, 0, 0]);
        let uav_array = uav_desc(&[61, 3, 0, 1, 2, 0, 0, 0, 0, 0]);
        // SAFETY: live textures and slots of our heap; null resources are
        // allowed with a description.
        unsafe {
            host.device.CreateShaderResourceView(&one, Some(&srv), at(0));
            host.device.CreateShaderResourceView(&array, Some(&srv_array), at(1));
            host.device.CreateUnorderedAccessView(&one, None::<&ID3D12Resource>, Some(&uav), at(2));
            host.device.CreateUnorderedAccessView(&array, None::<&ID3D12Resource>, Some(&uav_array), at(3));
        }
        make_null_kind(NullKind::texture_uav(Some(2)), at(4));
        make_null_kind(NullKind::texture_uav(Some(3)), at(5));
        make_null_kind(NullKind::like(2), at(6));
        make_null_kind(NullKind::like(3), at(7));
        // SAFETY: a live device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// What the texture code asks of the spans (levels on demand): where
    /// the next texture starts inside a span, the lowest level a view of a
    /// texture shows, and a texture's slots.
    #[test]
    fn spans_say_the_next_start_and_the_lowest_level_viewed() {
        let mut spans = Spans::default();
        // A 512x512 texture of 0x56000 bytes viewed from levels 3 and 1,
        // and the next one 0x6000 after it (run 188).
        spans.note(1, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(1, 3);
        spans.note(2, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(2, 1);
        spans.note(3, 0x1_6000, 0x1_6000 + 0x5_6000, true);
        assert_eq!(spans.next_start(0x1_0000, 0x1_0000 + 0x5_6000), Some(0x1_6000));
        assert_eq!(spans.next_start(0x1_6000, 0x1_6000 + 0x5_6000), None);
        assert_eq!(spans.first_viewed(0x1_0000), Some(1));
        assert_eq!(spans.first_viewed(0x1_6000), None);
        assert_eq!(spans.slots_at(0x1_0000), vec![1, 2]);
        // The view from level 1 goes: the texture is viewed from level 3.
        spans.forget(2);
        assert_eq!(spans.first_viewed(0x1_0000), Some(3));
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_templates_copy_what_null_for_makes() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let all = NullKind::all();
        assert_eq!(all.len(), NULL_KINDS);
        for (k, kind) in all.iter().enumerate() {
            assert_eq!(kind.slot(), k, "{kind:?}");
        }
        assert_eq!(NullKind::texture_uav(None), NullKind::texture_uav(Some(4)));
        assert_eq!(NullKind::texture_uav(Some(77)).slot(), NullKind::texture_uav(None).slot());
        assert_eq!(NullKind::like(1), NullKind::Plain);
        let Some(templates) = null_templates() else {
            return;
        };
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation; descriptors of a heap of the CPU's
        // are plain memory at their handles.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: 2,
                    ..Default::default()
                })
                .unwrap();
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            let step = templates.step as usize;
            let second = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + step };
            for null in all {
                make_null_kind(null, start);
                host.device.CopyDescriptorsSimple(1, second, templates.at(null.slot() as u32 * NULL_RUN + 5), kind);
                let a = std::slice::from_raw_parts(start.ptr as *const u8, step);
                let b = std::slice::from_raw_parts(second.ptr as *const u8, step);
                assert_eq!(a, b, "{null:?}");
            }
        }
    }

    /// Records mirrored in runs ([`HostHeap::mirror_picked`]) leave a heap
    /// as mirroring them one by one does: what each slot is made from, its
    /// entry in the table of strides, the buffer slots and the ones changed.
    #[test]
    fn runs_mirror_as_records_one_by_one_do() {
        if host().is_none() {
            return; // no host GPU
        }
        let (Some(one), Some(runs)) = (HostHeap::new(false, 300), HostHeap::new(false, 300)) else {
            return;
        };
        let null = std::ptr::null();
        let slots = vec![[0u64; 4]; 300];
        let handle = |i: u32| slots.as_ptr() as u64 + 32 * u64::from(i);
        let resource: [u32; 14] = std::array::from_fn(|k| if k == 0 { 1 } else { 0x4400 + k as u32 });
        let write = |round: u64| {
            for i in 0..300u32 {
                let at = handle(i);
                let n = u64::from(i) + round * 1000;
                match (i + round as u32) % 7 {
                    0..=2 => super::super::create_cbv(null, &[(0x7_0000 + n * 256) as u32, 3, 512, 0], at),
                    3 | 4 => {
                        let mut desc = [0u32; 16];
                        (desc[1], desc[8]) = (1, 16);
                        let address = 0x2000_0000 + n * 256;
                        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
                        super::super::create_placed_srv(null, &resource, &desc, at);
                    }
                    // A view that is not a buffer's: by `mirror` itself.
                    5 => super::super::create_srv(null, null, std::ptr::null(), at),
                    // Nothing (or what was there: left as it is).
                    _ if round == 1 => {}
                    _ => {
                        // SAFETY: a slot of `slots`.
                        unsafe { (at as *mut [u64; 4]).write([0; 4]) }
                    }
                }
            }
        };
        let picked: Vec<u32> = (0..300).collect();
        for round in 0..3 {
            write(round);
            for &i in &picked {
                one.mirror(i, handle(i), false);
            }
            let quick = runs.mirror_picked(&picked, handle, &mut |i, g| view_at(i, g), &mut PickScratch::default());
            assert!(quick > 0);
            for i in 0..300 {
                assert_eq!(one.made_key(i), runs.made_key(i), "slot {i}, round {round}");
                // and the table of strides says the same of each slot
                assert_eq!(one.stride(i), runs.stride(i), "stride of slot {i}, round {round}");
            }
            assert_eq!(one.buffer_slots(), runs.buffer_slots());
            let (mut a, mut b) = (one.buffer_changes(), runs.buffer_changes());
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Host constant views are whole 256s and at most 64 KiB (a larger one
    /// removes the device on NVIDIA); one of no bytes is a null view. Made
    /// on the device, an oversized one and one of nothing leave it alive.
    #[test]
    fn constant_views_are_at_most_64_kib_and_none_is_null() {
        assert_eq!(constant_view_bytes(0), None);
        assert_eq!(constant_view_bytes(16), Some(256));
        assert_eq!(constant_view_bytes(480), Some(512));
        assert_eq!(constant_view_bytes(65_536), Some(65_536));
        assert_eq!(constant_view_bytes(969_472), Some(65_536));
        assert_eq!(constant_view_bytes(u32::MAX), Some(65_536));
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let Some((buffer, _)) = host.upload_buffer(1 << 20) else {
            return;
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        heap.constant_at(0, gpu, 969_472);
        heap.constant_at(1, gpu, 0);
        heap.constant_at(2, 0, 256);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Our sampler and view heaps, for lists whose title binds none, are
    /// made once each; a list that binds both and runs leaves the device
    /// alive.
    #[test]
    fn our_heaps_are_made_once_and_bind() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let samplers = fallback_samplers().expect("a sampler heap of ours");
        let views = fallback_views().expect("a view heap of ours");
        assert!(std::ptr::eq(samplers, fallback_samplers().unwrap()));
        assert!(std::ptr::eq(views, fallback_views().unwrap()));
        assert!(samplers.sampler && !views.sampler);
        let list = host.new_list().unwrap();
        // SAFETY: an open list and shader-visible heaps of ours.
        unsafe { list.list.SetDescriptorHeaps(&[Some(views.heap.clone()), Some(samplers.heap.clone())]) };
        list.close();
        host.execute(&[&list]);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a
    /// slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| q.due(u64::MAX, 100).iter().find(|d| d.0 == index).unwrap().1;
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(q.due(frame, 100)[0].0, 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept, with what was there.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100)[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100)[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).iter().any(|d| d.0 == 98));
    }

    /// Slots looked at before and slots never looked at are served in
    /// turns, one looked at before first: a pass that can take only some
    /// of a backlog of slots streaming in a part at a time still reaches
    /// the slots never looked at (fxceless/victory-fork pull request #1246).
    #[test]
    fn the_texture_queue_serves_looked_at_and_fresh_slots_in_turns() {
        let mut q = TextureQueue::default();
        // Slots 100..110 looked at before (their memory not there yet).
        for i in 0..10u32 {
            q.written(100 + i, 1, 0, false);
        }
        for i in 0..10u32 {
            let w = q.due(u64::MAX, 100).iter().find(|d| d.0 == 100 + i).unwrap().1;
            q.later(100 + i, w, 1, 1);
        }
        // Slots 1..4 never looked at.
        for i in 1..4u32 {
            q.written(i, 2, 0, false);
        }
        let order: Vec<u32> = q.due(10, 6).into_iter().map(|d| d.0).collect();
        assert_eq!(order.len(), 6);
        assert!(order[0] >= 100, "a slot looked at before comes first: {order:?}");
        assert_eq!(order.iter().filter(|&&i| i < 100).count(), 3, "every fresh slot is reached: {order:?}");
        // In turns: looked at, fresh, looked at, fresh...
        assert!(order.iter().step_by(2).all(|&i| i >= 100) && order.iter().skip(1).step_by(2).all(|&i| i < 100));
    }

    /// Slots held back for lack of room on the card are not handed out
    /// until their frame, keep their place, are not counted as waiting
    /// for their memory, and all come back at once when the card has room;
    /// a slot written again meanwhile is the title's new write.
    #[test]
    fn slots_held_for_lack_of_room_come_back_when_there_is_room() {
        let mut q = TextureQueue::default();
        for i in 0..4u32 {
            q.written(20 + i, 1, 0, false);
        }
        let due = q.due(1, 100);
        assert_eq!(due.len(), 4);
        // The card is full: slots 20 and 21 are held back until frame 101.
        for &(index, write) in due.iter().filter(|d| d.0 < 22) {
            q.hold(index, write, 101);
        }
        let now: Vec<u32> = q.due(2, 100).into_iter().map(|d| d.0).collect();
        assert!(!now.contains(&20) && !now.contains(&21));
        assert!(now.contains(&22) && now.contains(&23));
        assert_eq!(q.census(2)[2], 0, "held slots do not wait for their memory");
        // Due again at their frame, in the order they had.
        assert_eq!(q.due(101, 100), due);
        // Slot 21 is written again while held: due as written.
        q.written(21, 3, 0, false);
        assert!(q.due(3, 100).iter().any(|d| d.0 == 21));
        // Room again at frame 5: slot 20 is due at once, and only it was
        // still held as it was.
        assert_eq!(q.room(5), 1);
        assert!(q.due(5, 100).iter().any(|d| d.0 == 20));
        assert_eq!(q.room(6), 0);
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn would_stale_says_what_note_does() {
        for rule in [Rule::Start, Rule::Overlap, Rule::Never] {
            let mut spans = Spans { rule, ..Spans::default() };
            let mut x = 0x2545_f491_4f6c_dd1du64;
            let (mut stale, mut quiet) = (0, 0);
            for _ in 0..20_000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let index = (x % 40) as u32;
                // Starts on a grid of 0x1000 in a small range, so that
                // they meet; lengths of 0 to 0x7000.
                let start = 0x10_0000 + ((x >> 8) % 48) * 0x1000;
                let end = start + ((x >> 16) % 8) * 0x1000;
                let newer = !(x >> 24).is_multiple_of(4);
                if (x >> 28).is_multiple_of(9) {
                    spans.forget(index);
                    continue;
                }
                let would = spans.would_stale(index, start, end, newer);
                let named = spans.clone().note(index, start, end, newer);
                assert_eq!(would, !named.is_empty(), "{rule:?} slot {index} {start:#x}..{end:#x} newer {newer}");
                if would { stale += 1 } else { quiet += 1 }
                spans.note(index, start, end, newer);
                spans.shows(index, (x >> 40) as u32 % 4);
                // `known` holds every slot the maps hold.
                assert!(spans.by_slot.keys().chain(spans.first.keys()).all(|&k| spans.known.contains(k)));
            }
            if rule != Rule::Never {
                assert!(stale > 100 && quiet > 100, "{rule:?}: {stale} {quiet}");
            }
        }
    }

    /// The queue's bit for a slot is set exactly while the slot is in it.
    #[test]
    fn queued_bits_follow_the_queue() {
        let bits: std::sync::Arc<[std::sync::atomic::AtomicU64]> =
            (0..4).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
        let mut q = TextureQueue { queued: Some(bits.clone()), ..TextureQueue::default() };
        let bit = |i: u32| bits[i as usize / 64].load(std::sync::atomic::Ordering::Relaxed) & (1 << (i % 64)) != 0;
        q.written(5, 1, 0, false);
        q.changed(130, 1, 2, crate::census::told(Cause::Mapped));
        assert!(bit(5) && bit(130) && !bit(6));
        let due = q.due(9, 8);
        q.remove(5);
        assert!(!bit(5) && bit(130));
        let (_, write) = due.into_iter().find(|d| d.0 == 130).unwrap();
        q.done(130, write);
        assert!(!bit(130));
        for i in 0..256 {
            assert_eq!(bit(i), q.slots.contains_key(&i));
        }
    }

    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// A write names the textures that hold its bytes: the one that
    /// starts where the write starts, not the neighbour before it, whose
    /// held levels end there.
    #[test]
    fn a_write_at_the_start_of_a_packed_texture_names_it_alone() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // A request that fills the second texture's held levels.
        assert_eq!(spans.slots_in(at + 0x2000, at + 0x6000), [1, 2]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x6000), [2]);
        // From inside the first one's held levels on: both.
        assert_eq!(spans.slots_holding(at + 0x1000, at + 0x3000), [1, 2]);
        // A whole 64 KiB page: every texture that starts in it.
        assert_eq!(spans.slots_holding(at, at + 0x10000), [1, 2, 3]);
        // Past the last start, the last one; past all, or no bytes, none.
        assert_eq!(spans.slots_holding(at + 0x20000, at + 0x21000), [3]);
        assert_eq!(spans.slots_holding(at + 0x32000, at + 0x33000), [] as [u32; 0]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x2000), [] as [u32; 0]);
    }

    /// A write behind the guards' back under a placed texture, by the
    /// writes' numbers: a slot never uploaded, or whose upload has not read
    /// yet, or read after the write, holds its bytes; one whose upload read
    /// before it is stale, whatever the queue knows of it, until uploaded
    /// again. (The ranges are handed over here, not through the list all
    /// heaps share.)
    #[test]
    fn a_page_write_after_an_uploads_read_makes_its_slot_stale() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicU64;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false).is_empty());
        let write = |number: u64| [(at, at + 0x1000, number, Cause::Pages as u8)];
        // Waiting for its first upload: nothing is stale, it stays as it is.
        assert_eq!(heap.slots_written_under(&write(10), 50), [] as [u32; 0]);
        let (queued, w) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        // Uploaded by the pass, the list not submitted yet: its read comes
        // after any write numbered now.
        let stamp = Arc::new(AtomicU64::new(crate::writes::NOT_READ));
        let asked = || Arc::new(AtomicU64::new(0));
        heap.texture_slot_stamp(1, stamp.clone(), asked());
        heap.texture_slot_done(1, w, None);
        assert_eq!(heap.slots_written_under(&write(11), 50), [] as [u32; 0]);
        assert_eq!(heap.texture_slots_stale_by(), [0, 0]);
        // Its read began at 12: a write numbered 12 is in it, 13 is not.
        stamp.store(12, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(heap.slots_written_under(&write(12), 60), [] as [u32; 0]);
        // Memory elsewhere: nothing.
        assert_eq!(heap.slots_written_under(&[(at + 0x10_0000, at + 0x10_1000, 13, Cause::Pages as u8)], 60), [] as [u32; 0]);
        assert_eq!(heap.slots_written_under(&write(13), 60), [1]);
        // Stale, and out of the queue all the same (`VICTORY_PAGE_WRITES`
        // bit 2 is not asked for here).
        assert_eq!(slots(u64::MAX), [] as [u32; 0]);
        assert_eq!(heap.texture_slots_stale_by(), [1, 0]);
        // Stale until uploaded again.
        heap.texture_slot_stamp(1, Arc::new(AtomicU64::new(14)), asked());
        assert_eq!(heap.texture_slots_stale_by(), [0, 0]);
    }

    /// A copy of the title's that the layer does into a texture's memory
    /// after an upload's read began makes the slot stale, by copy; one done
    /// before the read does not. The numbers are the ones `title_copied`
    /// gives (the ranges themselves are handed over here, not taken from
    /// the list all heaps share).
    #[test]
    fn a_copy_after_an_uploads_read_makes_it_stale_and_one_before_does_not() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicU64;
        let _one = REMAPPED_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fec_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false).is_empty());
        let (_, w) = heap.texture_slots_due(u64::MAX, 8).0[0];
        heap.texture_slot_done(1, w, None);
        // The copy first, then the upload's read: it holds the copy.
        let before = title_copied(at as usize, 0x1000);
        assert_ne!(before, 0, "the view's memory counts as spanned");
        heap.texture_slot_stamp(1, Arc::new(AtomicU64::new(crate::writes::stamp())), Arc::new(AtomicU64::new(0)));
        assert_eq!(heap.slots_written_under(&[(at, at + 0x1000, before, Cause::Copy as u8)], 50), [] as [u32; 0]);
        // The read first, then the copy: stale, by a copy.
        let after = title_copied(at as usize + 0x800, 0x100);
        assert!(after > before);
        assert_eq!(heap.slots_written_under(&[(at + 0x800, at + 0x900, after, Cause::Copy as u8)], 51), [1]);
        assert_eq!(heap.texture_slots_stale_by(), [0, 1]);
        // No bytes: no number.
        assert_eq!(title_copied(at as usize, 0), 0);
    }

    /// Every span that reaches into a range is found whatever starts inside
    /// it, long ones behind many short ones too; what is taken to hold the
    /// range's bytes is fewer: a texture ends there where the next starts.
    #[test]
    fn spans_over_a_range_are_all_found() {
        let mut spans = Spans::default();
        // A texture of 8 MiB, an older view of 64 KiB left standing inside
        // it, and small ones before it.
        let big = 0x1000_0000u64;
        assert!(spans.note(1, big, big + (8 << 20), true).is_empty());
        assert!(spans.note(2, big + 0x10_0000, big + 0x11_0000, true).is_empty());
        for k in 0..40u64 {
            assert!(spans.note(100 + k as u32, big - 0x8_0000 + k * 0x2000, big - 0x8_0000 + k * 0x2000 + 0x2000, true).is_empty());
        }
        // A write into the big one's largest level, past the old view.
        let (from, to) = (big + 0x40_0000, big + 0x50_0000);
        let mut over = spans.slots_over(from, to);
        over.sort_unstable();
        assert_eq!(over, [1]);
        assert_eq!(spans.slots_holding(from, to), [] as [u32; 0]);
        assert_eq!(spans.slots_in(from, to), [] as [u32; 0]);
        // One that begins in the old view's memory: both are over it, and
        // the old view alone is taken to hold it.
        let mut over = spans.slots_over(big + 0x10_8000, big + 0x12_0000);
        over.sort_unstable();
        assert_eq!(over, [1, 2]);
        assert_eq!(spans.slots_holding(big + 0x10_8000, big + 0x12_0000), [2]);
        // At the big one's first byte both ways name it.
        assert_eq!(spans.slots_holding(big, big + 0x1000), [1]);
        assert_eq!(spans.slots_over(big, big + 0x1000), [1]);
        // A range that only touches a span's end, or lies before every
        // span, has none; a slot forgotten is found no more.
        assert_eq!(spans.slots_over(big + (8 << 20), big + (9 << 20)), [] as [u32; 0]);
        assert_eq!(spans.slots_over(0x1000, 0x2000), [] as [u32; 0]);
        spans.forget(1);
        assert_eq!(spans.slots_over(from, to), [] as [u32; 0]);
        assert_eq!(spans.classed.len(), spans.by_start.len());
        // A span made stale by a newer view at its start leaves its class.
        assert_eq!(spans.note(3, big + 0x10_0000, big + 0x18_0000, true), [2]);
        assert_eq!(spans.classed.len(), spans.by_start.len());
        assert_eq!(spans.slots_over(big + 0x17_0000, big + 0x17_1000), [3]);
    }

    /// No lookup over the spans panics on a range that is none or lies at
    /// an end of the address space: the texture pass asks them of every
    /// texture it uploads and of every range it is told, on threads that
    /// record the title's commands. (`next_start` did, for every texture
    /// shorter than its first block: the map's range lookup panics when the
    /// start lies above the end.)
    #[test]
    fn lookups_over_the_spans_take_any_range() {
        let mut spans = Spans::default();
        let at = 0x2000_0000u64;
        // A texture of 16 bytes (1x1), one of a block, one at address 0 and
        // one that ends at the top of the address space.
        assert!(spans.note(1, at, at + 16, true).is_empty());
        assert!(spans.note(2, at + 0x10_0000, at + 0x10_1000, true).is_empty());
        assert!(spans.note(3, 0, 0x4_0000, true).is_empty());
        assert!(spans.note(4, u64::MAX - 0x2_0000, u64::MAX, true).is_empty());
        // A view of no bytes, and one whose end lies below its start, are
        // no span.
        assert!(spans.note(5, at + 0x20_0000, at + 0x20_0000, true).is_empty());
        assert!(spans.note(6, at + 0x30_0000, at + 0x2f_0000, true).is_empty());
        let ranges = [
            (at, at + 16),                       // shorter than the first block
            (at, at),                            // no bytes
            (at + 16, at),                       // an end below the start
            (0, 0),                              // address 0, no bytes
            (0, 16),                             // address 0
            (0, u64::MAX),                       // everything
            (u64::MAX - 8, u64::MAX),            // the top of the address space
            (u64::MAX, u64::MAX),
            (u64::MAX, 0),
            (at + 0x10_0000, at + 0x10_1000),    // exactly a block
            (at + 0x10_0000, at + 0x10_1001),
        ];
        for (start, end) in ranges {
            let _ = spans.next_start(start, end);
            let _ = spans.slots_over(start, end);
            let _ = spans.slots_holding(start, end);
            let _ = spans.slots_in(start, end);
            let _ = spans.overlapping(start, end);
            let _ = spans.would_stale(9, start, end, true);
        }
        // What they say of the ones that are none: nothing.
        assert_eq!(spans.next_start(at, at + 16), None);
        assert_eq!(spans.next_start(at + 16, at), None);
        assert_eq!(spans.next_start(u64::MAX, u64::MAX), None);
        assert_eq!(spans.slots_over(at + 16, at), [] as [u32; 0]);
        assert_eq!(spans.slots_in(at + 8, at + 8), [1]);
        assert_eq!(spans.slots_in(at + 16, at + 16), [] as [u32; 0]);
        // And of real ones what they said before: the texture inside the
        // one at address 0 past its first block, and everything over all.
        assert!(spans.note(7, 0x2_0000, 0x3_0000, true).is_empty());
        assert_eq!(spans.next_start(0, 0x4_0000), Some(0x2_0000));
        let mut all = spans.slots_over(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [1, 2, 3, 4, 7]);
    }

    /// The heap's questions for the census take the same ranges, and count
    /// the span that is none.
    #[test]
    fn the_census_questions_take_any_range() {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let at = 0x7fe8_0000_0000u64;
        let before = crate::census::FLAGS[crate::census::NO_SPAN].load(Relaxed);
        for (start, end, own) in [(at, at + 16, 16), (at, at, 0), (at + 16, at, 16), (0, 0, 0), (u64::MAX, u64::MAX, u64::MAX), (0, u64::MAX, 1)] {
            let (inner, held, _) = heap.texture_span_others(start, end, own);
            assert!(!held || inner.is_some());
            let _ = heap.texture_slot_found(1, start, end.wrapping_sub(start));
        }
        assert!(crate::census::FLAGS[crate::census::NO_SPAN].load(Relaxed) >= before + 4);
        let _ = heap.texture_queue_kinds(0);
        assert_eq!(heap.slots_written_under(&[(at + 16, at, 1, 200), (u64::MAX, u64::MAX, 2, 0)], 1), [] as [u32; 0]);
    }

    /// The slots due at a frame come from the index by due frame as the
    /// walk over every slot gave them, whatever was done to the queue: the
    /// same slots in the same order, and the index holds each slot once.
    #[test]
    fn the_due_slots_are_found_without_a_walk() {
        // The walk the index replaces.
        fn walked(q: &TextureQueue, frame: u64) -> Vec<u32> {
            let mut due: Vec<u32> = q.slots.iter().filter(|(_, w)| w.due <= frame).map(|(&i, _)| i).collect();
            due.sort_unstable();
            due
        }
        fn indexed(q: &TextureQueue, frame: u64) -> Vec<u32> {
            let mut due: Vec<u32> = q.due(frame, usize::MAX).into_iter().map(|d| d.0).collect();
            due.sort_unstable();
            due
        }
        let mut q = TextureQueue::default();
        let write = |q: &TextureQueue, slot: u32| q.slots[&slot].write;
        // A pseudo-random run of everything the pass and the title do.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        for step in 0..4000u64 {
            let (slot, frame) = (next(64) as u32, step / 8);
            if !q.slots.contains_key(&slot) || next(8) == 0 {
                q.written(slot, frame, frame + next(4), next(2) == 0);
                continue;
            }
            let w = write(&q, slot);
            match next(9) {
                0 => q.done(slot, w),
                1 => q.later(slot, w, frame, 16),
                2 => q.hold(slot, w, frame + 40),
                3 => q.after(slot, w, frame + 1),
                4 => q.partly(slot, w, frame + 1, 0x1000),
                5 => q.recheck(slot, w, frame + 4, (0x1000, 0x2000), 7, 0, frame),
                6 => q.changed(slot, frame, frame + 1, crate::census::VIEW),
                7 => {
                    q.room(frame);
                }
                _ => q.remove(slot),
            }
            assert_eq!(q.by_due.len(), q.slots.len(), "each slot once, step {step}");
            assert!(q.slots.iter().all(|(&i, w)| q.by_due.contains(&(w.due, i))), "step {step}");
            if step % 16 == 0 {
                assert_eq!(indexed(&q, frame), walked(&q, frame), "step {step}");
                assert_eq!(indexed(&q, frame + 100), walked(&q, frame + 100));
            }
        }
        // A write of another pass's (a stale write number) changes nothing.
        q.written(1, 600, 600, false);
        let stale = write(&q, 1) + 1000;
        q.later(1, stale, 600, 16);
        q.hold(1, stale, 700);
        assert_eq!(q.slots[&1].due, 600);
        assert!(q.by_due.contains(&(600, 1)));
        // The census is counted now and then, with the slots as they are.
        let all = q.census(700);
        assert_eq!(q.census_lately(700, 10_000), all);
        q.remove(1);
        let lately = q.census_lately(700, 10_100);
        assert_eq!((lately[0], lately[1]), (all[0] - 1, all[1]), "the slots now, the rest as last counted");
        assert_eq!(q.census_lately(700, 10_400), q.census(700));
    }

    /// The queue says what each slot waits for, and what made it due
    /// stays with it until it is written or told again.
    #[test]
    fn the_queue_says_what_its_slots_wait_for() {
        use crate::census::{LEVELS, MORE, VIEW, told};
        let mut q = TextureQueue::default();
        for slot in 1..=6 {
            q.written(slot, 10, 0, false);
        }
        // 1: never looked at and due. 2: due later (a table's slot).
        q.written(2, 10, 30, false);
        let write = |q: &TextureQueue, slot: u32| q.slots[&slot].write;
        // 3: held for room. 4: its memory not mapped. 5: uploaded in
        // part. 6: uploaded whole, after one look that found it the same.
        q.hold(3, write(&q, 3), 40);
        q.later(4, write(&q, 4), 10, 16);
        q.partly(5, write(&q, 5), 11, 0x1000);
        q.recheck(6, write(&q, 6), 22, (0x1000, 0x2000), 77, 1, 10);
        let (kinds, away) = q.kinds(10);
        assert_eq!(kinds, [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0]);
        assert_eq!(away, [4]);
        // The view was written: that is why each is due; uploaded in part
        // is said beside it.
        assert_eq!(q.why(1), Some((VIEW, false)));
        assert_eq!(q.why(5), Some((VIEW, true)));
        assert_eq!(q.why(9), None);
        // Told of its memory: that is why from then on.
        q.changed(5, 12, 13, told(Cause::Mapped));
        assert_eq!(q.why(5), Some((told(Cause::Mapped), false)));
        q.changed(6, 12, 13, LEVELS);
        assert_eq!(q.why(6), Some((LEVELS, false)));
        assert_ne!(MORE, VIEW);
    }

    /// A write that lies in a texture's span and is taken for another
    /// texture's (an older view stands between) is noted for the slot the
    /// pass uploaded before it, and says so when a look finds the slot's
    /// memory changed; uploaded again, the slot is as any other.
    #[test]
    fn a_write_taken_for_another_texture_is_known_of_the_one_it_lies_in() {
        use crate::census::{F_HIDDEN, F_OPEN, F_STALE, F_UNREAD};
        use std::sync::Arc;
        use std::sync::atomic::AtomicU64;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // 256x256 BC7 with its levels at an address no test and no title
        // uses, and an older 64x64 view left inside its span.
        let at = 0x7fea_0000_0000u64;
        let view = |at: u64, side: u32, levels: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, side, 0, side, 0x1_0000 | levels, 98, 1, 0, 0x105, 0, 0], desc)
        };
        assert!(heap.note_buffer(1, Some(&view(at, 256, 9)), false).is_empty());
        let inner = at + 0x4000;
        assert!(heap.note_buffer(2, Some(&view(inner, 64, 1)), false).is_empty());
        let size = {
            let placed = heap.placed.lock().unwrap_or_else(|e| e.into_inner());
            placed.by_start[&at].0 - at
        };
        assert!(size > 0x8000, "the large texture reaches past the small one");
        // Not uploaded yet: nothing is known of it.
        assert_eq!(heap.texture_slot_found(1, at, size), F_UNREAD);
        // Uploaded, its read begun; then a write past the inner view.
        let read = crate::writes::stamp();
        heap.texture_slot_stamp(1, Arc::new(AtomicU64::new(read)), Arc::new(AtomicU64::new(0)));
        heap.texture_slot_stamp(2, Arc::new(AtomicU64::new(read)), Arc::new(AtomicU64::new(0)));
        let number = crate::writes::next();
        let past = [(at + 0x6000, at + 0x7000, number, Cause::Pages as u8)];
        // Nobody is taken to hold it (the inner view's texture ends before
        // it, and the large one is taken to end where the inner starts):
        // it is known of the large one all the same.
        assert_eq!(heap.slots_written_under(&past, 50), [] as [u32; 0]);
        assert_eq!(heap.texture_slot_found(1, at, size), F_HIDDEN + Cause::Pages as usize);
        // Uploaded again after the write: it holds it, and nothing is noted
        // (the memory here is no view the guards hold: nothing could tell).
        heap.texture_slot_stamp(1, Arc::new(AtomicU64::new(crate::writes::stamp())), Arc::new(AtomicU64::new(0)));
        assert_eq!(heap.texture_slot_found(1, at, size), F_OPEN);
        // A write before an upload's read is not held against it.
        assert_eq!(heap.slots_written_under(&past, 51), [] as [u32; 0]);
        assert_eq!(heap.texture_slot_found(1, at, size), F_OPEN);
        // One into the inner view's memory is taken for its texture alone,
        // which is stale by it; the large one lies under it as well.
        let number = crate::writes::next();
        let into = [(inner + 0x800, inner + 0x900, number, Cause::Copy as u8)];
        assert_eq!(heap.slots_written_under(&into, 52), [2]);
        assert_eq!(heap.texture_slot_found(2, inner, 0x1000), F_STALE + Cause::Copy as usize);
        assert_eq!(heap.texture_slot_found(1, at, size), F_HIDDEN + Cause::Copy as usize);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let _one = REMAPPED_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0]).unwrap();
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000, Cause::Mapped);
        remapped(at as usize, 0x1000, Cause::Mapped);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [1]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// The tests that take [`REMAPPED`] and [`OVERFLOWED`], one at a time.
    static REMAPPED_TESTS: Mutex<()> = Mutex::new(());

    /// Ranges dropped on overflow make every placed slot due at the next
    /// pass, once; the pass after has nothing to do (PastiiJay/klbf-lab
    /// 74315e765).
    #[test]
    fn overflowed_ranges_make_every_placed_slot_due_once() {
        let _one = REMAPPED_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(3, Some(&view), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 3);
        heap.texture_slot_done(3, write, None);
        REMAPPED.lock().unwrap_or_else(|e| e.into_inner()).clear();
        OVERFLOWED.store(true, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(51), [3]);
        assert_eq!(heap.texture_slots_remapped(52), 0);
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, frame);
        heap.queue_texture(3, true, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    const SKIP: u32 = victory_shader::emit::SLOT_STRIDES_SKIP;

    /// A view of title memory as a structured buffer of 4 elements of
    /// `stride` bytes (format unknown, dimension buffer, the default
    /// mapping).
    fn placed_buffer(stride: u32) -> View {
        let mut desc = [0u32; 16];
        desc[..9].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 4, stride, 0]);
        View::Placed(
            false,
            [1, 0, 0, 0, 64, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
            desc,
        )
    }

    /// Until a slot holds a buffer view, the table of strides says it
    /// holds none: so it says for every slot of a new heap.
    #[test]
    fn the_stride_table_starts_with_every_slot_skipped() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        for index in [0, 7, victory_shader::emit::SLOT_STRIDES_COUNT - 1] {
            assert_eq!(heap.stride(index), Some(SKIP), "slot {index}");
        }
    }

    /// Below a view heap's null slot sit the null views of each class that
    /// a buffer picked where the stride table says no buffer view reads
    /// (`NULL_BUFFER_SRV_BELOW`): the title's slots and the dynamic ones
    /// end below them, also on a host that caps its heap.
    #[test]
    fn the_title_slots_end_below_the_null_buffer_views() {
        let lowest = victory_shader::emit::NULL_CBV_BELOW;
        assert_eq!(view_heap_fit(1_100_000, DYNAMIC, (1 << 21) - 1), 1_100_000);
        let capped = PORTABLE_VIEWS - 1;
        assert_eq!(
            view_heap_fit(1_100_000, DYNAMIC, capped) + DYNAMIC,
            capped - lowest
        );
        let just = capped - lowest - DYNAMIC;
        assert_eq!(view_heap_fit(just, DYNAMIC, capped), just);
        assert_eq!(view_heap_fit(just + 1, DYNAMIC, capped), just);
    }

    /// What a slot is to hold is noted before `mirror` writes the slot:
    /// from then until `mirror` has written it, its entry says it holds no
    /// buffer view, neither the old view's stride (the slot is about to
    /// change) nor the new one's (beside the old descriptor).
    #[test]
    fn the_stride_of_a_noted_slot_stays_skipped_until_mirror_writes_it() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let mut record = [0u32; 8];
        let handle = record.as_mut_ptr() as u64;
        crate::descriptor::record(handle, placed_buffer(16));
        heap.mirror(1, handle, false);
        assert_eq!(heap.stride(1), Some(16));
        let wider = placed_buffer(32);
        assert!(heap.note_buffer(1, Some(&wider), false).is_empty());
        assert_eq!(heap.stride(1), Some(SKIP));
    }

    /// Once `mirror` has written a slot, its entry says what the slot
    /// holds: a buffer view's stride (its null view's too, before a draw
    /// makes the view), 0 for a constant buffer view, none for a texture's,
    /// and none once its record names no view.
    #[test]
    fn mirror_publishes_the_stride_of_what_the_slot_holds() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fef_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let texture = View::Placed(
            false,
            [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0],
            desc,
        );
        let mut records = [[0u32; 8]; 4];
        let handles: Vec<u64> = records.iter_mut().map(|r| r.as_mut_ptr() as u64).collect();
        let views = [
            (1, placed_buffer(16)),
            (2, View::Constant(at + 0x1000_0000, 256)),
            (3, texture),
        ];
        for (slot, view) in views {
            crate::descriptor::record(handles[slot], view);
            heap.mirror(slot as u32, handles[slot], false);
        }
        assert_eq!(heap.stride(1), Some(16));
        assert_eq!(heap.stride(2), Some(0));
        assert_eq!(heap.stride(3), Some(SKIP));
        // Slot 1's record names no view any more; a draw reads it.
        // SAFETY: record 1, alive in `records`.
        unsafe { (handles[1] as *mut [u32; 8]).write([0; 8]) };
        heap.mirror(1, handles[1], true);
        assert_eq!(heap.stride(1), Some(SKIP));
    }

    /// A buffer's own view (made without a description) is a buffer view
    /// whose element size the host does not know: entry 0, as for every
    /// buffer view before the table told texture slots apart.
    #[test]
    fn a_buffers_own_view_is_a_buffer_view_of_unknown_element_size() {
        let buffer = crate::resource::test_buffer(256) as usize;
        assert_eq!(stride_entry(&View::Shader(buffer, None)), 0);
        assert_eq!(stride_entry(&View::Unordered(buffer, 0, None)), 0);
    }
}

#[cfg(test)]
mod record_print_tests {
    use super::record_print;

    /// Equal records print alike; a record that differs in any one byte,
    /// or in its length, prints otherwise.
    #[test]
    fn a_record_that_changed_prints_otherwise() {
        let record: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37).wrapping_add(5)).collect();
        let print = record_print(&record);
        assert_eq!(print, record_print(&record.clone()));
        for at in 0..32 {
            for bit in 0..8 {
                let mut other = record.clone();
                other[at] ^= 1 << bit;
                assert_ne!(record_print(&other), print, "byte {at} bit {bit}");
            }
        }
        assert_ne!(record_print(&record[..16]), print);
        assert_ne!(record_print(&[0u8; 32]), record_print(&[0u8; 16]));
        // A record of zeros is told from "never looked at" by the caller's `| 1`.
        assert_ne!(record_print(&[0u8; 32]) | 1, 0);
    }
}

#[cfg(test)]
mod own_copy_tests {
    use super::own_bytes_if;

    /// A buffer view gets a copy of its own bytes only when it has no host
    /// view on its buffer's copy: off that copy's elements, and not a
    /// structured view that can be a raw view of the same bytes.
    #[test]
    fn a_view_off_its_copys_elements_gets_its_own_bytes() {
        let base = 0xd_9000_0000u64;
        // On an element: the buffer's copy (the two cases of every run's
        // log are the ones below).
        assert_eq!(own_bytes_if(true, true, 24, base, 3, 10, 12, 0, false, false), None);
        assert_eq!(own_bytes_if(true, true, 0, base, 0, 10, 12, 0, false, false), None);
        // A typed view of 12-byte elements, 5,222,240 bytes into its copy
        // (8 past an element): its own bytes, from its first element.
        assert_eq!(own_bytes_if(true, true, 5_222_240, base, 3, 10, 12, 0, false, false), Some((base + 36, 120)));
        // A structured view of 32-byte elements 8 bytes in: not on 16, so
        // no raw view of the copy either.
        assert_eq!(own_bytes_if(true, true, 8, base, 2, 4, 32, 32, true, false), Some((base + 64, 128)));
        // The same 16 bytes in: off its elements, but a raw view of the
        // copy's bytes does (as before).
        assert_eq!(own_bytes_if(true, true, 16, base, 2, 4, 32, 32, true, false), None);
        // No elements, no size, or the switch off: as before.
        assert_eq!(own_bytes_if(true, true, 8, base, 2, 0, 32, 32, true, false), None);
        assert_eq!(own_bytes_if(true, true, 8, base, 2, 4, 0, 0, false, false), None);
        assert_eq!(own_bytes_if(false, true, 8, base, 2, 4, 32, 32, true, false), None);
    }

    /// A raw view whose bytes do not start on 16 in its buffer's copy (the
    /// run's log: `first=2 count=16 raw=true`) gets a copy of its own bytes;
    /// one that starts on 16 does not, nor any with its switch off.
    #[test]
    fn a_raw_view_off_sixteen_bytes_gets_its_own_bytes() {
        let base = 0xd_9000_0000u64;
        assert_eq!(own_bytes_if(true, true, 0, base, 2, 16, 4, 0, false, true), Some((base + 8, 64)));
        assert_eq!(own_bytes_if(true, true, 8, base, 0, 16, 4, 0, false, true), Some((base, 64)));
        assert_eq!(own_bytes_if(true, true, 0, base, 4, 16, 4, 0, false, true), None);
        assert_eq!(own_bytes_if(true, true, 8, base, 2, 16, 4, 0, false, true), None);
        assert_eq!(own_bytes_if(true, false, 0, base, 2, 16, 4, 0, false, true), None);
        assert_eq!(own_bytes_if(false, true, 0, base, 2, 16, 4, 0, false, true), None);
        // Not raw (a typed view of dwords): as before.
        assert_eq!(own_bytes_if(true, true, 0, base, 2, 16, 4, 0, false, false), None);
    }
}

#[cfg(test)]
mod quick_print_tests {
    use super::quick_print;

    /// Any one byte of a record changed changes its print, for records of
    /// views (32 bytes) and of samplers (16).
    #[test]
    fn a_changed_byte_changes_the_print() {
        for len in [16usize, 32] {
            let mut record: Vec<u8> = (0..len as u8).map(|b| b.wrapping_mul(37).wrapping_add(11)).collect();
            let print = quick_print(&record);
            for at in 0..len {
                for flip in [1u8, 0x80, 0xff, 0x5a] {
                    record[at] ^= flip;
                    assert_ne!(quick_print(&record) | 1, print | 1, "byte {at} ^ {flip:#x} of {len}");
                    record[at] ^= flip;
                }
            }
            assert_eq!(quick_print(&record), print);
        }
    }
}
