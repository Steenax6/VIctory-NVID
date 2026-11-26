//! Textures shaders write that the title's own code then reads on the CPU.
//!
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it
//! stays in the texture's host copy. The title's water is one: its wave
//! passes write a 256x256 height map (R32_FLOAT) and a displacement map
//! (R32G32_FLOAT), with two smaller maps beside them, and its CPU code
//! samples them, which is what boats float on. Without the copy back the
//! physics read memory nothing had written, and boats sat in a pit in the
//! water or sank.
//!
//! For textures of the kinds copied back ([`kinds`]), a list in which
//! shaders write one through unordered access copies the levels and layers
//! its views expose into a read-back buffer of ours when it closes; once
//! the list has run, those texels are tiled into the texture's title
//! memory in the console's layout, as the console's GPU would have left
//! them. A list that writes none of them costs nothing beyond a look at the
//! written texture's size and format.
//!
//! The copy needs the console layout to be addressable: a texture in a
//! swizzle mode `victory-formats` has no address pattern for cannot be
//! tiled; such textures are counted and left alone (the log names the
//! first). The pipe/bank XOR modes have theirs.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title
//! never reads are copied back too (about a megabyte per such list). The
//! copy shows what the list's last write left; a title read in between
//! sees the frame before, as it would before the console's GPU has run.
//!
//! From Geckomacabre/vx-layer eee5667d7, as merged in fxceless/victory-fork
//! PR #999 (895c0b8, with 04df435's tickets per allocation) and with
//! BradRoland's fxceless PR #1152 (only the levels and layers bound views
//! expose are copied).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// A kind of texture copied back: width, height and the title's format
/// number (the DXGI number).
pub(crate) type Kind = (u64, u32, u32);

/// The kinds the title's CPU was found reading after its shaders wrote
/// them (the water's maps; Geckomacabre's read watch, confirmed in several
/// forks: 256x256 R32_FLOAT, 256x256 R32G32_FLOAT, 64x64 R32G32_FLOAT,
/// 128x128 R16G16_UINT).
pub(crate) const BUILT_IN: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

/// What `VICTORY_TEXTURE_BACK` asks for ([`kinds`]): the kinds, and the
/// entries that were not understood.
fn parse(spec: Option<&str>) -> (Vec<Kind>, Vec<String>) {
    fn kind(entry: &str) -> Option<Kind> {
        let (size, format) = entry.split_once('f')?;
        let (w, h) = size.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
    }
    let built_in = || BUILT_IN.split(',').filter_map(kind).collect::<Vec<_>>();
    let spec = match spec.map(str::trim) {
        None | Some("1" | "on") => return (built_in(), Vec::new()),
        Some("" | "0" | "off") => return (Vec::new(), Vec::new()),
        Some(spec) => spec,
    };
    let entries: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();
    // With `+` or `-` the entries change the built-in list; without, they
    // are the list.
    let editing = entries.iter().any(|e| e.starts_with(['+', '-']));
    let mut kinds = if editing { built_in() } else { Vec::new() };
    let mut bad = Vec::new();
    for entry in entries {
        let (sign, body) = match entry.as_bytes()[0] {
            b'+' | b'-' => (Some(entry.as_bytes()[0]), &entry[1..]),
            _ => (None, entry),
        };
        match (kind(body), sign, editing) {
            (Some(k), Some(b'-'), _) => kinds.retain(|&have| have != k),
            (Some(k), Some(_), _) | (Some(k), None, false) => {
                if !kinds.contains(&k) {
                    kinds.push(k);
                }
            }
            // A plain entry among edits would be read as either.
            (Some(_), None, true) | (None, ..) => bad.push(entry.to_owned()),
        }
    }
    (kinds, bad)
}

/// The kinds of texture copied back. On by default with [`BUILT_IN`];
/// `VICTORY_TEXTURE_BACK=0` copies none; a list (`<width>x<height>f<format>`,
/// comma-separated) is the kinds instead; entries with `+` or `-` add to or
/// take from the built-in list (`+512x512f41,-64x64f16`).
pub(crate) fn kinds() -> &'static [Kind] {
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let spec = std::env::var("VICTORY_TEXTURE_BACK").ok();
        let (kinds, bad) = parse(spec.as_deref());
        if !bad.is_empty() {
            tracing::warn!(?bad, "VICTORY_TEXTURE_BACK: entries not understood, left out (<width>x<height>f<format>, all plain or all with + or -)");
        }
        let kinds_text: Vec<String> = kinds.iter().map(|(w, h, f)| format!("{w}x{h}f{f}")).collect();
        tracing::info!(
            kinds = %kinds_text.join(","),
            "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK: 0 none, a list instead, +/- entries change the built-in list)"
        );
        kinds
    })
}

/// The kinds named in the log so far: each the first time one of its
/// textures is written, with its swizzle mode (an XOR mode needs the pipe
/// and bank address patterns of `victory-formats`).
fn said_kind(resource: This, layout: &victory_formats::texture::Layout) {
    static SAID: Mutex<Vec<Kind>> = Mutex::new(Vec::new());
    let Some(kind) = crate::resource::texture_kind(resource) else {
        return;
    };
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.contains(&kind) {
        return;
    }
    said.push(kind);
    let (w, h, f) = kind;
    let swizzle = layout.desc.swizzle;
    tracing::info!(
        kind = %format_args!("{w}x{h}f{f}"),
        swizzle = swizzle.0,
        xor = swizzle.xor(),
        levels = layout.desc.mips,
        layers = layout.desc.depth,
        "a texture of a kind copied back is written by shaders for the first time"
    );
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding copy back into one title allocation. When the title
/// releases the texture, makes another texture at its memory or the
/// runtime remaps that memory, the copies of the old one are dropped:
/// their texels would land in memory that is no longer the texture's.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
/// Whether a ticket was ever made: until then the hooks below do nothing.
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|weak| {
            let Some(ticket) = weak.upgrade() else {
                return false;
            };
            if matches(&ticket) {
                ticket.valid.store(false, Ordering::Release);
            }
            true
        });
}

/// The title released the texture at `owner`.
pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

/// The title made a texture whose memory starts at `at`.
pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

/// The runtime mapped or unmapped `[at, at + len)`.
pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

/// A texture of a kind copied back that a list's shaders write, and the
/// levels and layers (as host subresource numbers, title level and layer)
/// its views expose.
pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
    subresources: std::collections::BTreeSet<u32>,
}

/// What a view of a texture of `mips` levels and `slices` layers can write
/// through unordered access: its level and first and end layer; `None` for
/// a view that writes none of it (another plane, a level or layer past the
/// texture's). `words` are the title's view description (the host's layout,
/// the format first); none is the whole texture's default view, level 0 of
/// every layer.
fn exposed(words: Option<&[u32; 10]>, mips: u32, slices: u32) -> Option<(u32, u32, u32)> {
    let (mip, first, count, plane) = match words {
        None => (0, 0, slices, 0),
        Some(w) => match D3D12_UAV_DIMENSION(w[1] as i32) {
            // MipSlice, PlaneSlice.
            D3D12_UAV_DIMENSION_TEXTURE2D => (w[2], 0, 1, w[3]),
            // MipSlice, FirstArraySlice, ArraySize, PlaneSlice.
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => (w[2], w[3], w[4], w[5]),
            _ => return None,
        },
    };
    // An array size of all ones: every layer from the first.
    let end = if count == u32::MAX {
        slices
    } else {
        first.checked_add(count)?
    };
    (plane == 0 && mip < mips && first < end && end <= slices).then_some((mip, first, end))
}

impl Target {
    /// `resource` written through a view described by `words`
    /// ([`exposed`]); `None` if it has no host copy or console layout, or
    /// the view writes none of it.
    pub(crate) fn new(resource: This, words: Option<&[u32; 10]>) -> Option<Self> {
        // Held by the target, apart from the title's resource.
        let copy = crate::resource::host_copy(resource)?;
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        said_kind(resource, &layout);
        if !layout.addressable() {
            UNADDRESSABLE.fetch_add(1, Ordering::Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    swizzle = ?layout.desc.swizzle,
                    width = layout.desc.width,
                    height = layout.desc.height,
                    "a texture to copy back into title memory could not be tiled e=Unsupported: its swizzle mode has no \
                     address pattern (victory-formats), so it is not copied back"
                );
            });
            return None;
        }
        // SAFETY: a live host texture.
        let host = unsafe { copy.texture.GetDesc() };
        if u32::from(host.MipLevels.max(1)) != layout.desc.mips
            || u32::from(host.DepthOrArraySize.max(1)) != layout.desc.depth
        {
            // A host copy with other levels or layers than the title's
            // (not seen for textures shaders write): nothing to match.
            return None;
        }
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        let mut target = Self {
            copy,
            layout,
            ticket,
            subresources: Default::default(),
        };
        target.include(words);
        (!target.subresources.is_empty()).then_some(target)
    }

    /// The title texture's address.
    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }

    /// Another view of the same texture the list's shaders write through.
    pub(crate) fn include(&mut self, words: Option<&[u32; 10]>) {
        let (mips, slices) = (self.layout.desc.mips, self.layout.desc.depth);
        if let Some((mip, first, end)) = exposed(words, mips, slices) {
            self.subresources
                .extend((first..end).map(|slice| slice * mips + mip));
        }
    }
}

struct Subresource {
    index: u32,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: u32,
}

/// A copy recorded into a list: the read-back buffer, the subresources in
/// it and where they go.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    subresources: Vec<Subresource>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    /// The title memory it writes when it lands: address, end.
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records the copy of `target`'s host texture, which the list has moved to
/// the copy-source state, into a read-back buffer.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if target.subresources.is_empty() || !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture the list holds.
    let desc = unsafe { texture.GetDesc() };
    let mut subresources = Vec::with_capacity(target.subresources.len());
    let mut total = 0u64;
    for index in target.subresources {
        let offset =
            total.checked_next_multiple_of(u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT))?;
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0;
        // SAFETY: outputs for one subresource of the live texture.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                index,
                1,
                offset,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                None,
            );
        }
        // The tiler reads whole pitched rows, the last one's padding too.
        let f = footprint.Footprint;
        let pitched = u64::from(f.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(f.Depth))?;
        total = footprint.Offset.checked_add(pitched)?;
        subresources.push(Subresource {
            index,
            footprint,
            rows,
        });
    }
    let buffer = host.read_back_buffer(total)?;
    for sub in &subresources {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: sub.footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub.index,
            },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    RECORDED.fetch_add(1, Ordering::Relaxed);
    Some(Back {
        buffer,
        subresources,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// Since the last line: copies back recorded, landed in title memory and
/// their bytes.
static RECORDED: AtomicU64 = AtomicU64::new(0);
static LANDED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Since the start: copies landed, dropped because the allocation changed
/// under them, not tiled, and textures not copied back for want of an
/// address pattern.
static TOTALS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`TOTALS`], for the tests: landed, dropped, not tiled.
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn totals() -> [u64; 3] {
    [0, 1, 2].map(|i| TOTALS[i].load(Ordering::Relaxed))
}
static UNADDRESSABLE: AtomicU64 = AtomicU64::new(0);

/// Once the list has run: its texels into the same title allocation.
pub(crate) fn land(back: &Back) {
    let _mapping = crate::watch::title_memory_reading();
    // Held to the end: a release or a new texture in the same memory waits.
    let _tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        TOTALS[1].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // The title's bytes the subresources do not cover stay as they are.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        TOTALS[1].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let before = tiled.clone();
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    let mips = back.layout.desc.mips;
    let mut tiled_all = true;
    for Subresource {
        index,
        footprint,
        rows,
    } in &back.subresources
    {
        let (mip, slice) = (index % mips, index / mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let depth_pitch = pitch * *rows as usize;
        let len = depth_pitch * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe {
            std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len)
        };
        if let Err(e) = back
            .layout
            .tile(&mut tiled, mip, slice, linear, pitch, depth_pitch)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
            });
            tiled_all = false;
            break;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, None) };
    if !tiled_all {
        TOTALS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    // Only the bytes the texels changed go to title memory: the rest of the
    // span (padding, the tail, a texture placed inside it) may have been
    // written by the title since it was read here.
    let mut written = 0;
    for (from, to) in changed_runs(&before, &tiled) {
        crate::host::copy_mapped(at.wrapping_add(from) as *mut u8, tiled[from..].as_ptr(), to - from, true);
        written += to - from;
    }
    LANDED[0].fetch_add(1, Ordering::Relaxed);
    LANDED[1].fetch_add(written as u64, Ordering::Relaxed);
    TOTALS[0].fetch_add(1, Ordering::Relaxed);
}

/// Where `after` differs from `before` (as long): runs `[from, to)`, those
/// less than 64 bytes apart joined.
fn changed_runs(before: &[u8], after: &[u8]) -> Vec<(usize, usize)> {
    const GAP: usize = 64;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    let n = before.len().min(after.len());
    while i < n {
        if before[i] == after[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && before[i] != after[i] {
            i += 1;
        }
        match runs.last_mut() {
            Some(last) if start - last.1 < GAP => last.1 = i,
            _ => runs.push((start, i)),
        }
    }
    runs
}

/// The line on the copies back since the last one, per frame of `frames`;
/// nothing while there are none.
pub(crate) fn line(frames: f64) {
    let recorded = RECORDED.swap(0, Ordering::Relaxed);
    let [landed, bytes] = [0, 1].map(|i| LANDED[i].swap(0, Ordering::Relaxed));
    if recorded + landed == 0 {
        return;
    }
    let [all, dropped, untiled] = [0, 1, 2].map(|i| TOTALS[i].load(Ordering::Relaxed));
    tracing::info!(
        "textures shaders wrote, copied back into title memory per frame: {:.1} recorded, {:.1} landed ({:.0} KB); since \
         the start: {all} landed, {dropped} dropped (their memory changed under them), {untiled} not tiled, {} views of \
         textures without an address pattern passed over",
        recorded as f64 / frames,
        landed as f64 / frames,
        bytes as f64 / 1e3 / frames,
        UNADDRESSABLE.load(Ordering::Relaxed)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_list_is_the_default_and_the_switch_widens_narrows_or_disables_it() {
        let built_in = parse(None).0;
        assert_eq!(
            built_in,
            vec![(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)]
        );
        // `1` is on, as unset: never an empty list.
        assert_eq!(parse(Some("1")).0, built_in);
        assert_eq!(parse(Some("on")).0, built_in);
        for off in ["0", "off", "", "  "] {
            assert_eq!(parse(Some(off)), (Vec::new(), Vec::new()), "{off:?}");
        }
        // A plain list is the list.
        assert_eq!(
            parse(Some("512x512f41, 256x256f41")),
            (vec![(512, 512, 41), (256, 256, 41)], Vec::new())
        );
        // Edits change the built-in list.
        let (kinds, bad) = parse(Some("+512x512f41,-64x64f16,+256x256f41"));
        assert!(bad.is_empty());
        assert_eq!(
            kinds,
            vec![
                (256, 256, 41),
                (256, 256, 16),
                (128, 128, 36),
                (512, 512, 41)
            ]
        );
        // What is not understood is named, and a plain entry among edits too.
        let (kinds, bad) = parse(Some("-64x64f16,128x128f36,+bogus"));
        assert_eq!(kinds, vec![(256, 256, 41), (256, 256, 16), (128, 128, 36)]);
        assert_eq!(bad, vec!["128x128f36".to_owned(), "+bogus".to_owned()]);
    }

    #[test]
    fn only_changed_bytes_are_written_back() {
        let before = [0u8; 300];
        let mut after = before;
        assert_eq!(changed_runs(&before, &after), []);
        after[10] = 1;
        after[20] = 1;
        after[200..210].fill(2);
        assert_eq!(changed_runs(&before, &after), [(10, 21), (200, 210)]);
        after[299] = 3;
        assert_eq!(changed_runs(&before, &after), [(10, 21), (200, 210), (299, 300)]);
        after[250] = 4;
        assert_eq!(changed_runs(&before, &after), [(10, 21), (200, 300)]);
    }

    #[test]
    fn views_expose_their_level_and_layers() {
        let two_d = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
        let array = D3D12_UAV_DIMENSION_TEXTURE2DARRAY.0 as u32;
        // The default view: level 0 of every layer.
        assert_eq!(exposed(None, 3, 4), Some((0, 0, 4)));
        assert_eq!(
            exposed(Some(&[41, two_d, 2, 0, 0, 0, 0, 0, 0, 0]), 3, 1),
            Some((2, 0, 1))
        );
        assert_eq!(
            exposed(Some(&[41, array, 1, 1, 2, 0, 0, 0, 0, 0]), 3, 4),
            Some((1, 1, 3))
        );
        // All ones: every layer from the first.
        assert_eq!(
            exposed(Some(&[41, array, 0, 2, u32::MAX, 0, 0, 0, 0, 0]), 1, 4),
            Some((0, 2, 4))
        );
        // Past the texture, another plane, an empty range, a buffer view.
        assert_eq!(
            exposed(Some(&[41, two_d, 3, 0, 0, 0, 0, 0, 0, 0]), 3, 1),
            None
        );
        assert_eq!(
            exposed(Some(&[41, two_d, 0, 1, 0, 0, 0, 0, 0, 0]), 1, 1),
            None
        );
        assert_eq!(
            exposed(Some(&[41, array, 0, 4, u32::MAX, 0, 0, 0, 0, 0]), 1, 4),
            None
        );
        assert_eq!(
            exposed(Some(&[41, array, 0, 3, 2, 0, 0, 0, 0, 0]), 1, 4),
            None
        );
        assert_eq!(exposed(Some(&[41, 1, 0, 0, 0, 0, 0, 0, 0, 0]), 1, 1), None);
    }

    #[test]
    fn only_copies_into_changed_allocations_are_dropped() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);
        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));
        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }
}
