//! Host textures the title let go of while a list that names them had not
//! run yet: counted at each submission, said with the frame-rate line.
//!
//! When the title releases a texture, its host copy waits for the
//! submissions running at that moment (`crate::host::retire`) and is freed
//! after them, at once if none runs. A list recorded before the release and
//! submitted after it runs later than that: it draws into, or reads, the
//! host texture only if the list itself holds it. A list holds a host copy
//! it moved to a state of its own (`HostList::transition`: every texture a
//! draw draws into, samples or writes) and every object it keeps
//! (`HostList::keep`: since 2336572 the targets it binds).
//!
//! The release marks the host texture (private data on the host object,
//! which dies with it, so a later texture at the same address is not taken
//! for it). At each submission every texture a list holds is looked at: one
//! marked released is counted, as held by a state of the list (it was held
//! before 2336572 too) or by a keep alone (without the keep it would have
//! been freed before the list ran).
//!
//! Both counts were 0 in every run through 2026-11-26 (22 runs after the
//! counter came in, entries into the open world with and without the
//! dump, with the keep of bound targets): the line, said only when one is
//! above 0, never came. The losses of the host device at the world's
//! entry that day are not explained by a texture let go under a list.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::GUID;

/// The private data a released host texture carries: the frame of its
/// release.
const RELEASED: GUID = GUID::from_u128(0x5f0c_2a71_9d4e_4b83_a6c1_77e2_0b59_d3a4);

/// Since the start: lists submitted with a released texture, counted once
/// per list and texture: [held by a state of the list, held by a keep
/// alone]; and the same since the last line.
static COUNTS: [[AtomicU64; 2]; 2] = [const { [const { AtomicU64::new(0) }; 2] }; 2];

/// The first few found, for the line.
static FIRST: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// How many are named.
const NAMED: usize = 8;

/// Who let a host texture go while lists may still name it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Path {
    /// The title released its texture (`Drop for Resource`).
    Title = 1,
    /// The layer's sweep let the host copy of a texture the title fills go
    /// while the title's texture lives (`resource::let_go_idle`): its next
    /// use makes a new host copy.
    Sweep = 2,
}

impl Path {
    fn of(code: u64) -> Option<Path> {
        match code {
            1 => Some(Path::Title),
            2 => Some(Path::Sweep),
            _ => None,
        }
    }

    fn text(self) -> &'static str {
        match self {
            Path::Title => "the title released it",
            Path::Sweep => "the sweep let its host copy go",
        }
    }
}

/// `texture` (a host object) is let go by `path`: marked with the frame
/// and the path.
pub(crate) fn mark(texture: &ID3D12Resource, path: Path) {
    let data = [crate::frame::current(), path as u64];
    // SAFETY: plain private data of a live object, 16 bytes read from a local.
    let _ = unsafe { texture.SetPrivateData(&RELEASED, 16, Some(data.as_ptr().cast())) };
}

/// `texture` is kept to be another title texture's host copy (reserved
/// textures reused): its mark goes, so the next owner's lists are not
/// counted for the last one's.
pub(crate) fn unmark(texture: &ID3D12Resource) {
    // SAFETY: plain private data of a live object; no data removes it.
    let _ = unsafe { texture.SetPrivateData(&RELEASED, 0, None) };
}

/// The frame `texture` was let go in and by which path, if it was.
pub(crate) fn released(texture: &ID3D12Resource) -> Option<(u64, Path)> {
    let (mut data, mut size) = ([0u64; 2], 16u32);
    // SAFETY: at most `size` bytes are written into the local.
    unsafe { texture.GetPrivateData(&RELEASED, &mut size, Some(data.as_mut_ptr().cast())) }.ok()?;
    if size != 16 {
        return None;
    }
    Some((data[0], Path::of(data[1])?))
}

/// At a submission: a list holds `texture`, which `path` let go in frame
/// `frame`; `kept_only`: held by a keep alone, no state of the list.
pub(crate) fn found(texture: &ID3D12Resource, (frame, path): (u64, Path), kept_only: bool) {
    let i = usize::from(kept_only);
    for c in &COUNTS {
        c[i].fetch_add(1, Ordering::Relaxed);
    }
    let mut first = FIRST.lock().unwrap_or_else(|e| e.into_inner());
    if first.len() < NAMED {
        // SAFETY: a plain query of a live object.
        let desc = unsafe { texture.GetDesc() };
        first.push(format!(
            "{}x{}x{} format {} let go in frame {frame} ({}), submitted in frame {}{}",
            desc.Width,
            desc.Height,
            desc.DepthOrArraySize,
            desc.Format.0,
            path.text(),
            crate::frame::current(),
            if kept_only { ", held by the keep alone" } else { ", held by a state of the list" }
        ));
    }
}

/// With the frame-rate line: the counts, whenever one is above 0.
pub(crate) fn line() {
    let total = COUNTS[0].each_ref().map(|c| c.load(Ordering::Relaxed));
    if total == [0, 0] {
        return;
    }
    let now = COUNTS[1].each_ref().map(|c| c.swap(0, Ordering::Relaxed));
    let first = FIRST.lock().unwrap_or_else(|e| e.into_inner());
    tracing::info!(
        "lists submitted with a host texture let go before (by the title or the sweep; 0 in every run through 2026-11-26): since the start {} held by a state of the list and {} by a keep alone \
         (without it freed before the list ran); since the last line {} and {}; the first: {}",
        total[0],
        total[1],
        now[0],
        now[1],
        first.join("; ")
    );
}

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::core::Interface;

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

    fn target(device: &ID3D12Device) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 32,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a plain committed texture on the software adapter.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
                .expect("a texture");
        }
        texture.expect("made")
    }

    #[test]
    fn warp_a_released_texture_is_known_as_released_through_any_reference_and_a_new_one_is_not() {
        let Some(device) = warp() else {
            return;
        };
        let texture = target(&device);
        // What a list holds: another reference to the same host object.
        let held = texture.clone();
        assert_eq!(super::released(&held), None);
        super::mark(&texture, super::Path::Sweep);
        drop(texture);
        assert_eq!(super::released(&held), Some((crate::frame::current(), super::Path::Sweep)));
        drop(held);
        // A texture made after it (maybe at the same address) is not marked.
        assert_eq!(super::released(&target(&device)), None);
    }
}
