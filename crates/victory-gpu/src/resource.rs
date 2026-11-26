//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use victory_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// The host dimension of a title resource: 1D textures (and 1D arrays) are
/// 2D on the host, one texel high. The console reads a 1D texture with 2D
/// image instructions too (its hardware goes by the descriptor and ignores
/// the extra coordinate); D3D12 wants the shader's resource type to match
/// the view's, and the view's the texture's, else the read is undefined (on
/// an RTX 5090 every lane read 0.0; GPU-based validation id 940). So the
/// textures are made 2D, the title's 1D views of them become 2D views
/// (`descriptor::host`), and the translation declares every 1D image 2D.
pub(crate) fn host_dimension(dimension: u32) -> u32 {
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    if dimension == TEXTURE_1D { TEXTURE_2D } else { dimension }
}

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::for_gpu(
        &TextureDesc {
            dimension,
            width: u32::try_from(desc.width).ok()?,
            height: desc.height,
            depth_or_array: u32::from(desc.depth_or_array_size),
            mips: u32::from(desc.mip_levels),
            format: desc.format,
            samples: desc.sample_count.max(1),
            swizzle,
        },
        crate::console::current().pipes(),
        0,
    )
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB).
        // The title keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in.
pub(crate) fn pc_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// The format of a texture the host GPU renders to or writes as unordered
/// access, for the title's `format`. The console's GPU does both with the
/// 16-bit colour formats B5G5R5A1 (86) and B4G4R4A4 (115), and
/// the title has full-screen targets in both; a host that refuses either use
/// (NVIDIA) refused the whole texture, so every pass that wrote it drew
/// nothing. There 8-bit RGBA stands in: the same channels in the same
/// order, every 4-bit level exact and every 5-bit level within 1/510.
/// The title also renders to R9G9B9E5 (67), which no PC GPU does: 16-bit
/// float RGBA stands in, with at least R9G9B9E5's precision and range.
/// The title's pipelines and views of such targets name the stand-in too
/// ([`view_format`], the host view checks).
/// shortcut: copies between such a target and a texture the title fills
/// itself in the 16-bit format are skipped ([`stands_in`]); convert them
/// if a pass turns out to need one.
pub(crate) fn target_format(format: u32) -> u32 {
    const STAND_INS: [(u32, u32); 3] = [(86, 28), (115, 28), (67, 10)];
    let Some(i) = STAND_INS.iter().position(|&(f, _)| f == format) else {
        return format;
    };
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::*;
        static REFUSED: std::sync::OnceLock<[bool; 3]> = std::sync::OnceLock::new();
        let refused = REFUSED.get_or_init(|| {
            STAND_INS.map(|(f, to)| {
                let Some(host) = crate::host::host() else {
                    return false;
                };
                let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(f as i32),
                    ..Default::default()
                };
                // SAFETY: a plain query into our own structure.
                let asked = unsafe {
                    host.device.CheckFeatureSupport(
                        D3D12_FEATURE_FORMAT_SUPPORT,
                        (&raw mut support).cast(),
                        size_of_val(&support) as u32,
                    )
                };
                let both = D3D12_FORMAT_SUPPORT1_RENDER_TARGET.0
                    | D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0;
                let refused = asked.is_err() || support.Support1.0 & both != both;
                if refused {
                    tracing::info!(format = f, stand_in = to, "the host does not render to this format: a stand-in is used");
                }
                refused
            })
        });
        if refused[i] {
            return STAND_INS[i].1;
        }
    }
    format
}

/// Whether the host copy of the title texture `this` is in a stand-in
/// format ([`target_format`]): its texels are not laid out as the title's.
#[cfg(windows)]
pub(crate) fn stands_in(this: This) -> bool {
    if this.is_null() {
        return false;
    }
    let own = pc_format(resource(this).desc.format);
    target_format(own) != own
        && host_resource(this)
            // SAFETY: a live resource.
            .is_some_and(|host| unsafe { host.GetDesc() }.Format.0 as u32 == target_format(own))
}

/// Depth textures the title also asks unordered access for (the console
/// allows both; the host copy loses the unordered access): how many were
/// made, and how often such a texture was bound as the depth target,
/// cleared as depth, and asked for an unordered access view the host could
/// not give. Measured to decide what such textures need (a colour twin, or
/// to be colour textures altogether).
pub(crate) static DEPTH_UNORDERED: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Whether the title asked for both depth and unordered access for `this`.
#[cfg(windows)]
pub(crate) fn depth_and_unordered(this: This) -> bool {
    !this.is_null() && resource(this).desc.flags & 6 == 6
}

/// The colour twin of a depth texture the title also writes as unordered
/// access. The console allows both on one texture; the host gives a depth
/// texture no unordered access, so shaders write a colour texture of the
/// same size and format family instead, and one is copied into the other
/// when a side is used after the other was written
/// (`HostList::twin_use`).
/// shortcut: whole textures are copied, not the mips and slices written.
/// A depth texture with a stencil plane gets a twin for each plane (
/// the title's tiled lighting writes each pixel's material class into
/// the depth plane and masks into the stencil plane, then draws each class
/// with the depth test "equal"): the stencil plane's is copied through a
/// buffer, the depth plane's is drawn into the plane by a pass of ours
/// (`HostList::pass_plane`). Unordered access views of a multisampled
/// texture stay null.
#[cfg(windows)]
pub(crate) struct Twin {
    /// The colour texture and the state it is in, as a host copy's.
    pub(crate) colour: std::sync::Arc<HostTexture>,
    /// Which side lacks what the other was written last, after the lists
    /// submitted so far: bit 0 the depth texture, bit 1 the twin.
    pub(crate) behind: std::sync::atomic::AtomicU32,
    /// For a depth texture with a stencil plane, `colour` is the twin of
    /// that plane (8 bits a texel) and this the buffer the two are copied
    /// through: no colour format shares a family with a depth-stencil
    /// format, but the host copies between a buffer and one plane.
    pub(crate) through: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// For a depth texture with a stencil plane: the twin of its depth
    /// plane, 16 bits a texel for the console's 16-bit depth-stencil
    /// formats and a 32-bit float otherwise. No host colour format holds
    /// the bits of a 24-bit depth plane, so the two are not copied: a pass
    /// draws the twin into the plane.
    pub(crate) plane: Option<std::sync::Arc<HostTexture>>,
}

#[cfg(windows)]
impl Drop for Twin {
    fn drop(&mut self) {
        TWINS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Colour twins alive, depth textures with unordered access left without
/// one, and the copies recorded into depth textures and into twins since
/// the last frame-rate line.
#[cfg(windows)]
pub(crate) static TWINS: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Depth planes drawn from their twins by the pass since the last
/// frame-rate line; and since the start, how often a depth plane's twin
/// lacked what was written to the plane (not given to it), and how often a
/// depth plane was used in the list that had written its twin, before the
/// pass (which runs when the list closes).
#[cfg(windows)]
pub(crate) static PLANES: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The colour twin of title texture `this`, if it has one.
#[cfg(windows)]
pub(crate) fn twin(this: This) -> Option<std::sync::Arc<Twin>> {
    if this.is_null() {
        return None;
    }
    hosted(this, true).and_then(|made| made.1)
}

/// The twin's texture, for the unordered access views of `this`.
#[cfg(windows)]
pub(crate) fn twin_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    let (_, twin) = hosted(this, true)?;
    // SAFETY: as in `host_resource`: the twin lives as long as the host
    // copy, which a depth texture keeps until the title releases it.
    twin.map(|t| unsafe { &*std::ptr::from_ref(&t.colour.texture) })
}

/// The texture of the twin of the depth plane of `this`, a depth texture
/// with a stencil plane, for the unordered access views of that plane.
#[cfg(windows)]
pub(crate) fn twin_plane_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    let (_, twin) = hosted(this, true)?;
    let twin = twin?;
    // SAFETY: as in `twin_resource`.
    twin.plane
        .as_ref()
        .map(|p| unsafe { &*std::ptr::from_ref(&p.texture) })
}

/// The colour twin for a texture described by `desc` whose host copy is
/// `depth`, if the title asks for depth and unordered access both and the
/// host can copy between the two: the same size, mips and slices in the
/// depth format's typeless family (R16 for 16-bit depth, R32 for 32-bit),
/// or R8 for the stencil plane of a format that has one, with unordered
/// access only. A format with a stencil plane gets a twin for its depth
/// plane too, R16 for the console's 16-bit depth-stencil formats and R32
/// otherwise. `VICTORY_DEPTH_TWINS=0` makes none (such views stay null),
/// `VICTORY_DEPTH_TWINS=stencil` only those of formats with a stencil
/// plane, `VICTORY_DEPTH_TWINS=nodepth` none for the depth planes of such
/// formats.
#[cfg(windows)]
fn twin_texture(
    desc: &ResourceDesc,
    depth: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
) -> Option<std::sync::Arc<Twin>> {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::Win32::Graphics::Direct3D12::*;
    const DEPTH_AND_UNORDERED: u32 = 6;
    const UNORDERED: u32 = 4;
    const UNORDERED_ACCESS: u32 = 0x8;
    // R32 and R16 typeless: the families of the depth formats of one plane.
    const ONE_PLANE: [u32; 2] = [39, 53];
    if desc.dimension == BUFFER || desc.flags & DEPTH_AND_UNORDERED != DEPTH_AND_UNORDERED {
        return None;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_none_or(|v| v != "0")) {
        return None;
    }
    // R32G8X24 and R24G8 typeless: depth with a stencil plane, whose twin
    // is R8 typeless.
    const WITH_STENCIL: [u32; 2] = [19, 44];
    const R8_TYPELESS: u32 = 60;
    let typeless = family(pc_format(desc.format));
    let stencil = WITH_STENCIL.contains(&typeless);
    static STENCIL_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let stencil_only = *STENCIL_ONLY
        .get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_some_and(|v| v == "stencil"));
    if !stencil && stencil_only {
        return None;
    }
    if !(stencil || ONE_PLANE.contains(&typeless)) || desc.sample_count > 1 {
        TWINS[1].fetch_add(1, Relaxed);
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags = UNORDERED;
    d.format = if stencil { R8_TYPELESS } else { typeless };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description has the host layout, as in `host_texture`;
    // the out pointer is ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            (&raw const d).cast(),
            D3D12_RESOURCE_STATES(UNORDERED_ACCESS as i32),
            None,
            &mut texture,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no colour twin for a depth texture");
    }
    let Some(texture) = texture else {
        TWINS[1].fetch_add(1, Relaxed);
        return None;
    };
    crate::host::name(&texture, || {
        format!("depth twin {}x{} f{}", d.width, d.height, d.format)
    });
    // The buffer a stencil plane and its twin are copied through: as big
    // as the plane's mips and slices laid out for a copy (they follow the
    // depth plane's in the texture's subresources).
    let through = if stencil {
        // SAFETY: a live texture.
        let of = unsafe { depth.GetDesc() };
        let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
        let mut bytes = 0;
        // SAFETY: a plain query; the out pointer is ours.
        unsafe {
            host.device.GetCopyableFootprints(
                &of,
                planes,
                planes,
                0,
                None,
                None,
                None,
                Some(&mut bytes),
            );
        }
        let Some(buffer) = host.uav_buffer(bytes) else {
            TWINS[1].fetch_add(1, Relaxed);
            return None;
        };
        Some(buffer)
    } else {
        None
    };
    // The twin of the depth plane of a format with a stencil plane: R16
    // typeless (53) for the console's 16-bit depth-stencil formats (118 to
    // 120), whose shaders write 16-bit steps, R32 typeless (39) otherwise.
    static NO_PLANE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let no_plane = *NO_PLANE
        .get_or_init(|| std::env::var_os("VICTORY_DEPTH_TWINS").is_some_and(|v| v == "nodepth"));
    let plane = if stencil && !no_plane {
        d.format = if (118..=120).contains(&desc.format) {
            53
        } else {
            39
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: as for the first twin.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                (&raw const d).cast(),
                D3D12_RESOURCE_STATES(UNORDERED_ACCESS as i32),
                None,
                &mut texture,
            )
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, desc = ?d, "no twin for a depth plane");
        }
        texture.map(|texture| {
            crate::host::name(&texture, || {
                format!("depth plane twin {}x{} f{}", d.width, d.height, d.format)
            });
            std::sync::Arc::new(HostTexture::new(texture, UNORDERED_ACCESS))
        })
    } else {
        None
    };
    TWINS[0].fetch_add(1, Relaxed);
    Some(std::sync::Arc::new(Twin {
        colour: std::sync::Arc::new(HostTexture::new(texture, UNORDERED_ACCESS)),
        behind: std::sync::atomic::AtomicU32::new(0),
        through,
        plane,
    }))
}

/// The typeless format of a format's family (itself if it has none).
/// A8_UNORM (65) has none: the host casts it to no R8 format, and an R8
/// view reads its bytes as red where A8 has them as alpha (views of one
/// read as the other: `descriptor::host::other_channels`).
pub(crate) fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 64),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
pub(crate) fn view_format(this: This) -> Option<u32> {
    // First: a view of no resource (a typed null view) has none.
    let host = host_resource(this)?;
    let own = pc_format(resource(this).desc.format);
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    // A stand-in target is typed.
    (format != own && format != target_format(own)).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

impl PlacedFootprint {
    /// Its format and its texels: width, height, depth.
    pub(crate) fn extent(&self) -> (u32, [u32; 3]) {
        (self.format, [self.width, self.height, self.depth])
    }

    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }

    /// For a read back through a host buffer: this footprint at offset 0
    /// with its rows pitched as the host wants them (multiples of 256
    /// bytes), then the bytes that buffer needs, the bytes of a row, the
    /// rows, and the title's own pitch.
    pub(crate) fn host_rows(&self) -> Option<(PlacedFootprint, [u64; 4])> {
        let (element, block) = element(self.format)?;
        let lines = (u64::from(self.height).div_ceil(block) * u64::from(self.depth)).max(1);
        let row = u64::from(self.width).div_ceil(block) * element;
        let pitch = row.max(1).next_multiple_of(256);
        let host = PlacedFootprint {
            offset: 0,
            row_pitch: u32::try_from(pitch).ok()?,
            ..*self
        };
        Some((host, [pitch * lines, row, lines, u64::from(self.row_pitch)]))
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's size in an upload buffer: footprint (offset 0), rows
/// and bytes per row. Planes of depth-stencil formats are not told apart.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let (mip, slice) = (index % mips, index / mips);
    if slice >= slices {
        return None;
    }
    let (element, block) = element(desc.format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format: desc.format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(PITCH)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Device method: where each subresource goes in an upload buffer, rows
/// 256-byte and subresources 512-byte aligned, as on PC. Unknown sizes give
/// the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (the title places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        // The big ones are named in the log: the memory line counts our own
        // allocations by size, and these are the title's memory given out
        // of ours (a heap it made through the device, a descriptor heap).
        if info.size >= 32 << 20 {
            tracing::info!(mb = info.size >> 20, at = ?ptr, "memory for the title from our allocator (a heap or a descriptor heap)");
        }
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet. With, for a depth texture the title also writes as unordered
    /// access, the colour texture shaders write instead ([`Twin`]).
    #[cfg(windows)]
    host: Hosted,
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
    /// The bytes the host gives it.
    bytes: u64,
    /// Whether its contents come from title memory ([`fills`]): counted in
    /// [`KINDS`] while it lives.
    filled: bool,
    /// Made without memory of its own (`crate::tiles`): the tiles its
    /// levels have been given. Its `bytes` are 0: what it holds of the card
    /// is counted there.
    pub(crate) tiled: Option<crate::tiles::Tiled>,
    /// For such a texture: kept for reuse when it goes ([`keep_reserved`]).
    reuse: Option<ReuseKey>,
    /// Made in the PC's memory the card reads over the bus ([`spill_on`]):
    /// counted in [`SPILLED`] while it lives, not as the card's.
    spilled: bool,
    /// What the card itself wrote of it since the title last wrote the
    /// texture's memory (`crate::texture::CardState`): uploads from title
    /// memory leave that alone.
    pub(crate) card: crate::texture::CardState,
    /// What it holds of each part of the texture's memory
    /// (`crate::texture::Sent`): uploads bring the levels it lacks. Shared
    /// with the texture's staging buffer.
    pub(crate) sent: std::sync::Arc<crate::texture::Sent>,
}

/// What a title texture has on the host: its host copy and, for a depth
/// texture shaders also write, the colour twin.
#[cfg(windows)]
type Made = (std::sync::Arc<HostTexture>, Option<std::sync::Arc<Twin>>);

/// A title texture's host side. The host copy is made when something first
/// uses the texture, not when the title creates it: the title
/// had created 3,871 textures (9.2 GB on the host) before its world's first
/// frame, of which a frame's lists name some hundreds (run 23), on a card
/// with 12 GB. A title texture is memory the title has either way; its host
/// copy is the card's.
#[cfg(windows)]
struct Hosted {
    /// The host copy (and twin) once made. Taken out again only by
    /// [`let_go_idle`] and when the title releases the resource.
    made: std::sync::Mutex<Option<Made>>,
    /// The state and clear value the title created the texture with.
    state: u32,
    clear: Option<[u32; 5]>,
    /// When the host copy was last asked for (`crate::host::moment`),
    /// written under `made`'s lock.
    used: std::sync::atomic::AtomicU64,
    /// The host refused to make it: not tried again.
    refused: std::sync::atomic::AtomicBool,
    /// Its host copy was let go as idle once.
    gone: std::sync::atomic::AtomicBool,
    /// A texture of [`PLACED_TEXTURES`].
    placed: std::sync::atomic::AtomicBool,
    /// When the sweep first found its host copy held by nothing (by list
    /// or slot) since it was last found held (`crate::host::moment`; 0:
    /// held, or not looked at yet): [`let_go_idle`].
    unheld: std::sync::atomic::AtomicU64,
}

/// Counts for the memory line: host copies of textures the title fills
/// alive and their bytes; such copies let go as idle since the start and
/// their bytes; made again after that and their bytes; and host copies made
/// since the start.
#[cfg(windows)]
pub(crate) static KINDS: [std::sync::atomic::AtomicU64; 7] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Nanoseconds spent making host textures of the title's placed textures,
/// and how many, since the line that says so.
#[cfg(windows)]
pub(crate) static MAKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Every title texture alive (not buffers), by address: for the sweep and
/// the memory line. A texture leaves the table before it is freed.
#[cfg(windows)]
static TEXTURES: std::sync::Mutex<std::collections::BTreeSet<usize>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

#[cfg(windows)]
fn textures() -> std::sync::MutexGuard<'static, std::collections::BTreeSet<usize>> {
    TEXTURES.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title textures the host renders to or writes (targets, depth,
/// unordered access), by the address of their memory: what a view the
/// title places in that memory in another format shows ([`alias_of`]). A
/// texture leaves the table before it is freed.
#[cfg(windows)]
static TARGETS_AT: std::sync::Mutex<std::collections::BTreeMap<u64, usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// For `this`, a texture made for a view the title placed in memory
/// ([`texture_at`]): the title texture the host writes that lies in the
/// same memory, if there is one of the same size, levels and layers with
/// texels of the same width. The console reads such a texture through the
/// view as it is
/// (the title packs light as R9G9B9E5 into R32_UINT textures
/// with compute shaders, which cannot write R9G9B9E5, and samples them
/// through R9G9B9E5 views placed at the textures' addresses: the sky, the
/// sky and light probes, the bounced light the lighting pass adds). On the
/// host the two are two textures and what the shaders wrote exists in one
/// of them only: the other was filled from title memory, which nothing
/// writes, so everything read through such a view was black.
/// Textures of several levels or layers count too (the screen probes'
/// light atlas has four levels, the camera's sky light cache six layers,
/// the cloud and sky tables are volumes): only textures of one level and
/// one layer were found at first, and the programs behind the others read
/// the title's unwritten memory - the sky light cache's second stage wrote
/// all zeros from a first stage that held light (run 88).
/// `VICTORY_ALIAS_COPY=0`: none is found, as before; `=1`: only textures
/// of one level and one layer, as at first.
#[cfg(windows)]
pub(crate) fn alias_of(this: This) -> Option<This> {
    use std::sync::atomic::Ordering::Relaxed;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if this.is_null()
        || !*ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_COPY").is_none_or(|v| v != "0"))
    {
        return None;
    }
    static WHOLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let whole =
        *WHOLE.get_or_init(|| std::env::var_os("VICTORY_ALIAS_COPY").is_none_or(|v| v != "1"));
    let r = resource(this);
    // A level count of 0 asks for a full chain: not counted here.
    let one = |d: &ResourceDesc| {
        d.dimension != BUFFER
            && d.mip_levels >= 1
            && d.depth_or_array_size >= 1
            && (whole || (d.mip_levels == 1 && d.depth_or_array_size == 1))
            && d.sample_count <= 1
    };
    if !r.host.placed.load(Relaxed) || !fills(&r.desc) || !one(&r.desc) {
        return None;
    }
    // Under the table's lock: a texture leaves it before it is freed.
    let targets = TARGETS_AT.lock().unwrap_or_else(|e| e.into_inner());
    let other = *targets.get(&(r.data() as u64))? as This;
    let o = resource(other);
    const DEPTH: u32 = 0x2;
    let same = one(&o.desc)
        && o.desc.dimension == r.desc.dimension
        && o.desc.width == r.desc.width
        && o.desc.height == r.desc.height
        && o.desc.mip_levels == r.desc.mip_levels
        && o.desc.depth_or_array_size == r.desc.depth_or_array_size
        && o.desc.flags & DEPTH == 0
        && target_format(o.desc.format) == o.desc.format
        && element(o.desc.format).is_some_and(|e| e.1 == 1 && Some(e) == element(r.desc.format));
    same.then_some(other)
}

/// Whether host copies are made on first use. On by default;
/// `VICTORY_LAZY_TEXTURES=0` makes each when the title creates its texture,
/// as before.
#[cfg(windows)]
fn lazy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_TEXTURES").is_none_or(|v| v != "0"))
}

/// Whether a texture described by `desc` takes its contents from title
/// memory: one the host neither renders to nor writes (no target, depth or
/// unordered access flag). Its host copy can be made again from that
/// memory; any other texture's contents exist on the host alone.
#[cfg(windows)]
fn fills(desc: &ResourceDesc) -> bool {
    const TARGET_DEPTH_UNORDERED: u32 = 0x7;
    desc.dimension != BUFFER && desc.flags & TARGET_DEPTH_UNORDERED == 0
}

/// Whether the title fills texture `this` itself ([`fills`]).
#[cfg(windows)]
pub(crate) fn title_fills(this: This) -> bool {
    !this.is_null() && fills(&resource(this).desc)
}

/// Width, height and format of `this` if it is a 2D texture of one sample
/// and not depth (`None` else), for the kinds `crate::texback` copies back.
#[cfg(windows)]
pub(crate) fn texture_kind(this: This) -> Option<(u64, u32, u32)> {
    const TEXTURE_2D: u32 = 3;
    const DEPTH: u32 = 0x2;
    if this.is_null() {
        return None;
    }
    let d = &resource(this).desc;
    (d.dimension == TEXTURE_2D && d.sample_count <= 1 && d.flags & DEPTH == 0).then_some((d.width, d.height, d.format))
}

/// Texture `this`'s console layout (one plane), where it lies in title
/// memory and the bytes that layout takes, for `crate::texback`; `None` for
/// a buffer or a texture of several planes.
#[cfg(windows)]
pub(crate) fn title_texture(this: This) -> Option<(texture::Layout, usize, usize)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    if r.desc.dimension == BUFFER {
        return None;
    }
    let t = texture_layout(&r.desc)?;
    let [plane] = &t.planes[..] else {
        return None;
    };
    Some((plane.layout.clone(), r.data() as usize, usize::try_from(t.size).ok()?))
}

/// Whether `this` is a texture whose host copy is not made yet (and may
/// still be): a descriptor slot written with a view of it waits
/// (`crate::descriptor::make_waiting`).
#[cfg(windows)]
pub(crate) fn host_waits(this: This) -> bool {
    if this.is_null() || !lazy() {
        return false;
    }
    let r = resource(this);
    r.desc.dimension != BUFFER
        && !r.host.refused.load(std::sync::atomic::Ordering::Relaxed)
        && r.host.made.lock().unwrap_or_else(|e| e.into_inner()).is_none()
}

/// The host side of title texture `this`: made now if it is not there and
/// `make` says so. `None` for buffers and for textures the host refuses.
#[cfg(windows)]
fn hosted(this: This, make: bool) -> Option<Made> {
    use std::sync::atomic::Ordering::Relaxed;
    let r = resource(this);
    if r.desc.dimension == BUFFER {
        return None;
    }
    let mut made = r.host.made.lock().unwrap_or_else(|e| e.into_inner());
    if made.is_none() {
        if !make || r.host.refused.load(Relaxed) {
            return None;
        }
        let clear = r
            .host
            .clear
            .as_ref()
            .map_or(std::ptr::null(), |c| std::ptr::from_ref(c).cast::<c_void>());
        // Levels on demand (`crate::tiles`): a texture the title places in
        // memory and fills, of a megabyte or more, that does not hold all
        // its levels now, is made without memory and given it level by
        // level. It does not hold them all if its memory is not all
        // mapped, if the next texture starts inside its span, or if a
        // view of it starts below level 0. Every other texture, and this
        // one if the card will not have it so, is made as before.
        const TEXTURE_2D: u32 = 3;
        let mut tiled = None;
        let mut reused_state = None;
        let mut texture = None;
        let shape_fits = fills(&r.desc)
            && r.desc.dimension == TEXTURE_2D
            && r.desc.depth_or_array_size == 1
            && r.desc.mip_levels > 1
            && r.desc.sample_count <= 1;
        // Without its most detailed levels (`crate::tiles::DropPolicy`):
        // how many this texture leaves out, by the bytes of host copies of
        // title textures so far. Such a texture is made the same way.
        let mut leave_out = 0;
        if shape_fits && r.host.placed.load(Relaxed) && crate::tiles::card_can() {
            let shape = crate::tiles::Shape {
                width: r.desc.width,
                height: r.desc.height,
                levels: u32::from(r.desc.mip_levels),
                layers: u32::from(r.desc.depth_or_array_size),
                plain_2d: true,
                block_compressed: block_compressed(pc_format(r.desc.format)),
            };
            leave_out = drop_policy().levels(&shape, filled_for_policy());
        }
        // Levels are left out by the policy, as without the spill; a
        // texture the title fills that finds no room on the card even so
        // goes whole into the PC's memory the card reads over the bus
        // ([`spill_on`], [`placement`]).
        let spill = fills(&r.desc)
            && placement(room_lately(), spill_on() && shared_room_lately()) == Placement::Shared;
        if spill {
            leave_out = 0;
        }
        if shape_fits && r.host.placed.load(Relaxed) && (crate::tiles::wanted() || leave_out > 0) {
            let bytes = footprint(&r.desc).map_or(0, |f| f.size);
            let start = r.data() as u64;
            // What the upload can read: a console layout, no XOR mode
            // (`crate::texture` `staging`).
            let readable = texture_layout(&r.desc).is_some_and(|t| !t.planes[0].layout.desc.swizzle.xor());
            let partly = crate::tiles::wanted()
                && readable
                && bytes >= crate::tiles::floor()
                && (!crate::texture::committed(r.data().cast_const(), bytes as usize)
                    || crate::descriptor::host::next_start(start, start.saturating_add(bytes)).is_some()
                    || crate::descriptor::host::first_viewed(start).is_some_and(|first| first > 0));
            // One kept from a texture of the same description that went
            // (`take_reserved`), or a new one.
            let reserved = if partly || leave_out > 0 {
                match take_reserved(ReuseKey::of(&r.desc)) {
                    Some((texture, state)) => {
                        RESERVED_REUSE[0].fetch_add(1, Relaxed);
                        reused_state = Some(state);
                        Some(texture)
                    }
                    None => {
                        RESERVED_REUSE[1].fetch_add(1, Relaxed);
                        host_texture_as(&r.desc, r.host.state, clear, true, false, false).map(|(t, _)| t)
                    }
                }
            } else {
                None
            };
            let cut = reserved
                .as_ref()
                .zip(crate::host::host())
                .and_then(|(t, host)| crate::tiles::Tiled::make(&host.device, t, u32::from(r.desc.mip_levels)));
            // (Both or neither: a reserved texture the card will not cut
            // into tiles is dropped here, and the old kind is made below.)
            if let Some(cut) = cut {
                cut.leave_out(leave_out);
                tiled = Some(cut);
                texture = reserved;
            } else {
                reused_state = None;
            }
        }
        // A texture the title views in its own memory, whose contents are
        // uploaded whole, may be placed in memory another had. (One that is
        // not uploaded must read zeros.)
        let placed = r.host.placed.load(Relaxed)
            && crate::placed::on()
            && fills(&r.desc)
            && texture_layout(&r.desc).is_some_and(|t| !t.planes[0].layout.desc.swizzle.xor());
        let began = std::time::Instant::now();
        let (texture, spilled) = match texture {
            Some(texture) => (Some(texture), false),
            None => host_texture(&r.desc, r.host.state, clear, placed, spill).map_or((None, false), |(t, s)| (Some(t), s)),
        };
        if r.host.placed.load(Relaxed) {
            MAKING[0].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
            MAKING[1].fetch_add(1, Relaxed);
        }
        let Some(texture) = texture else {
            r.host.refused.store(true, Relaxed);
            return None;
        };
        let twin = twin_texture(&r.desc, &texture);
        // One line per depth texture with unordered access: its address is
        // the one target dumps and the captured frame's list of commands
        // name.
        if r.desc.flags & 6 == 6 {
            tracing::info!(
                resource = ?this,
                desc = ?r.desc,
                twin = twin.is_some(),
                "depth texture with unordered access"
            );
        }
        // (A texture kept for reuse is in the state its last list left it.)
        let state = reused_state
            .unwrap_or_else(|| crate::host::legal_state(&texture, crate::host::texture_state(r.host.state)));
        let mut host = match tiled {
            Some(tiled) => HostTexture::new_tiled(texture, state, tiled, ReuseKey::of(&r.desc)),
            None => HostTexture::new(texture, state),
        };
        host.filled = fills(&r.desc);
        if host.filled {
            KINDS[0].fetch_add(1, Relaxed);
            KINDS[1].fetch_add(host.bytes, Relaxed);
        }
        if spilled {
            host.spilled = true;
            SPILLED[0].fetch_add(1, Relaxed);
            SPILLED[1].fetch_add(host.bytes, Relaxed);
            SPILLED[2].fetch_add(1, Relaxed);
            SPILLED[3].fetch_add(host.bytes, Relaxed);
        }
        if r.host.gone.load(Relaxed) {
            KINDS[4].fetch_add(1, Relaxed);
            KINDS[5].fetch_add(host.bytes, Relaxed);
        }
        KINDS[6].fetch_add(1, Relaxed);
        let host = std::sync::Arc::new(host);
        if host.tiled.as_ref().is_some_and(|t| t.dropped() > 0) {
            REDUCED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((r.data() as u64, std::sync::Arc::downgrade(&host)));
        }
        *made = Some((host, twin));
        // Slots the title wrote with views of it while it had no host copy
        // are made before the next submission runs.
        crate::descriptor::texture_made(this as usize);
    }
    let moment = crate::host::moment();
    r.host.used.store(moment, Relaxed);
    if let Some((host, _)) = made.as_ref()
        && let Some(tiled) = &host.tiled
    {
        tiled.touch(moment);
    }
    made.clone()
}

/// Host copies made without their most detailed levels
/// (`crate::tiles::DropPolicy`) that may still get them back
/// ([`promote_reduced`]): where the title texture's memory starts, and
/// the copy (gone once the title lets the texture go and no list holds
/// it).
#[cfg(windows)]
static REDUCED: std::sync::Mutex<Vec<(u64, std::sync::Weak<HostTexture>)>> = std::sync::Mutex::new(Vec::new());

/// Whether a host format (DXGI) is block-compressed: BC1 to BC7.
#[cfg(windows)]
fn block_compressed(format: u32) -> bool {
    matches!(format, 70..=84 | 94..=99)
}

/// The policy of host copies made without their most detailed levels
/// (`crate::tiles::DropPolicy`), settled at its first use (by then the
/// host device exists and says the card's budget), from the switches:
/// `VICTORY_TEXTURE_DROP_LEVELS` (`0`: never, as before; `auto`, the
/// default; `force1`, `force2`: from every large texture whatever the
/// room), `VICTORY_TEXTURE_DROP_MIN_SIDE`, `VICTORY_TEXTURE_DROP_MIN_LEVELS`,
/// `VICTORY_TEXTURE_DROP_FROM` (percent of the line, `50,80`) and
/// `VICTORY_TEXTURE_DROP_LINE_MB`. The line, the bytes of host copies of
/// title textures the card has room for: `VICTORY_TEXTURE_DROP_LINE_MB`,
/// or `VICTORY_TEXTURE_BUDGET_MB` if that is set, or else the share of the
/// card's budget up to which such copies are made ([`room_percent`]) less
/// what the other things on the card take, measured in the runs of the
/// first room (a budget of 10,131 to 10,659 MB): render targets and depth
/// up to [`targets_room_kept`]'s 3,072 MB, card copies of title buffers
/// about 1,000 MB, and about 1,200 MB of everything else (pipelines, heaps,
/// lists, staging). A budget of 10,400 MB gives a line of about 4,060 MB;
/// textures start leaving a level out at about 2,030 MB, two from about
/// 3,250 MB. (Settled once: the budget the system gives moves a little
/// from run to run, and the line should not move under the textures.)
#[cfg(windows)]
fn drop_policy() -> &'static crate::tiles::DropPolicy {
    static POLICY: std::sync::OnceLock<crate::tiles::DropPolicy> = std::sync::OnceLock::new();
    POLICY.get_or_init(|| {
        const MB: u64 = 1 << 20;
        const OTHERS: u64 = (3072 + 1000 + 1200) * MB;
        let env_mb = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|mb| mb * MB);
        let budget = crate::host::host().and_then(|h| h.video_memory()).map(|[budget, ..]| budget);
        let line = env_mb("VICTORY_TEXTURE_DROP_LINE_MB")
            .or_else(|| env_mb("VICTORY_TEXTURE_BUDGET_MB"))
            .or_else(|| budget.map(|b| (b / 100 * room_percent()).saturating_sub(OTHERS)))
            .unwrap_or(4096 * MB)
            .max(1024 * MB);
        let var = |name: &str| std::env::var(name).ok();
        let policy = crate::tiles::DropPolicy::parse(
            var("VICTORY_TEXTURE_DROP_LEVELS").as_deref(),
            var("VICTORY_TEXTURE_DROP_MIN_SIDE").as_deref(),
            var("VICTORY_TEXTURE_DROP_MIN_LEVELS").as_deref(),
            var("VICTORY_TEXTURE_DROP_FROM").as_deref(),
            line,
        );
        tracing::info!(
            mode = ?policy.mode,
            line_mb = line >> 20,
            budget_mb = budget.map(|b| b >> 20),
            one_level_from_mb = policy.one_from >> 20,
            two_levels_from_mb = policy.two_from >> 20,
            min_side = policy.min_side,
            min_levels = policy.min_levels,
            tiled_resources_tier_2 = crate::tiles::card_can(),
            "host copies of large block-compressed title textures are made without their most detailed levels past a line (VICTORY_TEXTURE_DROP_LEVELS)"
        );
        policy
    })
}

/// The bytes of host copies of title textures the policy goes by: those
/// made whole, the tiles given to those made level by level, and what the
/// ones made that way and not uploaded yet will take.
#[cfg(windows)]
fn filled_for_policy() -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    (KINDS[1].load(Relaxed) + crate::tiles::given_total() + crate::tiles::STATS[11].load(Relaxed))
        .saturating_sub(SPILLED[1].load(Relaxed))
}

/// Brings levels back to host copies made without them, while the card
/// has room (`room`, as the texture pass found it) and no texture picked
/// by index waits for room (`waiting`: slots the pass held back): the
/// copies asked for most lately first, a level each, while the bytes of
/// host copies of title textures stay under the policy's first line
/// (where textures begin to leave levels out), within
/// `VICTORY_TEXTURE_PROMOTE_MB` (64) a frame, and at most every half
/// second. A copy not asked for in the last 10 s is left as it is. A level
/// brought back is given tiles and written by the texture's next upload,
/// and the views go down to it after that
/// (`crate::descriptor::host::levels_changed`); the texture's slots are
/// made due again here for that upload (`levels_waiting`), and a texture
/// read through tables uploads again by itself (`texture.rs` `upload`).
/// Returns the levels brought back.
#[cfg(windows)]
pub(crate) fn promote_reduced(room: bool, waiting: u64) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static PER_FRAME: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let per_frame = *PER_FRAME.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_PROMOTE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64) << 20
    });
    // (`room` may be the PC's memory's: levels brought back are the card's.)
    if !room || waiting > 0 || per_frame == 0 || crate::tiles::STATS[12].load(Relaxed) == 0 || !room_lately() {
        return 0;
    }
    let now = crate::host::now_ms();
    if now < NEXT.load(Relaxed) {
        return 0;
    }
    NEXT.store(now + 500, Relaxed);
    let policy = drop_policy();
    if policy.mode != crate::tiles::DropMode::Auto {
        return 0;
    }
    let mut candidates: Vec<(u64, u64, std::sync::Arc<HostTexture>)> = {
        let mut reduced = REDUCED.lock().unwrap_or_else(|e| e.into_inner());
        reduced.retain(|(_, weak)| weak.upgrade().is_some_and(|h| h.tiled.as_ref().is_some_and(|t| t.dropped() > 0)));
        reduced
            .iter()
            .filter_map(|(memory, weak)| {
                let host = weak.upgrade()?;
                let used = host.tiled.as_ref()?.last_used();
                Some((used, *memory, host))
            })
            .collect()
    };
    candidates.retain(|(used, ..)| !crate::host::long_ago(*used, 10_000, 2));
    // Most lately asked for first; by memory where that is the same, so
    // the order does not hang on the table's.
    candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut filled = filled_for_policy();
    let mut spent = 0u64;
    let mut promoted = 0;
    for (_, memory, host) in candidates {
        let Some(tiled) = &host.tiled else {
            continue;
        };
        let bytes = tiled.promotion_bytes();
        if bytes == 0 || spent + bytes > per_frame || filled + bytes >= policy.one_from {
            continue;
        }
        if tiled.promote().is_some() {
            spent += bytes;
            filled += bytes;
            promoted += 1;
            crate::descriptor::host::levels_waiting(memory);
        }
    }
    promoted
}

/// The host copy of title texture `this` if it is made, without making it
/// (for diagnostics, which must not count as a use).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn host_made<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    let (host, _) = hosted(this, false)?;
    // SAFETY: as in `host_resource`.
    Some(unsafe { &*std::ptr::from_ref(&host.texture) })
}

/// Whether more host copies of textures the title fills may be made for
/// shaders that pick textures by index (`draw.rs` `heap_textures`): while
/// the card has room for them ([`card_room`]) or, past that, the PC's
/// memory the card reads over the bus has ([`spill_on`]).
#[cfg(windows)]
pub(crate) fn room() -> bool {
    card_room() || spill_on() && shared_room()
}

/// Whether the card has room for more host copies of textures the title
/// fills: always up to `FLOOR` bytes of them on the card, beyond that while
/// the card's memory in use by this process is under nine tenths of the
/// budget the system gives it ([`room_percent`] says another share;
/// `VICTORY_TEXTURE_BUDGET_MB` sets a limit on those copies' bytes
/// instead). Past the budget the system moves the card's memory into the
/// PC's, which ended runs 20 and 24.
#[cfg(windows)]
pub(crate) fn card_room() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    const FLOOR: u64 = 2 << 30;
    const WITHOUT_BUDGET: u64 = 4 << 30;
    static LIMIT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        let mb = std::env::var("VICTORY_TEXTURE_BUDGET_MB").ok()?;
        mb.trim().parse::<u64>().ok().map(|mb| mb << 20)
    });
    // (With the tiles of the copies made level by level, whose own bytes
    // count as none: `crate::tiles`; without the copies in the PC's
    // memory.)
    let filled = (KINDS[1].load(Relaxed) + crate::tiles::given_total()).saturating_sub(SPILLED[1].load(Relaxed));
    if let Some(limit) = limit {
        return filled < limit;
    }
    if filled < FLOOR {
        return true;
    }
    match crate::host::host().and_then(|h| h.video_memory()) {
        // What is kept for card copies of title buffers not made yet
        // (`crate::shadow::card_room_kept`) and for the render targets a
        // world makes in its first frame ([`targets_room_kept`]) counts as
        // used.
        // What the heaps of placed textures have free is room.
        Some([budget, used, ..]) => {
            used.saturating_sub(crate::placed::free_bytes()) + crate::shadow::card_room_kept() + targets_room_kept() < budget / 100 * room_percent()
        }
        None => filled < WITHOUT_BUDGET,
    }
}

/// The bytes of the card's memory the texture queue leaves free for
/// render targets and other host copies the host itself writes that are
/// not made yet: `VICTORY_CARD_ROOM_FOR_TARGETS_MB` (3,072; 0: none, as
/// before) less what such copies hold now.
///
/// Measured (runs 193, 197, 198): 652 MB of them before the world, 2,255
/// to 2,352 MB right after its first frame, 2,800 to 3,100 MB two minutes
/// on, none of them idle (run 198's census). They are made with no room
/// test, in the same frame as the card copies of title buffers (762 MB),
/// after the texture queue has filled the card to its line with textures
/// written while the world loaded (7.3 GB of them at the first world frame
/// of run 193, of which 2.3 GB were let go again within 13 seconds). The
/// card then stood at 99 to 100.5 % of its budget, buffer copies were
/// turned down for good (31 of 95 made: the frame 355 ms where 95 give
/// 238), and the queue was refused from the first world frame on. With
/// their room kept, the queue stops that much earlier while the world
/// loads and gets it back as the targets are made.
#[cfg(windows)]
pub(crate) fn targets_room_kept() -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let most = *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_ROOM_FOR_TARGETS_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(3072 << 20, |mb| mb.min(8192) << 20)
    });
    // Every host texture's bytes less those of the copies the title fills.
    let targets = HOST_TEXTURES[1].load(Relaxed).saturating_sub(KINDS[1].load(Relaxed));
    most.saturating_sub(targets)
}

/// [`card_room`] as it was at most a tenth of a second ago, for a question
/// asked of many textures a frame ([`card_room`] asks the system each
/// time).
#[cfg(windows)]
pub(crate) fn room_lately() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    // The tenth of a second of the last asking, counted from 1, and its
    // answer in the lowest bit.
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tenth = crate::host::now_ms() / 100 + 1;
    let last = LAST.load(Relaxed);
    if last >> 1 == tenth {
        return last & 1 != 0;
    }
    let room = card_room();
    LAST.store((tenth << 1) | u64::from(room), Relaxed);
    room
}

/// Where a new host copy of a texture the title fills goes
/// ([`placement`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// On the card, with the levels the policy leaves out
    /// (`crate::tiles::DropPolicy`) left out.
    Card,
    /// Whole, in the PC's memory the card reads over the bus.
    Shared,
}

/// The order as memory fills: on the card while it has room
/// ([`card_room`]), levels left out past the policy's line as without the
/// spill; once the card has no room even so, in the PC's memory while
/// that has room (`shared`, [`shared_room`]); and with neither, on the card
/// again, as before the spill (a texture picked by index waits for room
/// then, `draw.rs` `heap_textures`).
fn placement(card: bool, shared: bool) -> Placement {
    if !card && shared { Placement::Shared } else { Placement::Card }
}

/// Whether host copies of textures the title fills that find no room on
/// the card go into the PC's memory the system lets the card reach over the
/// bus, made whole, instead of waiting for room (on by default; a texture
/// there is read over the bus, slower than the card's own memory, but it is
/// there; `VICTORY_TEXTURE_SPILL=0`: they wait, as before). The last step:
/// levels are left out first ([`placement`]). Spill first, then levels,
/// measured on the RTX 4070 Ti: the opening cutscene 170 ms a frame
/// against 106, 50,457 textures made there in 205 s for 6,795 alive, and
/// the picture the same.
/// From fxceless/victory-fork pull request #601 (DiogoPCastelos: on a
/// 6 GB card skin, walls and clothes sampled black while their textures
/// waited for room; with the spill they had their colours) and
/// NitricAcidDev/VIctory-Omega-Fork 4237574084 (the textures made there
/// explicitly: on an RTX 5080 the texels right, none of the card's memory
/// taken, read at 46.9 GB/s against 218.8).
#[cfg(windows)]
pub(crate) fn spill_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_SPILL").is_none_or(|v| v != "0"))
}

/// Whether the PC's memory shared with the card has room for more host
/// copies of textures: its use by this process under
/// `VICTORY_TEXTURE_SPILL_PERCENT` (50; 10 to 90) of the budget the system
/// gives it (the rest for our upload and read-back buffers and the copies
/// of title buffers kept there), and more than [`SPILL_RAM_FLOOR`] of the
/// PC's memory free (the title commits much of it itself).
#[cfg(windows)]
fn shared_room() -> bool {
    static PERCENT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let percent = *PERCENT.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_SPILL_PERCENT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(50, |p| p.clamp(10, 90))
    });
    let Some([.., budget, used]) = crate::host::host().and_then(|h| h.video_memory()) else {
        return false;
    };
    shared_has_room(budget, used, percent, crate::host::available_physical_ram())
}

/// The PC's memory left free that a texture in it may not take.
#[cfg(windows)]
const SPILL_RAM_FLOOR: u64 = 4 << 30;

/// [`shared_room`]'s rule: `used` of the shared `budget` under `percent` of
/// it, and (if known) more than [`SPILL_RAM_FLOOR`] of the PC's memory free.
#[cfg(windows)]
fn shared_has_room(budget: u64, used: u64, percent: u64, free_ram: Option<u64>) -> bool {
    used < budget / 100 * percent && free_ram.is_none_or(|free| free > SPILL_RAM_FLOOR)
}

/// [`shared_room`] as it was at most a tenth of a second ago (as
/// [`room_lately`]).
#[cfg(windows)]
fn shared_room_lately() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tenth = crate::host::now_ms() / 100 + 1;
    let last = LAST.load(Relaxed);
    if last >> 1 == tenth {
        return last & 1 != 0;
    }
    let room = shared_room();
    LAST.store((tenth << 1) | u64::from(room), Relaxed);
    room
}

/// Host copies in the PC's memory ([`spill_on`]): alive and their bytes,
/// made since the start and their bytes, and those that memory refused
/// (made on the card instead).
#[cfg(windows)]
pub(crate) static SPILLED: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// The line on host copies in the PC's memory, every few seconds; nothing
/// until there is one, or a pass of textures picked by index found no room
/// at all.
#[cfg(windows)]
pub(crate) fn spill_line() {
    use std::sync::atomic::Ordering::Relaxed;
    let [alive, bytes, made, made_bytes, refused] = [0, 1, 2, 3, 4].map(|i| SPILLED[i].load(Relaxed));
    let full = crate::draw::HEAP_TEXTURES_FULL.load(Relaxed);
    if made + refused + full == 0 {
        return;
    }
    let shared = crate::host::host()
        .and_then(|h| h.video_memory())
        .map_or_else(|| "unknown".to_owned(), |[.., budget, used]| format!("{} of {} MB", used >> 20, budget >> 20));
    tracing::info!(
        "textures that found no room on the card, levels left out and all (VICTORY_TEXTURE_SPILL): {made} made whole in \
         the PC's memory the card reads over the bus instead of waiting since the start ({} MB), {alive} of them alive \
         ({} MB), {refused} refused there and made on the card; {full} passes over textures picked by index that found \
         no room in either, so that they waited; {} heaps of placed textures there; the PC's shared memory in use: \
         {shared}",
        made_bytes >> 20,
        bytes >> 20,
        crate::placed::shared_heaps(),
    );
}


/// The share of the card's budget, in percent, up to which host copies of
/// textures picked by index are made ([`room`]): 90 unless
/// `VICTORY_TEXTURE_ROOM_PERCENT` says another (50 to 97). At 90 about a
/// tenth of the budget, 1,020 MB of 10,200, stays unused while thousands of
/// textures wait (runs 170-174); the card copies of title buffers from
/// 1 MiB up take 449 MB more of the card than before (941 against 492),
/// and 94 gives the textures about that much back.
#[cfg(windows)]
fn room_percent() -> u64 {
    static PERCENT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PERCENT.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_ROOM_PERCENT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(90, |percent| percent.clamp(50, 97))
    })
}

/// For the card accounting line ([`card_accounting`]): the title texture
/// made for a view of memory at `address` with the resource description
/// `desc`, if there is one ([`texture_at`] makes it; this does not).
#[cfg(windows)]
pub(crate) fn placed_known(address: u64, desc: &[u32; 14]) -> Option<This> {
    // Without the filler words and with them: whichever `texture_at` keeps.
    let mut kept = *desc;
    kept[12] &= !PLACED_WRITE_FLAGS;
    let mut key = kept;
    key[1] = 0;
    key[13] = 0;
    let map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    map.get(&(address, key)).or_else(|| map.get(&(address, kept))).map(|&r| r as This)
}

/// For the card accounting line: whether title texture `this` has a host
/// copy now (2), had one that was let go and not made again (1), or never
/// had one (0); the bytes (its host copy's on the card, or without one
/// what its console layout takes in title memory); and where its title
/// memory is. `None` for buffers. Not a use of the texture.
#[cfg(windows)]
pub(crate) fn copy_state(this: This) -> Option<(u8, u64, u64)> {
    use std::sync::atomic::Ordering::Relaxed;
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    if r.desc.dimension == BUFFER {
        return None;
    }
    let memory = r.data() as u64;
    let made = r.host.made.lock().unwrap_or_else(|e| e.into_inner());
    Some(match made.as_ref() {
        Some((host, _)) => (2, host.bytes, memory),
        None => (u8::from(r.host.gone.load(Relaxed)), footprint(&r.desc).map_or(0, |f| f.size), memory),
    })
}

/// One line of where the card's memory is and what the queue of textures
/// picked by index would need (`VICTORY_CARD_ACCOUNTING=1`; `draw.rs`
/// `card_accounting` says when, and hands over the queue's side: `room`
/// as the pass found it, `due` its counts, `never` and `gone` the memory
/// of the due textures without a host copy). The line holds
/// - the host copies of textures the title fills that are alive: how
///   many, their bytes, and the bytes of the levels whose title memory is
///   not mapped (a host copy is made with every level of the description;
///   a texture streaming in has its small levels only). A copy's bytes
///   are shared out to its levels by their texels, so that part is an
///   estimate;
/// - the due slots: how many, and their textures by what they have on
///   the card (a host copy alive: no card memory needed; never one; one
///   that was let go), with the bytes the last two would need and how
///   much of their memory is mapped now;
/// - the host copies let go, made again and made since the line before.
///
/// The textures are noted under the table's lock (their memory, their
/// description, their copy's bytes: about a millisecond for 18,000) and
/// the kernel is asked about their memory on a thread of its own, after
/// the lock is given back: one question for a texture that is all there
/// (up to 256 pages each; 18,000 textures of 5.5 GB are 1.4 million
/// pages, some 20,000 questions), one per level and array slice more for
/// one that is not. Not measured: a tenth to half a second is expected,
/// and the line says what it took. No line is begun while one is walking.
#[cfg(windows)]
pub(crate) fn card_accounting(room: bool, due: [u64; 5], never: Vec<(u64, u64)>, gone: Vec<(u64, u64)>) {
    use std::sync::atomic::Ordering::Relaxed;
    static WALKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    // Host copies let go and their bytes, made again and their bytes, and
    // made in all, at the line before.
    static BEFORE: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];
    if WALKING.swap(true, Relaxed) {
        return;
    }
    let began = std::time::Instant::now();
    // (The last: made level by level, its bytes are those of its tiles.)
    let mut copies: Vec<(usize, ResourceDesc, u64, bool)> = Vec::new();
    // Textures the title fills without a host copy: never one, one let go.
    let mut without = [0u64; 2];
    {
        let all = textures();
        for &at in all.iter() {
            // SAFETY: as in `let_go_idle`.
            let r = unsafe { &*(at as *const Resource) };
            if !fills(&r.desc) {
                continue;
            }
            let made = r.host.made.lock().unwrap_or_else(|e| e.into_inner());
            match made.as_ref() {
                Some((host, _)) => copies.push(match &host.tiled {
                    Some(tiled) => (r.data() as usize, r.desc, tiled.given_bytes(), true),
                    None => (r.data() as usize, r.desc, host.bytes, false),
                }),
                None => without[usize::from(r.host.gone.load(Relaxed))] += 1,
            }
        }
    }
    let kinds = KINDS.each_ref().map(|k| k.load(Relaxed));
    let mut since = [0u64; 5];
    for ((slot, now), before) in since.iter_mut().zip([kinds[2], kinds[3], kinds[4], kinds[5], kinds[6]]).zip(&BEFORE) {
        *slot = now.saturating_sub(before.swap(now, Relaxed));
    }
    let video = crate::host::host().and_then(|h| h.video_memory());
    let noted_us = began.elapsed().as_micros() as u64;
    let walk = move || {
        fn texels(m: &texture::Mip) -> u64 {
            u64::from(m.width) * u64::from(m.height) * u64::from(m.depth)
        }
        let began = std::time::Instant::now();
        // Copies; their bytes; the bytes of levels not mapped; copies with
        // a level not mapped; with the largest level not mapped; with no
        // level mapped; copies without a console layout (not asked).
        let mut c = [0u64; 7];
        // Of the copies: made level by level, and the bytes of their tiles.
        let mut on_demand = [0u64; 2];
        for &(memory, desc, bytes, by_level) in &copies {
            c[0] += 1;
            c[1] += bytes;
            if by_level {
                on_demand[0] += 1;
                on_demand[1] += bytes;
                continue;
            }
            let Some(layout) = texture_layout(&desc).map(|t| t.planes[0].layout.clone()) else {
                c[6] += 1;
                continue;
            };
            if layout.size == 0 || crate::texture::committed(memory as *const u8, layout.size as usize) {
                continue;
            }
            c[3] += 1;
            // Every slice holds the whole chain: the levels of the tail in
            // its first block, the others after it from the smallest up
            // (`victory_formats::texture::Layout`).
            let mips = &layout.mips;
            let tail = (layout.first_tail_mip as usize).min(mips.len());
            let there = |from: u64, to: u64| {
                (0..u64::from(layout.slices)).all(|slice| {
                    let at = memory as u64 + slice * layout.slice_size + from;
                    crate::texture::committed(at as *const u8, (to - from) as usize)
                })
            };
            let all: u64 = mips.iter().map(texels).sum::<u64>().max(1);
            let (mut missing, mut any) = (0u64, false);
            for (level, m) in mips.iter().enumerate().take(tail) {
                let end = if level == 0 { layout.slice_size } else { mips[level - 1].offset };
                if end <= m.offset {
                    continue;
                }
                if there(m.offset, end) {
                    any = true;
                } else {
                    missing += texels(m);
                    c[4] += u64::from(level == 0);
                }
            }
            if tail < mips.len() {
                if there(0, layout.alignment.min(layout.slice_size).max(1)) {
                    any = true;
                } else {
                    missing += mips[tail..].iter().map(texels).sum::<u64>();
                }
            }
            c[2] += (u128::from(bytes) * u128::from(missing) / u128::from(all)) as u64;
            c[5] += u64::from(!any);
        }
        // Of the due textures without a host copy: how many; the bytes of
        // their memory; with all of it mapped, and those bytes; with part
        // of it mapped, and the bytes that are; with none of it mapped.
        let mapped = |list: &[(u64, u64)]| {
            let mut n = [0u64; 7];
            for &(at, len) in list {
                n[0] += 1;
                n[1] += len;
                let there = crate::texture::mapped_bytes(at as *const u8, len as usize);
                if len > 0 && there >= len {
                    n[2] += 1;
                    n[3] += len;
                } else if there > 0 {
                    n[4] += 1;
                    n[5] += there;
                } else {
                    n[6] += 1;
                }
            }
            n
        };
        let (never, gone) = (mapped(&never), mapped(&gone));
        let [budget_mb, used_mb] = video.map_or([0, 0], |v| [v[0] >> 20, v[1] >> 20]);
        let [copies_n, copies_mb, unmapped_mb, partly, largest, nothing, unknown] =
            [c[0], c[1] >> 20, c[2] >> 20, c[3], c[4], c[5], c[6]];
        let [without_never, without_gone] = without;
        let [due_slots, due_uploaded, due_other, alive_n, alive_mb] = [due[0], due[1], due[2], due[3], due[4] >> 20];
        let [never_n, never_mb, never_whole, never_whole_mb, never_part, never_part_mb, never_none] =
            [never[0], never[1] >> 20, never[2], never[3] >> 20, never[4], never[5] >> 20, never[6]];
        let [gone_n, gone_mb, gone_whole, gone_whole_mb, gone_part, gone_part_mb, gone_none] =
            [gone[0], gone[1] >> 20, gone[2], gone[3] >> 20, gone[4], gone[5] >> 20, gone[6]];
        let [let_go, let_go_mb, again, again_mb, made] = [since[0], since[1] >> 20, since[2], since[3] >> 20, since[4]];
        let walk_ms = began.elapsed().as_millis() as u64;
        let [by_level_n, by_level_mb] = [on_demand[0], on_demand[1] >> 20];
        tracing::info!(
            "card accounting: room {room}, the card {used_mb} MB in use of a budget of {budget_mb} MB; host copies of textures the title \
             fills: {copies_n} alive {copies_mb} MB, of which {unmapped_mb} MB are levels whose title memory is not mapped ({partly} copies \
             with such a level, {largest} of them the largest level, {nothing} with no level mapped; {unknown} without a console layout, not \
             asked); title textures without a host copy: {without_never} never had one, {without_gone} had one let go; due slots: \
             {due_slots} ({due_uploaded} uploaded before, {due_other} of nothing the title fills), their textures: {alive_n} with a host \
             copy alive ({alive_mb} MB, no card memory needed); {never_n} that never had one ({never_mb} MB of title memory: {never_whole} \
             all mapped {never_whole_mb} MB, {never_part} in part {never_part_mb} MB mapped, {never_none} not mapped); {gone_n} whose host \
             copy was let go ({gone_mb} MB: {gone_whole} all mapped {gone_whole_mb} MB, {gone_part} in part {gone_part_mb} MB mapped, \
             {gone_none} not mapped); since the line before: {let_go} host copies let go {let_go_mb} MB, {again} made again {again_mb} MB, \
             {made} made in all; noting took {noted_us} us on the frame's thread, the walk {walk_ms} ms on its own; of the host \
             copies {by_level_n} are made level by level and hold {by_level_mb} MB of tiles (in the MB alive, not in the levels not mapped)"
        );
        WALKING.store(false, Relaxed);
    };
    if std::thread::Builder::new().name("card accounting".into()).spawn(walk).is_err() {
        WALKING.store(false, Relaxed);
    }
}

/// Lets go of the host copies of textures the title fills that nothing
/// names any more: placed textures were never released (one host texture
/// per address and description ever viewed: 18,639 of them, 2.5 GB, after
/// nine minutes of run 24), nor those of texture objects before the title
/// released them. A host copy goes when
/// - the title fills the texture ([`fills`]) and the host never copied
///   into it: its contents are in title memory;
/// - only the resource holds it: no list not yet reset names it (a list
///   holds the copies of the textures it uses, `HostList::transition`);
/// - only the host copy holds the host texture: no descriptor slot views
///   it (a slot keeps what it views, `HostHeap::kept`), so no shader can
///   read it through a slot made earlier, by a table or by an index;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next use makes the host copy anew and uploads the title's memory
/// again (its staging state goes with it). The title resource stays: its
/// address is what views and lists name.
/// shortcut: [`host_resource`] hands out the host texture without holding
/// it; a thread that asked for it and has not yet recorded its use (which
/// makes the list or the slot hold it) is protected by that idle time alone.
/// Every caller uses it at once, within the title's call.
/// Called before a submission starts, by the thread that submits.
/// `VICTORY_TEXTURE_SWEEP=0` lets nothing go.
#[cfg(windows)]
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    // A moment with nothing in flight (the sweep's): the slots of textures
    // whose views changed are made again now (`crate::tiles::publishing`).
    crate::descriptor::remake_at_quiet_point();
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    let idle_ms = crate::host::vram_idle_ms(crate::host::host().and_then(|h| h.video_memory()));
    // Host copies made without memory of their own (`crate::tiles`: levels
    // left out, or on demand) wait longer while the card is not pressed
    // (`vram_idle_ms` above its least): they hold little of it, and the
    // opening cutscene made 12,168 copies again in 110 s with them where it
    // made 2,938 without. `VICTORY_TILED_IDLE_MS` (5,000; 0: as the others).
    static TILED_IDLE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let tiled_idle = *TILED_IDLE.get_or_init(|| {
        std::env::var("VICTORY_TILED_IDLE_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(5000).min(60_000)
    });
    let tiled_idle_ms = if idle_ms > 1000 { idle_ms.max(tiled_idle) } else { idle_ms };
    let spilled_idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    // For the trail (`crate::host::trail`): how many went, their bytes,
    // and the first one.
    let (mut went, mut bytes, mut first) = (0u64, 0u64, None);
    let all = textures();
    for &at in all.iter() {
        // SAFETY: a texture leaves the table before it is freed, under the
        // lock held here (`Drop for Resource`).
        let r = unsafe { &*(at as *const Resource) };
        if !fills(&r.desc) || crate::texture::host_wrote(at) {
            continue;
        }
        let gone = {
            let mut made = r.host.made.lock().unwrap_or_else(|e| e.into_inner());
            // A slot that shows a copy holds it without asking for it: the
            // time since it was asked for can be minutes when the title
            // moves the texture to another slot, and the copy went at the
            // next sweep, to be made and uploaded again a frame later with
            // nothing shown meanwhile (tippythetaidum/VIppy 70bdfa4: 19,054
            // of 75,053 copies let go in 600 s were made again). So the idle
            // time also runs from when the copy was first found held by
            // nothing.
            let idle = made.as_ref().is_some_and(|(host, _)| {
                // A copy in the PC's memory takes none of the card's: by
                // the card's pressure it went after a second idle and was
                // made and uploaded again (50,457 made for 6,795 alive in
                // 205 s). It goes by the PC's memory, as other caches there
                // do; once gone, its next use makes it on the card if that
                // has room, which is how it gets back there.
                let ms = if host.spilled {
                    spilled_idle_ms
                } else if host.tiled.is_some() {
                    tiled_idle_ms
                } else {
                    idle_ms
                };
                let unheld =
                    std::sync::Arc::strong_count(host) == 1 && crate::host::references(&host.texture) == 1;
                if !unheld {
                    r.host.unheld.store(0, Relaxed);
                    return false;
                }
                let since = r.host.unheld.load(Relaxed);
                if since == 0 {
                    r.host.unheld.store(crate::host::moment().max(1), Relaxed);
                    return false;
                }
                crate::host::long_ago(r.host.used.load(Relaxed), ms, IDLE_FRAMES)
                    && crate::host::long_ago(since, ms, IDLE_FRAMES)
            });
            if idle {
                // Under the lock: the next use must not find the staging
                // state of the texture let go (it says "uploaded").
                crate::texture::unstage_later(at as This);
                made.take()
            } else {
                None
            }
        };
        if let Some((host, _)) = gone {
            r.host.gone.store(true, Relaxed);
            r.host.unheld.store(0, Relaxed);
            KINDS[2].fetch_add(1, Relaxed);
            KINDS[3].fetch_add(host.bytes, Relaxed);
            went += 1;
            bytes += host.bytes;
            if first.is_none() {
                first = Some(at);
            }
            crate::host::retire_later(host);
        }
    }
    if went > 0 {
        crate::host::trail(|| {
            format!(
                "sweep: {went} host copies of textures the title fills let go ({} KiB), each held by its title texture alone (no list, no slot); the first: title texture {:#x}",
                bytes >> 10,
                first.unwrap_or(0)
            )
        });
    }
}

/// For the memory line: the host copies the sweep never lets go
/// ([`let_go_idle`] takes only copies of textures the title fills that the
/// host never copied into), by what would say they are dead weight. A
/// count and the bytes (twins included) of: the copies the host renders to
/// or writes; of them, those whose title memory is not mapped now (its
/// first page); those nothing has used for 30 seconds and 60 frames and
/// that no list and no slot holds; those that are both; the copies of
/// textures the title fills that the host copied into; and, of the first,
/// the placed ones. Render-target copies grew from 2,319 MB at the world's
/// first frame to 3,075 MB 150 s later in run 197 and never shrank, while
/// 26,517 copies of filled textures were let go to stay under the card's
/// line.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn targets_census() -> [u64; 12] {
    use std::sync::atomic::Ordering::Relaxed;
    let mut c = [0u64; 12];
    let all = textures();
    for &at in all.iter() {
        // SAFETY: as in `let_go_idle`.
        let r = unsafe { &*(at as *const Resource) };
        let made = r.host.made.lock().unwrap_or_else(|e| e.into_inner());
        let Some((host, twin)) = made.as_ref() else {
            continue;
        };
        let copied_into = host.filled && crate::texture::host_wrote(at);
        if host.filled && !copied_into {
            continue;
        }
        let bytes = host.bytes + twin.as_ref().map_or(0, |t| t.colour.bytes);
        let mut add = |i: usize| {
            c[i] += 1;
            c[i + 1] += bytes;
        };
        if copied_into {
            add(8);
            continue;
        }
        add(0);
        let placed = r.host.placed.load(Relaxed);
        // Its first page: what is there of a texture starts there.
        let unmapped = placed && !crate::texture::committed(r.data().cast_const(), 1);
        let idle = crate::host::long_ago(r.host.used.load(Relaxed), 30_000, 60)
            && std::sync::Arc::strong_count(host) == 1
            && crate::host::references(&host.texture) == 1;
        if unmapped {
            add(2);
        }
        if idle {
            add(4);
        }
        if unmapped && idle {
            add(6);
        }
        if placed {
            add(10);
        }
    }
    c
}

/// For the memory line: title textures with a host copy the host renders
/// to or writes and those copies' bytes (twins included); title textures
/// without a host copy and the bytes of title memory they are; host copies
/// of textures the title fills and their bytes; of all title textures,
/// the placed ones with a host copy; the bytes of the twins alone (the
/// layer's own: the console has no second texture for a depth texture
/// shaders write); and, of the host copies of placed textures the title
/// fills, the bytes of those whose title memory is not mapped any more and
/// of those a descriptor slot still shows (the card holds a texture the
/// title has given up for as long as a slot shows it). The last two are
/// estimates from every 16th such texture: asking Windows about the memory
/// of all 16,000 takes too long.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn census() -> [u64; 10] {
    const EVERY: u64 = 16;
    let mut count = [0u64; 10];
    let mut placed_filled = 0u64;
    let all = textures();
    for &at in all.iter() {
        // SAFETY: as in `let_go_idle`.
        let r = unsafe { &*(at as *const Resource) };
        let made = r.host.made.lock().unwrap_or_else(|e| e.into_inner());
        match made.as_ref() {
            Some((host, twin)) => {
                let kind = if host.filled { 4 } else { 0 };
                count[kind] += 1;
                count[kind + 1] += host.bytes + twin.as_ref().map_or(0, |t| t.colour.bytes);
                count[7] += twin.as_ref().map_or(0, |t| t.colour.bytes);
                let placed = r.host.placed.load(std::sync::atomic::Ordering::Relaxed);
                if placed {
                    count[6] += 1;
                }
                if placed && host.filled {
                    placed_filled += 1;
                    // Its first page: a texture's smallest mips lie there,
                    // and are mapped first and unmapped last.
                    if placed_filled.is_multiple_of(EVERY) && !crate::texture::committed(r.data().cast_const(), 1) {
                        count[8] += host.bytes * EVERY;
                        if crate::host::references(&host.texture) > 1 {
                            count[9] += host.bytes * EVERY;
                        }
                    }
                }
            }
            None => {
                count[2] += 1;
                count[3] += footprint(&r.desc).map_or(0, |f| f.size);
            }
        }
    }
    count
}

/// Host textures alive (the title's and the twins) and the bytes the host
/// gives them, for the memory line.
#[cfg(windows)]
pub(crate) static HOST_TEXTURES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The bytes the host allocates for `texture`.
#[cfg(windows)]
fn host_bytes(texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> u64 {
    let Some(host) = crate::host::host() else {
        return 0;
    };
    // SAFETY: plain queries on a live resource and device.
    unsafe {
        let desc = texture.GetDesc();
        host.device.GetResourceAllocationInfo(0, &[desc]).SizeInBytes
    }
}

#[cfg(windows)]
impl HostTexture {
    fn new(texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource, state: u32) -> HostTexture {
        use std::sync::atomic::Ordering::Relaxed;
        let bytes = host_bytes(&texture);
        HOST_TEXTURES[0].fetch_add(1, Relaxed);
        HOST_TEXTURES[1].fetch_add(bytes, Relaxed);
        HostTexture {
            texture,
            state: std::sync::atomic::AtomicU32::new(state),
            bytes,
            filled: false,
            tiled: None,
            reuse: None,
            spilled: false,
            card: crate::texture::CardState::default(),
            sent: Default::default(),
        }
    }

    /// A texture made without memory of its own (`crate::tiles`): counted
    /// as a host texture of no bytes. `reuse`: what it is kept under for
    /// the next texture of that description when it goes
    /// ([`take_reserved`]).
    fn new_tiled(
        texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        state: u32,
        tiled: crate::tiles::Tiled,
        reuse: ReuseKey,
    ) -> HostTexture {
        HOST_TEXTURES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        HostTexture {
            texture,
            state: std::sync::atomic::AtomicU32::new(state),
            bytes: 0,
            filled: false,
            tiled: Some(tiled),
            reuse: Some(reuse),
            spilled: false,
            card: crate::texture::CardState::default(),
            sent: Default::default(),
        }
    }
}

#[cfg(windows)]
impl Drop for HostTexture {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        HOST_TEXTURES[0].fetch_sub(1, Relaxed);
        HOST_TEXTURES[1].fetch_sub(self.bytes, Relaxed);
        if self.filled {
            KINDS[0].fetch_sub(1, Relaxed);
            KINDS[1].fetch_sub(self.bytes, Relaxed);
        }
        if self.spilled {
            SPILLED[0].fetch_sub(1, Relaxed);
            SPILLED[1].fetch_sub(self.bytes, Relaxed);
        }
        // A texture without memory of its own is kept for the next one of
        // its description: its tiles are taken from it on the queue (ahead
        // of anything a later list does with it) and go back to the pool.
        // A host copy goes only when nothing on the card uses it any more
        // (a list holds it until it is reset, and the sweep's and the
        // title's releases wait for the next landing).
        // (Only a texture nothing else names: a descriptor slot keeps the
        // texture it views, and a slot of a texture the title released may
        // still be read.)
        if reserved_reuse_most() > 0
            && self.reuse.is_some()
            && crate::host::references(&self.texture) == 1
            && let (Some(key), Some(tiled)) = (self.reuse, self.tiled.take())
            && tiled.unmap_all(&self.texture)
        {
            drop(tiled);
            keep_reserved(key, self.texture.clone(), self.state.load(Relaxed));
        }
    }
}

/// What a texture without memory of its own is kept under for reuse: the
/// title description's dimension, width, height, layers, levels, format
/// and flags (the host's description follows from them).
#[cfg(windows)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) struct ReuseKey(u32, u64, u32, u16, u16, u32, u32);

#[cfg(windows)]
impl ReuseKey {
    fn of(desc: &ResourceDesc) -> ReuseKey {
        ReuseKey(
            desc.dimension,
            desc.width,
            desc.height,
            desc.depth_or_array_size,
            desc.mip_levels,
            desc.format,
            desc.flags,
        )
    }
}

/// Textures without memory of their own kept for reuse, by description:
/// the texture and the state its last list left it in; and how many in all.
/// Making one is two calls into the kernel (its allocation and its range
/// of the card's addresses: `NtGdiDdDDICreateAllocation`, 33 ms a frame of
/// the texture pass in the opening cutscene with some 700 made a second),
/// releasing it two more; it holds no memory of the card, only addresses.
#[cfg(windows)]
static SPARE_RESERVED: std::sync::Mutex<(
    std::collections::BTreeMap<ReuseKey, Vec<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)>>,
    usize,
)> = std::sync::Mutex::new((std::collections::BTreeMap::new(), 0));

/// Since the start: textures without memory taken from the spare ones,
/// made anew, kept when let go, and released because the spare ones were
/// as many as allowed ([`reserved_reuse_most`]).
#[cfg(windows)]
pub(crate) static RESERVED_REUSE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// How many textures without memory are kept spare at most:
/// `VICTORY_RESERVED_REUSE` (4,096; 0: none is kept, each is made and
/// released as before).
#[cfg(windows)]
fn reserved_reuse_most() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_RESERVED_REUSE")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4096)
            .min(1 << 16)
    })
}

/// Keeps `texture` (without memory, no tiles mapped any more, in host
/// state `state`) for the next texture of description `key`, or releases
/// it if as many are kept as allowed.
#[cfg(windows)]
fn keep_reserved(key: ReuseKey, texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource, state: u32) {
    use std::sync::atomic::Ordering::Relaxed;
    let most = reserved_reuse_most();
    let mut spare = SPARE_RESERVED.lock().unwrap_or_else(|e| e.into_inner());
    if spare.1 >= most {
        drop(spare);
        RESERVED_REUSE[3].fetch_add(1, Relaxed);
        drop(texture);
        return;
    }
    spare.1 += 1;
    spare.0.entry(key).or_default().push((texture, state));
    RESERVED_REUSE[2].fetch_add(1, Relaxed);
}

/// A spare texture without memory of description `key`, and its host
/// state ([`keep_reserved`]).
#[cfg(windows)]
fn take_reserved(key: ReuseKey) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)> {
    if reserved_reuse_most() == 0 {
        return None;
    }
    let mut spare = SPARE_RESERVED.lock().unwrap_or_else(|e| e.into_inner());
    let taken = spare.0.get_mut(&key).and_then(Vec::pop);
    if taken.is_some() {
        spare.1 -= 1;
    }
    taken
}

/// For the 5-second line: textures without memory taken from the spare
/// ones, made anew, kept, released for want of places; and spare now.
#[cfg(windows)]
pub(crate) fn reserved_reuse_stats() -> [u64; 5] {
    use std::sync::atomic::Ordering::Relaxed;
    let spare = SPARE_RESERVED.lock().unwrap_or_else(|e| e.into_inner()).1 as u64;
    [
        RESERVED_REUSE[0].load(Relaxed),
        RESERVED_REUSE[1].load(Relaxed),
        RESERVED_REUSE[2].load(Relaxed),
        RESERVED_REUSE[3].load(Relaxed),
        spare,
    ]
}

/// How many textures placed in title memory have a resource of ours
/// (never released), for the memory line.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn placed_textures() -> u64 {
    PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner()).len() as u64
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
#[cfg(windows)]
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let (host, _) = hosted(this, true)?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The host GPU's copy of a title resource, if it has one: made now if
/// this is the texture's first use ([`Hosted`]). Asking is a use; what must
/// not count as one asks `host_made`.
#[cfg(windows)]
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    let (host, _) = hosted(this, true)?;
    // SAFETY: the resource holds the host copy until the title releases it
    // (lists and slots that name the texture hold it past that), or until
    // `let_go_idle` takes it: only from a texture the title fills, that no
    // list and no slot holds and that nothing asked for in seconds (the
    // asking just made is noted under the same lock).
    Some(unsafe { &*std::ptr::from_ref(&host.texture) })
}

/// A title texture's host copy, shared, for a list to hold.
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    hosted(this, true).map(|made| made.0)
}

/// Whether the host takes a clear value on a texture of these (host) flags:
/// render targets and depth buffers only.
fn takes_clear_value(flags: u32) -> bool {
    const TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    flags & (TARGET | DEPTH) != 0
}

/// A host texture like the title's: same description, in GPU memory,
/// without the console's layout and flags; with `spill`, in the PC's
/// memory the card reads over the bus ([`spill_on`]). Also whether it went
/// there (a texture that memory refuses is made on the card).
#[cfg(windows)]
fn host_texture(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
    placed: bool,
    spill: bool,
) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, bool)> {
    host_texture_as(desc, state, clear, false, placed, spill)
}

/// [`host_texture`]; `reserved`: without memory of its own, to be given
/// tiles level by level (`crate::tiles`; never spilled).
#[cfg(windows)]
fn host_texture_as(
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
    reserved: bool,
    placed: bool,
    spill: bool,
) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, bool)> {
    use windows::Win32::Graphics::Direct3D12::*;
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // One texel high; a 1D array's size stays its slice count.
    if host_dimension(d.dimension) != d.dimension {
        d.dimension = host_dimension(d.dimension);
        d.height = 1;
    }
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; the title's shadow maps ask for it).
    // The host copy loses the unordered access: shaders write a colour
    // twin instead ([`Twin`]).
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    if d.flags & DEPTH != 0 {
        if d.flags & UNORDERED != 0 {
            DEPTH_UNORDERED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        d.flags &= !UNORDERED;
    }
    d.format = pc_format(d.format);
    const TARGET: u32 = 1;
    if d.flags & (TARGET | UNORDERED) != 0 {
        d.format = target_format(d.format);
    }
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    // The console allows one on any texture; the host only on render
    // targets and depth buffers, and refuses the texture (E_INVALIDARG)
    // otherwise: it is dropped there (Marvisak/VIctory pull request #3065).
    let clear_value = (!clear.is_null() && takes_clear_value(d.flags)).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = pc_format(value[0]);
        if d.flags & (TARGET | UNORDERED) != 0 {
            value[0] = target_format(value[0]);
        }
        value
    });
    // Only a texture the title fills goes to the PC's memory (the host
    // never renders to it or writes it there).
    const TARGET_DEPTH_UNORDERED: u32 = 0x7;
    let spill = spill && !reserved && d.flags & TARGET_DEPTH_UNORDERED == 0;
    if placed && !reserved {
        let state = D3D12_RESOURCE_STATES(crate::host::legal_for(d.flags, crate::host::texture_state(state)) as i32);
        // SAFETY: the description has the host layout (checked size, same
        // fields).
        let of = unsafe { &*(&raw const d).cast::<D3D12_RESOURCE_DESC>() };
        if let Some(t) = crate::placed::make(&host.device, of, state, spill) {
            crate::host::name(&t, || format!("placed texture {}x{} f{}", d.width, d.height, d.format));
            return Some((t, spill));
        }
    }
    let mut texture: Option<ID3D12Resource> = None;
    if spill {
        // SAFETY: as below, in the PC's memory.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &crate::placed::shared_heap(),
                crate::placed::not_resident(),
                (&raw const d).cast(),
                D3D12_RESOURCE_STATES(crate::host::legal_for(d.flags, crate::host::texture_state(state)) as i32),
                None,
                &mut texture,
            )
        };
        let made = made.and_then(|()| texture.as_ref().map_or(Ok(()), |t| crate::placed::resident(&host.device, t)));
        match made {
            Ok(()) if texture.is_some() => {
                crate::host::name(texture.as_ref()?, || {
                    format!("texture in the PC's memory {}x{} f{}", d.width, d.height, d.format)
                });
                return texture.map(|t| (t, true));
            }
            Ok(()) => {}
            Err(e) => {
                SPILLED[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(error = %e, desc = ?d, "a texture could not be made in the PC's memory: made on the card")
                });
            }
        }
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let made = if reserved {
        // The card's own layout of 64 KiB tiles, which a reserved texture
        // must have.
        d.layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE.0 as u32;
        // SAFETY: as below; a texture the title fills has no clear value.
        unsafe {
            host.device.CreateReservedResource(
                (&raw const d).cast(),
                D3D12_RESOURCE_STATES(
                    crate::host::legal_for(d.flags, crate::host::texture_state(state)) as i32,
                ),
                None,
                &mut texture,
            )
        }
    } else {
        // SAFETY: the description has the host layout (checked size, same
        // fields); the out pointer is ours.
        unsafe {
            host.device.CreateCommittedResource(
                &heap,
                crate::placed::not_resident(),
                (&raw const d).cast(),
                D3D12_RESOURCE_STATES(
                    crate::host::legal_for(d.flags, crate::host::texture_state(state)) as i32,
                ),
                // The same layout: format, then color or depth and stencil.
                clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast()),
                &mut texture,
            )
        }
        // Resident before anything names it (`crate::placed::resident`).
        .and_then(|()| texture.as_ref().map_or(Ok(()), |t| crate::placed::resident(&host.device, t)))
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, reserved, "no host texture");
        texture = None;
    }
    if let Some(t) = &texture {
        crate::host::name(t, || {
            format!("texture {}x{} f{} flags {:#x}", d.width, d.height, d.format, d.flags)
        });
    }
    texture.map(|t| (t, false))
}

/// Whether title texture `this` is one of a view the title placed in
/// memory ([`texture_at`]), not a texture object of the title's.
#[cfg(windows)]
pub(crate) fn is_placed(this: This) -> bool {
    !this.is_null() && resource(this).host.placed.load(std::sync::atomic::Ordering::Relaxed)
}

/// The host copy of title texture `this` if it is made and was made
/// without memory of its own (`crate::tiles`); it is not made here.
#[cfg(windows)]
pub(crate) fn host_tiled(this: This) -> Option<OnDemand> {
    if this.is_null() || !crate::tiles::any_made() {
        return None;
    }
    hosted(this, false).map(|(host, _)| host).filter(|host| host.tiled.is_some()).map(OnDemand)
}

/// What a view of title texture `this` may show, if its host copy is made
/// level by level and some level is held and has memory
/// (`crate::tiles::Tiled::view_levels`).
#[cfg(windows)]
pub(crate) fn view_levels(this: This) -> Option<crate::tiles::ViewLevels> {
    host_tiled(this)?.tiled()?.view_levels()
}

/// A host copy made without memory of its own (`crate::tiles`), as a
/// submission-time step of a list holds it.
#[cfg(windows)]
pub(crate) struct OnDemand(std::sync::Arc<HostTexture>);

// SAFETY: the host texture is named only in queue calls, which any thread
// may make; what it has been given is behind a lock of its own.
#[cfg(windows)]
unsafe impl Send for OnDemand {}
// SAFETY: as above.
#[cfg(windows)]
unsafe impl Sync for OnDemand {}

#[cfg(windows)]
impl OnDemand {
    pub(crate) fn tiled(&self) -> Option<&crate::tiles::Tiled> {
        self.0.tiled.as_ref()
    }

    /// [`crate::tiles::Tiled::uploaded`]; `memory`: where the texture's
    /// title memory starts (its slots are made again if the first level
    /// they may show changed).
    /// Held levels get tiles only while the card has room ([`card_room`]).
    /// `copied`: the levels the upload's list wrote.
    /// Returns whether a level is still without its tiles: the caller
    /// records the upload again later.
    pub(crate) fn uploaded(&self, held: u32, copied: u32, memory: u64) -> bool {
        let Some(tiled) = &self.0.tiled else {
            return false;
        };
        let (changed, waiting) = tiled.uploaded(&self.0.texture, held, copied, card_room());
        if changed {
            self.views_changed(memory);
        }
        if waiting {
            crate::tiles::STATS[9].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        waiting
    }

    /// [`crate::tiles::Tiled::card_wrote`]; `memory` as above.
    pub(crate) fn card_wrote(&self, levels: u32, memory: u64) {
        if let Some(tiled) = &self.0.tiled
            && tiled.card_wrote(&self.0.texture, levels)
        {
            self.views_changed(memory);
        }
    }

    /// The tiles changed what the views may show, in a step of the
    /// submission being prepared now. With `crate::tiles::publishing`: the
    /// views follow once that submission has landed ([`Publish`]), and
    /// their slots are made again at the next moment with nothing in
    /// flight (`crate::descriptor::host::levels_changed`). Without: at
    /// once, as before.
    fn views_changed(&self, memory: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if !crate::tiles::publishing() {
            crate::descriptor::host::levels_changed(memory);
            return;
        }
        crate::tiles::ORDER[0].fetch_add(1, Relaxed);
        // (The step runs while its submission is prepared: it counts as
        // started, so `retire` keeps this until it lands.)
        if crate::host::submissions_in_flight() == 0 {
            crate::tiles::ORDER[1].fetch_add(1, Relaxed);
        }
        crate::host::retire(Publish {
            host: self.0.clone(),
            memory,
        });
    }
}

/// A change of what the views of a texture made level by level may show,
/// held until the submission whose step made it has landed: by then its
/// copies into the levels newly given tiles have run, and the views go by
/// them (`crate::tiles::Tiled::publish`).
#[cfg(windows)]
struct Publish {
    host: std::sync::Arc<HostTexture>,
    memory: u64,
}

#[cfg(windows)]
impl Drop for Publish {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        crate::tiles::ORDER[2].fetch_add(1, Relaxed);
        if let Some(tiled) = &self.host.tiled
            && tiled.publish()
        {
            crate::tiles::ORDER[3].fetch_add(1, Relaxed);
            crate::descriptor::host::levels_changed(self.memory);
        }
    }
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// A title texture's dimension as the title made it (2 1D, 3 2D, 4 3D;
/// the host's copy of a 1D one is 2D: [`host_dimension`]); `None` for
/// buffers.
pub(crate) fn title_dimension(this: This) -> Option<u32> {
    texture_memory(this).map(|(_, d)| d.dimension)
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear);
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host: Hosted {
            made: std::sync::Mutex::new(None),
            state,
            // SAFETY: the title passes a whole clear value (the format,
            // then color or depth and stencil: 5 words) or null.
            clear: (!clear.is_null()).then(|| unsafe { clear.cast::<[u32; 5]>().read_unaligned() }),
            used: std::sync::atomic::AtomicU64::new(0),
            refused: std::sync::atomic::AtomicBool::new(false),
            gone: std::sync::atomic::AtomicBool::new(false),
            placed: std::sync::atomic::AtomicBool::new(false),
            unheld: std::sync::atomic::AtomicU64::new(0),
        },
    });
    // The host copy waits for the texture's first use ([`Hosted`]).
    #[cfg(windows)]
    if desc.dimension != BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        textures().insert(at);
        // Copies back into the memory of a texture there before are not
        // this one's.
        crate::texback::replaced(resource.data() as usize);
        if !fills(&desc) {
            TARGETS_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(resource.data() as u64, at);
        }
        if !lazy() {
            let _ = hosted(at as This, true);
        }
    }
    #[cfg(windows)]
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        crate::shadow::register(at, resource.data() as u64, desc.width);
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        // Out of the table of textures first: the sweep looks at the ones
        // in it, under its lock.
        if self.desc.dimension != BUFFER {
            let address = std::ptr::from_ref::<Resource>(self) as usize;
            crate::texback::released(address);
            textures().remove(&address);
            if !fills(&self.desc) {
                let mut targets = TARGETS_AT.lock().unwrap_or_else(|e| e.into_inner());
                let at = self.data() as u64;
                if targets.get(&at) == Some(&(std::ptr::from_ref::<Resource>(self) as usize)) {
                    targets.remove(&at);
                }
            }
        }
        let made = self
            .host
            .made
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        #[cfg(feature = "diagnostics")]
        if made
            .as_ref()
            .is_some_and(|(h, _)| std::sync::Arc::strong_count(h) > 1)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset names its host copy"
                );
            }
        }
        // A running submission may still read the host copy (and a depth
        // texture's colour twin).
        if let Some((host, twin)) = made {
            crate::host::retire(host);
            if let Some(twin) = twin {
                crate::host::retire(twin);
            }
        }
        crate::texture::forget(std::ptr::from_ref::<Resource>(self) as usize);
        crate::descriptor::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        }
    }
}

/// A committed buffer of `width` zeroed bytes, for tests elsewhere in the
/// crate that need a title buffer object.
#[cfg(test)]
pub(crate) fn test_buffer(width: u64) -> This {
    let properties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let desc = ResourceDesc {
        dimension: BUFFER,
        alignment: 0,
        width,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
        flags: 0,
    };
    let null = std::ptr::null();
    let mut buffer = null;
    let hr = create_committed(
        null,
        &properties,
        0,
        &desc,
        0,
        null.cast(),
        &RESOURCE_IIDS[0],
        &mut buffer,
    );
    assert_eq!(hr, S_OK);
    buffer
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
    )
}

/// Device method 0x3f8, inferred from the title's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From the title's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if address == 0 {
        return E_INVALIDARG;
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, the title's streamed and
/// interface textures), by address and description: a resource at that
/// address, made the first time a draw reads one.
/// The resource is kept for the run (views and lists name its address);
/// its host texture is let go when nothing names it any more
/// ([`let_go_idle`]) and made again on the next use.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The flags of a resource description that make a texture one the host
/// writes (render target, depth, unordered access: [`fills`]).
#[cfg(windows)]
const PLACED_WRITE_FLAGS: u32 = 0x7;

/// The placed textures asked for with filler words set, as asked
/// (address and description): for [`PLACED_TWINS`].
#[cfg(windows)]
static PLACED_RAW: std::sync::Mutex<std::collections::BTreeSet<(u64, [u32; 14])>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Placed textures asked for a second time with other filler words in
/// their description: each was a second resource and host copy before the
/// filler was left out of the key ([`texture_at`]).
#[cfg(windows)]
static PLACED_TWINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See [`PLACED_TWINS`].
#[cfg(windows)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn placed_filler_twins() -> u64 {
    PLACED_TWINS.load(std::sync::atomic::Ordering::Relaxed)
}

/// The bytes of title memory a texture with the resource description
/// `desc` (as words) takes where the title placed it; `None` for a buffer
/// and for a description without a size.
#[cfg(windows)]
pub(crate) fn placed_bytes(desc: &[u32; 14]) -> Option<u64> {
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    footprint(&d).map(|f| f.size)
}

/// The placed textures by address ([`PLACED_TEXTURES`]), for the count of
/// `crate::census`: textures in all; addresses with textures of several
/// descriptions; the most at one address; and such addresses with several
/// host copies alive now. One walk of the table under its lock, asked
/// every five seconds; the host copies are asked after it, by a look at
/// whether each texture has one. (They were asked through [`copy_state`],
/// which works out a texture's footprint when it has none: 40,000 of
/// those every second, on a thread that records the frame, were most of
/// what the pass's line calls the queue's order, 1.8 ms a frame.)
#[cfg(windows)]
pub(crate) fn placed_descriptions() -> [u64; 4] {
    let mut several: Vec<Vec<usize>> = Vec::new();
    let (mut all, mut most) = (0u64, 0u64);
    {
        let map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
        let mut run: Vec<usize> = Vec::new();
        let mut at = u64::MAX;
        for (&(address, _), &texture) in map.iter() {
            all += 1;
            if address != at {
                if run.len() > 1 {
                    most = most.max(run.len() as u64);
                    several.push(std::mem::take(&mut run));
                }
                run.clear();
                at = address;
            }
            run.push(texture);
        }
        if run.len() > 1 {
            most = most.max(run.len() as u64);
            several.push(run);
        }
    }
    let has_copy = |t: &&usize| resource(**t as This).host.made.lock().unwrap_or_else(|e| e.into_inner()).is_some();
    let alive = |textures: &Vec<usize>| textures.iter().filter(has_copy).count() > 1;
    [all, several.len() as u64, most, several.iter().filter(|t| alive(t)).count() as u64]
}

/// How many of the textures [`texture_at`] made at `address` (one for each
/// description viewed there) have a host copy now.
#[cfg(windows)]
pub(crate) fn placed_copies(address: u64) -> usize {
    let textures: Vec<usize> = {
        let map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
        map.range((address, [0; 14])..=(address, [u32::MAX; 14])).map(|(_, &t)| t).collect()
    };
    if textures.len() < 2 {
        return textures.len();
    }
    textures.iter().filter(|&&t| copy_state(t as This).is_some_and(|state| state.0 == 2)).count()
}

/// The textures [`texture_at`] made for views placed in the title memory
/// `[address, address + bytes)`, `this` itself left out, each with where
/// it is placed: each has a host copy of its own, filled from that memory
/// (for `crate::texture::CARD_WRITTEN`, when the card first copies into
/// the texture that owns the memory).
#[cfg(windows)]
pub(crate) fn placed_in(address: u64, bytes: u64, this: This) -> Vec<(u64, This)> {
    // (No bytes: nothing inside; the end never lies below the start.)
    if bytes == 0 {
        return Vec::new();
    }
    let map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    map.range((address, [0; 14])..(address.saturating_add(bytes), [0; 14]))
        .filter(|&(_, &texture)| texture != this as usize)
        .map(|(&(at, _), &texture)| (at, texture as This))
        .collect()
}

/// The host copies, where they are made, of the textures [`texture_at`]
/// made for views placed at `address` (one for each description viewed
/// there). None is made here. The textures stay for the run, so this may
/// be asked at a submission.
#[cfg(windows)]
pub(crate) fn placed_copies_at(address: u64) -> Vec<std::sync::Arc<HostTexture>> {
    let textures: Vec<usize> = {
        let map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
        map.range((address, [0; 14])..=(address, [u32::MAX; 14])).map(|(_, &t)| t).collect()
    };
    textures.into_iter().filter_map(|t| hosted(t as This, false).map(|made| made.0)).collect()
}

/// The resource [`texture_at`] made for a texture placed at `address` with
/// the description `desc`, if it has made one: never made here (for the
/// frame listing, which must not count as a use).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn texture_at_made(address: u64, desc: &[u32; 14]) -> Option<This> {
    let mut kept = *desc;
    kept[12] &= !PLACED_WRITE_FLAGS;
    let mut key = kept;
    key[1] = 0;
    key[13] = 0;
    let map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    let found = map.get(&(address, key)).or_else(|| map.get(&(address, kept)))?;
    Some(*found as This)
}

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use.
#[cfg(windows)]
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<This> {
    // Words 1 and 13 of the description are the filler of its layout
    // (after the dimension, and at the end): the title leaves there what
    // its stack held, and two views of one texture that differed only in
    // them each got a resource and a host copy of their own, on a card
    // that fills at world entry. They are not part of what a texture is.
    // `VICTORY_PLACED_FILLER=1`: kept in the key, as before.
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // A texture a view is placed to sample is one the title fills: the
    // target, depth and unordered access flags its resource description may
    // still carry say what the title wrote it with, and would make the host
    // copy one of the title textures the host writes ([`TARGETS_AT`]), which
    // is then the very texture the views sample ([`alias_of`]) and not their
    // twin (fxceless/victory-fork pull request #114, uiliannn/VItoriaBr
    // pull request #72).
    let mut raw = *desc;
    raw[12] &= !PLACED_WRITE_FLAGS;
    let mut key = raw;
    if !*KEPT.get_or_init(|| std::env::var_os("VICTORY_PLACED_FILLER").is_some_and(|v| v == "1")) {
        key[1] = 0;
        key[13] = 0;
    }
    let desc = &key;
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    // How many resources the filler alone would have added (a count for
    // the log: `placed_filler_twins`).
    if raw != key {
        let mut seen = PLACED_RAW.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((address, raw)) {
            let others = seen.range((address, [0; 14])..=(address, [u32::MAX; 14])).filter(|(_, r)| {
                let mut blank = *r;
                blank[1] = 0;
                blank[13] = 0;
                blank == key
            });
            if others.count() > 1 {
                PLACED_TWINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
    if let Some(&r) = map.get(&(address, *desc)) {
        return Some(r as This);
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let mut out = std::ptr::null();
    let made = create_at_address(
        std::ptr::null(),
        address,
        &d,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    if made != S_OK {
        return None;
    }
    map.insert((address, *desc), out as usize);
    crate::texture::placed_in_host_written(address, desc);
    resource(out)
        .host
        .placed
        .store(true, std::sync::atomic::Ordering::Relaxed);
    Some(out)
}

/// The textures [`uav_texture_at`] made, by address and description: like
/// [`PLACED_TEXTURES`], never released, but not in it: those have a host
/// copy filled from title memory, these only what shaders write.
#[cfg(windows)]
static PLACED_UAV_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<(u64, [u32; 14]), usize>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Bytes of title memory the textures of [`PLACED_UAV_TEXTURES`] take where
/// the title placed them (what their host copies take on the card is of that
/// order), and how many there are.
#[cfg(windows)]
static PLACED_UAV_MADE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the textures of [`PLACED_UAV_TEXTURES`] may take together: past it
/// a placed unordered view is a null one again, as every one was.
/// shortcut: nothing releases such a texture when the title reuses its
/// memory, as it lets go the host copies of placed views it reads.
#[cfg(windows)]
const PLACED_UAV_LIMIT: u64 = 1 << 30;

/// The format of the resource made for an unordered access view of a texture
/// described as `own` (a title format) that names `view` (0: its own): the
/// resource's own, or what the view names where that is the same texels in
/// another type (the title packs light as R9G9B9E5 and writes it as R32_UINT,
/// which the host cannot do to the one it is).
#[cfg(windows)]
fn uav_resource_format(own: u32, view: u32) -> u32 {
    let wanted = if view == 0 { own } else { pc_format(view) };
    if view == 0 || target_format(own) == target_format(wanted) {
        return own;
    }
    if family(own) == family(wanted) {
        family(wanted)
    } else if element(own).is_some() && element(own) == element(wanted) {
        wanted
    } else {
        own
    }
}

/// The resource for an unordered access view of a texture the title placed
/// at `address` with resource description `desc` and view format
/// `view_format` (0: the resource's own): an unordered-access texture of the
/// host at that address when its shape and format allow the view, else one
/// made here on first use (starting in `UNORDERED_ACCESS`), which is then
/// the texture [`alias_of`] finds for the views of the same memory that the
/// title places to sample it. `None` for buffers and when the limit
/// ([`PLACED_UAV_LIMIT`]) is reached.
///
/// Before, such a view was a null one (writes dropped), and what the title
/// read of that memory through a view placed at it was black or stale:
/// fxceless/victory-fork pull request #114 and uiliannn/VItoriaBr pull
/// request #72, whose restore of the table's entry is taken here.
#[cfg(windows)]
pub(crate) fn uav_texture_at(address: u64, desc: &[u32; 14], view_format: u32) -> Option<This> {
    const DEPTH: u32 = 0x2;
    const UNORDERED: u32 = 0x4;
    const UNORDERED_ACCESS: u32 = 0x8;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    use std::sync::atomic::Ordering::Relaxed;
    if address == 0 {
        return None;
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let own = pc_format(d.format);
    let wanted = if view_format == 0 { own } else { pc_format(view_format) };
    {
        let targets = TARGETS_AT.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&other) = targets.get(&address) {
            let o = resource(other as This);
            let o_fmt = target_format(pc_format(o.desc.format));
            let w_fmt = target_format(wanted);
            let typeless = family(o_fmt) == o_fmt && family(o_fmt + 1) == o_fmt;
            let srgb = SRGB.iter().any(|&(s, _)| s == o_fmt);
            let compatible = o_fmt == w_fmt || ((typeless || srgb) && family(w_fmt) == family(o_fmt));
            if o.desc.dimension == d.dimension
                && o.desc.width == d.width
                && o.desc.height == d.height
                && o.desc.mip_levels == d.mip_levels
                && o.desc.depth_or_array_size == d.depth_or_array_size
                && o.desc.sample_count <= 1
                && d.sample_count <= 1
                && o.desc.flags & (DEPTH | UNORDERED) == UNORDERED
                && compatible
            {
                return Some(other as This);
            }
        }
    }
    let mut r = *desc;
    // The filler words are not part of what a texture is ([`texture_at`]).
    (r[1], r[13]) = (0, 0);
    r[12] = (r[12] & !DEPTH) | UNORDERED;
    r[8] = uav_resource_format(own, view_format);
    let mut map = PLACED_UAV_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&existing) = map.get(&(address, r)) {
        // A temporary target at this address may have replaced this texture
        // in the table and been released since: put it back if the entry is
        // vacant.
        TARGETS_AT
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(address)
            .or_insert(existing);
        return Some(existing as This);
    }
    // SAFETY: 56 bytes in the description's layout.
    let made_desc = unsafe { r.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    let bytes = footprint(&made_desc).map_or(0, |f| f.size);
    if PLACED_UAV_MADE[0].load(Relaxed).saturating_add(bytes) > PLACED_UAV_LIMIT {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::warn!(
                limit = PLACED_UAV_LIMIT,
                "placed unordered texture views: no more textures made (a null view is given)"
            )
        });
        return None;
    }
    let mut out = std::ptr::null();
    let made = hand_out_resource(
        made_desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        UNORDERED_ACCESS,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    if made != S_OK {
        return None;
    }
    map.insert((address, r), out as usize);
    PLACED_UAV_MADE[0].fetch_add(bytes, Relaxed);
    let n = PLACED_UAV_MADE[1].fetch_add(1, Relaxed) + 1;
    if n <= 16 || n.is_power_of_two() {
        tracing::info!(
            n,
            address = format_args!("{address:#x}"),
            dimension = made_desc.dimension,
            width = made_desc.width,
            height = made_desc.height,
            layers = made_desc.depth_or_array_size,
            levels = made_desc.mip_levels,
            format = made_desc.format,
            view_format,
            bytes,
            all_bytes = PLACED_UAV_MADE[0].load(Relaxed),
            "a texture made for an unordered access view the title placed in memory"
        );
    }
    Some(out)
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed::<Resource>(this)
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// The console lets a title set a clear value on any texture; the host
    /// refuses one (E_INVALIDARG) unless the texture is a render target or
    /// a depth buffer (Marvisak/VIctory pull request #3065).
    #[test]
    fn only_targets_and_depth_buffers_keep_a_clear_value() {
        assert!(!takes_clear_value(0));
        assert!(!takes_clear_value(4), "unordered access alone");
        assert!(!takes_clear_value(8));
        assert!(takes_clear_value(1));
        assert!(takes_clear_value(2));
        assert!(takes_clear_value(1 | 4));
    }

    /// The R8 family is R8 typeless (60) with UNORM, UINT, SNORM and SINT
    /// (61 to 64); A8_UNORM (65) is a family of its own (Marvisak 97c8783).
    #[test]
    fn a8_unorm_is_not_in_the_r8_family() {
        assert_eq!(family(60), 60);
        for f in 61..=64 {
            assert_eq!(family(f), 60);
        }
        assert_eq!(family(65), 65);
        assert_eq!(family(59), 53);
        assert_eq!(family(66), 66);
    }

    /// The order: levels left out first, the PC's memory only for what
    /// finds no room on the card even so, and with no room in either the
    /// card as before.
    #[test]
    fn textures_go_to_the_pcs_memory_last() {
        assert_eq!(placement(true, true), Placement::Card, "room on the card: there, levels by the policy");
        assert_eq!(placement(true, false), Placement::Card);
        assert_eq!(placement(false, true), Placement::Shared, "no room on the card: the PC's memory");
        assert_eq!(placement(false, false), Placement::Card, "no room anywhere: as before (it waits)");
    }

    /// The PC's shared memory takes textures up to its share of the budget
    /// and while the PC keeps its floor free.
    #[cfg(windows)]
    #[test]
    fn the_shared_memory_takes_textures_up_to_its_share() {
        const GB: u64 = 1 << 30;
        assert!(shared_has_room(32 * GB, 2 * GB, 50, Some(40 * GB)));
        assert!(shared_has_room(32 * GB, 2 * GB, 50, None));
        assert!(!shared_has_room(32 * GB, 16 * GB, 50, Some(40 * GB)));
        assert!(shared_has_room(32 * GB, 15 * GB, 50, Some(40 * GB)));
        assert!(!shared_has_room(32 * GB, 2 * GB, 50, Some(SPILL_RAM_FLOOR)));
        assert!(!shared_has_room(0, 0, 50, None));
    }

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    /// The resource of an unordered access view the title places is the type
    /// its shaders write: R32_UINT for the packed light, not R9G9B9E5.
    #[cfg(windows)]
    #[test]
    fn a_placed_unordered_view_names_the_type_shaders_write() {
        assert_eq!(uav_resource_format(67, 0), 67, "the view names none");
        assert_eq!(uav_resource_format(67, 42), 42, "the same texels as R32_UINT");
        assert_eq!(uav_resource_format(28, 28), 28);
        assert_eq!(uav_resource_format(27, 28), 27, "the family's typeless one");
        assert_eq!(uav_resource_format(10, 42), 10, "texels of another size stay");
    }

    /// A placed unordered view gets a texture of the host's own, and a view
    /// placed to sample the same memory takes its texels from it, though
    /// its description still carries the flags the title wrote it with
    /// (fxceless/victory-fork pull request #114, uiliannn/VItoriaBr #72).
    #[cfg(windows)]
    #[test]
    fn a_placed_unordered_view_is_what_the_sampling_view_aliases() {
        use std::sync::atomic::Ordering::Relaxed;
        static MEMORY: [u8; 4096] = [0; 4096];
        let address = MEMORY.as_ptr() as u64;
        let words = |format: u32, flags: u32| {
            let desc = ResourceDesc { layout: 1, flags, ..texture(4, 4, format, 1) };
            // SAFETY: 56 bytes in the description's layout.
            unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() }
        };
        let uav = uav_texture_at(address, &words(67, 4), 42).unwrap();
        assert_eq!(resource(uav).desc.format, 42);
        assert_eq!(resource(uav).desc.flags & 7, 4);
        assert!(!resource(uav).host.placed.load(Relaxed), "not one filled from title memory");
        assert_eq!(uav_texture_at(address, &words(67, 4), 42), Some(uav), "made once");
        assert_eq!(uav_texture_at(address, &words(67, 0), 42), Some(uav), "the flags the title left do not matter");
        let view = texture_at(address, &words(67, 4)).unwrap();
        assert_eq!(resource(view).desc.flags & 7, 0, "a sampled texture is one the title fills");
        assert_eq!(texture_at(address, &words(67, 0)), Some(view), "the same one with the flags gone");
        assert_eq!(placed_known(address, &words(67, 4)), Some(view));
        assert_eq!(alias_of(view), Some(uav));
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as the title asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[crate::host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    /// For a texture both the title and the card write, what was written
    /// last is what it holds, by the order the lists are submitted in and
    /// not the order they were recorded in: an upload from title memory
    /// recorded before a card copy and submitted after it leaves the
    /// card's texels (it overwrote them: 27 times in a run of the title,
    /// whose bake copies its result into textures the texture pass had
    /// just asked uploads for; Marvisak/VIctory issue #1891 and pull
    /// request #1934). The title's bytes come back when it writes the
    /// memory after the card; a level the card did not write is the
    /// title's; a card copy recorded and never submitted changes nothing.
    #[cfg(windows)]
    #[test]
    fn a_texture_holds_what_was_written_last_by_the_order_of_submission() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        const SHADER_READ: u32 = 0x80;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        // Fills every texel of a texture's title memory with one colour.
        let fill = |t: This, colour: [u8; 4]| {
            let desc = resource(t).desc;
            let size = texture_layout(&desc).unwrap().planes[0].layout.size as usize;
            let (memory, _) = texture_memory(t).unwrap();
            // SAFETY: the texture's own committed memory, `size` bytes.
            let bytes = unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), size) };
            for texel in bytes.chunks_exact_mut(4) {
                texel.copy_from_slice(&colour);
            }
        };
        // A 4x4 texture the title fills, of `levels` levels, in one colour.
        let make = |levels: u16, colour: [u8; 4]| -> This {
            let mut t = null;
            let desc = ResourceDesc { layout: 1, ..texture(4, 4, 28, levels) };
            assert_eq!(create_committed(null, &PROPERTIES, 0, &desc, 0, null.cast(), &RESOURCE_IIDS[0], &mut t), S_OK);
            fill(t, colour);
            t
        };
        // The first row of subresource `sub` of a texture's host copy.
        let row = |t: This, sub: u32| -> Vec<[u8; 4]> {
            let image = host_resource(t).unwrap();
            // SAFETY: live resources, an open list, a finished copy.
            unsafe {
                let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let mut total = 0u64;
                host.device.GetCopyableFootprints(&image.GetDesc(), sub, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
                let mut readback: Option<ID3D12Resource> = None;
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: total.max(256),
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut readback,
                    )
                    .unwrap();
                let readback = readback.unwrap();
                let back = host.new_list().unwrap();
                if let Some((_, before)) = host_transition(t, COPY_SOURCE) {
                    back.list.ResourceBarrier(&[crate::host::transition(
                        image,
                        D3D12_RESOURCE_STATES(before as i32),
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                }
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(image),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
                };
                back.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                back.close();
                host.execute(&[&back]);
                let mut mapped = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut mapped)).unwrap();
                let first = mapped.cast::<u8>().add(footprint.Offset as usize).cast::<[u8; 4]>();
                let texels = (0..footprint.Footprint.Width as usize).map(|x| first.add(x).read_unaligned()).collect();
                readback.Unmap(0, None);
                texels
            }
        };
        // Its first texel.
        let texel = |t: This, sub: u32| row(t, sub)[0];
        // A list that samples `t`: its upload is asked for, as a draw's
        // table read asks.
        let reads = |t: This| {
            let list = host.new_list().unwrap();
            crate::texture::upload(&list, t, host_resource(t).unwrap());
            crate::list::ensure_state(&list, t, SHADER_READ);
            list.close();
            list
        };
        // A list that copies `from` into `to` on the card: all of it, or
        // all of one subresource (as `copy_resource` and
        // `copy_texture_region` record it).
        let copies = |to: This, from: This, sub: Option<u32>| {
            let list = host.new_list().unwrap();
            let (host_to, host_from) = (host_resource(to).unwrap(), host_resource(from).unwrap());
            crate::texture::upload(&list, from, host_from);
            crate::list::ensure_state(&list, to, COPY_DEST);
            crate::list::ensure_state(&list, from, COPY_SOURCE);
            crate::texture::host_written(&list, to, host_to, sub);
            // SAFETY: an open list, live textures in the copy states.
            unsafe {
                match sub {
                    None => list.list.CopyResource(host_to, host_from),
                    Some(sub) => {
                        let part = |texture: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(texture),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
                        };
                        list.list.CopyTextureRegion(&part(host_to), 0, 0, 0, &part(host_from), None);
                    }
                }
            }
            list.close();
            list
        };
        let (red, green, blue) = ([200, 10, 10, 255], [10, 200, 10, 255], [10, 10, 200, 255]);
        let counts = || crate::texture::CARD_WRITTEN.each_ref().map(|c| c.load(std::sync::atomic::Ordering::Relaxed));
        let before = counts();

        // The upload recorded first, the card copy submitted first: the
        // card wrote last.
        let (t, s) = (make(1, red), make(1, green));
        let upload = reads(t);
        let copy = copies(t, s, None);
        host.execute(&[&copy]);
        host.execute(&[&upload]);
        assert_eq!(texel(t, 0), green, "an upload recorded before a card copy and submitted after it overwrote the card's texels");
        assert_eq!(host_copy(t).unwrap().card.subs(), 1);
        // The same in one submission.
        let (t2, s2) = (make(1, red), make(1, green));
        let upload = reads(t2);
        let copy = copies(t2, s2, None);
        host.execute(&[&copy, &upload]);
        assert_eq!(texel(t2, 0), green, "in one submission");
        // And the other order: the upload submitted first, then the copy.
        let (t3, s3) = (make(1, red), make(1, green));
        let copy = copies(t3, s3, None);
        let upload = reads(t3);
        host.execute(&[&upload, &copy]);
        assert_eq!(texel(t3, 0), green, "the copy submitted last");
        // A list that samples it later leaves the card's texels, whether
        // it was recorded before or after.
        let later = reads(t);
        host.execute(&[&later]);
        assert_eq!(texel(t, 0), green, "a later reader");
        let now = counts();
        assert_eq!(now[2], before[2], "nothing copied from title memory over what the card wrote last");
        assert!(now[5] + now[6] >= before[5] + before[6] + 3, "the uploads withheld or not asked for are counted");

        // The title writes the memory after the card: its bytes are the
        // last written. The list that samples it then has that found when
        // it is submitted, and the next one uploads them.
        fill(t, blue);
        let again = reads(t);
        host.execute(&[&again]);
        assert_eq!(host_copy(t).unwrap().card.subs(), 0, "the title wrote after the card: found at a submission");
        let again = reads(t);
        host.execute(&[&again]);
        assert_eq!(texel(t, 0), blue, "the title wrote after the card");
        assert!(counts()[7] > before[7], "an upload made again because the title wrote after the card is counted");
        // And the card after the title again.
        let copy = copies(t, s, None);
        host.execute(&[&copy]);
        assert_eq!(texel(t, 0), green, "the card wrote after the title");
        let later = reads(t);
        host.execute(&[&later]);
        assert_eq!(texel(t, 0), green);

        // A card copy into one level: the other level is the title's, and
        // comes from its memory though the upload is submitted after the
        // copy.
        let (levels, source) = (make(2, red), make(2, green));
        let upload = reads(levels);
        let copy = copies(levels, source, Some(0));
        host.execute(&[&copy]);
        host.execute(&[&upload]);
        assert_eq!(texel(levels, 0), green, "the level the card wrote");
        assert_eq!(texel(levels, 1), red, "the level it did not write is the title's");
        assert_eq!(host_copy(levels).unwrap().card.subs(), 1);

        // A card copy recorded and never submitted changes nothing: the
        // texture is the title's.
        let (never, source2) = (make(1, red), make(1, green));
        let dropped = copies(never, source2, None);
        dropped.reset();
        let upload = reads(never);
        host.execute(&[&upload]);
        assert_eq!(texel(never, 0), red, "a copy that was never submitted");
        assert_eq!(host_copy(never).unwrap().card.subs(), 0);

        // Two card copies into different parts of one level, and an upload
        // between them. A copy of a part asks for the destination's upload
        // first, in its own list (as `copy_texture_region` does): title
        // memory goes under the level once, before the card's first copy
        // into it, and never again, so what the first copy wrote stays
        // under the second.
        let part = |to: This, from: This, columns: std::ops::Range<u32>| {
            let list = host.new_list().unwrap();
            let (host_to, host_from) = (host_resource(to).unwrap(), host_resource(from).unwrap());
            crate::texture::upload(&list, from, host_from);
            crate::texture::upload(&list, to, host_to);
            crate::list::ensure_state(&list, to, COPY_DEST);
            crate::list::ensure_state(&list, from, COPY_SOURCE);
            crate::texture::host_written(&list, to, host_to, Some(0));
            let columns = D3D12_BOX { left: columns.start, top: 0, front: 0, right: columns.end, bottom: 4, back: 1 };
            // SAFETY: an open list, live textures in the copy states, a box
            // inside both.
            unsafe {
                let of = |texture: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&of(host_to), columns.left, 0, 0, &of(host_from), Some(&columns));
            }
            list.close();
            list
        };
        let (halves, left, right) = (make(1, red), make(1, green), make(1, blue));
        let own_before = counts();
        let between = reads(halves);
        let first = part(halves, left, 0..2);
        let second = part(halves, right, 2..4);
        host.execute(&[&first]);
        host.execute(&[&between]);
        assert_eq!(row(halves, 0), [green, green, red, red], "title memory under the part the card did not write");
        host.execute(&[&second]);
        assert_eq!(row(halves, 0), [green, green, blue, blue], "the first copy's part stays under the second");
        let later = reads(halves);
        host.execute(&[&later]);
        assert_eq!(row(halves, 0), [green, green, blue, blue]);
        let own = counts();
        assert_eq!(own[2], own_before[2], "nothing copied over what the card wrote last");
        assert!(own[8] > own_before[8], "the level uploaded ahead of its own list's copy is counted as that");

        // A view the title placed in the memory of a texture the card
        // writes shows what the card wrote, though it is a host texture of
        // its own and title memory never holds those bytes: one whose host
        // copy is there when the card writes, and one made afterwards.
        let object = make(1, red);
        let address = resource(object).data() as u64;
        let placed = |format: u32| {
            let desc = ResourceDesc { layout: 1, ..texture(4, 4, format, 1) };
            // SAFETY: 56 bytes in the description's layout.
            let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
            texture_at(address, &words).unwrap()
        };
        let fed_before = counts()[9];
        let view = placed(29);
        let upload = reads(view);
        host.execute(&[&upload]);
        assert_eq!(texel(view, 0), red, "filled from title memory while nothing else wrote the memory");
        let copy = copies(object, left, None);
        host.execute(&[&copy]);
        // The copy into the view runs with the next list submitted.
        let next = reads(right);
        host.execute(&[&next]);
        assert_eq!(texel(view, 0), green, "a placed view of memory the card wrote shows what the card wrote");
        let late = placed(30);
        let upload = reads(late);
        host.execute(&[&upload]);
        assert_eq!(texel(late, 0), green, "and so does one made after the card wrote");
        // An upload asked for either leaves the card's bytes.
        let (again, again2) = (reads(view), reads(late));
        host.execute(&[&again, &again2]);
        assert_eq!((texel(view, 0), texel(late, 0)), (green, green));
        assert!(counts()[9] >= fed_before + 2, "the views fed from the card are counted");

        // An upload brings the levels whose bytes the host copy lacks, and
        // no others. (Level 0 of the host copy is painted here behind the
        // layer's back, by a copy on the card that nothing notes: an
        // upload that copied the level again would put the title's colour
        // back.)
        let big = |colour: [u8; 4]| -> This {
            let mut t = null;
            let desc = ResourceDesc { layout: 1, ..texture(256, 256, 28, 9) };
            assert_eq!(create_committed(null, &PROPERTIES, 0, &desc, 0, null.cast(), &RESOURCE_IIDS[0], &mut t), S_OK);
            fill(t, colour);
            t
        };
        // Fills level `n` of a texture's title memory with one colour.
        let level = |t: This, n: usize, colour: [u8; 4]| {
            let layout = texture_layout(&resource(t).desc).unwrap().planes[0].layout.clone();
            let (memory, _) = texture_memory(t).unwrap();
            let from = layout.mips[n].offset as usize;
            let to = if n == 0 { layout.size as usize } else { layout.mips[n - 1].offset as usize };
            // SAFETY: the level's bytes of the texture's own committed memory.
            let bytes = unsafe { std::slice::from_raw_parts_mut(memory.cast_mut().add(from), to - from) };
            for texel in bytes.chunks_exact_mut(4) {
                texel.copy_from_slice(&colour);
            }
        };
        let holds = |t: This| -> Vec<u64> {
            let copy = host_copy(t).unwrap();
            (0..crate::census::PARTS).map(|part| copy.sent.key(part)).collect()
        };
        let (mips, paint) = (big(red), big(green));
        let (first, source_first) = (reads(mips), reads(paint));
        host.execute(&[&first, &source_first]);
        assert_eq!((texel(mips, 0), texel(mips, 1), texel(mips, 8)), (red, red, red));
        let held = holds(mips);
        assert!(held.iter().filter(|&&key| key != 0).count() >= 2, "what the host copy holds is noted level by level");
        let behind = {
            let list = host.new_list().unwrap();
            let (to, from) = (host_resource(mips).unwrap(), host_resource(paint).unwrap());
            crate::list::ensure_state(&list, mips, COPY_DEST);
            crate::list::ensure_state(&list, paint, COPY_SOURCE);
            // SAFETY: an open list, live textures in the copy states.
            unsafe {
                let of = |texture: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&of(to), 0, 0, 0, &of(from), None);
            }
            list.close();
            list
        };
        host.execute(&[&behind]);
        assert_eq!(texel(mips, 0), green);
        // Nothing written: nothing is copied.
        let again = reads(mips);
        host.execute(&[&again]);
        assert_eq!(texel(mips, 0), green, "an upload with no level changed copied a level");
        assert_eq!(holds(mips), held);
        // The title writes level 1: it goes up, and it alone.
        level(mips, 1, blue);
        let again = reads(mips);
        host.execute(&[&again]);
        assert_eq!((texel(mips, 1), texel(mips, 0), texel(mips, 2)), (blue, green, red), "the level written, and it alone");
        // Then level 0.
        level(mips, 0, blue);
        let again = reads(mips);
        host.execute(&[&again]);
        assert_eq!((texel(mips, 0), texel(mips, 1), texel(mips, 2)), (blue, blue, red));
        // The card writes level 1, and this time it is noted: the level
        // is the card's, whatever the host copy held of title memory there;
        // the title's bytes come back when it writes them.
        let copy = copies(mips, paint, Some(1));
        host.execute(&[&copy]);
        let later = reads(mips);
        host.execute(&[&later]);
        assert_eq!((texel(mips, 1), texel(mips, 0)), (green, blue), "the level the card wrote last");
        level(mips, 1, red);
        let (once, twice) = (reads(mips), reads(mips));
        host.execute(&[&once]);
        host.execute(&[&twice]);
        assert_eq!((texel(mips, 1), texel(mips, 0)), (red, blue), "the title wrote the level after the card");
        assert_eq!(counts()[2], before[2], "nothing copied from title memory over what the card wrote last");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        for t in [t, s, t2, s2, t3, s3, levels, source, never, source2, halves, left, right, object, mips, paint] {
            resource_release(t);
        }
    }

    /// A submission that runs one list twice gives each run a prelude of
    /// its own: the second run's transitions are recorded while the first
    /// run's prelude still waits to be sent, and in the list's own prelude
    /// they would have replaced the first's. Each run starts the texture in
    /// the state the list starts with, and the list ends it in its own.
    #[cfg(windows)]
    #[test]
    fn a_list_run_twice_in_one_submission_has_a_prelude_per_run() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        const COPY_DEST: u32 = 0x400;
        const SHADER_READ: u32 = 0x80;
        const COPY_SOURCE: u32 = 0x800;
        let mut image = null;
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(null, &PROPERTIES, 0, &rgba, COPY_DEST, null.cast(), &RESOURCE_IIDS[0], &mut image),
            S_OK
        );
        let list = host.new_list().unwrap();
        // The list starts with the texture read by shaders and leaves it a
        // copy source.
        assert!(list.transition(image, SHADER_READ).is_none());
        let (host_image, before) = list.transition(image, COPY_SOURCE).unwrap();
        // SAFETY: an open list and a live texture.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                host_image,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES(before as i32),
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        list.close();
        host.execute(&[&list, &list]);
        assert_eq!(list.preludes_again(), 1, "the second run's prelude");
        assert_eq!(host_copy(image).unwrap().state.load(std::sync::atomic::Ordering::Relaxed), COPY_SOURCE);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        list.reset();
        assert_eq!(list.preludes_again(), 0, "let go with the reset");
    }

    /// An earlier title-memory upload must not suppress a later alias copy
    /// from a GPU-written probe atlas into the same sampled texture
    /// (fxceless/victory-fork pull request #1241).
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn alias_after_title_upload_copies_the_gpu_atlas() {
        use std::sync::atomic::Ordering::Relaxed;
        use windows::Win32::Graphics::Direct3D12::*;

        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let mut source = null;
        let mut view = null;
        let source_desc = ResourceDesc {
            flags: 4,
            ..texture(4, 4, 42, 1)
        };
        assert_eq!(
            create_committed(
                null, &PROPERTIES, 0, &source_desc, 0, null.cast(), &RESOURCE_IIDS[0], &mut source,
            ),
            S_OK
        );
        assert_eq!(
            create_at_address(
                null, resource(source).data() as u64, &texture(4, 4, 67, 1), 0,
                null.cast(), &RESOURCE_IIDS[0], &mut view,
            ),
            S_OK
        );
        resource(view).host.placed.store(true, Relaxed);
        assert_eq!(alias_of(view), Some(source));
        let written = host_resource(source).unwrap();
        let sampled = host_resource(view).unwrap();
        let (upload, data) = host.upload_buffer(1024).unwrap();
        let seed = host.new_list().unwrap();
        let packed_light = 0x8c02_0100_u32;
        // SAFETY: four rows of four texels in the upload's 1024 mapped bytes.
        unsafe {
            for row in 0..4 {
                for x in 0..4 {
                    data.add(row * 256 + x * 4).cast::<u32>().write_unaligned(packed_light);
                }
            }
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(written),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: written.GetDesc().Format,
                            Width: 4,
                            Height: 4,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            crate::list::ensure_state(&seed, source, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32);
            seed.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        seed.close();
        host.execute(&[&seed]);

        let list = host.new_list().unwrap();
        crate::texture::upload(&list, view, sampled);
        assert!(crate::texture::alias(&list, view, source));
        crate::list::ensure_state(&list, view, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let (readback, rows) = crate::capture::read_back(&host.device, &list.list, sampled).unwrap();
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the completed copy contains at least one 32-bit texel.
        let actual = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let texel = mapped.cast::<u8>().add(rows.Offset as usize).cast::<u32>().read_unaligned();
            readback.Unmap(0, None);
            texel
        };
        resource_release(view);
        resource_release(source);
        assert_eq!(actual, packed_light, "the sampled texture kept the CPU upload instead of the GPU atlas");
    }

    /// A 16-bit depth texture the title also asks unordered access for gets
    /// a colour twin, and the host copies one into the other, both ways: a
    /// pattern put into the twin is copied to the depth texture, the twin
    /// is overwritten with zeros, and the pattern comes back from the depth
    /// texture. A copy the host does not allow ends the test (with the
    /// debug layer's message under `VICTORY_D3D12_DEBUG=1` in a diagnostics
    /// build).
    #[cfg(windows)]
    #[test]
    fn a_depth_texture_and_its_twin_copy_into_each_other() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        const DEPTH_WRITE: u32 = 0x10;
        let desc = ResourceDesc {
            flags: 6,
            ..texture(4, 4, 55, 1)
        };
        let mut depth = null;
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &desc,
                DEPTH_WRITE,
                null.cast(),
                &RESOURCE_IIDS[0],
                &mut depth
            ),
            S_OK
        );
        let host_depth = host_resource(depth).unwrap();
        let Some(twin) = twin_resource(depth) else {
            return; // twins are turned off
        };
        // SAFETY: live resources.
        let (d, t) = unsafe { (host_depth.GetDesc(), twin.GetDesc()) };
        // Both R16 typeless: one a depth texture, one with unordered access.
        assert_eq!((d.Format.0, d.Flags.0), (53, 2));
        assert_eq!((t.Format.0, t.Flags.0), (53, 4));

        // Four rows of four 16-bit texels, 256 bytes apart as copies place
        // them: the pattern, then as many zeros.
        let mut pattern = vec![0u8; 1024];
        for row in 0..4 {
            for b in 0..8 {
                pattern[row * 256 + b] = (1 + row * 8 + b) as u8;
            }
        }
        let (upload, data) = host.upload_buffer(2048).unwrap();
        // SAFETY: the buffer's mapping, 2048 bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 1024);
            std::ptr::write_bytes(data.add(1024), 0, 1024);
        }
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();

        // A texture's one subresource; the four rows of a buffer at `offset`.
        let whole = |texture: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let rows = |buffer: &ID3D12Resource, offset: u64| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT(53),
                        Width: 4,
                        Height: 4,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        let (written, tested, source, dest) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let moved = crate::host::transition;
        let list = host.new_list().unwrap();
        // SAFETY: an open list and live resources; the two textures are one
        // size in one format family.
        unsafe {
            let l = &list.list;
            // The pattern into the twin, the twin into the depth texture.
            l.ResourceBarrier(&[moved(twin, written, dest), moved(host_depth, tested, dest)]);
            l.CopyTextureRegion(&whole(twin), 0, 0, 0, &rows(&upload, 0), None);
            l.ResourceBarrier(&[moved(twin, dest, source)]);
            l.CopyResource(host_depth, twin);
            // Zeros over the twin, then the depth texture back into it.
            l.ResourceBarrier(&[moved(twin, source, dest)]);
            l.CopyTextureRegion(&whole(twin), 0, 0, 0, &rows(&upload, 1024), None);
            l.ResourceBarrier(&[moved(host_depth, dest, source)]);
            l.CopyResource(twin, host_depth);
            // The twin read back; both textures to the states they had.
            l.ResourceBarrier(&[moved(twin, dest, source)]);
            l.CopyTextureRegion(&rows(&readback, 0), 0, 0, 0, &whole(twin), None);
            l.ResourceBarrier(&[
                moved(twin, source, written),
                moved(host_depth, source, tested),
            ]);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 8], pattern[at..at + 8], "row {row}");
        }
    }

    /// A depth texture with a stencil plane that the title also asks
    /// unordered access for gets a twin of the stencil plane, and what
    /// shaders write there is in the stencil plane once the list is closed:
    /// a pattern put into the twin is read back from the plane.
    #[cfg(windows)]
    #[test]
    fn a_stencil_planes_twin_reaches_the_plane() {
        use std::sync::atomic::Ordering::Relaxed;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        const DEPTH_WRITE: u32 = 0x10;
        let desc = ResourceDesc {
            flags: 6,
            ..texture(4, 4, 45, 1)
        };
        let mut depth = null;
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &desc,
                DEPTH_WRITE,
                null.cast(),
                &RESOURCE_IIDS[0],
                &mut depth
            ),
            S_OK
        );
        let host_depth = host_resource(depth).unwrap();
        let Some(twin) = twin_resource(depth) else {
            return; // twins are turned off
        };
        // SAFETY: live resources.
        let (d, t) = unsafe { (host_depth.GetDesc(), twin.GetDesc()) };
        // R24G8 typeless with depth, R8 typeless with unordered access.
        assert_eq!((d.Format.0, d.Flags.0), (44, 2));
        assert_eq!((t.Format.0, t.Flags.0), (60, 4));

        // Four rows of four texels, 256 bytes apart as copies place them.
        let mut pattern = vec![0u8; 1024];
        for row in 0..4 {
            for b in 0..4 {
                pattern[row * 256 + b] = (1 + row * 4 + b) as u8;
            }
        }
        let (upload, data) = host.upload_buffer(1024).unwrap();
        // SAFETY: the buffer's mapping, 1024 bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 1024) };
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            // Room for the rows as the host places them (512 bytes apart
            // for the stencil plane on NVIDIA, not 256).
            Width: 4096,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        // How the host lays the stencil plane (subresource 1) out in a buffer.
        let mut plane = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        // SAFETY: a plain query writing one footprint.
        unsafe {
            host.device
                .GetCopyableFootprints(&d, 1, 1, 0, Some(&mut plane), None, None, None);
        }
        let pattern_rows = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: DXGI_FORMAT(60),
                Width: 4,
                Height: 4,
                Depth: 1,
                RowPitch: 256,
            },
        };

        // A subresource of a texture; rows of a buffer.
        let whole = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        };
        let rows = |buffer: &ID3D12Resource, placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: placed,
                },
            }
        };
        let (written, source, dest) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let moved = crate::host::transition;

        // The pattern into the twin, where a shader would write it.
        let fill = host.new_list().unwrap();
        // SAFETY: an open list and live resources.
        unsafe {
            let l = &fill.list;
            l.ResourceBarrier(&[moved(twin, written, dest)]);
            l.CopyTextureRegion(&whole(twin, 0), 0, 0, 0, &rows(&upload, pattern_rows), None);
            l.ResourceBarrier(&[moved(twin, dest, written)]);
        }
        fill.close();
        host.execute(&[&fill]);

        // A list in which a shader writes the twin: closing it copies the
        // twin into the stencil plane.
        let list = host.new_list().unwrap();
        assert!(list.twin_use(depth, true, true));
        list.twins_written();
        list.close();
        host.execute(&[&list]);

        // The stencil plane read back from the state the list left it in.
        let state = host_copy(depth).unwrap().state.load(Relaxed);
        let state = D3D12_RESOURCE_STATES(state as i32);
        let back = host.new_list().unwrap();
        // SAFETY: an open list and live resources; a whole subresource.
        unsafe {
            let l = &back.list;
            l.ResourceBarrier(&[moved(host_depth, state, source)]);
            let (to, from) = (rows(&readback, plane), whole(host_depth, 1));
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            l.ResourceBarrier(&[moved(host_depth, source, state)]);
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copies finished (execute waits); the buffer is 4096 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 4096,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 4096)
        };
        let pitch = plane.Footprint.RowPitch as usize;
        for row in 0..4 {
            let (at, from) = (row * pitch, row * 256);
            assert_eq!(read[at..at + 4], pattern[from..from + 4], "row {row}");
        }
    }

    /// The depth plane of a depth-stencil texture is drawn from the plane's
    /// twin when the list that wrote the twin closes: class numbers are put
    /// into the twin of a texture in the console's 16-bit depth-stencil
    /// format, and draws with the depth test "equal" at the depth the
    /// title's lighting computes for a class (the number times the float
    /// nearest 1/65535) then cover the texels of that class and no other.
    #[cfg(windows)]
    #[test]
    fn a_depth_planes_twin_is_drawn_into_the_plane() {
        use std::sync::atomic::Ordering::Relaxed;
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
        use windows::core::s;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        const DEPTH_WRITE: u32 = 0x10;
        let desc = ResourceDesc {
            flags: 6,
            ..texture(4, 4, 118, 1)
        };
        let mut depth = null;
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &desc,
                DEPTH_WRITE,
                null.cast(),
                &RESOURCE_IIDS[0],
                &mut depth
            ),
            S_OK
        );
        let host_depth = host_resource(depth).unwrap();
        let Some(plane) = twin_plane_resource(depth) else {
            return; // these twins are turned off
        };
        // SAFETY: live resources.
        let (d, t) = unsafe { (host_depth.GetDesc(), plane.GetDesc()) };
        // R24G8 typeless with depth, R16 typeless with unordered access.
        assert_eq!((d.Format.0, d.Flags.0), (44, 2));
        assert_eq!((t.Format.0, t.Flags.0), (53, 4));

        // Four rows of four class numbers (16 bits each), 256 bytes apart
        // as copies place them. 128 and 129 are the classes whose 24-bit
        // depth lies nearest a half step.
        const CLASSES: [u16; 16] = [
            8, 65, 128, 129, 65, 8, 129, 128, 0, 255, 65, 64, 129, 130, 8, 65,
        ];
        let mut pattern = vec![0u8; 1024];
        for (i, class) in CLASSES.into_iter().enumerate() {
            let at = i / 4 * 256 + i % 4 * 2;
            pattern[at..at + 2].copy_from_slice(&class.to_le_bytes());
        }
        let (upload, data) = host.upload_buffer(1024).unwrap();
        // SAFETY: the buffer's mapping, 1024 bytes.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), data, 1024) };
        let buffer = |kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a buffer.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut made,
                    )
                    .unwrap();
            }
            made.unwrap()
        };
        let readback = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        // The colour target the draws cover: 4x4, 8 bits a texel.
        let target = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT(61),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a texture.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut made,
                    )
                    .unwrap();
            }
            made.unwrap()
        };
        let rows_of = |format: i32| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: DXGI_FORMAT(format),
                Width: 4,
                Height: 4,
                Depth: 1,
                RowPitch: 256,
            },
        };
        // A texture's one subresource; rows of a buffer.
        let whole = |texture: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let rows = |buffer: &ID3D12Resource, placed: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: placed,
                },
            }
        };
        let (written, source, dest, tested, drawn) = (
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let moved = crate::host::transition;

        // The class numbers into the twin, where a shader would write them.
        let fill = host.new_list().unwrap();
        // SAFETY: an open list and live resources.
        unsafe {
            let l = &fill.list;
            l.ResourceBarrier(&[moved(plane, written, dest)]);
            l.CopyTextureRegion(&whole(plane), 0, 0, 0, &rows(&upload, rows_of(53)), None);
            l.ResourceBarrier(&[moved(plane, dest, written)]);
        }
        fill.close();
        host.execute(&[&fill]);

        // A list in which a shader writes the twins: closing it draws the
        // depth plane's twin into the plane.
        let list = host.new_list().unwrap();
        assert!(list.twin_use(depth, true, true));
        list.twins_written();
        list.close();
        host.execute(&[&list]);

        // The draws: a triangle over the target at the depth of a class,
        // as the title's lighting places its rectangles.
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants = 1, b0)"
cbuffer Drawn : register(b0) { uint number; };
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float z = float(number) * asfloat(0x37800080);
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, z, 1.0);
}
float4 lit() : SV_Target { return 1.0; }
"#;
        let vs = crate::host::compile(SOURCE, s!("corner"), s!("vs_5_1")).unwrap();
        let ps = crate::host::compile(SOURCE, s!("lit"), s!("ps_5_1")).unwrap();
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT(61);
        // SAFETY: plain creation; the description points at live data.
        let (signature, pipeline) = unsafe {
            let signature: ID3D12RootSignature =
                host.device.CreateRootSignature(0, &vs[..]).unwrap();
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&signature),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // As the title's lighting: test "equal", nothing written.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT(45),
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState =
                host.device.CreateGraphicsPipelineState(&desc).unwrap();
            (signature, pipeline)
        };
        let state = host_copy(depth).unwrap().state.load(Relaxed);
        let state = D3D12_RESOURCE_STATES(state as i32);
        let colour = host.rtv(Some(&target), None);
        let depth_view = host.dsv(
            host_depth,
            Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: DXGI_FORMAT(45),
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                ..Default::default()
            }),
        );
        let back = host.new_list().unwrap();
        // SAFETY: an open list and live resources and views.
        unsafe {
            let l = &back.list;
            if state != tested {
                l.ResourceBarrier(&[moved(host_depth, state, tested)]);
            }
            l.ClearRenderTargetView(colour, &[0.0; 4], None);
            l.SetGraphicsRootSignature(&signature);
            l.SetPipelineState(&pipeline);
            l.OMSetRenderTargets(1, Some(&colour), false, Some(&depth_view));
            l.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 4.0,
                Height: 4.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            l.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            }]);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            // Classes 65 and 129 are drawn; the others are not.
            for class in [65u32, 129] {
                l.SetGraphicsRoot32BitConstant(0, class, 0);
                l.DrawInstanced(3, 1, 0, 0);
            }
            l.ResourceBarrier(&[moved(&target, drawn, source)]);
            let (to, from) = (rows(&readback, rows_of(61)), whole(&target));
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for (i, class) in CLASSES.into_iter().enumerate() {
            let covered = read[i / 4 * 256 + i % 4] != 0;
            let drawn = matches!(class, 65 | 129);
            assert_eq!(covered, drawn, "texel {i}, class {class}");
        }
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }

    /// A list cut after the title set its targets: the next host list binds
    /// the views the title set, though it wrote the render target or depth
    /// descriptor (or both) again in between (DexNom PR #48, by Cuebert).
    #[cfg(windows)]
    #[test]
    fn a_list_continued_after_a_cut_draws_into_the_views_that_were_set() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        let image = |format: u32, flags: u32, state: u32| {
            let desc = ResourceDesc {
                flags,
                ..texture(4, 4, format, 1)
            };
            let mut out = std::ptr::null();
            let made = create_committed(
                std::ptr::null(),
                &PROPERTIES,
                0,
                &desc,
                state,
                std::ptr::null(),
                &RESOURCE_IIDS[0],
                &mut out,
            );
            assert_eq!(made, S_OK);
            out
        };
        // Two colour targets and two depth targets (render target; depth
        // without shader access).
        let colours = [0, 1].map(|_| image(28, 1, RENDER_TARGET));
        let depths = [0, 1].map(|_| image(40, 2 | 8, DEPTH_WRITE));
        let colour = colours.map(|r| host_resource(r).unwrap().clone());
        let depth = depths.map(|r| host_resource(r).unwrap().clone());
        let source = r#"
            [RootSignature("RootFlags(0)")]
            float4 vs(uint id : SV_VertexID) : SV_Position {
                float2 p = float2((id << 1) & 2, id & 2);
                return float4(2 * p - 1, 0.5, 1);
            }
            float4 ps() : SV_Target { return float4(1, 0, 0, 1); }
        "#;
        let vs = crate::host::compile(source, s!("vs"), s!("vs_5_1")).unwrap();
        let ps = crate::host::compile(source, s!("ps"), s!("ps_5_1")).unwrap();
        let code = |s: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: s.as_ptr().cast(),
            BytecodeLength: s.len(),
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        // SAFETY: plain creation; the description points at live data.
        let (root, pipeline) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &vs).unwrap();
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Drawn where the depth target holds 1 (the first), not 0.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = host.device.CreateGraphicsPipelineState(&desc).unwrap();
            (root, pipeline)
        };
        let readback = host.read_back_buffer(2048).unwrap();
        let mut results = Vec::new();
        for (colour_again, depth_again) in [(false, false), (true, false), (false, true), (true, true)] {
            // The title's descriptor slots (at least 112 bytes each).
            let mut colour_slot = [0u64; 16];
            let mut depth_slot = [0u64; 16];
            let (c, d) = (colour_slot.as_mut_ptr() as u64, depth_slot.as_mut_ptr() as u64);
            let write = |ci: usize, di: usize| {
                crate::descriptor::create_rtv(std::ptr::null(), colours[ci], std::ptr::null(), c);
                crate::descriptor::create_dsv(std::ptr::null(), depths[di], std::ptr::null(), d);
            };
            write(0, 0);
            let first = host.new_list().unwrap();
            let mut state = crate::draw::State::default();
            state.set_targets(&first, &[c], d);
            // SAFETY: an open list; the targets are in the states named.
            unsafe {
                for i in 0..2 {
                    first.list.ClearRenderTargetView(host.rtv(Some(&colour[i]), None), &[0.0; 4], None);
                    let far = if i == 0 { 1.0 } else { 0.0 };
                    first
                        .list
                        .ClearDepthStencilView(host.dsv(&depth[i], None), D3D12_CLEAR_FLAG_DEPTH, far, 0, None);
                }
            }
            first.close();
            host.execute(&[&first]);
            write(usize::from(colour_again), usize::from(depth_again));
            let list = host.new_list().unwrap();
            state.replay(&list);
            // SAFETY: an open list; what it uses lives past its execution.
            unsafe {
                let l = &list.list;
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pipeline);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 4.0,
                    Height: 4.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: 4,
                    bottom: 4,
                }]);
                l.DrawInstanced(3, 1, 0, 0);
                for (i, image) in colour.iter().enumerate() {
                    l.ResourceBarrier(&[crate::host::transition(
                        image,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&readback),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: 1024 * i as u64,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                    Width: 4,
                                    Height: 4,
                                    Depth: 1,
                                    RowPitch: 256,
                                },
                            },
                        },
                    };
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(image),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    };
                    l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                    l.ResourceBarrier(&[crate::host::transition(
                        image,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                    )]);
                }
            }
            list.close();
            host.execute(&[&list]);
            let mut mapped = std::ptr::null_mut();
            let mut red = [0; 2];
            // SAFETY: the copies finished (execute waits); both images lie
            // in rows of 256 bytes in the 2048.
            unsafe {
                readback.Map(0, None, Some(&mut mapped)).unwrap();
                for (i, count) in red.iter_mut().enumerate() {
                    for y in 0..4 {
                        for x in 0..4 {
                            let texel = mapped.cast::<u8>().add(i * 1024 + y * 256 + x * 4).cast::<[u8; 4]>().read();
                            assert!(matches!(texel, [0, 0, 0, 0] | [255, 0, 0, 255]), "texel {texel:?}");
                            *count += usize::from(texel[0] == 255);
                        }
                    }
                }
                readback.Unmap(0, Some(&D3D12_RANGE::default()));
            }
            results.push((colour_again, depth_again, red));
        }
        for r in colours.into_iter().chain(depths) {
            slot::<Release>(r, 0x10)(r);
        }
        assert_eq!(
            results,
            [(false, false), (true, false), (false, true), (true, true)].map(|(c, d)| (c, d, [16, 0])),
            "a descriptor written again changed the targets of a list continued after a cut"
        );
    }
}
