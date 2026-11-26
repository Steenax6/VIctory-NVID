//! A stand-in for the indirect light while no rays run.
//!
//! The title's indirect diffuse light comes from screen probes it fills by
//! tracing rays. Those need the ray-tracing structures, whose builder is off
//! by default (`VICTORY_BVH_BUILDER`), so the program that resolves the
//! probes leaves next to nothing, or whatever its inputs' memory held, and
//! whatever the sun does not reach directly is black. Right after that
//! program, a small compute pass of ours replaces its output (R9G9B9E5 as
//! 32-bit integers) with a light of a set level and colour, shaded in
//! creases and under objects from the scene's depth. The title's own
//! lighting then applies it with each surface's colours. Replaced, not
//! added: the result is the next frame's history, and what the resolve
//! wrote without rays is garbage (bright magenta on translucent surfaces,
//! blocks of colours indoors).
//!
//! Without rays the reflections are the sky's too, unoccluded, wherever a
//! surface is glossy: indoors a wooden floor, a counter and glass tops wore
//! a white glaze five to ten times brighter than the room. After the
//! program that writes the reflections for the last time in a frame, they
//! are held to a part of the ambient light.
//!
//! And the program that tells the screen probes which rays to trace is left
//! out while the stand-in is on: with no rays behind it, the picture lost
//! light, shadows and reflections.
//!
//! The programs are known by the first 16 hex digits of their containers'
//! BLAKE3 hash, as the run's log and `victory translate --list` print them
//! (as `draw::BAKE_PIXEL_SHADERS`); `VICTORY_AMBIENT_PROGRAMS` adds others
//! (see [`role`]).
//!
//! `VICTORY_AMBIENT`: unset, a constant level of 0.3; `<level>` that level;
//! `sky` a level that follows the sky the title draws ([`follow`]); `off`
//! the title's own output and programs. Always off while the structure
//! builder may run (`VICTORY_BVH_BUILDER=1`).
//!
//! From Belobug/VIctory-Belobug-AMD (`ambient.rs`, 0603123 to 4f62c2a,
//! e48eb83, 44dcd43, c3bd561, 6986a57; GPL-3.0), which fxceless/victory-fork
//! PR #1093 and VItoriaBr d0af132 took over; its programs are matched by
//! name there, by hash here. The default level is the one compared against
//! console footage (Marvisak/VIctory PR #2619: 0.3).
//! shortcut: the same light everywhere, from everywhere; it goes once the
//! title's rays run.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// What a program is to the stand-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// It resolves the screen probes' indirect diffuse light over time:
    /// its output is replaced.
    Resolve,
    /// It writes the reflections for the last time in a frame (the
    /// specular denoiser's history update): they are held.
    Reflections,
    /// It draws the sky as the title reflects it (a paraboloid map): the
    /// light may follow it.
    Sky,
    /// It tells the screen probes which rays to trace: left out.
    ActivateRays,
}

/// The title's programs by role, by the first 16 hex digits of their
/// hash: the six variants of the resolve (plain, with reflective caustics,
/// each in three builds), the specular denoiser's history update, the sky
/// dome's pixel program and the probes' ray activation.
const PROGRAMS: &[(&str, Role)] = &[
    ("2fc5d11ece80c884", Role::Resolve),
    ("34fb51b8a0312e82", Role::Resolve),
    ("5053352ac4370ca1", Role::Resolve),
    ("2bed88a03a2f5558", Role::Resolve),
    ("cf76630573822d04", Role::Resolve),
    ("3b4f6eb5edb79848", Role::Resolve),
    ("6fad2980c2774c90", Role::Reflections),
    ("a7cfe758751c20aa", Role::Sky),
    ("39f0e9aeb156a61c", Role::ActivateRays),
];

/// The role of the program with hash `hash` (16 hex digits), if any:
/// [`PROGRAMS`], and those `VICTORY_AMBIENT_PROGRAMS` names as
/// `<hash>=<role>` pairs separated by commas (`resolve`, `reflections`,
/// `sky`, `rays`), for another build of the title.
pub(crate) fn role(hash: &str) -> Option<Role> {
    static MORE: OnceLock<Vec<(String, Role)>> = OnceLock::new();
    let more = MORE.get_or_init(|| {
        std::env::var("VICTORY_AMBIENT_PROGRAMS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|pair| {
                let (hash, role) = pair.split_once('=')?;
                let role = match role.trim() {
                    "resolve" => Role::Resolve,
                    "reflections" => Role::Reflections,
                    "sky" => Role::Sky,
                    "rays" => Role::ActivateRays,
                    _ => return None,
                };
                Some((hash.trim().to_lowercase(), role))
            })
            .collect()
    });
    PROGRAMS
        .iter()
        .map(|&(h, r)| (h, r))
        .chain(more.iter().map(|(h, r)| (h.as_str(), *r)))
        .find(|&(h, _)| h == hash)
        .map(|(_, r)| r)
}

/// Where the resolve writes the indirect diffuse light (R9G9B9E5 as 32-bit
/// integers): root parameter and dword in its table.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
/// Where the resolve reads the scene's depth (reversed, D32 with stencil).
pub(crate) const DEPTH: (u8, u32) = (0, 464);
/// Where the reflections' last program writes them (R9G9B9E5 as 32-bit
/// integers).
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

/// How the light's level is set.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Level {
    /// The same level always.
    Constant(f32),
    /// Following the sky: `gain * mean^power + least` of the sky image's
    /// mean luminance, in its colour ([`follow`]).
    Sky([f32; 3]),
}

/// The default constant level. Against console footage of three prison
/// shots 0.6 came out 12 to 16 display-luma steps too bright and 0.3
/// within 2 to 4 (tippythetaidum, Marvisak/VIctory PR #2619, whose constant
/// is ours: the R9G9B9E5 value the lighting reads, before the surfaces'
/// colours); 0.6 "looked balanced" by eye on an RTX 4070 (fxceless/
/// victory-fork PR #1093) and in Belobug's run a2.
const DEFAULT: f32 = 0.3;

/// The level, from `VICTORY_AMBIENT` ([`parse_level`]); `None`: off.
fn level() -> Option<Level> {
    static LEVEL: OnceLock<Option<Level>> = OnceLock::new();
    *LEVEL.get_or_init(|| parse_level(std::env::var("VICTORY_AMBIENT").ok().as_deref(), follow()))
}

/// `VICTORY_AMBIENT` read: unset (or not a level) the default constant,
/// `off` none, `sky` following the sky with `sky`'s settings, a number
/// that constant (0 for no indirect diffuse light at all).
fn parse_level(set: Option<&str>, sky: [f32; 3]) -> Option<Level> {
    match set.map(str::trim) {
        Some("off") => None,
        Some("sky") => Some(Level::Sky(sky)),
        Some(v) => Some(Level::Constant(
            v.parse().ok().filter(|l: &f32| l.is_finite() && *l >= 0.0).unwrap_or(DEFAULT),
        )),
        None => Some(Level::Constant(DEFAULT)),
    }
}

/// Whether the stand-in runs: set so, and the structure builder may not
/// run (its rays would fill the probes).
pub(crate) fn on() -> bool {
    level().is_some() && !crate::pipeline::builder_allowed()
}

/// The light's colour: warm white, or `VICTORY_AMBIENT_TINT=r,g,b`.
fn tint() -> [f32; 3] {
    static SET: OnceLock<[f32; 3]> = OnceLock::new();
    *SET.get_or_init(|| numbers("VICTORY_AMBIENT_TINT", [1.0, 0.98, 0.94]))
}

/// The most a reflection may hold, as a part of the ambient light's level:
/// `VICTORY_AMBIENT_REFLECT=<part>[,<least part>]` (0.5 and 0 by default;
/// 0 leaves the title's reflections alone). The least part gives what
/// mirrors nothing (still water, which without rays sees no sky and is
/// black) a part of the ambient light.
/// shortcut: stands in for what the title's traced reflections would see.
fn reflect() -> [f32; 2] {
    static SET: OnceLock<[f32; 2]> = OnceLock::new();
    *SET.get_or_init(|| numbers("VICTORY_AMBIENT_REFLECT", [0.5, 0.0]).map(|k| k.max(0.0)))
}

/// Following the sky (`VICTORY_AMBIENT=sky`): gain, least level and power
/// (`VICTORY_AMBIENT_SKY=gain,least,power`; 0.36, 0.05, 0.4). Hand-set
/// levels that looked right were 0.07 at night, 0.3 at dusk and 1.4 at
/// noon, with sky means of 0.001 at night and 0.65 at dawn; the power is
/// under one because the title's exposure follows the light (Belobug,
/// runs lum1, pr1, av1, ciel3, opt1b).
/// shortcut: the sky's mean stands in for the light the traced probes
/// would gather; indoors it is the sky's too.
fn follow() -> [f32; 3] {
    numbers("VICTORY_AMBIENT_SKY", [0.36, 0.05, 0.4])
}

/// How much less of the light reaches creases and corners, from the depth
/// image alone: strength (0 for none), the radius looked at in metres, the
/// near plane's distance, and the tangent of half the vertical field of
/// view (the last two are guesses the picture is not sensitive to).
/// `VICTORY_AMBIENT_SHADE=s,r,n,t`.
/// shortcut: stands in for the occlusion the traced probes carry.
fn shade() -> [f32; 4] {
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| numbers("VICTORY_AMBIENT_SHADE", [1.0, 0.6, 0.1, 0.45]))
}

/// The numbers of switch `name`, separated by commas, each in place of the
/// default's at its place.
fn numbers<const N: usize>(name: &str, default: [f32; N]) -> [f32; N] {
    let set = std::env::var(name).unwrap_or_default();
    let mut out = default;
    for (to, v) in out.iter_mut().zip(set.split(',')) {
        if let Ok(v) = v.trim().parse::<f32>()
            && v.is_finite()
        {
            *to = v;
        }
    }
    out
}

/// The sky image the sky program last drew into: its host copy, held,
/// since the title's object may go at any time.
///
/// The program draws twice a frame, on two lists recorded at once: the dome
/// (128x128, R16G16B16A16_FLOAT) and another image of the sky (R11G11B10).
/// Following whichever was recorded last made the light jump between the
/// two, three times apart, from frame to frame (Belobug e48eb83): only the
/// dome is followed.
static SKY_IMAGE: std::sync::Mutex<Option<std::sync::Arc<crate::resource::HostTexture>>> = std::sync::Mutex::new(None);

/// The sky program draws into title texture `image`.
pub(crate) fn sky_drawn(image: usize) {
    if !matches!(level(), Some(Level::Sky(_))) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(image as This) else { return };
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    if d.Format != DXGI_FORMAT_R16G16B16A16_FLOAT || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        return;
    }
    let mut held = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner());
    if !held.as_ref().is_some_and(|h| std::sync::Arc::ptr_eq(h, &copy)) {
        *held = Some(copy);
    }
}

/// Passes recorded since the start: lights replaced, reflections held.
pub(crate) static PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=13, b0), DescriptorTable(UAV(u0), SRV(t0), SRV(t1))"
cbuffer Light : register(b0) {
    float level; float3 tint;
    float strength; float radius; float near; float tan_y;
    float gain; float least;
    float held; float floor_part; float power;
};
Texture2DArray<float4> sky : register(t1);
// The light, from the sky's mean when it follows the sky (gain above 0).
float3 from_around() {
    if (gain <= 0.0) return tint * level;
    uint sw, sh, slices;
    sky.GetDimensions(sw, sh, slices);
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            // The sun's own texels are held down: it lights by itself.
            float3 there = sky.Load(int4((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0, 0)).rgb;
            sum += clamp(there, 0.0, 8.0);
        }
    float3 mean = sum / (N * N);
    float m = dot(mean, float3(0.2126, 0.7152, 0.0722));
    // The sky's colour, half washed out (the ground lights too); a dark
    // sky has none worth following.
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.5), saturate(m / 0.05));
    return hue * (gain * pow(max(m, 0.0), power) + least);
}
RWTexture2D<uint> light : register(u0);
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = near / max(d, 1e-7);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
// How much of the light from around reaches the texel: 1 in the open.
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    light.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    if (held > 0.0) {
        // Reflections: no brighter than a part of the light around, and
        // no darker than another part of it.
        const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
        float3 seen_there = unpack(light[at.xy]);
        float3 around = from_around();
        float most = dot(around, LUMA) * held;
        float has = dot(seen_there, LUMA);
        float3 floor_light = around * floor_part;
        if (has > most) light[at.xy] = pack(seen_there * (most / has));
        else if (has < dot(floor_light, LUMA)) light[at.xy] = pack(floor_light);
        return;
    }
    float open = strength > 0.0 ? open_to_light(int2(at.xy), float2(w, h)) : 1.0;
    light[at.xy] = pack(from_around() * open);
}
"#;

/// View triples in our heap, used in turn: more than the passes of every
/// list in flight.
const SLOTS: u32 = 1024;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let code = crate::host::compile(SOURCE, s!("main"), s!("cs_5_1"))?;
    // SAFETY: the root signature is inside the shader; plain creations.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * 3,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        Ok(Pass {
            root,
            pso,
            views,
            step: device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Replaces what title texture `light` holds with the stand-in light, in
/// `list`, after the program that wrote it as an image; `depth` is the
/// depth the program read. Whether it recorded anything (the list's
/// pipeline, root signature and heaps are then ours).
pub(crate) fn replace(list: &HostList, light: This, depth: Option<This>) -> bool {
    let done = record(list, light, depth, 0.0);
    if done {
        PASSES[0].fetch_add(1, Ordering::Relaxed);
    }
    done
}

/// Holds the reflections in title texture `image` to a part of the
/// ambient light, in `list`, after the program that wrote them last.
/// Whether it recorded anything, as [`replace`].
pub(crate) fn hold(list: &HostList, image: This) -> bool {
    let held = reflect()[0];
    if held <= 0.0 || !record(list, image, None, held) {
        return false;
    }
    PASSES[1].fetch_add(1, Ordering::Relaxed);
    static HELD: std::sync::Once = std::sync::Once::new();
    HELD.call_once(|| tracing::info!(held, "ambient light: reflections held to it"));
    true
}

fn record(list: &HostList, light: This, depth: Option<This>, held: f32) -> bool {
    let (Some(level), Some(host)) = (level(), host()) else {
        return false;
    };
    let Some(light) = crate::resource::host_resource(light) else {
        return false;
    };
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device)
                .inspect_err(|e| tracing::warn!(error = %e, "no ambient light: its pass was not made"))
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.GetDesc() };
    static ONCE: std::sync::Once = std::sync::Once::new();
    if d.Format != DXGI_FORMAT_R32_UINT || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
        ONCE.call_once(|| tracing::warn!(format = d.Format.0, "no ambient light: the light image is in another format"));
        return false;
    }
    ONCE.call_once(|| {
        tracing::info!(?level, width = d.Width, height = d.Height, "ambient light: stands in for the indirect light while no rays run");
    });
    let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS * 3;
    // The sky image, if the light follows it and the title has drawn it.
    let sky_image = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let follow = match level {
        Level::Sky(f) => sky_image.as_ref().map(|s| (s, f)),
        Level::Constant(_) => None,
    };
    if let Some((copy, f)) = follow {
        static FOLLOWS: std::sync::Once = std::sync::Once::new();
        FOLLOWS.call_once(|| tracing::info!(gain = f[0], least = f[1], power = f[2], "ambient light: follows the sky"));
        list.keep(&copy.texture);
        const READ: u32 = (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0) as u32;
        if let Some(before) = list.transition_copy(copy, READ) {
            let after = crate::host::legal_state(&copy.texture, READ);
            let barrier = crate::host::transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(after as i32),
            );
            // SAFETY: an open list and a live texture the list keeps.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
    }
    list.keep(light);
    // The depth the resolve just read, in the state it read it in; of the
    // light's size, or the shading is left out.
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    if let Some(depth) = depth {
        list.keep(depth);
    }
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: an open list; live textures the list keeps, the light in the
    // unordered-access state the program before left it in; views in
    // slots no list in flight reads.
    unsafe {
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot * pass.step) as usize,
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device
            .CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&view), at);
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        let beside = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + pass.step as usize };
        host.device.CreateShaderResourceView(depth, Some(&seen), beside);
        let around = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
            },
        };
        let third = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at.ptr + 2 * pass.step as usize };
        host.device
            .CreateShaderResourceView(follow.map(|(s, _)| &s.texture), Some(&around), third);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        let [r, g, b] = tint();
        let constant = match level {
            Level::Constant(c) => c,
            // Before the sky is drawn: the default.
            Level::Sky(_) => DEFAULT,
        };
        let [strength, radius, near, tan_y] = shade();
        let strength = if depth.is_some() { strength } else { 0.0 };
        let [gain, least, power] = follow.map_or([0.0, 0.0, 1.0], |(_, f)| f);
        let floor = reflect()[1];
        let constants = [constant, r, g, b, strength, radius, near, tan_y, gain, least, held, floor, power];
        l.SetComputeRoot32BitConstants(0, 13, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(slot * pass.step),
            },
        );
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
    }
    uav_barrier(l);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pass compiles with the system's compiler and the host takes it.
    #[test]
    fn ambient_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        make(&host.device).expect("the ambient pass");
    }

    /// `VICTORY_AMBIENT`: unset or unreadable the default constant, `off`
    /// none, `sky` following the sky, a number that constant.
    #[test]
    fn the_level_is_the_default_constant_unless_set() {
        let sky = [0.36, 0.05, 0.4];
        assert_eq!(parse_level(None, sky), Some(Level::Constant(DEFAULT)));
        assert_eq!(parse_level(Some(" off "), sky), None);
        assert_eq!(parse_level(Some("sky"), sky), Some(Level::Sky(sky)));
        assert_eq!(parse_level(Some("0.6"), sky), Some(Level::Constant(0.6)));
        assert_eq!(parse_level(Some("0"), sky), Some(Level::Constant(0.0)));
        assert_eq!(parse_level(Some("-1"), sky), Some(Level::Constant(DEFAULT)));
        assert_eq!(parse_level(Some("bright"), sky), Some(Level::Constant(DEFAULT)));
    }

    /// The programs are known by 16 hex digits of their hash, each once.
    #[test]
    fn the_programs_are_known_by_their_hash() {
        for (k, &(hash, role)) in PROGRAMS.iter().enumerate() {
            assert_eq!(hash.len(), 16, "{hash}");
            assert!(hash.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()), "{hash}");
            assert!(PROGRAMS[..k].iter().all(|&(h, _)| h != hash), "{hash} twice");
            assert_eq!(super::role(hash), Some(role));
        }
        assert_eq!(super::role("0000000000000000"), None);
    }
}
