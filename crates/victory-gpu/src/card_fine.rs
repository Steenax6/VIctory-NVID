//! Fine flags for copies in the card's memory (`crate::shadow::cards`):
//! the compare on the card flags 4 KiB blocks of a 64 KiB piece, not the
//! piece, and only the flagged blocks are fetched and merged.
//!
//! Measured in the intro before (log e10_all): every frame 16,788 pieces
//! (1.1 GB) were hashed on the card, 1,596 of them (104 MB) flagged and
//! fetched whole over the bus, and after the CPU's merge 0.7 MB in 143
//! stretches really went into title memory: a piece was fetched for a few
//! kilobytes that changed in it.
//!
//! With [`on`] the compare is [`CARD_COMPARE_FINE`]: one thread per 4 KiB
//! block (16 a piece, so a pass also takes the time of one block, not of a
//! piece), a 16-byte hash per block (`crate::shadow::card_hash_bytes`: 256
//! bytes a piece), and the piece's dword in the results holds the mask of
//! its blocks that differ (bit `b`: block `b`). The dword is nonzero
//! exactly where the old compare wrote 1, so whatever reads it as "the
//! piece differs" (the fetch inside flights, `crate::host::CARD_FETCH`)
//! keeps working and fetches the piece whole. The landings
//! (`Host::relay_card_back`, `Host::card_back`) fetch the flagged blocks
//! only, joined into runs where they touch, and merge those runs.
//!
//! Why no shader write is lost: every byte of a piece lies in exactly one
//! block's cut (`crate::shadow::card_blocks_of`, the same cut as the
//! shader's), and that block's hash covers it; a block whose hash is what
//! was stored holds what it held when it was stored, as a piece did. A
//! range that was not compared, and a block fetched beside a later flight
//! (`crate::host::BESIDE`, noted per block), are fetched whatever the
//! compare says, as pieces were.
//!
//! `VICTORY_CARD_FINE=0`: the compare per piece and whole pieces fetched,
//! as before. Read once: the hashes' layout is fixed for the process.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// A block: the compare's grain with fine flags, in bytes of title memory.
pub(crate) const BLOCK: usize = 4 << 10;

/// Blocks in a 64 KiB piece.
pub(crate) const BLOCKS: u32 = 16;

/// Every block of a piece.
pub(crate) const FULL: u32 = (1 << BLOCKS) - 1;

/// Whether the compare flags 4 KiB blocks (`VICTORY_CARD_FINE=0`: pieces,
/// as before; on by default).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_FINE").is_none_or(|v| v != "0"))
}

/// The blocks a dword of the compare's results flags: its mask with fine
/// flags, else every block of a flagged piece.
pub(crate) fn mask_of(dword: u32) -> u32 {
    mask_in(on(), dword)
}

fn mask_in(fine: bool, dword: u32) -> u32 {
    match (fine, dword) {
        (_, 0) => 0,
        (true, d) => d & FULL,
        (false, _) => FULL,
    }
}

/// Groups to dispatch for a pass over `pieces` pieces: the wide fine
/// compare takes a piece a group of 256 threads, the fine one 4 pieces a
/// group of 64, the old one 64.
pub(crate) fn groups(pieces: u32) -> u32 {
    match (on(), wide()) {
        (true, true) => pieces,
        (true, false) => pieces.div_ceil(64 / BLOCKS),
        _ => pieces.div_ceil(64),
    }
}

/// Whether the fine compare is [`CARD_COMPARE_WIDE`] (`VICTORY_CARD_WIDE_HASH=0`:
/// [`CARD_COMPARE_FINE`], one thread per block, as before; on by default).
///
/// The e23 timing run put our hash passes after send-ups at 4.48 ms for 85
/// a frame, about 53 us each for some 18 pieces each: a pass of few
/// pieces has few threads, and each walked its 4 KiB block 16 bytes a
/// load, one load after the other, 256 trips to memory.
pub(crate) fn wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_WIDE_HASH").is_none_or(|v| v != "0"))
}

/// The fine compare, wide: the same blocks, the same constants, results
/// and hash layout as [`CARD_COMPARE_FINE`], but a group of 256 threads per
/// piece, 16 per block, each hashing 256 bytes of it (the last one up to
/// the block's widened end) with all its loads issued before the first is
/// used; a block's hash is the XOR of its 16 parts' hashes (each seeded by
/// its part's number, so bytes moved between parts change it). Another
/// hash function than the narrow shader's: the two are never mixed in one
/// process (the switch is read once).
pub(crate) const CARD_COMPARE_WIDE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint hs[64];
groupshared uint mask;
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    uint block = t >> 4;
    uint part = t & 15u;
    uint i = group.x;
    if (t == 0u) mask = 0u;
    if (t < 64u) hs[t] = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint piece = first + i;
    uint from = piece * 65536u + block * 4096u;
    uint to = from + 4096u;
    uint lo = from > skew ? from - skew : 0u;
    uint hi = to > skew ? min(to - skew, size) : 0u;
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint a = lo + part * 256u;
    uint b = part == 15u ? hi : min(a + 256u, hi);
    // Up to 17 steps: the last part holds the widening (at most 4112 bytes
    // a block).
    // Every load inside the buffer: HLSL's `?:` evaluates both sides, and
    // a root view has no bounds (a load past the end faults the card), so
    // the address is clamped and the value dropped where it lies outside.
    uint4 v[17];
    [unroll]
    for (uint k = 0u; k < 17u; k++) {
        uint at = a + k * 16u;
        uint4 x = now.Load4(min(at, size - 16u));
        v[k] = (i < pieces && at < b) ? x : uint4(0u, 0u, 0u, 0u);
    }
    uint4 h = uint4(0x9E3779B9u ^ (part * 0x01000193u), 0x85EBCA6Bu + part, 0xC2B2AE35u, 0x27D4EB2Fu ^ (part << 16));
    [unroll]
    for (uint s = 0u; s < 17u; s++) {
        if (a + s * 16u < b) {
            h = (h ^ v[s]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
    }
    uint was;
    InterlockedXor(hs[block * 4u + 0u], h.x, was);
    InterlockedXor(hs[block * 4u + 1u], h.y, was);
    InterlockedXor(hs[block * 4u + 2u], h.z, was);
    InterlockedXor(hs[block * 4u + 3u], h.w, was);
    GroupMemoryBarrierWithGroupSync();
    if (part == 0u && i < pieces) {
        uint4 whole = uint4(hs[block * 4u], hs[block * 4u + 1u], hs[block * 4u + 2u], hs[block * 4u + 3u]);
        uint slot = (piece * 16u + block) * 16u;
        if (compare != 0u) {
            if (any(whole != hashes.Load4(slot))) InterlockedOr(mask, 1u << block, was);
            if (compare == 2u) hashes.Store4(slot, whole);
        } else {
            hashes.Store4(slot, whole);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0u && i < pieces && compare != 0u) differs.Store(out_at + i * 4u, mask);
}
"#;

/// The fine compare's source, as [`wide`] says.
pub(crate) fn compare_source() -> &'static str {
    if wide() { CARD_COMPARE_WIDE } else { CARD_COMPARE_FINE }
}

/// The compare with fine flags: as `crate::host::CARD_COMPARE` (the same
/// root signature and constants, the same hash), one thread per 4 KiB
/// block, cut at multiples of 4 KiB of title memory (piece `p`'s block `b`
/// is title bytes `p * 64 KiB + b * 4 KiB` on, offset by `skew` into the
/// copy) and widened to 16 bytes. The hash of block `b` of piece `p` is at
/// `(p * 16 + b) * 16` in `hashes`; told 1 or 2, piece `i`'s dword of
/// `differs` gets the mask of its blocks whose hash is another (2: and the
/// new hashes are stored); told 0 the hashes are stored and `differs` is
/// left alone.
pub(crate) const CARD_COMPARE_FINE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per 4 KiB block of the copy.
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint masks[4];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    uint local = t >> 4;
    uint block = t & 15u;
    uint i = group.x * 4u + local;
    if (block == 0u) masks[local] = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (i < pieces) {
        uint piece = first + i;
        uint from = piece * 65536u + block * 4096u;
        uint to = from + 4096u;
        uint lo = from > skew ? from - skew : 0u;
        uint hi = to > skew ? min(to - skew, size) : 0u;
        lo = lo & ~15u;
        hi = (hi + 15u) & ~15u;
        uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
        [loop]
        for (uint at = lo; at < hi; at += 16u) {
            h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
        uint slot = (piece * 16u + block) * 16u;
        if (compare != 0u) {
            if (any(h != hashes.Load4(slot))) {
                uint was;
                InterlockedOr(masks[local], 1u << block, was);
            }
            if (compare == 2u) hashes.Store4(slot, h);
        } else {
            hashes.Store4(slot, h);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (block == 0u && i < pieces && compare != 0u) differs.Store(out_at + i * 4u, masks[local]);
}
"#;

/// Since the last frame-rate line, of the ranges of copies in the card's
/// memory compared after submissions: pieces in ranges declared only for
/// shaders that store by address in copies no dispatch of the submission
/// named (`crate::host::AddressWrites`), blocks the compare flagged in
/// those (with fine flags; a flagged piece is 16 without), and pieces
/// compared in all.
pub(crate) static ADDRESS_STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Whether shaders that store by address declare written only the copies
/// their dispatch names (`VICTORY_NAMED_WRITES=1`; off by default), and on
/// every [`named_net`]th frame all of them, as before (`crate::memory::ready`).
///
/// Measured first: the line "copies written by address" says how many
/// blocks the compare flags in copies no dispatch named. If that is
/// always 0, the narrowing loses nothing. If a shader does write such a
/// copy through an address it read from memory, between two full frames:
/// the copy is not closed to refreshes, so a refresh may send title bytes
/// up over the shader's write, and the next full compare finds only what
/// is left of it; the title does not see the write until then, if ever.
pub(crate) fn named_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NAMED_WRITES").is_some_and(|v| v == "1"))
}

/// With [`named_writes`]: every how many frames all copies are declared
/// (`VICTORY_NAMED_WRITES_NET`, 8; at least 1, 1 being every frame).
pub(crate) fn named_net() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_NAMED_WRITES_NET").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8).max(1))
}

/// Since the last frame-rate line, with [`on`]: pieces with a flagged
/// block, blocks flagged, runs fetched, their bytes, blocks fetched beside
/// later flights whatever the compare said; then, of every merge of a
/// fetched run or piece (with or without fine flags, [`merged`]): runs
/// whose merge put nothing into title memory, their bytes, and the bytes
/// the merges put there.
pub(crate) static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Per copy in the card's memory, since the last line, by its title
/// address: its bytes, then of its merged runs: bytes fetched, bytes put
/// into title memory, runs, runs that put nothing there.
static BY_COPY: std::sync::Mutex<Option<FastMap<u64, [u64; 5]>>> = std::sync::Mutex::new(None);

/// A fetched run of `fetched` bytes of `shadow` was merged and put
/// `written` bytes into title memory (the merge's stretch from its first
/// changed byte to its last): for the line that says what the fetches
/// bring and to which copies.
pub(crate) fn merged(shadow: &crate::shadow::Shadow, fetched: usize, written: usize) {
    if written == 0 {
        STATS[5].fetch_add(1, Ordering::Relaxed);
        STATS[6].fetch_add(fetched as u64, Ordering::Relaxed);
    }
    STATS[7].fetch_add(written as u64, Ordering::Relaxed);
    let (start, len) = shadow.span();
    let mut by = BY_COPY.lock().unwrap_or_else(|e| e.into_inner());
    let e = by.get_or_insert_with(Default::default).entry(start).or_insert([len, 0, 0, 0, 0]);
    e[1] += fetched as u64;
    e[2] += written as u64;
    e[3] += 1;
    e[4] += u64::from(written == 0);
}

/// Runs of offsets (from, to), sorted and joined where they overlap or
/// touch.
pub(crate) fn join_runs(mut runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    runs.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (a, b) in runs {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// A cheap hasher for keys of our own (addresses of copies, piece numbers):
/// the relay thread hashed some 20,000 keys a landing with the standard
/// one (`hash_one` 1.7 ms a frame in e16's profile). Not for keys anyone
/// else chooses.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A map with [`Fx`].
pub(crate) type FastMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;

/// The stretches where `now` differs from `was` (one length), as (offset,
/// bytes) in order: every differing byte is in one, each starts and ends
/// with a differing byte, and stretches closer than `gap` equal bytes are
/// one. Eight bytes at a time (32 while they are equal), so a landing that
/// fetched 60 MB a frame of 4 KiB blocks finds the few bytes that changed
/// without a byte loop over all of them.
pub(crate) fn changed_stretches(now: &[u8], was: &[u8], gap: usize, out: &mut Vec<(usize, usize)>) {
    #[cfg(target_arch = "x86_64")]
    if gap >= 62 && wide_compare() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2 (asked just now).
        unsafe { changed_stretches_avx2(now, was, gap, out) };
        return;
    }
    changed_stretches_scalar(now, was, gap, out);
}

/// Whether [`changed_stretches`] compares 64 bytes a step with AVX2 where
/// the processor has it (`VICTORY_WIDE_COMPARE=0`: 8 at a time, as
/// before; on by default). e25's profile: 9.7 ms a frame for some 64 MB,
/// 6.6 GB/s.
pub(crate) fn wide_compare() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_COMPARE").is_none_or(|v| v != "0"))
}

/// [`changed_stretches`] 64 bytes a step: two 32-byte compares and their
/// byte masks; a step with a difference is one stretch from its first
/// differing byte to its last (at most 62 equal bytes lie between, within
/// `gap`, which must be 62 or more). The tail goes 8 bytes and then 1 at a
/// time, as [`changed_stretches_scalar`].
///
/// # Safety
/// The processor has AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn changed_stretches_avx2(now: &[u8], was: &[u8], gap: usize, out: &mut Vec<(usize, usize)>) {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    out.clear();
    let n = now.len().min(was.len());
    let mut cur: Option<(usize, usize)> = None;
    let add = |first: usize, last: usize, cur: &mut Option<(usize, usize)>, out: &mut Vec<(usize, usize)>| match cur {
        Some((_, to)) if first - *to <= gap => *to = last + 1,
        Some((from, to)) => {
            out.push((*from, *to - *from));
            *cur = Some((first, last + 1));
        }
        None => *cur = Some((first, last + 1)),
    };
    let (a, b) = (now.as_ptr(), was.as_ptr());
    let mut i = 0;
    while i + 64 <= n {
        // SAFETY: `i + 64 <= n`, inside both slices; unaligned loads.
        let equal = unsafe {
            let e0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                _mm256_loadu_si256(a.add(i).cast::<__m256i>()),
                _mm256_loadu_si256(b.add(i).cast::<__m256i>()),
            )) as u32;
            let e1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(
                _mm256_loadu_si256(a.add(i + 32).cast::<__m256i>()),
                _mm256_loadu_si256(b.add(i + 32).cast::<__m256i>()),
            )) as u32;
            u64::from(e0) | (u64::from(e1) << 32)
        };
        let differs = !equal;
        if differs != 0 {
            let first = i + differs.trailing_zeros() as usize;
            let last = i + 63 - differs.leading_zeros() as usize;
            add(first, last, &mut cur, out);
        }
        i += 64;
    }
    while i < n {
        if now[i] != was[i] {
            add(i, i, &mut cur, out);
        }
        i += 1;
    }
    if let Some((from, to)) = cur {
        out.push((from, to - from));
    }
}

/// Whether most of a fetched run changed: more than half of the 64-byte
/// steps of its first 4 KiB differ (`now` against `was`). Such a run (a
/// buffer a shader rewrites whole, 18 MB a frame in e23) is merged and
/// written whole, without looking for stretches.
pub(crate) fn mostly_changed(now: &[u8], was: &[u8]) -> bool {
    let n = now.len().min(was.len()).min(BLOCK);
    if n < BLOCK {
        return false;
    }
    let differing = (0..n / 64).filter(|&k| now[k * 64..k * 64 + 64] != was[k * 64..k * 64 + 64]).count();
    differing * 2 > n / 64
}

fn changed_stretches_scalar(now: &[u8], was: &[u8], gap: usize, out: &mut Vec<(usize, usize)>) {
    out.clear();
    let n = now.len().min(was.len());
    // The stretch being grown: [from, to).
    let mut cur: Option<(usize, usize)> = None;
    let mut add = |first: usize, last: usize, cur: &mut Option<(usize, usize)>| match cur {
        Some((_, to)) if first - *to <= gap => *to = last + 1,
        Some((from, to)) => {
            out.push((*from, *to - *from));
            *cur = Some((first, last + 1));
        }
        None => *cur = Some((first, last + 1)),
    };
    let word = |i: usize| {
        let a = u64::from_le_bytes(now[i..i + 8].try_into().unwrap_or([0; 8]));
        let b = u64::from_le_bytes(was[i..i + 8].try_into().unwrap_or([0; 8]));
        a ^ b
    };
    let mut i = 0;
    while i + 8 <= n {
        if i + 32 <= n && now[i..i + 32] == was[i..i + 32] {
            i += 32;
            continue;
        }
        let mut x = word(i);
        if x != 0 && gap >= 6 {
            // At most 6 equal bytes lie between two differing ones of a
            // word: within the gap, so the word is one stretch.
            let first = i + (x.trailing_zeros() / 8) as usize;
            let last = i + 7 - (x.leading_zeros() / 8) as usize;
            add(first, last, &mut cur);
        } else {
            while x != 0 {
                let b = (x.trailing_zeros() / 8) as usize;
                add(i + b, i + b, &mut cur);
                x &= !(0xffu64 << (8 * b));
            }
        }
        i += 8;
    }
    while i < n {
        if now[i] != was[i] {
            add(i, i, &mut cur);
        }
        i += 1;
    }
    if let Some((from, to)) = cur {
        out.push((from, to - from));
    }
}

/// With fine flags, the relay's notes of blocks fetched on the copy queue
/// beside later flights (`crate::host`'s `BESIDE`, which keys a note per
/// block and was cloned whole every landing): per (copy, piece) the
/// submission number up to which each of its 16 blocks must be fetched
/// again (0: no note).
pub(crate) type BesideFine = FastMap<(usize, u32), [u64; BLOCKS as usize]>;

/// The blocks of piece `(copy, piece)` that the flight numbered `number`
/// must fetch whatever its compare says: noted when an earlier fetch ran
/// while submissions up to the note's number had started, at or after
/// this one (`crate::host::fetched_beside`).
pub(crate) fn beside_mask(noted: &BesideFine, copy: usize, piece: u32, number: u64) -> u32 {
    let Some(upto) = noted.get(&(copy, piece)) else {
        return 0;
    };
    let mut mask = 0;
    for (b, &u) in upto.iter().enumerate() {
        if u != 0 && u >= number {
            mask |= 1 << b;
        }
    }
    mask
}

/// `crate::host::note_beside` for [`BesideFine`]: after a landing of
/// flights up to `landed` whose fetch brought the blocks `fetched` (copy,
/// piece, mask) down while submissions up to `started` had started. Notes
/// whose flights have all landed go; the blocks just fetched are noted if
/// anything later had started. Returns the blocks noted now.
pub(crate) fn note_beside_fine(noted: &mut BesideFine, fetched: &[(usize, u32, u32)], landed: u64, started: u64) -> u64 {
    noted.retain(|_, upto| {
        for u in upto.iter_mut() {
            if *u <= landed {
                *u = 0;
            }
        }
        upto.iter().any(|&u| u != 0)
    });
    if started <= landed {
        return 0;
    }
    let mut count = 0;
    for &(copy, piece, mask) in fetched {
        let upto = noted.entry((copy, piece)).or_insert([0; BLOCKS as usize]);
        for (b, u) in upto.iter_mut().enumerate() {
            if mask & (1 << b) != 0 {
                *u = started;
                count += 1;
            }
        }
    }
    count
}

/// The frame-rate line's part.
pub(crate) fn line(frames: f64) {
    if !crate::shadow::cards() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let by: Vec<(u64, [u64; 5])> = BY_COPY.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default().into_iter().collect();
    if s.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |b: f64| b / (1 << 20) as f64;
    {
        let mut by = by;
        by.sort_by_key(|(_, v)| std::cmp::Reverse(v[1]));
        let top: Vec<String> = by
            .iter()
            .take(10)
            .map(|(at, v)| {
                format!(
                    "{at:#x}+{:#x}: {:.2} MB fetched in {:.0} runs ({:.0} put nothing back), {:.2} MB put back",
                    v[0],
                    mb(v[1] as f64 / frames),
                    v[3] as f64 / frames,
                    v[4] as f64 / frames,
                    mb(v[2] as f64 / frames)
                )
            })
            .collect();
        tracing::info!(
            "card copies' merges per frame: {:.2} MB put into title memory (first to last changed byte of each run); \
             {:.0} fetched runs ({:.2} MB) put nothing back; {} copies, the most fetched: {}",
            mb(s[7]),
            s[5],
            mb(s[6]),
            by.len(),
            top.join("; ")
        );
    }
    {
        let [unnamed, flagged, all] = [0, 1, 2].map(|i| ADDRESS_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        if unnamed > 0.0 || named_writes() {
            tracing::info!(
                "copies written by address per frame: {unnamed:.0} of {all:.0} pieces compared lie in copies declared only for shaders \
                 that store by address and named by no dispatch of the submission; {flagged:.0} blocks of 4 KiB flagged in them \
                 (writes through addresses read from memory; VICTORY_NAMED_WRITES={}, all copies every {} frames)",
                u8::from(named_writes()),
                named_net()
            );
        }
    }
    if !on() {
        return;
    }
    tracing::info!(
        "card copies' fine flags per frame (VICTORY_CARD_FINE=1): {:.0} blocks of 4 KiB flagged in {:.0} pieces, \
         fetched as {:.0} runs ({:.2} MB, whole pieces would be {:.2} MB); {:.0} blocks fetched beside later flights \
         whatever the compare said",
        s[1],
        s[0],
        s[2],
        s[3] / (1 << 20) as f64,
        s[0] * 65536.0 / (1 << 20) as f64,
        s[4],
    );
}

#[cfg(test)]
mod tests {
    use super::{BLOCK, BLOCKS, CARD_COMPARE_FINE, CARD_COMPARE_WIDE, FULL, join_runs, mask_in};
    use crate::shadow::{CardSpan, card_blocks_of};
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    const PIECE: usize = 64 << 10;

    #[test]
    fn a_dword_means_its_mask_with_fine_flags_and_every_block_without() {
        assert_eq!(mask_in(true, 0), 0);
        assert_eq!(mask_in(false, 0), 0);
        assert_eq!(mask_in(true, 0b1010), 0b1010);
        assert_eq!(mask_in(false, 1), FULL);
        assert_eq!(mask_in(true, 0xffff_0001), 1);
    }

    /// The stretches hold every differing byte, start and end on one, and
    /// are apart by more than the gap; against a byte-by-byte reference.
    #[test]
    fn changed_stretches_are_the_differing_bytes_joined_over_the_gap() {
        use super::changed_stretches;
        let mut rng = 0x1234_5678_9abc_def1u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut out = Vec::new();
        for round in 0..400 {
            let n = (next() as usize) % 9000;
            let was: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut now = was.clone();
            let changes = if round % 7 == 0 { n } else { (next() % 30) as usize };
            for _ in 0..changes {
                if n == 0 {
                    break;
                }
                let at = (next() as usize) % n;
                now[at] = now[at].wrapping_add(1 + (next() % 255) as u8);
            }
            let gap = [0usize, 1, 7, 64][round % 4];
            changed_stretches(&now, &was, gap, &mut out);
            // Reference: differing bytes, joined where the equal bytes
            // between are at most `gap`.
            let mut want: Vec<(usize, usize)> = Vec::new();
            for i in (0..n).filter(|&i| now[i] != was[i]) {
                match want.last_mut() {
                    Some((from, len)) if i - (*from + *len) <= gap => *len = i + 1 - *from,
                    _ => want.push((i, 1)),
                }
            }
            assert_eq!(out, want, "round {round}, n {n}, gap {gap}");
        }
    }

    /// The 64-byte compare gives what the 8-byte one gives, at the gaps it
    /// takes, on random data with changes at step edges and in the tail.
    #[test]
    fn the_wide_compare_finds_the_same_stretches() {
        use super::{changed_stretches, changed_stretches_scalar, mostly_changed};
        let mut rng = 0x5eed_0000_1111_2222u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let (mut wide, mut narrow) = (Vec::new(), Vec::new());
        for round in 0..300 {
            let n = (next() as usize) % 20_000;
            let was: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut now = was.clone();
            let changes = if round % 9 == 0 { n } else { (next() % 40) as usize };
            for k in 0..changes {
                if n == 0 {
                    break;
                }
                let at = match k % 3 {
                    0 => (next() as usize) % n,
                    1 => ((next() as usize) % n.div_ceil(64)) * 64 + if next() % 2 == 0 { 0 } else { 63 },
                    _ => n - 1 - (next() as usize) % n.min(70),
                }
                .min(n - 1);
                now[at] = now[at].wrapping_add(1 + (next() % 255) as u8);
            }
            for gap in [62usize, 64, 200] {
                changed_stretches(&now, &was, gap, &mut wide);
                changed_stretches_scalar(&now, &was, gap, &mut narrow);
                assert_eq!(wide, narrow, "round {round}, n {n}, gap {gap}");
            }
        }
        // Mostly changed: more than half the 64-byte steps of 4 KiB.
        let was = vec![0u8; 8192];
        let mut now = was.clone();
        assert!(!mostly_changed(&now, &was));
        for k in 0..33 {
            now[k * 64 + 5] = 1;
        }
        assert!(mostly_changed(&now, &was));
        assert!(!mostly_changed(&now[..4000], &was[..4000]), "a run under 4 KiB is looked at as it is");
    }

    /// Fine notes of blocks fetched beside later flights: a block is
    /// fetched again by flights that had started by then, by no later one,
    /// and the note goes once they have landed.
    #[test]
    fn fine_notes_name_blocks_until_their_flights_have_landed() {
        use super::{BesideFine, beside_mask, note_beside_fine};
        let mut noted = BesideFine::default();
        let (a, p) = (0x1000usize, 7u32);
        assert_eq!(note_beside_fine(&mut noted, &[(a, p, 0b101)], 5, 7), 2);
        assert_eq!(beside_mask(&noted, a, p, 6), 0b101);
        assert_eq!(beside_mask(&noted, a, p, 7), 0b101);
        assert_eq!(beside_mask(&noted, a, p, 8), 0);
        assert_eq!(beside_mask(&noted, a, p + 1, 6), 0);
        // Flight 6 lands and fetches block 0 again with 9 started: block 0
        // is noted up to 9, block 2 stays up to 7.
        assert_eq!(note_beside_fine(&mut noted, &[(a, p, 0b001)], 6, 9), 1);
        assert_eq!(beside_mask(&noted, a, p, 8), 0b001);
        assert_eq!(beside_mask(&noted, a, p, 7), 0b101);
        // Up to 7 landed: block 2's note goes, block 0's stays.
        assert_eq!(note_beside_fine(&mut noted, &[], 7, 7), 0);
        assert_eq!(beside_mask(&noted, a, p, 8), 0b001);
        // Everything landed: nothing is noted.
        assert_eq!(note_beside_fine(&mut noted, &[], 9, 9), 0);
        assert!(noted.is_empty());
    }

    #[test]
    fn runs_join_where_they_touch_or_overlap() {
        assert_eq!(join_runs(vec![(10, 20), (0, 5), (20, 30), (25, 27), (40, 41)]), [(0, 5), (10, 30), (40, 41)]);
        assert!(join_runs(Vec::new()).is_empty());
    }

    /// The blocks' runs cover exactly the bytes of the flagged blocks in
    /// the piece's part of the span, for a copy that does not start on a
    /// block or a 16-byte line.
    #[test]
    fn block_runs_cover_the_flagged_blocks_and_nothing_else() {
        let start = 0x1_0000_0000usize + 3 * BLOCK + 40;
        let len = 5 * PIECE;
        // A span from inside piece 0 to inside piece 3.
        let span = CardSpan {
            from: 1000,
            to: (3 * PIECE + 7 * BLOCK + 5) as u32,
            first: 0,
            pieces: 4,
            skew: (start % PIECE) as u32,
        };
        let piece0 = start / PIECE * PIECE;
        let mut rng = 0x1234_5678u32;
        for _ in 0..200 {
            let mut out = Vec::new();
            let mut masks = [0u32; 4];
            for (i, m) in masks.iter_mut().enumerate() {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                *m = rng & FULL;
                card_blocks_of(start, len, &span, i as u32, *m, &mut out);
            }
            // Runs are in order, apart, and do not touch.
            for w in out.windows(2) {
                assert!(w[0].1 < w[1].0);
            }
            for o in 0..len {
                let inside = o >= span.from as usize && o < span.to as usize;
                let g = (start + o - piece0) / BLOCK;
                let (piece, block) = (g / BLOCKS as usize, g % BLOCKS as usize);
                let wanted = inside && piece < 4 && masks[piece] & (1 << block) != 0;
                let got = out.iter().any(|&(a, b)| o >= a && o < b);
                assert_eq!(got, wanted, "offset {o}");
            }
        }
    }

    struct Gpu {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        sent: u64,
        event: HANDLE,
    }

    impl Gpu {
        /// The card if there is one (`warp`: false), else the software
        /// adapter.
        fn new(warp: bool) -> Option<Gpu> {
            // SAFETY: plain DXGI and D3D12 creation calls.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                if warp {
                    let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                    let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                } else {
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                }
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                Some(Gpu {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    sent: 0,
                    event,
                })
            }
        }

        /// A buffer in the PC's memory, mapped (as the host's own), that
        /// shaders may read and write.
        fn shared(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and a persistent mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        fn compile(&self, source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: the source and names outlive the call; the blob's own
            // bytes; the root signature is inside the shader.
            unsafe {
                let made = D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("card compare fine"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    Some(&mut errors),
                );
                if made.is_err() {
                    let text = errors
                        .map(|e| String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned())
                        .unwrap_or_default();
                    panic!("the fine compare does not compile: {text}");
                }
                let code = code.unwrap();
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes).unwrap();
                let pso: ID3D12PipelineState = self
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                (root, pso)
            }
        }

        fn run(&mut self) {
            self.sent += 1;
            // SAFETY: our list, queue, fence and event.
            unsafe {
                self.list.Close().unwrap();
                self.queue.ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>().unwrap())]);
                self.queue.Signal(&self.fence, self.sent).unwrap();
                self.fence.SetEventOnCompletion(self.sent, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    /// The fine compare against a CPU reference, on the card and on the
    /// software adapter: random copies (one that starts off a block and off
    /// a 16-byte line), a few random bytes changed; every changed byte's
    /// block is flagged, mode 1 keeps flagging until the hashes are stored,
    /// mode 2 stores them, mode 0 writes no result, and the fetch of the
    /// flagged blocks' runs (`card_blocks_of`) holds every changed byte.
    fn the_fine_compare_flags_every_changed_byte(warp: bool, wide: bool) {
        let Some(mut gpu) = Gpu::new(warp) else {
            eprintln!("no {} device: the fine compare is not tested on it", if warp { "software" } else { "hardware" });
            return;
        };
        let (root, pso) = gpu.compile(if wide { CARD_COMPARE_WIDE } else { CARD_COMPARE_FINE });
        for &skew in &[0usize, 40, 3 * BLOCK + 8, PIECE - 24] {
            const PIECES: usize = 6;
            let size = PIECES * PIECE;
            // The copy's title memory starts `skew` into a piece; the copy
            // holds `size - skew` bytes (its buffer `size`).
            let start = 0x2_0000_0000usize + skew;
            let len = size - skew;
            let pieces = (start + len - 1) / PIECE - start / PIECE + 1;
            let (_now, now_ptr, now_gpu) = gpu.shared(size as u64);
            let (_hashes, _, hashes_gpu) = gpu.shared(((size / PIECE + 2) * 16 * BLOCKS as usize) as u64);
            let (_differs, differs_ptr, differs_gpu) = gpu.shared(4 * 64);
            let mut rng = 0x9e37_79b9u64 ^ skew as u64;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut content: Vec<u8> = (0..size).map(|_| next() as u8).collect();
            // SAFETY: the mapping holds `size` bytes.
            unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
            let pass = |gpu: &mut Gpu, mode: u32| -> Vec<u32> {
                // SAFETY: our open list, live buffers, the shader's own root
                // signature; the mappings hold what is read and written.
                unsafe {
                    for i in 0..pieces {
                        differs_ptr.cast::<u32>().add(i).write_volatile(0xdead);
                    }
                    gpu.list.SetComputeRootSignature(&root);
                    gpu.list.SetPipelineState(&pso);
                    let constants = [0u32, skew as u32, size as u32, pieces as u32, 0, mode];
                    gpu.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                    gpu.list.SetComputeRootShaderResourceView(1, now_gpu);
                    gpu.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                    gpu.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                    gpu.list.Dispatch(if wide { pieces as u32 } else { (pieces as u32).div_ceil(4) }, 1, 1);
                }
                gpu.run();
                // SAFETY: as above.
                (0..pieces).map(|i| unsafe { differs_ptr.cast::<u32>().add(i).read_volatile() }).collect()
            };
            assert!(pass(&mut gpu, 0).iter().all(|&d| d == 0xdead), "mode 0 writes no result");
            let quiet = pass(&mut gpu, 1);
            // SAFETY: a plain query.
            let removed = unsafe { gpu.device.GetDeviceRemovedReason() };
            assert!(quiet.iter().all(|&d| d == 0), "nothing changed: {quiet:x?}, device {removed:?}");
            let span = CardSpan {
                from: 0,
                to: len as u32,
                first: 0,
                pieces: pieces as u32,
                skew: (start % PIECE) as u32,
            };
            // The CPU's reference: the blocks whose bytes, as the shader
            // cuts and widens them, differ from those last stored.
            let expected = |content: &[u8], stored: &[u8]| -> Vec<u32> {
                (0..pieces)
                    .map(|i| {
                        let mut mask = 0u32;
                        for b in 0..BLOCKS as usize {
                            let from = i * PIECE + b * BLOCK;
                            let to = from + BLOCK;
                            let lo = from.saturating_sub(skew);
                            let hi = if to > skew { (to - skew).min(size) } else { 0 };
                            let (lo, hi) = (lo & !15, (hi + 15) & !15);
                            if lo < hi && content[lo..hi] != stored[lo..hi] {
                                mask |= 1 << b;
                            }
                        }
                        mask
                    })
                    .collect()
            };
            let mut stored = content.clone();
            let mut since = 0usize;
            for round in 0..24 {
                // A few bytes, some in the same block, some at block and
                // piece edges.
                let mut changed = Vec::new();
                let count = 1 + (next() % 6) as usize;
                for k in 0..count {
                    let at = match (round + k) % 4 {
                        0 => (next() as usize) % len,
                        1 => {
                            // Right at the start or end of a block.
                            let g = (next() as usize) % (len / BLOCK).max(1);
                            let edge = (g * BLOCK + BLOCK - start % BLOCK) % len;
                            if next() % 2 == 0 { edge } else { edge.saturating_sub(1) }
                        }
                        2 => len - 1 - (next() as usize) % 32,
                        _ => (next() as usize) % 64,
                    };
                    content[at] ^= 1 + (next() % 255) as u8;
                    // SAFETY: inside the mapping.
                    unsafe { now_ptr.add(at).write_volatile(content[at]) };
                    changed.push(at);
                }
                let mode = if round % 2 == 0 { 1 } else { 2 };
                let flagged = pass(&mut gpu, mode);
                // Exactly the CPU's blocks (a hash collision aside).
                assert_eq!(flagged, expected(&content, &stored), "skew {skew} round {round}");
                // Every byte that differs from what was stored is in a
                // flagged block and in the fetch of the flagged blocks.
                let mut runs = Vec::new();
                for (i, &m) in flagged.iter().enumerate() {
                    assert_eq!(m & !FULL, 0);
                    card_blocks_of(start, len, &span, i as u32, m, &mut runs);
                }
                let piece0 = start / PIECE * PIECE;
                for at in (0..len).filter(|&at| content[at] != stored[at]) {
                    let g = (start + at - piece0) / BLOCK;
                    let (piece, block) = (g / BLOCKS as usize, g % BLOCKS as usize);
                    assert!(
                        flagged[piece] & (1 << block) != 0,
                        "skew {skew} round {round}: byte {at} (block {block} of piece {piece}) not flagged: {flagged:x?}"
                    );
                    assert!(runs.iter().any(|&(a, b)| at >= a && at < b), "skew {skew}: byte {at} not fetched");
                }
                let fetched: usize = runs.iter().map(|&(a, b)| b - a).sum();
                // A changed byte flags its own block, and the next one if it
                // lies in the 16 bytes that block's hash is widened over.
                since += changed.len();
                assert!(fetched <= since * 2 * BLOCK, "{fetched} bytes fetched for {since} changes");
                if mode == 2 {
                    since = 0;
                    stored.copy_from_slice(&content);
                    // Stored: the next compare sees nothing.
                    assert!(pass(&mut gpu, 1).iter().all(|&d| d == 0), "skew {skew}: hashes stored with mode 2");
                } else {
                    // Mode 1 does not store: the same blocks again.
                    assert_eq!(pass(&mut gpu, 1), flagged);
                }
            }
        }
    }

    #[test]
    fn the_fine_compare_flags_every_changed_byte_on_the_card() {
        the_fine_compare_flags_every_changed_byte(false, false);
    }

    #[test]
    fn the_wide_compare_flags_every_changed_byte_on_the_card() {
        the_fine_compare_flags_every_changed_byte(false, true);
    }

    #[test]
    fn the_wide_compare_flags_every_changed_byte_on_warp() {
        the_fine_compare_flags_every_changed_byte(true, true);
    }

    #[test]
    fn the_fine_compare_flags_every_changed_byte_on_warp() {
        the_fine_compare_flags_every_changed_byte(true, false);
    }
}
