//! Host textures placed in heaps of 16 MiB (64 in the pull request this
//! came from, fxceless/victory-fork #903: smaller ones run empty sooner), for the textures the pass over
//! the descriptor heap makes by the hundred: a committed resource is a
//! kernel allocation of its own (about half a millisecond each), a placed
//! one is not.
//!
//! The range a texture has belongs to the COM object itself: a small object
//! given to it as private data, which it lets go of when it is destroyed (a
//! descriptor slot, a list and the title's texture each may hold it). From
//! then on the range waits for the submissions running now
//! ([`crate::host::retire`]) before another texture gets it: the system does
//! that for a committed resource.
// shortcut: no aliasing barrier between two textures that have a range one
// after the other: what used the first has finished when the second gets it,
// and every texture made here is uploaded whole before a shader reads it.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, MutexGuard};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface};

/// A heap's bytes.
const HEAP: u64 = 16 << 20;
/// The largest texture that goes into one.
const LARGEST: u64 = HEAP / 2;
/// Where textures may start: any, and small ones (most detailed level of at
/// most 64 KiB). Each kind has its own heaps.
const NORMAL: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
const SMALL: u64 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT as u64;
/// An empty heap is freed after this, but for one of each kind.
const EMPTY_MS: u64 = 5_000;

/// The free ranges of a heap, by offset: none touch.
struct Free(Vec<(u64, u64)>);

impl Free {
    /// The start of `size` bytes in the smallest range that has them.
    fn take(&mut self, size: u64) -> Option<u64> {
        let fits = (0..self.0.len()).filter(|&i| self.0[i].1 >= size);
        let best = fits.min_by_key(|&i| self.0[i].1)?;
        let (offset, has) = self.0[best];
        if has == size {
            self.0.remove(best);
        } else {
            self.0[best] = (offset + size, has - size);
        }
        Some(offset)
    }

    fn give(&mut self, offset: u64, size: u64) {
        let at = self.0.partition_point(|&(o, _)| o < offset);
        self.0.insert(at, (offset, size));
        if self.0.get(at + 1).is_some_and(|&(next, _)| next == offset + size) {
            self.0[at].1 += self.0.remove(at + 1).1;
        }
        if at > 0 && self.0[at - 1].0 + self.0[at - 1].1 == offset {
            self.0[at - 1].1 += self.0.remove(at).1;
        }
    }
}

struct Block {
    id: u64,
    heap: ID3D12Heap,
    align: u64,
    /// In the PC's memory the card reaches over the bus ([`shared_heap`]).
    shared: bool,
    free: Free,
    free_bytes: u64,
    /// When something was last given back.
    given: u64,
}

static BLOCKS: Mutex<Vec<Block>> = Mutex::new(Vec::new());
/// Bytes of the heaps in the card's memory no texture has.
static FREE: AtomicU64 = AtomicU64::new(0);
/// Heaps in the PC's memory alive.
static SHARED_HEAPS: AtomicU64 = AtomicU64::new(0);
/// For [`stats`]: heaps made, textures placed, textures committed instead.
static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn blocks() -> MutexGuard<'static, Vec<Block>> {
    BLOCKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A texture's range, given back when dropped.
struct Range {
    block: u64,
    offset: u64,
    size: u64,
    shared: bool,
}

impl Drop for Range {
    fn drop(&mut self) {
        let mut blocks = blocks();
        if let Some(b) = blocks.iter_mut().find(|b| b.id == self.block) {
            b.free.give(self.offset, self.size);
            b.free_bytes += self.size;
            b.given = crate::host::now_ms();
            if !self.shared {
                FREE.fetch_add(self.size, Relaxed);
            }
        }
    }
}

/// The object a texture holds as private data: `IUnknown` and no more.
#[repr(C)]
struct Ticket {
    vtable: &'static IUnknown_Vtbl,
    references: AtomicU32,
    range: Range,
}

static TICKET: IUnknown_Vtbl = IUnknown_Vtbl {
    QueryInterface: ticket_query,
    AddRef: ticket_add,
    Release: ticket_release,
};

fn ticket(range: Range) -> IUnknown {
    let ticket = Box::new(Ticket {
        vtable: &TICKET,
        references: AtomicU32::new(1),
        range,
    });
    // SAFETY: an object whose first field points to the three methods, with
    // one reference, which the caller gets.
    unsafe { IUnknown::from_raw(Box::into_raw(ticket).cast()) }
}

unsafe extern "system" fn ticket_query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: the caller passes a live ticket, an identifier and an out
    // pointer.
    unsafe {
        if *iid != IUnknown::IID {
            *out = std::ptr::null_mut();
            return E_NOINTERFACE;
        }
        ticket_add(this);
        *out = this;
    }
    HRESULT(0)
}

unsafe extern "system" fn ticket_add(this: *mut c_void) -> u32 {
    // SAFETY: a live ticket.
    unsafe { &*this.cast::<Ticket>() }.references.fetch_add(1, Relaxed) + 1
}

unsafe extern "system" fn ticket_release(this: *mut c_void) -> u32 {
    use std::sync::atomic::Ordering::AcqRel;
    let this = this.cast::<Ticket>();
    // SAFETY: a live ticket.
    let left = unsafe { &*this }.references.fetch_sub(1, AcqRel) - 1;
    if left == 0 {
        // SAFETY: made by `ticket` from a box, and the last reference.
        let ticket = unsafe { Box::from_raw(this) };
        // Not before the submissions that run now have finished.
        crate::host::retire(ticket.range);
    }
    left
}

/// Whether new texture memory is made resident, and waited for, before
/// anything can name it (on by default; `VICTORY_WAIT_RESIDENT=0`: as
/// before). Memory of the default kind is resident once it is made, but
/// its mapping into the card's address space is the system's work that
/// runs later, and only lists sent after it wait for it. Lists already on
/// the card read the descriptor heaps as they are: a slot that shows a
/// texture in new memory was a page fault for them until it was mapped
/// (avatarmanz/VIctory-7900XTX 0322b39ab: the losses there lay in textures
/// made in the same tenth of a second). For heaps (placed textures, tiles
/// of textures made level by level) and committed host textures; a texture
/// placed in a heap that is there needs no more.
fn wait_resident() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_RESIDENT").is_none_or(|v| v != "0"))
}

/// The flag new texture memory is made with for [`resident`].
pub(crate) fn not_resident() -> D3D12_HEAP_FLAGS {
    if wait_resident() { D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT } else { D3D12_HEAP_FLAG_NONE }
}

/// Memory made with [`not_resident`] is resident, and mapped, when this
/// returns ([`wait_resident`]).
pub(crate) fn resident(device: &ID3D12Device, object: &impl Interface) -> windows::core::Result<()> {
    if !wait_resident() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let object: ID3D12Pageable = object.cast()?;
    // SAFETY: a live heap or resource of the device's.
    let made = unsafe { device.MakeResident(&[Some(object)]) };
    let us = started.elapsed().as_micros() as u64;
    RESIDENT[0].fetch_add(1, Relaxed);
    RESIDENT[1].fetch_add(us, Relaxed);
    RESIDENT[2].fetch_max(us, Relaxed);
    made
}

/// Since the start: memory waited for by [`resident`], the microseconds
/// and the longest wait.
static RESIDENT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The line on [`resident`], every few seconds; nothing before the first.
pub(crate) fn resident_line() {
    let [n, us, longest] = [0, 1, 2].map(|i| RESIDENT[i].load(Relaxed));
    if n > 0 {
        tracing::info!(
            "new texture memory made resident before anything names it (VICTORY_WAIT_RESIDENT=0: not), since the start:              {n} times, {:.1} ms, the longest {:.1} ms",
            us as f64 / 1e3,
            longest as f64 / 1e3
        );
    }
}

/// Whether such textures are placed (`VICTORY_PLACED_TEX=1`; off by default).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLACED_TEX").is_some_and(|v| v == "1"))
}

/// The heap properties of memory in the PC that the card reads and writes
/// over the bus and the processor does not see, so that textures there
/// have the card's own layout (`crate::resource` spill).
pub(crate) fn shared_heap() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    }
}

/// A texture as `desc` says (no target, depth or unordered access; in
/// `state`) placed in a heap, in the PC's memory with `shared`
/// ([`shared_heap`]); `None` if it must be committed.
pub(crate) fn make(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    shared: bool,
) -> Option<ID3D12Resource> {
    let made = try_make(device, desc, state, shared);
    COUNTS[if made.is_some() { 1 } else { 2 }].fetch_add(1, Relaxed);
    made
}

fn try_make(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    shared: bool,
) -> Option<ID3D12Resource> {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    if desc.Flags.0 & TARGET_DEPTH_UNORDERED != 0 || desc.SampleDesc.Count > 1 {
        return None;
    }
    let mut desc = *desc;
    desc.Alignment = SMALL;
    // SAFETY: a plain query.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    if info.Alignment != SMALL || info.SizeInBytes == u64::MAX {
        desc.Alignment = 0;
        // SAFETY: as above.
        info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    }
    let (size, align) = (info.SizeInBytes, info.Alignment);
    if size == 0 || size > LARGEST || (align != SMALL && align != NORMAL) {
        return None;
    }
    let (heap, range) = take(device, size.next_multiple_of(align), align, shared)?;
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: a live heap for such textures, and a range of it that is ours.
    let made = unsafe { device.CreatePlacedResource(&heap, range.offset, &desc, state, None, &mut texture) };
    if let Err(e) = made {
        tracing::warn!(error = %e, ?desc, "no placed host texture");
    }
    let texture = texture?;
    const RANGE: GUID = GUID::from_u128(0x6d2f1c1e_5a0b_4c7e_9a64_0f8d3b7e2a91);
    let ticket = ticket(range);
    // SAFETY: a live texture, which takes a reference of its own.
    if let Err(e) = unsafe { texture.SetPrivateDataInterface(&RANGE, &ticket) } {
        // The range must not go back while the texture lives.
        tracing::warn!(error = %e, "a placed host texture keeps its range for the run");
        std::mem::forget(ticket);
    }
    Some(texture)
}

/// `size` bytes of a heap of kind `align`, in the PC's memory with
/// `shared`; a new heap if none has them.
fn take(device: &ID3D12Device, size: u64, align: u64, shared: bool) -> Option<(ID3D12Heap, Range)> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let mut blocks = blocks();
    // The fullest heap first: the emptier ones run empty as their textures
    // go, and are freed ([`let_go_idle`]). In heap order the bytes a scene
    // let go of stayed spread over every heap (2.2 GB free inside 73 heaps
    // of 64 MiB after the opening cutscene, none of them empty).
    let mut order: Vec<usize> =
        (0..blocks.len())
            .filter(|&i| blocks[i].align == align && blocks[i].shared == shared && blocks[i].free_bytes >= size)
            .collect();
    order.sort_unstable_by_key(|&i| blocks[i].free_bytes);
    for i in order {
        let b = &mut blocks[i];
        if let Some(offset) = b.free.take(size) {
            b.free_bytes -= size;
            if !shared {
                FREE.fetch_sub(size, Relaxed);
            }
            let range = Range { block: b.id, offset, size, shared };
            return Some((b.heap.clone(), range));
        }
    }
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: HEAP,
        Properties: if shared {
            shared_heap()
        } else {
            D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            }
        },
        Alignment: NORMAL,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | not_resident(),
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: plain creation.
    if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
        tracing::warn!(error = %e, "no heap for host textures");
    }
    let heap = heap?;
    if let Err(e) = resident(device, &heap) {
        tracing::warn!(error = %e, "a heap for host textures is not resident");
        return None;
    }
    COUNTS[0].fetch_add(1, Relaxed);
    if shared {
        SHARED_HEAPS.fetch_add(1, Relaxed);
    } else {
        FREE.fetch_add(HEAP - size, Relaxed);
    }
    let id = NEXT.fetch_add(1, Relaxed);
    blocks.push(Block {
        id,
        heap: heap.clone(),
        align,
        shared,
        free: Free(vec![(size, HEAP - size)]),
        free_bytes: HEAP - size,
        given: 0,
    });
    let range = Range { block: id, offset: 0, size, shared };
    Some((heap, range))
}

/// Frees heaps that have been empty for a while.
pub(crate) fn let_go_idle() {
    let now = crate::host::now_ms();
    let mut gone = Vec::new();
    let mut blocks = blocks();
    for (align, shared) in [(NORMAL, false), (SMALL, false), (NORMAL, true), (SMALL, true)] {
        let mut first = true;
        let mut i = 0;
        while i < blocks.len() {
            let b = &blocks[i];
            let empty = b.align == align && b.shared == shared && b.free_bytes == HEAP;
            // Heaps in the PC's memory are kept empty longer: few textures
            // go there, in bursts, and a heap made again is a heap made
            // resident again.
            let after = if shared { 12 * EMPTY_MS } else { EMPTY_MS };
            if empty && !first && now.saturating_sub(b.given) > after {
                if shared {
                    SHARED_HEAPS.fetch_sub(1, Relaxed);
                } else {
                    FREE.fetch_sub(HEAP, Relaxed);
                }
                gone.push(blocks.remove(i));
                continue;
            }
            first &= !empty;
            i += 1;
        }
    }
    drop(blocks);
    // Freed outside the lock.
    drop(gone);
}

/// Bytes of the heaps in the card's memory no texture has.
pub(crate) fn free_bytes() -> u64 {
    FREE.load(Relaxed)
}

/// Heaps in the PC's memory alive now.
pub(crate) fn shared_heaps() -> u64 {
    SHARED_HEAPS.load(Relaxed)
}

/// Since the start: heaps made, textures placed, and textures asked for
/// that had to be committed.
pub(crate) fn stats() -> [u64; 3] {
    [0, 1, 2].map(|i| COUNTS[i].load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_come_back_whole() {
        let mut free = Free(vec![(0, HEAP)]);
        let sizes = [NORMAL, 3 * NORMAL, 16 * NORMAL, NORMAL];
        let at: Vec<u64> = sizes.iter().map(|&s| free.take(s).unwrap()).collect();
        for (i, &a) in at.iter().enumerate() {
            for (j, &b) in at.iter().enumerate() {
                assert!(i == j || a + sizes[i] <= b || b + sizes[j] <= a);
            }
        }
        // The smallest range that has room is taken first.
        free.give(at[1], sizes[1]);
        assert_eq!(free.take(2 * NORMAL), Some(at[1]));
        free.give(at[1], 2 * NORMAL);
        for i in [2, 0, 3] {
            free.give(at[i], sizes[i]);
        }
        assert_eq!(free.0, [(0, HEAP)]);
        assert_eq!(free.take(HEAP + 1), None);
    }

    /// Textures alive at once have ranges of their own, and a range comes
    /// back with the last reference to its texture, not before.
    #[test]
    fn a_range_lives_as_long_as_its_texture() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC7_UNORM, DXGI_SAMPLE_DESC};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let make = |width: u32, levels| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width.into(),
                Height: width,
                DepthOrArraySize: 1,
                MipLevels: levels,
                Format: DXGI_FORMAT_BC7_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            try_make(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON, false)
        };
        let big = make(2048, 1).unwrap();
        let small = make(64, 7).unwrap();
        // SAFETY: plain queries on live textures.
        let (a, b) = unsafe { (big.GetDesc(), small.GetDesc()) };
        assert_eq!((a.Width, b.MipLevels), (2048, 7));
        let free = free_bytes();
        let slot = big.clone();
        drop(big);
        assert_eq!(free_bytes(), free, "a slot still holds it");
        drop(slot);
        // Other tests' submissions may be running: then later.
        for _ in 0..200 {
            if free_bytes() > free {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(free_bytes() >= free + (4 << 20));
        assert!(make(8192, 14).is_none(), "too large to place");
    }

    /// A texture placed in the PC's memory takes the texels copied into it
    /// and gives them back, in a heap of that memory.
    #[test]
    fn a_texture_in_the_pcs_memory_keeps_its_texels() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const SIDE: u32 = 64;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: SIDE.into(),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let texture = try_make(&host.device, &desc, D3D12_RESOURCE_STATE_COPY_DEST, true).expect("placed in the PC's memory");
        assert!(shared_heaps() >= 1);
        // Rows of 256 bytes: the copies' pitch as it is.
        let pitch = SIDE * 4;
        let bytes = u64::from(pitch * SIDE);
        let (upload, at) = host.upload_buffer(bytes).unwrap();
        let texels: Vec<u8> = (0..bytes).map(|i| (i * 7 + 3) as u8).collect();
        // SAFETY: a mapped upload buffer of `bytes` bytes.
        unsafe { std::ptr::copy_nonoverlapping(texels.as_ptr(), at, texels.len()) };
        let back = host.read_back_buffer(bytes).unwrap();
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: desc.Format,
                Width: SIDE,
                Height: SIDE,
                Depth: 1,
                RowPitch: pitch,
            },
        };
        let buffer_at = |buffer: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let texture_at = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let list = host.new_list().unwrap();
        // SAFETY: an open list; live resources in the states named.
        unsafe {
            list.list.CopyTextureRegion(&texture_at, 0, 0, 0, &buffer_at(&upload), None);
            list.list.ResourceBarrier(&[crate::host::transition(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyTextureRegion(&buffer_at(&back), 0, 0, 0, &texture_at, None);
        }
        list.close();
        host.execute(&[&list]);
        let mut data = std::ptr::null_mut();
        // SAFETY: our read-back buffer, written by the list that has run.
        unsafe {
            back.Map(0, None, Some(&mut data)).unwrap();
            let read = std::slice::from_raw_parts(data.cast::<u8>(), texels.len());
            assert!(read == &texels[..], "the texels came back changed");
            back.Unmap(0, None);
        }
    }
}
