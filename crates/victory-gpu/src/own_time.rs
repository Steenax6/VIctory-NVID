//! The layer's own work on the card, timed (`VICTORY_GPU_TIMING`,
//! diagnostics builds): timestamp pairs around the passes of ours that are
//! not commands of a title list (those are timed with the list's notes,
//! `HostList::note` and `copystat::command_took`). What is marked:
//!
//! - the send-up of changed ranges to copies in the card's memory and the
//!   hash passes after it (`Host::card_before`'s sync list, per copy),
//! - the compare passes after a submission's lists, and the fetch inside
//!   flights behind them (its check list),
//! - the fetches of flagged pieces on the direct queue (`Host::card_back`,
//!   `Host::relay_card_back` the old way; a fetch on the copy queue of
//!   `crate::queue2` is not marked: its wait is in the "waits for the
//!   card" of the card copies' line),
//! - list preludes: their state barriers, depth/colour twin copies and
//!   the hoisted indirect clamp (`crate::indirect::record_hoisted`).
//!
//! A pair's two slots in the read-back buffer are zeroed when the first is
//! recorded; the frame-rate line takes the pairs whose list has run (both
//! slots set) and keeps the others for a later line. Nothing is recorded
//! with the switch off.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// What a pair of timestamps measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Own {
    SendUp = 0,
    Hash,
    Compare,
    InflightFetch,
    Fetch,
    PreludeBarriers,
    TwinCopies,
    Hoisted,
}

const KINDS: usize = 8;

const NAMES: [&str; KINDS] = [
    "send-ups",
    "hashes",
    "compares",
    "fetches inside flights",
    "fetches",
    "prelude barriers",
    "prelude twin copies",
    "prelude indirect clamps",
];

/// Timestamp pairs that can be in the air at once: some 30 submissions a
/// frame with a pair or two per copy sent up, and a line every second.
const PAIRS: u32 = 1 << 15;

/// A pair left unrun for this many lines is given up (its list was reset
/// without being submitted); its slots are not used again.
const GIVE_UP: u32 = 30;

/// A pair begun in a list, to end in the same list.
pub(crate) struct Mark {
    pair: u32,
    what: Own,
}

struct Stamps {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    data: *mut u64,
    free: Vec<u32>,
    /// Pairs recorded and not read yet, with the lines they have waited.
    pending: Vec<(u32, Own, u32)>,
    /// Since the last line, per kind: pairs and ticks.
    sums: [(u64, u64); KINDS],
    /// Since the last line: pairs not recorded for want of a free one.
    short: u64,
}

// SAFETY: D3D12 objects (free-threaded) and a persistent mapping of the
// read-back buffer, used under the mutex.
unsafe impl Send for Stamps {}

static STAMPS: Mutex<Option<Option<Stamps>>> = Mutex::new(None);

/// Tests time without the switch.
#[cfg(test)]
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether our passes are timed.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    #[cfg(feature = "diagnostics")]
    {
        crate::host::timing_on()
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        false
    }
}

fn make() -> Option<Stamps> {
    let device = &crate::host::host()?.device;
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(PAIRS) * 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: plain object creation and a persistent mapping of a read-back
    // buffer.
    unsafe {
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: PAIRS * 2,
                    NodeMask: 0,
                },
                &mut heap,
            )
            .ok()?;
        device
            .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
            .ok()?;
        let back = back?;
        back.Map(0, None, Some(&mut data)).ok()?;
        Some(Stamps {
            heap: heap?,
            back,
            data: data.cast(),
            free: (0..PAIRS).rev().collect(),
            pending: Vec::new(),
            sums: [(0, 0); KINDS],
            short: 0,
        })
    }
}

/// A timestamp in `list` (open, on the direct queue) before the work
/// `what`; `None` with timing off or no pair free.
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, what: Own) -> Option<Mark> {
    if !on() {
        return None;
    }
    let mut stamps = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
    let stamps = stamps.get_or_insert_with(make).as_mut()?;
    let Some(pair) = stamps.free.pop() else {
        stamps.short += 1;
        return None;
    };
    // SAFETY: the pair's two slots of the mapping; no list in the air
    // resolves into them (the pair was free).
    unsafe {
        stamps.data.add(2 * pair as usize).write_volatile(0);
        stamps.data.add(2 * pair as usize + 1).write_volatile(0);
        list.EndQuery(&stamps.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * pair);
    }
    Some(Mark { pair, what })
}

/// The timestamp after the work of `mark` (from [`begin`] in the same
/// list), and the pair resolved into the read-back buffer.
pub(crate) fn end(list: &ID3D12GraphicsCommandList, mark: Option<Mark>) {
    let Some(mark) = mark else {
        return;
    };
    let mut stamps = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(stamps)) = stamps.as_mut() else {
        return;
    };
    // SAFETY: an open list, our heap and read-back buffer (in the copy
    // destination state for good), the pair's two slots.
    unsafe {
        list.EndQuery(&stamps.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * mark.pair + 1);
        list.ResolveQueryData(&stamps.heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * mark.pair, 2, &stamps.back, 16 * u64::from(mark.pair));
    }
    stamps.pending.push((mark.pair, mark.what, 0));
}

/// The pairs whose lists have run since the last call, per kind (pairs,
/// ticks); the pairs still waiting; the pairs not taken for want of one.
fn collect() -> Option<([(u64, u64); KINDS], u64, u64)> {
    let mut stamps = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(stamps)) = stamps.as_mut() else {
        return None;
    };
    let Stamps { data, free, pending, sums, short, .. } = stamps;
    let mut waiting = 0u64;
    pending.retain_mut(|(pair, what, age)| {
        // SAFETY: the pair's slots of the mapping.
        let (a, b) = unsafe { (data.add(2 * *pair as usize).read_volatile(), data.add(2 * *pair as usize + 1).read_volatile()) };
        if a != 0 && b != 0 {
            let s = &mut sums[*what as usize];
            s.0 += 1;
            s.1 += b.saturating_sub(a);
            free.push(*pair);
            return false;
        }
        *age += 1;
        waiting += 1;
        // Given up, and its slots not used again.
        *age < GIVE_UP
    });
    let out = (*sums, waiting, *short);
    *sums = [(0, 0); KINDS];
    *short = 0;
    Some(out)
}

/// The frame-rate line's part: the pairs whose lists have run, per frame.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let Some(frequency) = crate::host::host().map(|h| h.timestamp_frequency()) else {
        return;
    };
    let Some((sums, waiting, short)) = collect() else {
        return;
    };
    let parts: Vec<String> = sums
        .iter()
        .zip(NAMES)
        .map(|(&(n, ticks), name)| {
            format!("{name} {:.2} ms in {:.0}", ticks as f64 * 1e3 / frequency.max(1) as f64 / frames, n as f64 / frames)
        })
        .collect();
    let all: u64 = sums.iter().map(|s| s.1).sum();
    tracing::info!(
        "the layer's own passes on the card per frame (VICTORY_GPU_TIMING): {:.2} ms in all; {}; {} timestamp pairs not run yet, {} not taken (none free)",
        all as f64 * 1e3 / frequency.max(1) as f64 / frames,
        parts.join(", "),
        waiting,
        short
    );
}

#[cfg(test)]
mod tests {
    use super::{FORCED, Own, begin, collect, end};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    /// A pair around a copy in a list of the host's device is read back
    /// once the list has run, under its kind, and not before.
    #[test]
    fn a_pair_is_read_once_its_list_has_run() {
        let Some(host) = crate::host::host() else {
            eprintln!("no host GPU: the timestamp pairs are not tested");
            return;
        };
        FORCED.store(true, std::sync::atomic::Ordering::Relaxed);
        let device = &host.device;
        // SAFETY: plain object creation and a list of our own, waited for.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
            let mark = begin(&list, Own::Compare);
            assert!(mark.is_some());
            end(&list, mark);
            list.Close().unwrap();
            // Recorded, not run: still waiting (other tests may time too).
            let (sums, waiting, _) = collect().unwrap();
            assert!(waiting >= 1);
            let _ = sums;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, event).unwrap();
            WaitForSingleObject(event, INFINITE);
            let (sums, _, _) = collect().unwrap();
            // (Other tests' compares may be timed meanwhile too.)
            assert!(sums[Own::Compare as usize].0 >= 1, "the pair was not read back");
        }
        FORCED.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
