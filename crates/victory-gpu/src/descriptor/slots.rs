//! Per-slot bookkeeping of the host heaps without trees, hashes of the
//! standard kind or one lock for a whole heap: flat sets by slot, striped
//! locks, and the ranges a batch of null descriptors is copied in. Pure
//! logic, tested without a device.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A set of slot numbers as a bitmap: O(1) insert, remove and lookup,
/// iteration in ascending order (as the `BTreeSet` it replaces).
#[derive(Default, Clone)]
pub(crate) struct SlotSet {
    words: Vec<u64>,
    len: usize,
}

impl SlotSet {
    /// Adds `slot`; true if it was not there.
    pub(crate) fn insert(&mut self, slot: u32) -> bool {
        let (word, bit) = (slot as usize / 64, 1u64 << (slot % 64));
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        let new = self.words[word] & bit == 0;
        self.words[word] |= bit;
        self.len += usize::from(new);
        new
    }

    /// Takes `slot` out; true if it was there.
    pub(crate) fn remove(&mut self, slot: u32) -> bool {
        let (word, bit) = (slot as usize / 64, 1u64 << (slot % 64));
        let Some(w) = self.words.get_mut(word) else {
            return false;
        };
        let was = *w & bit != 0;
        *w &= !bit;
        self.len -= usize::from(was);
        was
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn contains(&self, slot: u32) -> bool {
        self.words
            .get(slot as usize / 64)
            .is_some_and(|w| w & (1u64 << (slot % 64)) != 0)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The slots, ascending.
    pub(crate) fn to_vec(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len);
        if self.len == 0 {
            return out;
        }
        for (i, &w) in self.words.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let bit = w.trailing_zeros();
                out.push(i as u32 * 64 + bit);
                w &= w - 1;
            }
        }
        out
    }
}

/// Slots noted since they were last taken, each once, in the order first
/// noted (the `HashSet` this replaces gave them in no order at all).
#[derive(Default)]
pub(crate) struct SlotList {
    set: SlotSet,
    list: Vec<u32>,
}

impl SlotList {
    pub(crate) fn insert(&mut self, slot: u32) {
        if self.set.insert(slot) {
            self.list.push(slot);
        }
    }

    /// The slots noted, left noted (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn noted(&self) -> &[u32] {
        &self.list
    }

    /// The slots noted, and none from now on.
    pub(crate) fn take(&mut self) -> Vec<u32> {
        for &slot in &self.list {
            self.set.remove(slot);
        }
        std::mem::take(&mut self.list)
    }
}

/// A hasher for `u32` slot keys: one multiply instead of SipHash, which was
/// most of what a lookup by slot cost. Not for keys anyone can pick.
#[derive(Default, Clone, Copy)]
pub(crate) struct SlotHasher(u64);

const K: u64 = 0x9e37_79b9_7f4a_7c15;

impl std::hash::Hasher for SlotHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(K);
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.0 = (self.0 ^ u64::from(n)).wrapping_mul(K);
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(K);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn finish(&self) -> u64 {
        // The table takes its bucket from the low bits and its tag from
        // the top ones: both get the multiply's well-mixed high half.
        self.0 ^ (self.0 >> 32)
    }
}

/// A map keyed by slot with [`SlotHasher`].
pub(crate) type SlotMap<V> = std::collections::HashMap<u32, V, std::hash::BuildHasherDefault<SlotHasher>>;

/// Stripes of the per-slot locks: slot `i` is guarded by stripe `i % STRIPES`.
pub(crate) const STRIPES: usize = 1024;

/// Locks for per-slot state shared by threads that each touch one slot
/// (draws recording on five threads, the submit thread's passes): a gate
/// that work on one slot takes shared and work on many slots (or on slots
/// it cannot name up front) takes exclusive, and under the shared gate a
/// lock per stripe of slots. Before, one mutex per heap ordered all of it,
/// held while host views were made, and the recording threads stood behind
/// each other and behind the submit thread's pass over the heap.
///
/// Rules: the gate before any stripe; a thread holding a stripe takes no
/// other stripe of the same `SlotLocks`, except the one pass
/// ([`SlotLocks::stripes`]) that takes several in ascending order while no
/// other thread holds more than one; nothing waits for the exclusive gate
/// while holding it shared.
pub(crate) struct SlotLocks {
    gate: RwLock<()>,
    stripes: Box<[Stripe]>,
    /// `false` (`VICTORY_DESCRIPTOR_STRIPES=0`): every lock is the gate,
    /// exclusive: one lock for the heap, as before.
    striped: bool,
}

/// A stripe's lock, on a cache line of its own. A lock is a few bytes, and
/// eight of them shared a line: the slots a draw makes are handed out one
/// after the other, so the threads that record took neighbouring stripes
/// and passed that line between them at every slot (8.7 ms a frame of the
/// recording threads in `Striped::with`'s lock with nobody waiting for it,
/// the profile of the evening of 2026-11-26).
#[repr(align(64))]
struct Stripe(Mutex<()>);

/// A held lock of a single slot.
pub(crate) enum SlotGuard<'a> {
    Shared(#[allow(dead_code)] RwLockReadGuard<'a, ()>, #[allow(dead_code)] MutexGuard<'a, ()>),
    Exclusive(#[allow(dead_code)] RwLockWriteGuard<'a, ()>),
}

impl SlotGuard<'_> {
    /// Whether it excludes every other slot's work too.
    pub(crate) fn exclusive(&self) -> bool {
        matches!(self, SlotGuard::Exclusive(_))
    }
}

/// The most stripes [`SlotLocks::stripes`] holds at once.
pub(crate) const MOST_STRIPES: usize = 64;

/// Stripes held together, with the gate shared (or the gate exclusive).
/// (Fields are dropped in order: the stripes before the gate.)
pub(crate) struct StripesGuard<'a> {
    _stripes: [Option<MutexGuard<'a, ()>>; MOST_STRIPES],
    _gate: Option<RwLockReadGuard<'a, ()>>,
    _exclusive: Option<RwLockWriteGuard<'a, ()>>,
}

impl SlotLocks {
    pub(crate) fn new(striped: bool) -> Self {
        SlotLocks {
            gate: RwLock::new(()),
            stripes: (0..STRIPES).map(|_| Stripe(Mutex::new(()))).collect(),
            striped,
        }
    }

    /// The lock of slot `slot` alone.
    pub(crate) fn slot(&self, slot: u32) -> SlotGuard<'_> {
        if !self.striped {
            return self.all();
        }
        let gate = self.gate.read().unwrap_or_else(|e| e.into_inner());
        let stripe = self.stripes[slot as usize % STRIPES].0.lock().unwrap_or_else(|e| e.into_inner());
        SlotGuard::Shared(gate, stripe)
    }

    /// Every slot's lock.
    pub(crate) fn all(&self) -> SlotGuard<'_> {
        SlotGuard::Exclusive(self.gate.write().unwrap_or_else(|e| e.into_inner()))
    }

    /// The locks of all of `slots` at once (each stripe once, ascending),
    /// with nothing allocated; every slot's if there are more than
    /// [`MOST_STRIPES`]. For the one pass that holds several; see the
    /// rules above.
    pub(crate) fn stripes(&self, slots: &[u32]) -> StripesGuard<'_> {
        let mut guard = StripesGuard { _stripes: std::array::from_fn(|_| None), _gate: None, _exclusive: None };
        if !self.striped || slots.len() > MOST_STRIPES {
            guard._exclusive = Some(self.gate.write().unwrap_or_else(|e| e.into_inner()));
            return guard;
        }
        let mut held = [0u16; MOST_STRIPES];
        for (h, &s) in held.iter_mut().zip(slots) {
            *h = (s as usize % STRIPES) as u16;
        }
        let held = &mut held[..slots.len()];
        held.sort_unstable();
        guard._gate = Some(self.gate.read().unwrap_or_else(|e| e.into_inner()));
        let mut last = None;
        for (at, &s) in held.iter().enumerate() {
            if last != Some(s) {
                guard._stripes[at] = Some(self.stripes[s as usize].0.lock().unwrap_or_else(|e| e.into_inner()));
                last = Some(s);
            }
        }
        guard
    }
}

/// Per-slot values each guarded by a stripe lock of their own (not the
/// [`SlotLocks`] of the heap): slot `i` by lock `i % STRIPES`.
pub(crate) struct Striped<T> {
    cells: Box<[std::cell::UnsafeCell<T>]>,
    locks: Box<[Stripe]>,
}

// SAFETY: every access to a cell goes through `with`, under the lock of its
// stripe; values move between threads only as `T: Send` allows.
unsafe impl<T: Send> Sync for Striped<T> {}

impl<T> Striped<T> {
    pub(crate) fn new(n: usize, mut value: impl FnMut() -> T) -> Self {
        Striped {
            cells: (0..n).map(|_| std::cell::UnsafeCell::new(value())).collect(),
            locks: (0..STRIPES).map(|_| Stripe(Mutex::new(()))).collect(),
        }
    }

    /// `f` on slot `slot`'s value under its stripe's lock; `None` past the
    /// end. `f` must not reach this `Striped` again.
    pub(crate) fn with<R>(&self, slot: u32, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let cell = self.cells.get(slot as usize)?;
        let _held = self.locks[slot as usize % STRIPES].0.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the stripe's lock is held, and only `with` reaches cells.
        Some(f(unsafe { &mut *cell.get() }))
    }
}

/// A record's fingerprint against the one last seen in its slot (0: never
/// looked at): whether it is new, noting it if so. A slot never looked at
/// that holds only zeros is as it was (`HostHeap::mirror_copied`).
pub(crate) fn print_is_new(last: &mut u64, print: u64, record: &[u8]) -> bool {
    if *last == print || (*last == 0 && record.iter().all(|&b| b == 0)) {
        return false;
    }
    *last = print;
    true
}

/// The ranges that copy null descriptors to slots `slots` (ascending or
/// not), slot `i` from template `kinds[i]`: a template holds `run` copies
/// of each kind of null descriptor one after the other, kind `k` from
/// descriptor `k * run`. Destination ranges are runs of consecutive slots;
/// source ranges are runs of one kind, at most `run` long. Both cover the
/// same number of descriptors, in the same order, as `CopyDescriptors`
/// pairs them.
#[cfg(test)]
pub(crate) fn null_ranges(slots: &[(u32, usize)], run: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let (mut dst, mut src) = (Vec::new(), Vec::new());
    null_ranges_into(slots, run, &mut dst, &mut src);
    (dst, src)
}

/// [`null_ranges`] into `dst` and `src` (cleared first).
pub(crate) fn null_ranges_into(slots: &[(u32, usize)], run: u32, dst: &mut Vec<(u32, u32)>, src: &mut Vec<(u32, u32)>) {
    dst.clear();
    src.clear();
    let mut last: Option<(u32, usize)> = None;
    for &(slot, kind) in slots {
        match dst.last_mut() {
            Some((start, n)) if start.checked_add(*n) == Some(slot) => *n += 1,
            _ => dst.push((slot, 1)),
        }
        match src.last_mut() {
            Some((_, n)) if last.is_some_and(|(_, k)| k == kind) && *n < run => *n += 1,
            _ => src.push((kind as u32 * run, 1)),
        }
        last = Some((slot, kind));
    }
}

/// Slots in `queue` (a map by slot) that are in `slots` (ascending,
/// distinct), found by one walk over the map's range when the slots lie
/// close together and by a lookup each otherwise.
#[cfg(test)]
pub(crate) fn present_in<V>(queue: &std::collections::BTreeMap<u32, V>, slots: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    present_into(queue, slots, &mut out);
    out
}

/// [`present_in`] into `out` (cleared first).
pub(crate) fn present_into<V>(queue: &std::collections::BTreeMap<u32, V>, slots: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let (Some(&lo), Some(&hi)) = (slots.first(), slots.last()) else {
        return;
    };
    if queue.is_empty() {
        return;
    }
    let sorted = slots.windows(2).all(|w| w[0] < w[1]);
    if sorted && ((hi - lo) as usize) <= slots.len() * 8 {
        out.extend(queue.range(lo..=hi).map(|(&k, _)| k).filter(|k| slots.binary_search(k).is_ok()));
    } else {
        out.extend(slots.iter().copied().filter(|k| queue.contains_key(k)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_sets_are_sets_in_ascending_order() {
        let mut set = SlotSet::default();
        let mut model = std::collections::BTreeSet::new();
        let mut x = 12345u64;
        for _ in 0..20_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let slot = (x >> 40) as u32 % 5000;
            if x & 1 == 0 {
                assert_eq!(set.insert(slot), model.insert(slot));
            } else {
                assert_eq!(set.remove(slot), model.remove(&slot));
            }
            assert_eq!(set.contains(slot), model.contains(&slot));
        }
        assert_eq!(set.len(), model.len());
        assert_eq!(set.to_vec(), model.iter().copied().collect::<Vec<_>>());
        assert!(!set.remove(1 << 30), "past the end");
    }

    #[test]
    fn slot_lists_give_each_slot_once_until_taken() {
        let mut list = SlotList::default();
        for slot in [5, 3, 5, 9, 3, 0] {
            list.insert(slot);
        }
        assert_eq!(list.take(), [5, 3, 9, 0]);
        assert!(list.take().is_empty());
        list.insert(3);
        assert_eq!(list.take(), [3]);
    }

    #[test]
    fn slot_maps_are_maps() {
        let mut map: SlotMap<u64> = SlotMap::default();
        for i in 0..10_000u32 {
            map.insert(i * 7, u64::from(i));
        }
        for i in 0..10_000u32 {
            assert_eq!(map.get(&(i * 7)), Some(&u64::from(i)));
            assert_eq!(map.get(&(i * 7 + 1)), None);
        }
    }

    #[test]
    fn prints_are_new_once() {
        let mut last = 0;
        assert!(!print_is_new(&mut last, 7, &[0; 32]), "zeros never looked at");
        assert_eq!(last, 0);
        assert!(print_is_new(&mut last, 7, &[1; 32]));
        assert_eq!(last, 7);
        assert!(!print_is_new(&mut last, 7, &[1; 32]));
        assert!(print_is_new(&mut last, 9, &[0; 32]), "zeros after a record");
    }

    /// The ranges pair every slot with a template of its kind.
    #[test]
    fn null_ranges_pair_slots_with_their_kind() {
        let run = 4;
        let slots = [(10, 2), (11, 2), (12, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (30, 5), (29, 5)];
        let (dst, src) = null_ranges(&slots, run);
        let expand = |ranges: &[(u32, u32)]| -> Vec<u32> { ranges.iter().flat_map(|&(s, n)| s..s + n).collect() };
        let d = expand(&dst);
        let s = expand(&src);
        assert_eq!(d, slots.iter().map(|&(i, _)| i).collect::<Vec<_>>());
        assert_eq!(s.len(), d.len());
        for (k, &(_, kind)) in slots.iter().enumerate() {
            assert_eq!(s[k] / run, kind as u32, "slot {k} copies its own kind");
        }
        assert!(src.iter().all(|&(_, n)| n <= run));
        assert_eq!(dst, [(10, 3), (20, 5), (30, 1), (29, 1)]);
        assert!(null_ranges(&[], run).0.is_empty());
    }

    #[test]
    fn present_slots_are_found_both_ways() {
        let queue: std::collections::BTreeMap<u32, ()> = [3, 4, 9, 100, 5000].into_iter().map(|k| (k, ())).collect();
        assert_eq!(present_in(&queue, &[2, 3, 9, 10]), [3, 9]);
        assert_eq!(present_in(&queue, &[4, 5000]), [4, 5000], "far apart");
        assert_eq!(present_in(&queue, &[9, 3]), [9, 3], "unsorted");
        assert!(present_in(&queue, &[]).is_empty());
    }

    #[test]
    fn striped_values_and_locks() {
        let values = Striped::new(3000, || 0u32);
        assert_eq!(values.with(2999, |v| {
            *v += 1;
            *v
        }), Some(1));
        assert_eq!(values.with(3000, |v| *v), None);
        let locks = SlotLocks::new(true);
        {
            let one = locks.slot(5);
            assert!(!one.exclusive());
            // Another stripe is free meanwhile.
            let _other = locks.slot(6);
        }
        {
            let _many = locks.stripes(&[1, 1025, 7, 1]);
            // Stripe 1 and 7 are held; stripe 2 is not.
            let _free = locks.slot(2);
        }
        assert!(locks.all().exclusive());
        let one_lock = SlotLocks::new(false);
        assert!(one_lock.slot(5).exclusive());
    }
}
