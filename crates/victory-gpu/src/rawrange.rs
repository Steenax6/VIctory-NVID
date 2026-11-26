//! The range a raw buffer view may cover.
//!
//! A view's range has to lie inside its buffer: a description that runs past
//! it is not a valid call, and what a refused creation leaves in the slot is
//! undefined, not "whatever was there", which is enough to keep such a
//! description away from a slot shaders read by index. Not seen on this
//! machine; recorded as a possibility (fxceless/victory-fork pull request
//! #1266).

use std::sync::atomic::{AtomicU64, Ordering};

/// Where a raw buffer view starts and how many dwords it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RawRange {
    pub(crate) first: u64,
    pub(crate) elements: u32,
}

/// Why a request lost part of its range, for the counters below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cut {
    /// The start is at or past the buffer's last dword: nothing to view.
    Past,
    /// The length was cut to what the buffer holds.
    Length,
    /// The title named less than a dword: one element, as before this guard.
    Short,
    /// The start is not on the 16 bytes a raw view needs.
    Unaligned,
}

/// Where a raw buffer view starts and how many dwords it covers, both inside
/// `size` bytes of buffer: `None` when the request has nothing in the buffer.
///
/// `FirstElement` is 64-bit and `NumElements` 32-bit, so only the length is
/// cut at `u32::MAX`; the start is used as asked and never moved. Moving it
/// is what a clamp would do, and clamping the start of a request that lies
/// entirely past the buffer turns it into a view of the buffer's tail: 100
/// bytes of buffer with an offset of 112 would view bytes 96 to 99 as if they
/// had been asked for.
///
/// Callers pass `offset` on 16 bytes (`draw.rs` biases by `offset & 15`), so
/// a raw view's start stays on that boundary; an offset off it is refused.
pub(crate) fn raw_range(size: u64, offset: u64, len: u64) -> (Option<RawRange>, Option<Cut>) {
    let words = size / 4;
    if !offset.is_multiple_of(16) {
        return (None, Some(Cut::Unaligned));
    }
    let asked = offset / 4;
    if asked >= words {
        return (None, Some(Cut::Past));
    }
    let avail = words - asked;
    // A length under a dword made one element before this guard; the floor
    // stays, and the buffer's end caps it.
    let want = (len / 4).max(1);
    let elements = want.min(avail).min(u64::from(u32::MAX));
    let cut = if elements < want {
        Some(Cut::Length)
    } else if len < 4 {
        Some(Cut::Short)
    } else {
        None
    };
    (Some(RawRange { first: asked, elements: elements as u32 }), cut)
}

/// Counts what [`raw_range`] refused or cut, by kind (`"SRV"`, `"UAV"`) and
/// cause, so that a run can say whether the guard applied at all: said on the
/// first and every 4096th, nothing per view beyond one atomic.
pub(crate) fn raw_range_count(kind: &'static str, cut: Cut, offset: u64, len: u64, size: u64) {
    static SRV: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    static UAV: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    let cells = if kind == "SRV" { &SRV } else { &UAV };
    let n = cells[cut as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n.is_multiple_of(4096) {
        tracing::warn!(
            kind,
            cut = ?cut,
            offset,
            len,
            size,
            seen = n,
            "a raw buffer view the title's range did not cover"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rr(size: u64, offset: u64, len: u64) -> (Option<(u64, u32)>, Option<Cut>) {
        let (range, cut) = raw_range(size, offset, len);
        (range.map(|v| (v.first, v.elements)), cut)
    }

    fn range(size: u64, offset: u64, len: u64) -> Option<(u64, u32)> {
        rr(size, offset, len).0
    }

    #[test]
    fn the_range_stays_inside_the_buffer() {
        assert_eq!(range(1024, 0, 256), Some((0, 64)));
        assert_eq!(range(1024, 768, 512), Some((192, 64)));
        assert_eq!(range(1024, 1024, 16), None, "a start at the end");
        assert_eq!(range(1024, 1040, 16), None, "a start past the end");
        assert_eq!(range(100, 112, 16), None, "no intersection: never the tail");
        assert_eq!(range(100, 96, 16), Some((24, 1)), "the last dword");
        assert_eq!(range(100, 96, 4), Some((24, 1)));
        assert_eq!(range(0, 0, 16), None, "an empty buffer");
        assert_eq!(range(3, 0, 16), None, "no whole dword");
        assert_eq!(range(1024, 0, 0), Some((0, 1)), "the floor of one element stands");
        assert_eq!(range(1024, 0, 1), Some((0, 1)), "as does a length under a dword");
        assert_eq!(range(1024, 3, 16), None, "not on 16 bytes");
        assert_eq!(range(1024, u64::MAX, 16), None, "no wrap, no clamp back");
        assert_eq!(range(1024, 0, u64::MAX), Some((0, 256)), "the length is cut");
    }

    #[test]
    fn the_cause_is_the_one_the_counters_report() {
        assert_eq!(rr(100, 112, 16), (None, Some(Cut::Past)));
        assert_eq!(rr(1024, 3, 16), (None, Some(Cut::Unaligned)));
        assert_eq!(rr(1024, 0, 0), (Some((0, 1)), Some(Cut::Short)));
        assert_eq!(rr(1024, 768, 512), (Some((192, 64)), Some(Cut::Length)));
        assert_eq!(rr(1024, 0, 256), (Some((0, 64)), None));
        assert_eq!(rr(1024, 0, 1020), (Some((0, 255)), None), "255 of 256 dwords");
    }

    /// `FirstElement` is 64-bit: a start past the 32-bit element index is
    /// kept, not narrowed; `NumElements` is 32-bit: the length alone is cut.
    #[test]
    fn the_start_keeps_its_64_bits_and_the_length_cuts_at_32() {
        let size = 1u64 << 40;
        let offset = (u64::from(u32::MAX) + 10) * 16;
        assert_eq!(range(size, offset, 64), Some((offset / 4, 16)));
        assert_eq!(range(size, 0, u64::MAX), Some((0, u32::MAX)));
        assert_eq!(rr(size, 1u64 << 60, 64), (None, Some(Cut::Past)));
        assert_eq!(rr(size, size - 16, 64), (Some((size / 4 - 4, 4)), Some(Cut::Length)));
    }

    #[test]
    fn accepted_ranges_hold_their_invariant() {
        let sizes = [0u64, 1, 3, 4, 100, 1024, 4096, 1 << 30];
        let offsets = [0u64, 4, 16, 48, 96, 112, 1024, 1040, 4096, u64::MAX];
        let lens = [0u64, 1, 3, 4, 16, 512, 1 << 20, u64::MAX];
        for size in sizes {
            for offset in offsets {
                for len in lens {
                    let (range, _) = raw_range(size, offset, len);
                    let allowed = offset.is_multiple_of(16) && offset / 4 < size / 4;
                    match range {
                        Some(r) => {
                            assert!(r.elements > 0, "{size} {offset} {len}");
                            assert_eq!(r.first, offset / 4, "{size} {offset} {len}: moved");
                            assert!(r.first + u64::from(r.elements) <= size / 4, "{size} {offset} {len}");
                        }
                        None => assert!(!allowed, "{size} {offset} {len}: refused though allowed"),
                    }
                }
            }
        }
    }
}
