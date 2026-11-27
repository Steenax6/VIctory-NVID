//! Title buffer copies out of memory shaders write, made on the card in the
//! order of the lists (`VICTORY_CARD_COPY=0`: as before).
//!
//! Why: the submitting thread is the frame's bottleneck, and inside a
//! submission it stood 14.35 ms a frame waiting for flights (profile of the
//! first room). Every such wait counted (`VICTORY_READ_WAIT=2`) was the same
//! case: the title copies 2,048 bytes out of a buffer its shaders write into
//! one of three small buffers of its own, in rotation; the copy is left to
//! its submission's landing ([`super::copy_goes_late`]), and a later
//! submission refreshes the destination's host copy, which then waits for
//! that landing (0.45 to 0.7 waits a frame, 18.7 ms each). On the console the
//! copy is card to card, in the list's order, and so it is here now: when the
//! title records it and both ends have host copies, the host list copies
//! between the two host copies at that place, and the destination counts as
//! written by the list's shaders ([`crate::shadow::Shadow::gpu_writes`]). The
//! card then has the bytes in the destination's host copy for whatever draws
//! with them next, the landing brings them to title memory as it does any
//! range shaders write, and a later refresh of the destination is not waited
//! for (it leaves ranges flights' shaders write alone, and the title's own
//! writes there go in beside them: `stale_under_flights`). No copy on the CPU
//! is deferred for it.
//!
//! The source's host copy is current at the copy's place: a refresh brings up
//! what the title wrote on the CPU before the submission (before the lists
//! run), and what the shaders of earlier lists wrote is on the card already.
//! A copy that does not meet the condition ([`decide`]) goes the way it went
//! before, decided when its submission is prepared.
//!
//! A step of the same submission behind such a copy that reads its
//! destination on the CPU (a fill of upload memory, a texture's bytes, another
//! title copy, a step of the title's own) would take the bytes from before the
//! copy: the copy is then made on the CPU as well, at its place among the
//! steps, behind the wait the old way had for it ([`cpu_too`]). The card's
//! copy writes the same bytes.
//!
//! States: host copies of buffers are used by shaders without explicit
//! states (buffers move out of the common state implicitly). The copy moves
//! the source to the copy-source state (a read state, which joins others a
//! list may have put it in, as an indirect count buffer) and the destination
//! to the copy-destination state, both implicitly; one barrier call before it
//! (every unordered access done: the shaders that wrote the source) and one
//! after it (the destination back to the common state, which also has the
//! copy done before anything later reads the destination or writes the
//! source; and every unordered access done). The source is left in its read
//! state; buffers fall back to the common state when the queue's call is
//! done. A copy into a buffer an earlier copy of the same list read from goes
//! the old way, as that buffer is still in the copy-source state there.

use super::{HostList, Step, fill_ahead, relay_exact, transition};
use std::sync::OnceLock;
use windows::Win32::Graphics::Direct3D12::*;

/// Whether such copies are made on the card (`VICTORY_CARD_COPY=0`: they
/// are made on the CPU, at the submission or its landing, as before).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_COPY").is_none_or(|v| v != "0"))
}

/// Which way a title buffer copy goes, and if not on the card, why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Way {
    /// Made on the card.
    Card = 0,
    /// The switch is off, or the relay does not land flights by what they
    /// write ([`relay_exact`]): nothing to wait for that the card would save.
    Off,
    /// Its source is not memory shaders were seen to write
    /// ([`fill_ahead::hot`]): made at the submission, waiting for nothing.
    NotHot,
    /// No host copy of the source holds it.
    NoSource,
    /// No host copy of the destination holds it.
    NoDestination,
    /// One end is title memory in place (`crate::inplace`): its bytes are
    /// the title's, not a copy the card writes.
    InPlace,
    /// Both ends are in the same host copy, or the destination's host copy
    /// was the source of an earlier copy of the list (it is in the copy-source
    /// state there).
    Same,
    /// The list's shader writes are not given back to the title
    /// (`VICTORY_BVH_QUARANTINE=1`).
    Quarantined,
}

#[cfg(feature = "diagnostics")]
const WAYS: usize = 8;

/// One end of a copy as far as [`decide`] goes: its host copy's number,
/// whether it is title memory in place, whether it holds the range.
#[derive(Clone, Copy, Debug)]
pub(super) struct End {
    pub(super) id: u64,
    pub(super) in_place: bool,
    pub(super) holds: bool,
}

/// The way of a copy of `len` bytes, by what is known when the title
/// records it: `hot` whether shaders were seen to write its source, the two
/// ends' host copies, `used` whether the destination's host copy was the
/// source of an earlier card copy of the list, `quarantined` the list's
/// quarantine. (`copy_goes_late` also asks whether flights in the air write
/// the source and not the destination; that is known only at the
/// submission, and the card needs neither: it reads the source's host copy
/// in the order of the lists whatever is in the air, and writes the
/// destination there in the same order.)
pub(super) fn decide(on: bool, len: usize, hot: bool, src: Option<End>, dst: Option<End>, used: bool, quarantined: bool) -> Way {
    if !on {
        return Way::Off;
    }
    if len == 0 || !hot {
        return Way::NotHot;
    }
    let Some(src) = src.filter(|s| s.holds) else {
        return Way::NoSource;
    };
    let Some(dst) = dst.filter(|d| d.holds) else {
        return Way::NoDestination;
    };
    if src.in_place || dst.in_place {
        return Way::InPlace;
    }
    if src.id == dst.id || used {
        return Way::Same;
    }
    if quarantined {
        return Way::Quarantined;
    }
    Way::Card
}

/// Since the last frame line: copies recorded, by [`Way`]; copies made on
/// the CPU as well ([`cpu_too`]).
#[cfg(feature = "diagnostics")]
static COUNTS: [std::sync::atomic::AtomicU64; WAYS + 1] = [const { std::sync::atomic::AtomicU64::new(0) }; WAYS + 1];

fn count(way: Way) {
    #[cfg(feature = "diagnostics")]
    COUNTS[way as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "diagnostics"))]
    let _ = way;
}

/// The host copy of `[address, address + len)`, if one holds it, and its
/// [`End`].
fn end_of(address: usize, len: usize) -> Option<(std::sync::Arc<crate::shadow::Shadow>, End)> {
    let shadow = crate::shadow::existing(address as u64, len as u64)?;
    let (start, bytes) = shadow.span();
    let holds = address as u64 >= start && (address as u64).checked_add(len as u64).is_some_and(|end| end <= start + bytes);
    let end = End { id: shadow.id(), in_place: shadow.in_place(), holds };
    Some((shadow, end))
}

/// The title records a copy of `len` bytes from `src` to `dst` (title
/// memory) into `list`: whether it is made on the card. If so the host list
/// has the copy and the steps that go with it, and nothing is to be deferred
/// for it; if not, nothing was recorded.
pub(crate) fn record(list: &HostList, dst: usize, src: usize, len: usize) -> bool {
    if !on() || !relay_exact() {
        count(Way::Off);
        return false;
    }
    // Most copies end here (the hot pieces are few).
    if len == 0 || !fill_ahead::hot(src, len) {
        count(Way::NotHot);
        return false;
    }
    let from = end_of(src, len);
    let to = end_of(dst, len);
    let (quarantined, used) = {
        let deferred = list.deferred();
        let used = to.as_ref().is_some_and(|(to, _)| deferred.card_sources.contains(&to.id()));
        (deferred.quarantined, used)
    };
    let way = decide(true, len, true, from.as_ref().map(|f| f.1), to.as_ref().map(|t| t.1), used, quarantined);
    count(way);
    let (Way::Card, Some((from, _)), Some((to, _))) = (way, from, to) else {
        return false;
    };
    // What the title wrote on the CPU into the source before the submission
    // goes up before the lists run; the destination's host copy is brought
    // up to date as a range shaders write is, before its bytes are kept to
    // tell later what the card wrote there.
    from.refresh(list, src as u64, len as u64);
    to.refresh(list, dst as u64, len as u64);
    to.gpu_writes(list, dst as u64, len as u64);
    {
        let mut deferred = list.deferred();
        deferred.reads_read_back(src, len);
        deferred.copies.push(Step::Card(dst, src, len));
        deferred.card_sources.push(from.id());
    }
    let before = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    };
    let after = [
        transition(&to.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
            },
        },
    ];
    // SAFETY: an open list; both host copies are alive and held by the
    // list's steps (the refresh of the source, the write of the destination)
    // until it is reset, and hold the ranges (checked above).
    unsafe {
        list.list.ResourceBarrier(&[before]);
        list.list.CopyBufferRegion(&to.buffer, to.offset(dst as u64), &from.buffer, from.offset(src as u64), len as u64);
        list.list.ResourceBarrier(&after);
    }
    true
}

/// What a step of a submission does on the CPU, as far as a card copy before
/// it goes ([`cpu_among`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cpu {
    /// A copy made on the card: destination, source, bytes.
    Card(usize, usize, usize),
    /// A title copy on the CPU: destination, source, bytes.
    Copy(usize, usize, usize),
    /// Reads `[.0, .1)` of title memory on the CPU: a fill of upload memory,
    /// a texture's bytes.
    Reads(usize, usize),
    /// A step of the title's own, which may read anything.
    Any,
    /// Nothing on the CPU that a card copy's destination matters to: a
    /// refresh of a host copy (it leaves ranges shaders write alone), a
    /// range given to shaders (the card runs it in order), work of ours.
    Nothing,
}

fn cpu(step: &Step) -> Cpu {
    match step {
        &Step::Card(dst, src, len) => Cpu::Card(dst, src, len),
        &Step::Copy(dst, src, len) => Cpu::Copy(dst, src, len),
        &Step::Fill(_, src, len) => Cpu::Reads(src, src.saturating_add(len)),
        Step::Texture(staging) => {
            let (at, len) = staging.memory();
            Cpu::Reads(at, at.saturating_add(len))
        }
        Step::Run(_) => Cpu::Any,
        Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => Cpu::Nothing,
    }
}

/// Of a submission's steps in order, the card copies a later step reads or
/// writes the destination of on the CPU: made on the CPU as well, at their
/// place.
pub(super) fn cpu_among(steps: &[Cpu]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let &Cpu::Card(dst, src, len) = step else {
            continue;
        };
        let end = dst.saturating_add(len);
        let over = |a: usize, b: usize| a < end && dst < b;
        if steps.get(i + 1..).unwrap_or(&[]).iter().any(|later| match *later {
            Cpu::Copy(d, s, n) => over(d, d.saturating_add(n)) || over(s, s.saturating_add(n)),
            Cpu::Reads(a, b) => over(a, b),
            Cpu::Any => true,
            Cpu::Card(..) | Cpu::Nothing => false,
        }) {
            out.push((dst, src, len));
        }
    }
    out
}

/// The card copies of `runs` that are made on the CPU as well ([`cpu_among`]);
/// `count`: counted for the frame line (once a submission, where its steps
/// are run).
pub(super) fn cpu_too(runs: &[&HostList], count: bool) -> Vec<(usize, usize, usize)> {
    // Most submissions have no card copy: nothing is gathered for them.
    if !runs.iter().any(|list| list.deferred().copies.iter().any(|step| matches!(step, Step::Card(..)))) {
        return Vec::new();
    }
    let mut steps = Vec::new();
    for list in runs {
        steps.extend(list.deferred().copies.iter().map(cpu));
    }
    let out = cpu_among(&steps);
    #[cfg(feature = "diagnostics")]
    if count {
        COUNTS[WAYS].fetch_add(out.len() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = count;
    out
}

/// The check (`VICTORY_CARD_COPY_CHECK=1`, diagnostics): when the flight of
/// a card copy has landed, the destination's bytes in title memory are
/// compared with the source's there, which is what the copy on the CPU at the
/// landing would have written ([`super::late_copy`]).
#[cfg(feature = "diagnostics")]
pub(super) mod check {
    use super::super::{Flight, Step, read_into};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    pub(in super::super) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_COPY_CHECK").is_some_and(|v| v == "1"))
    }

    /// Since the start: copies compared, those that differed, and of those
    /// the ones whose source a shader of a later list or step of the same
    /// flight was declared to write.
    pub(super) static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    /// Where a copy's destination differs from its source: the first offset
    /// that differs, the last, and how many bytes do.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct Differs {
        pub(super) first: usize,
        pub(super) last: usize,
        pub(super) bytes: usize,
    }

    pub(super) fn differs(dst: &[u8], src: &[u8]) -> Option<Differs> {
        let mut found: Option<Differs> = None;
        for (at, _) in dst.iter().zip(src).enumerate().filter(|(_, (a, b))| a != b) {
            let d = found.get_or_insert(Differs { first: at, last: at, bytes: 0 });
            d.last = at;
            d.bytes += 1;
        }
        found
    }

    /// One copy: compared in title memory now, counted, and the first few
    /// that differ logged. `later` says whether shaders behind it in its
    /// flight were declared to write its source.
    pub(super) fn one(number: u64, dst: usize, src: usize, len: usize, later: bool) -> Option<Differs> {
        let (mut a, mut b) = (vec![0u8; len], vec![0u8; len]);
        if !read_into(dst as *const u8, &mut a) || !read_into(src as *const u8, &mut b) {
            return None;
        }
        COUNTS[0].fetch_add(1, Relaxed);
        let found = differs(&a, &b)?;
        let n = COUNTS[1].fetch_add(1, Relaxed) + 1;
        if later {
            COUNTS[2].fetch_add(1, Relaxed);
        }
        if n <= 8 || n.is_power_of_two() {
            tracing::warn!(
                n,
                flight = number,
                frame = crate::frame::current(),
                dst = format_args!("{dst:#x}"),
                src = format_args!("{src:#x}"),
                len,
                first = format_args!("{:#x}", found.first),
                last = format_args!("{:#x}", found.last),
                bytes = found.bytes,
                later_shader_write = later,
                dst_bytes = format_args!("{:02x?}", &a[found.first..(found.first + 16).min(len)]),
                src_bytes = format_args!("{:02x?}", &b[found.first..(found.first + 16).min(len)]),
                "card copy check: a landed copy's destination differs from its source in title memory"
            );
        }
        Some(found)
    }

    /// `flight` has landed and its shader writes are in title memory: its
    /// card copies are compared.
    pub(in super::super) fn landed(flight: &Flight) {
        if !on() {
            return;
        }
        // The flight's steps in order, and for each card copy whether a
        // later step declares a shader write over its source.
        let mut steps: Vec<(bool, usize, usize, usize)> = Vec::new();
        for &i in &flight.order {
            for step in &flight.lists[i].host().deferred().copies {
                match *step {
                    Step::Card(dst, src, len) => steps.push((true, dst, src, len)),
                    Step::GpuWrite(_, at, len) => steps.push((false, at, 0, len)),
                    _ => {}
                }
            }
        }
        for (k, &(card, dst, src, len)) in steps.iter().enumerate() {
            if !card {
                continue;
            }
            let later = steps[k + 1..].iter().any(|&(c, at, _, n)| !c && at < src + len && src < at + n);
            one(flight.number, dst, src, len, later);
        }
    }

    pub(super) fn line() -> String {
        if !on() {
            return String::new();
        }
        let [compared, differing, later] = [0, 1, 2].map(|i| COUNTS[i].load(Relaxed));
        format!(
            "; checked since the start (VICTORY_CARD_COPY_CHECK=1): {compared} compared at their landing, {differing} differed \
             from their source in title memory ({later} of them with a later shader write of the source in their flight)"
        )
    }
}

/// With the frame line (diagnostics).
#[cfg(feature = "diagnostics")]
pub(super) fn line(frames: f64) {
    let c: Vec<f64> = COUNTS.iter().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames).collect();
    if c[Way::Card as usize] == 0.0 && c[Way::NoSource as usize] + c[Way::NoDestination as usize] + c[Way::InPlace as usize] + c[Way::Same as usize] == 0.0 && !check::on() {
        return;
    }
    tracing::info!(
        "title buffer copies out of memory shaders write, made on the card (VICTORY_CARD_COPY=0: on the CPU): {:.2} per frame, \
         {:.2} per frame of them with a later step of their submission reading the destination on the CPU (made there as \
         well); not on the card: {:.1} per frame not out of such memory, {:.2} no host copy of the source, {:.2} none of the \
         destination, {:.2} in place, {:.2} the same host copy, {:.2} quarantined, {:.1} switched off{}",
        c[Way::Card as usize],
        c[WAYS],
        c[Way::NotHot as usize],
        c[Way::NoSource as usize],
        c[Way::NoDestination as usize],
        c[Way::InPlace as usize],
        c[Way::Same as usize],
        c[Way::Quarantined as usize],
        c[Way::Off as usize],
        check::line()
    );
}

#[cfg(test)]
mod tests {
    use super::{Cpu, End, Way, cpu_among, decide};

    /// Which copies go on the card: out of memory shaders write, both ends
    /// held by host copies of their own that are not title memory in place,
    /// in a list whose shader writes go back to the title.
    #[test]
    fn a_copy_goes_on_the_card_only_between_two_host_copies_out_of_shader_memory() {
        let a = Some(End { id: 1, in_place: false, holds: true });
        let b = Some(End { id: 2, in_place: false, holds: true });
        assert_eq!(decide(true, 2048, true, a, b, false, false), Way::Card);
        assert_eq!(decide(false, 2048, true, a, b, false, false), Way::Off);
        assert_eq!(decide(true, 2048, false, a, b, false, false), Way::NotHot);
        assert_eq!(decide(true, 0, true, a, b, false, false), Way::NotHot);
        assert_eq!(decide(true, 2048, true, None, b, false, false), Way::NoSource);
        assert_eq!(decide(true, 2048, true, a, None, false, false), Way::NoDestination);
        // A host copy that does not hold the whole range is none.
        let short = Some(End { id: 3, in_place: false, holds: false });
        assert_eq!(decide(true, 2048, true, short, b, false, false), Way::NoSource);
        assert_eq!(decide(true, 2048, true, a, short, false, false), Way::NoDestination);
        let placed = Some(End { id: 4, in_place: true, holds: true });
        assert_eq!(decide(true, 2048, true, placed, b, false, false), Way::InPlace);
        assert_eq!(decide(true, 2048, true, a, placed, false, false), Way::InPlace);
        assert_eq!(decide(true, 2048, true, a, a, false, false), Way::Same);
        assert_eq!(decide(true, 2048, true, a, b, true, false), Way::Same);
        assert_eq!(decide(true, 2048, true, a, b, false, true), Way::Quarantined);
    }

    /// A card copy is made on the CPU as well only when a later step of the
    /// submission reads or writes its destination there.
    #[test]
    fn a_card_copy_is_made_on_the_cpu_too_only_under_a_later_cpu_read_of_its_destination() {
        let card = Cpu::Card(0x1000, 0x9000, 0x800);
        let made = vec![(0x1000, 0x9000, 0x800)];
        // Alone, behind refreshes and shader ranges, behind another card
        // copy out of it, or with CPU work elsewhere: on the card only.
        assert!(cpu_among(&[card]).is_empty());
        assert!(cpu_among(&[card, Cpu::Nothing, Cpu::Card(0x5000, 0x1000, 0x800)]).is_empty());
        assert!(cpu_among(&[card, Cpu::Reads(0x1800, 0x2000), Cpu::Copy(0x3000, 0x17ff_0000, 0x10)]).is_empty());
        // A read of it before the copy is in order.
        assert!(cpu_among(&[Cpu::Reads(0x1000, 0x1800), Cpu::Any, card]).is_empty());
        // Behind it: a fill or a texture over its destination, a title copy
        // out of it or into it, a step of the title's own.
        assert_eq!(cpu_among(&[card, Cpu::Reads(0x17ff, 0x1800)]), made);
        assert_eq!(cpu_among(&[card, Cpu::Copy(0x5000, 0x1400, 0x10)]), made);
        assert_eq!(cpu_among(&[card, Cpu::Copy(0x0ff8, 0x5000, 0x10)]), made);
        assert_eq!(cpu_among(&[card, Cpu::Nothing, Cpu::Any]), made);
    }

    /// What a submission waits for around a card copy: a later refresh of
    /// its destination no longer waits (the flight's shaders are declared to
    /// write it, where before its copy was still to land there); the copy
    /// itself waits only for a flight's copy still to land in its source;
    /// made on the CPU as well behind a fill of its destination, it waits as
    /// a copy into memory flights write does.
    #[test]
    fn a_card_copy_waits_only_where_the_cpu_needs_a_flights_bytes() {
        use super::super::{HostList, Step, WHY_READ, blocked_by, host, join_ranges};
        let Some(host) = host() else { return };
        let memory = crate::resource::Memory::zeroed(1 << 16, 4096).expect("a test buffer");
        let start = memory.address();
        crate::shadow::register(start as usize, start, 1 << 16);
        let copy = crate::shadow::find(start, 1 << 16).expect("a host copy");
        let (dst, len) = (start as usize + 0x800, 0x800);
        let (shaders, src) = ((0x10_0000usize, 0x11_0000usize), 0x10_0800usize);
        let blocked = |written: &[(usize, usize)], copied: &[(usize, usize)], record: &dyn Fn(&HostList)| {
            let list = host.new_list().unwrap();
            record(&list);
            let why = blocked_by(written, copied, &[&list], false);
            list.close();
            why
        };
        let refresh = |list: &HostList| list.defer_refresh(&copy, dst, len);
        let card = |list: &HostList| list.deferred().copies.push(Step::Card(dst, src, len));
        // Before: the copy was left to its flight's landing, and a refresh
        // of its destination waited for that.
        assert_eq!(blocked(&[shaders], &[(dst, dst + len)], &refresh), Some(WHY_READ));
        // Now the flight's shaders are declared to write the destination.
        let mut written = vec![shaders];
        join_ranges(&mut written, [(dst, dst + len)]);
        assert_eq!(blocked(&written, &[], &refresh), None);
        // The copy out of what flights write, into what they write or not.
        assert_eq!(blocked(&[shaders], &[], &card), None);
        assert_eq!(blocked(&written, &[], &card), None);
        // A flight's copy is still to land in its source.
        assert_eq!(blocked(&[shaders], &[(src, src + 0x10)], &card), Some(WHY_READ));
        // A fill of its destination behind it in the same submission: made
        // on the CPU too, into memory a flight writes, it waits.
        let fill_behind = |list: &HostList| {
            card(list);
            list.defer_fill(0x70_0000 as *mut u8, dst as *const u8, 0x40);
        };
        assert_eq!(blocked(&written, &[], &fill_behind), Some(WHY_READ));
        drop(copy);
        crate::shadow::unregister(start as usize, start);
    }

    /// The check finds a seeded difference, where it lies, and nothing in
    /// equal bytes; read from memory as at a landing.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_check_finds_a_seeded_difference() {
        use super::check::{COUNTS, Differs, differs, one};
        use std::sync::atomic::Ordering::Relaxed;
        let src: Vec<u8> = (0..2048u32).map(|i| i as u8).collect();
        let mut dst = src.clone();
        assert_eq!(differs(&dst, &src), None);
        dst[0x123] ^= 0x40;
        dst[0x400] ^= 1;
        assert_eq!(differs(&dst, &src), Some(Differs { first: 0x123, last: 0x400, bytes: 2 }));
        let before = [0, 1, 2].map(|i| COUNTS[i].load(Relaxed));
        assert_eq!(one(1, dst.as_ptr() as usize, src.as_ptr() as usize, 2048, true), Some(Differs { first: 0x123, last: 0x400, bytes: 2 }));
        assert_eq!(one(1, src.as_ptr() as usize, src.as_ptr() as usize, 2048, false), None);
        let after = [0, 1, 2].map(|i| COUNTS[i].load(Relaxed));
        // (Other tests may count beside this one: at least ours.)
        assert!(after[0] >= before[0] + 2 && after[1] > before[1] && after[2] > before[2]);
    }
}
