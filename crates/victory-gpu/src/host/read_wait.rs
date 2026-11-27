//! Diagnostics: what the reads of a submission wait for flights for
//! (`Host::submit`, `relay_wait` with `WHY_READ`): which steps would wait
//! and which flights' copies or shaders' ranges they touch, and whether the
//! bytes of that overlap are other ones after the wait.
//!
//! The wait is about 12 ms a frame of the submitting thread. Most of it
//! comes through the steps that read where a flight has left a title buffer
//! copy to its landing (`waits` in `blocked_by`); if the bytes such a step
//! reads do not change under the wait, or the step reads a few bytes of a
//! range of megabytes, the wait is for nothing. Counts only: nothing about
//! waits, copies, landings or refreshes changes, with the switch or without.
//!
//! `VICTORY_READ_WAIT=1`: summary lines with the frame line and a line the
//! first time a pair of a step and a flight's range is seen; `=2`: also a
//! line for every wait. Every line starts with `read wait:`. The look at
//! the line and the snapshots of the overlapped bytes (1 MiB a pair, 8 MiB
//! a wait) take time of their own, which is not part of the wait counted.

use super::{Flight, HostList, LATE_STATS, Late, STARTED, Step, WHY_READ, fill_ahead, late_copied, late_copies, late_order, line, touches};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const KINDS: [&str; 7] = ["copy-src", "fill-src", "refresh", "copy-written", "fill-written", "texture-written", "none"];
/// Bytes snapshotted for one pair, and for one wait.
const PAIR_CAP: usize = 1 << 20;
const WAIT_CAP: usize = 8 << 20;
/// Pairs kept for one wait, distinct keys kept in a table, detail lines.
const PAIRS: usize = 64;
const KEYS: usize = 256;
const DETAILS: u64 = 24;

fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_READ_WAIT").as_deref() {
        Err(_) | Ok("") | Ok("0") => 0,
        Ok("2") => 2,
        Ok(_) => 1,
    })
}

/// A step that would wait: its kind (`KINDS`), where it is, the ranges
/// (address, bytes) of title memory it reads or writes that decide it, and
/// for a refresh the host copy's title address and length.
struct Hit {
    kind: usize,
    run: usize,
    index: usize,
    ranges: Vec<(usize, usize)>,
    host: Option<(u64, u64)>,
}

/// What a step's range was found to overlap.
#[derive(Clone, Copy)]
enum Source {
    /// A flight's title buffer copy left to its landing: destination,
    /// source, bytes.
    Copy(usize, usize, usize),
    /// A range its shaders may write (address, end).
    Shader(usize, usize),
}

/// The flights of the line that have something to land in title memory.
struct FlightInfo {
    number: u64,
    /// Its place among the flights of the line, from the front.
    place: usize,
    copies: Vec<(usize, usize, usize)>,
    shaders: Vec<(usize, usize)>,
}

struct Pair {
    hit: usize,
    range: (usize, usize),
    source: Source,
    flight: u64,
    place: usize,
    from: usize,
    to: usize,
    /// The bytes of `[from, from + snap.len())` when the wait began.
    snap: Vec<u8>,
}

struct Pending {
    frame: u64,
    started: u64,
    stood: u64,
    at: Instant,
    hits: Vec<Hit>,
    /// A `Step::Run` stops `blocked_by`: where, if one is reached.
    run: Option<(usize, usize)>,
    flights: Vec<FlightInfo>,
    runs: usize,
    pairs: Vec<Pair>,
    dropped: usize,
    capped: bool,
}

static PENDING: Mutex<Option<Pending>> = Mutex::new(None);

#[derive(Default)]
struct CopyRow {
    key: (usize, usize, usize),
    n: u64,
    us: u64,
    ov: u64,
    ch: u64,
    zero: u64,
}

#[derive(Default)]
struct StepRow {
    key: (usize, usize, usize, u64, u64),
    n: u64,
    us: u64,
    bytes: u64,
    ov: u64,
    ch: u64,
}

#[derive(Default)]
struct Stats {
    waits: u64,
    stood: u64,
    us: u64,
    by_first: [(u64, u64); 7],
    nothing: u64,
    some: u64,
    unknown: u64,
    ov: u64,
    ch: u64,
    place: [u64; 5],
    length: [u64; 5],
    copies: Vec<CopyRow>,
    steps: Vec<StepRow>,
    seen: Vec<[usize; 7]>,
    details: u64,
    /// At the last frame line: waits, microseconds.
    last: (u64, u64),
}

static STATS: Mutex<Stats> = Mutex::new(Stats {
    waits: 0,
    stood: 0,
    us: 0,
    by_first: [(0, 0); 7],
    nothing: 0,
    some: 0,
    unknown: 0,
    ov: 0,
    ch: 0,
    place: [0; 5],
    length: [0; 5],
    copies: Vec::new(),
    steps: Vec::new(),
    seen: Vec::new(),
    details: 0,
    last: (0, 0),
});

/// `[at, at + len)` against `[from, to)`: the overlap.
fn overlap(range: (usize, usize), from: usize, to: usize) -> Option<(usize, usize)> {
    let (a, b) = (range.0.max(from), range.0.saturating_add(range.1).min(to));
    (a < b).then_some((a, b))
}

/// Every step of one run's `steps` that `blocked_by` would stop at, if it
/// were asked again after each of them had been let go, added to `hits`;
/// the place of a step that runs ours, if one is reached (`blocked_by`
/// stops there). `late` are this submission's own copies left to its
/// landing, those not met yet. One run at a time, as `blocked_by` walks
/// them: a list may stand twice in a submission, and its steps are behind
/// its own lock.
fn walk(
    written: &[(usize, usize)],
    copied: &[(usize, usize)],
    late: &mut Vec<(usize, usize, usize)>,
    run: usize,
    steps: &[Step],
    hits: &mut Vec<Hit>,
) -> Option<(usize, usize)> {
    let waits = |at: usize, len: usize| late_order() && touches(copied, at, at.saturating_add(len));
    for (index, step) in steps.iter().enumerate() {
        let mut hit = |kind: usize, ranges: Vec<(usize, usize)>, host: Option<(u64, u64)>| {
            hits.push(Hit { kind, run, index, ranges, host });
        };
        match step {
            &Step::Copy(dst, src, len) => {
                if let Some(at) = late.iter().position(|c| *c == (dst, src, len)) {
                    late.swap_remove(at);
                    continue;
                }
                let (by_src, by_dst) = (touches(written, src, src + len), touches(written, dst, dst + len));
                let reads = waits(src, len);
                let writes = (by_src || by_dst) && (by_dst || !fill_ahead::ahead(src, len));
                if reads || writes {
                    let mut ranges = Vec::new();
                    if reads || by_src {
                        ranges.push((src, len));
                    }
                    if by_dst {
                        ranges.push((dst, len));
                    }
                    hit(if reads { 0 } else { 3 }, ranges, None);
                }
            }
            &Step::Fill(_, src, len) => {
                let reads = waits(src, len);
                if reads || (touches(written, src, src + len) && !fill_ahead::ahead(src, len)) {
                    hit(if reads { 1 } else { 4 }, vec![(src, len)], None);
                }
            }
            Step::Texture(staging) => {
                let (at, len) = staging.memory();
                if touches(written, at, at + len) {
                    hit(5, vec![(at, len)], None);
                }
            }
            Step::Run(_) => return Some((run, index)),
            Step::Refresh(shadow, address, len) => {
                if waits(*address, *len) {
                    hit(2, vec![(*address, *len)], Some(shadow.span()));
                }
            }
            // (Made on the card: only a flight's copy still to land in its
            // source waits, as in `blocked_by`; one made on the CPU as well
            // is not told apart here.)
            &Step::Card(_, src, len) => {
                if waits(src, len) {
                    hit(0, vec![(src, len)], None);
                }
            }
            Step::GpuWrite(..) | Step::Host(_) => {}
        }
    }
    None
}

/// The pairs of the steps that would wait and the ranges flights leave
/// (`flights` in the order of the line): where they overlap. Steps that
/// wait only for the copies of a flight are paired with those, the others
/// with the ranges its shaders may write as well.
fn pairs(hits: &[Hit], flights: &[FlightInfo]) -> (Vec<Pair>, usize) {
    let mut out = Vec::new();
    let mut dropped = 0;
    for (h, hit) in hits.iter().enumerate() {
        for &range in &hit.ranges {
            for flight in flights {
                let mut add = |source: Source, from: usize, to: usize| {
                    if out.len() < PAIRS {
                        out.push(Pair { hit: h, range, source, flight: flight.number, place: flight.place + 1, from, to, snap: Vec::new() });
                    } else {
                        dropped += 1;
                    }
                };
                for &(dst, src, len) in &flight.copies {
                    if let Some((a, b)) = overlap(range, dst, dst.saturating_add(len)) {
                        add(Source::Copy(dst, src, len), a, b);
                    }
                }
                if hit.kind >= 3 {
                    for &(from, to) in &flight.shaders {
                        if let Some((a, b)) = overlap(range, from, to) {
                            add(Source::Shader(from, to), a, b);
                        }
                    }
                }
            }
        }
    }
    (out, dropped)
}

/// Before the submission waits for flights because its reads need them
/// (`Host::submit`, `relay_exact` and `WHY_READ`).
pub(super) fn begin(runs: &[&HostList]) {
    if level() == 0 {
        return;
    }
    // What is needed of the line, copied out: the lock is let go before
    // anything is read.
    let (written, copied, flights, nruns) = {
        let line = line();
        let mut place = 0;
        let mut flights = Vec::new();
        for item in &line.line {
            if let Late::Run(f) = item {
                let Flight { number, copies, back, read_backs, .. } = f;
                let shaders: Vec<(usize, usize)> = back.iter().flat_map(|(_, r)| r.iter().copied()).chain(read_backs.iter().copied()).collect();
                if !copies.is_empty() || !shaders.is_empty() {
                    flights.push(FlightInfo { number: *number, place, copies: copies.clone(), shaders });
                }
                place += 1;
            }
        }
        (line.written.clone(), late_copied(&line), flights, place)
    };
    let mut late = late_copies(&written, runs);
    let (mut hits, mut run) = (Vec::new(), None);
    for (k, list) in runs.iter().enumerate() {
        run = walk(&written, &copied, &mut late, k, &list.deferred().copies, &mut hits);
        if run.is_some() {
            break;
        }
    }
    let (mut pairs, dropped) = pairs(&hits, &flights);
    let (mut budget, mut capped) = (WAIT_CAP, dropped > 0);
    for pair in &mut pairs {
        let n = (pair.to - pair.from).min(PAIR_CAP).min(budget);
        capped |= n < pair.to - pair.from;
        let mut bytes = vec![0u8; n];
        if n > 0 && super::read_into(pair.from as *const u8, &mut bytes) {
            budget -= n;
            pair.snap = bytes;
        }
    }
    *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Pending {
        frame: crate::frame::current(),
        started: STARTED.load(Relaxed),
        stood: LATE_STATS[WHY_READ].load(Relaxed),
        at: Instant::now(),
        hits,
        run,
        flights,
        runs: nruns,
        pairs,
        dropped,
        capped,
    });
}

fn view(at: usize) -> String {
    match crate::inplace::view_at(at) {
        Some((_, _, section, offset)) => format!("{section:#x}+{offset:#x}"),
        None => "none".to_string(),
    }
}

fn views(at: usize) -> usize {
    crate::inplace::pages_of(at & !0xffff).len()
}

/// After the wait (`relay_wait` has returned).
pub(super) fn end() {
    let Some(p) = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let us = p.at.elapsed().as_micros() as u64;
    // `relay_wait` counts a wait only when it stands.
    let stood = LATE_STATS[WHY_READ].load(Relaxed) != p.stood;
    let landed = {
        let line = line();
        p.flights
            .iter()
            .filter(|f| !line.line.iter().any(|i| matches!(i, Late::Run(r) if r.number == f.number)))
            .count()
    };
    // The overlaps again: per pair the bytes that differ.
    let changed: Vec<u64> = p
        .pairs
        .iter()
        .map(|pair| {
            let mut now = vec![0u8; pair.snap.len()];
            if now.is_empty() || !super::read_into(pair.from as *const u8, &mut now) {
                return 0;
            }
            pair.snap.iter().zip(&now).filter(|(a, b)| a != b).count() as u64
        })
        .collect();
    let compared: u64 = p.pairs.iter().map(|x| x.snap.len() as u64).sum();
    let full: u64 = p.pairs.iter().map(|x| (x.to - x.from) as u64).sum();
    let ch: u64 = changed.iter().sum();
    let first = p.hits.first().map_or(6, |h| h.kind);
    let place = p.pairs.iter().map(|x| x.place).max().unwrap_or(0);
    let range_of = |pair: &Pair| match pair.source {
        Source::Copy(dst, _, len) => format!("{dst:#x}+{len:#x}"),
        Source::Shader(from, to) => format!("shaders:{from:#x}+{:#x}", to - from),
    };
    if level() >= 2 {
        let mut text = format!(
            "read wait: w frame={} us={us} stood={} first={} steps={} copies={} overlap={full} changed={ch} landed={landed}/{} place={place} run={}",
            p.frame,
            u8::from(stood),
            KINDS[first],
            p.hits.len(),
            p.pairs.iter().filter(|x| matches!(x.source, Source::Copy(..))).count(),
            p.flights.len(),
            u8::from(p.run.is_some())
        );
        for (pair, ch) in p.pairs.iter().zip(&changed) {
            let hit = &p.hits[pair.hit];
            text.push_str(&format!(
                "; {} {:#x}+{:#x} ~ {} ov={} ch={ch}",
                KINDS[hit.kind],
                pair.range.0,
                pair.range.1,
                range_of(pair),
                pair.to - pair.from
            ));
        }
        tracing::info!("{text}");
    }
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = &mut *stats;
    s.waits += 1;
    if !stood {
        return;
    }
    s.stood += 1;
    s.us += us;
    s.by_first[first].0 += 1;
    s.by_first[first].1 += us;
    s.ov += compared;
    s.ch += ch;
    match (compared, ch) {
        (0, _) => s.unknown += 1,
        (_, 0) => s.nothing += 1,
        _ => s.some += 1,
    }
    s.place[place.min(4)] += 1;
    s.length[match us {
        0..1000 => 0,
        1000..4000 => 1,
        4000..10000 => 2,
        10000..20000 => 3,
        _ => 4,
    }] += 1;
    // Per distinct copy and step: once a wait.
    let mut done_copies: Vec<(usize, usize, usize)> = Vec::new();
    let mut done_steps: Vec<(usize, usize, usize, u64, u64)> = Vec::new();
    let mut zero: std::collections::HashMap<(usize, usize, usize), bool> = std::collections::HashMap::new();
    for (pair, &c) in p.pairs.iter().zip(&changed) {
        let hit = &p.hits[pair.hit];
        let host = hit.host.unwrap_or((0, 0));
        let skey = (hit.kind, pair.range.0 & !0xfff, pair.range.1, host.0, host.1);
        let row = match s.steps.iter().position(|r| r.key == skey) {
            Some(k) => Some(k),
            None if s.steps.len() < KEYS => {
                s.steps.push(StepRow { key: skey, ..Default::default() });
                Some(s.steps.len() - 1)
            }
            None => None,
        };
        if let Some(k) = row {
            let r = &mut s.steps[k];
            if !done_steps.contains(&skey) {
                done_steps.push(skey);
                r.n += 1;
                r.us += us;
                r.bytes += pair.range.1 as u64;
            }
            r.ov += pair.snap.len() as u64;
            r.ch += c;
        }
        if let Source::Copy(dst, src, len) = pair.source {
            let ckey = (dst & !0xfff, src & !0xfff, len);
            let row = match s.copies.iter().position(|r| r.key == ckey) {
                Some(k) => Some(k),
                None if s.copies.len() < KEYS => {
                    s.copies.push(CopyRow { key: ckey, ..Default::default() });
                    Some(s.copies.len() - 1)
                }
                None => None,
            };
            if let Some(k) = row {
                let r = &mut s.copies[k];
                if !done_copies.contains(&ckey) {
                    done_copies.push(ckey);
                    r.n += 1;
                    r.us += us;
                }
                r.ov += pair.snap.len() as u64;
                r.ch += c;
                *zero.entry(ckey).or_insert(true) &= c == 0;
            }
        }
        // The first time a distinct pair is seen.
        let (a, b, l) = match pair.source {
            Source::Copy(dst, src, len) => (dst & !0xfff, src & !0xfff, len),
            Source::Shader(from, to) => (from & !0xfff, 0, to - from),
        };
        let seen = [hit.kind, pair.range.0 & !0xfff, pair.range.1, usize::from(matches!(pair.source, Source::Copy(..))), a, b, l];
        if !s.seen.contains(&seen) && s.seen.len() < KEYS {
            s.seen.push(seen);
            if s.details < DETAILS {
                s.details += 1;
                let step_id = match hit.host {
                    Some((at, len)) => format!(" host={at:#x}+{len:#x}"),
                    None => String::new(),
                };
                let (src_view, src_at) = match pair.source {
                    Source::Copy(_, src, _) => (view(src), src),
                    Source::Shader(..) => ("-".to_string(), 0),
                };
                let (dst_at, flight_range) = match pair.source {
                    Source::Copy(dst, src, len) => (dst, format!("copy dst={dst:#x} src={src:#x} len={len:#x}")),
                    Source::Shader(from, to) => (from, format!("shaders {from:#x}+{:#x}", to - from)),
                };
                tracing::info!(
                    "read wait: pair {} {:#x}+{:#x} run={} step={}{step_id} ~ {flight_range} flight={} age={} place={}/{} overlap={:#x}..{:#x} \
                     compared={} changed={c} capped={} views(step dst src)={} {} {} pages(step dst src)={} {} {} frame={} stood={}",
                    KINDS[hit.kind],
                    pair.range.0,
                    pair.range.1,
                    hit.run,
                    hit.index,
                    pair.flight,
                    p.started.saturating_sub(pair.flight),
                    pair.place,
                    p.runs,
                    pair.from,
                    pair.to,
                    pair.snap.len(),
                    u8::from(p.capped),
                    view(pair.range.0),
                    view(dst_at),
                    src_view,
                    views(pair.range.0),
                    views(dst_at),
                    if src_at == 0 { 0 } else { views(src_at) },
                    p.frame,
                    u8::from(stood),
                );
            }
        }
    }
    for (key, all_zero) in zero {
        if all_zero && let Some(r) = s.copies.iter_mut().find(|r| r.key == key) {
            r.zero += 1;
        }
    }
    if p.dropped > 0 && level() >= 2 {
        tracing::info!("read wait: w frame={} dropped {} pairs beyond {PAIRS}", p.frame, p.dropped);
    }
}

/// With the frame line.
pub(super) fn line_for(frames: f64) {
    if level() == 0 {
        return;
    }
    let mut stats = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = &mut *stats;
    if s.waits == 0 {
        return;
    }
    let frames = frames.max(1.0);
    let (dw, dus) = (s.stood - s.last.0.min(s.stood), s.us - s.last.1.min(s.us));
    s.last = (s.stood, s.us);
    let firsts: Vec<String> = s
        .by_first
        .iter()
        .enumerate()
        .filter(|(_, f)| f.0 > 0)
        .map(|(k, f)| format!("{} {}x {:.1} ms", KINDS[k], f.0, f.1 as f64 / 1000.0))
        .collect();
    tracing::info!(
        "read wait: since the start: {} waits, {} stood, {:.1} ms; since the last line: {:.1} stood a frame, {:.2} ms a frame; \
         first blocking step: {}; none changed: {}, some changed: {}, no bytes compared: {}; overlap {} bytes compared, {} changed; \
         flights landed before the last one needed (0 1 2 3 4+): {:?}; length (<1 1-4 4-10 10-20 20+ ms): {:?}",
        s.waits,
        s.stood,
        s.us as f64 / 1000.0,
        dw as f64 / frames,
        dus as f64 / 1000.0 / frames,
        firsts.join(", "),
        s.nothing,
        s.some,
        s.unknown,
        s.ov,
        s.ch,
        s.place,
        s.length
    );
    let mut copies: Vec<&CopyRow> = s.copies.iter().collect();
    copies.sort_unstable_by_key(|r| std::cmp::Reverse(r.n));
    let copies: Vec<String> = copies
        .iter()
        .take(6)
        .map(|r| {
            format!(
                "{:#x}->{:#x}+{:#x} {}x {:.1} ms ov={} ch={} unchanged={}x",
                r.key.1,
                r.key.0,
                r.key.2,
                r.n,
                r.us as f64 / 1000.0,
                r.ov,
                r.ch,
                r.zero
            )
        })
        .collect();
    tracing::info!("read wait: {} distinct late copies, the most frequent: {}", s.copies.len(), copies.join(", "));
    let mut steps: Vec<&StepRow> = s.steps.iter().collect();
    steps.sort_unstable_by_key(|r| std::cmp::Reverse(r.n));
    let steps: Vec<String> = steps
        .iter()
        .take(6)
        .map(|r| {
            let host = if r.key.3 == 0 { String::new() } else { format!(" host {:#x}+{:#x}", r.key.3, r.key.4) };
            format!(
                "{} {:#x}+{:#x}{host} {}x {:.1} ms bytes={} ov={} ch={}",
                KINDS[r.key.0],
                r.key.1,
                r.key.2,
                r.n,
                r.us as f64 / 1000.0,
                r.bytes,
                r.ov,
                r.ch
            )
        })
        .collect();
    tracing::info!("read wait: {} distinct steps, the most frequent: {}", s.steps.len(), steps.join(", "));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlaps_are_cut_to_both_ranges() {
        assert_eq!(overlap((0x100, 0x40), 0x120, 0x200), Some((0x120, 0x140)));
        assert_eq!(overlap((0x100, 0x40), 0x140, 0x200), None);
        assert_eq!(overlap((0x100, 0x400), 0x180, 0x1c0), Some((0x180, 0x1c0)));
    }

    #[test]
    fn the_walk_names_every_step_that_waits_and_stops_at_a_run() {
        let at = |address: usize| address;
        let written = [(0x10_0000usize, 0x11_0000usize), (0x30_0000, 0x30_0100)];
        let copied = [(0x30_0000usize, 0x30_0100usize)];
        let steps = [
            // Reads where a flight's copy is to land.
            Step::Copy(at(0x50_0000), at(0x30_0080), 0x40),
            // Nothing to wait for.
            Step::Copy(at(0x50_0000), at(0x60_0000), 0x40),
            Step::Fill(at(0x70_0000), at(0x30_00ff), 0x40),
            // Into what shaders may write.
            Step::Copy(at(0x10_0100), at(0x50_0000), 0x40),
            Step::Run(Box::new(|| {})),
            Step::Fill(at(0x70_0000), at(0x30_0000), 0x40),
        ];
        let mut hits = Vec::new();
        let run = walk(&written, &copied, &mut Vec::new(), 0, &steps, &mut hits);
        let kinds: Vec<(usize, usize)> = hits.iter().map(|h| (h.kind, h.index)).collect();
        assert_eq!(kinds, [(0, 0), (1, 2), (3, 3)]);
        assert_eq!(run, Some((0, 4)));
        assert_eq!(hits[2].ranges, [(0x10_0100, 0x40)]);
        // Its own copy left to its landing is skipped.
        let mut own = Vec::new();
        walk(&written, &copied, &mut vec![(0x50_0000, 0x30_0080, 0x40)], 0, &steps[..1], &mut own);
        assert!(own.is_empty());
        // Pairs: a copy's destination, and for a step of the written kind the shaders' range.
        let flights = [FlightInfo { number: 7, place: 1, copies: vec![(0x30_0000, 0x50_0000, 0x100)], shaders: vec![(0x10_0000, 0x11_0000)] }];
        let (found, dropped) = pairs(&hits, &flights);
        assert_eq!(dropped, 0);
        let ov: Vec<(usize, usize, usize)> = found.iter().map(|p| (p.hit, p.from, p.to)).collect();
        assert_eq!(ov, [(0, 0x30_0080, 0x30_00c0), (1, 0x30_00ff, 0x30_0100), (2, 0x10_0100, 0x10_0140)]);
        assert!(found.iter().all(|p| p.place == 2));
    }
}
