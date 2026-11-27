//! The merge of a landing (`Host::relay_card_back`): what shaders wrote to
//! copies in the card's memory goes into title memory as jobs of about
//! 256 KiB of one copy each ([`JOB`]), on the threads kept for wide work and the
//! landing's own (`crate::shadow::wide_run`).
//!
//! The landing thread spent 28 to 43 ms a frame in landings, 23 to 34 of it
//! in this merge (some 50 MB a frame, the timeline of 2026-11-26), one run
//! after the other. Spread over the helpers as eighths of the list of runs
//! (`VICTORY_WIDE_MERGE=1`, gone with this) it gave nothing: every merge
//! into the heap's copy took that copy's one side lock, and the helpers
//! waited 15.4 ms a frame for it. The lock is per 64 KiB piece now
//! (`Shadow::side_of`), and a job holds one copy's runs of one stretch of
//! title memory.
//!
//! Why this is exact (rules R2 and R4 of `docs/architecture.md`):
//! - The runs of one landing come from one fetch of the card's bytes as
//!   they are after every flight landed with it; within a copy they are
//!   apart (joined per copy), and a copy's runs are cut where a line of
//!   256 KiB of title memory lies: each byte of a copy is merged once, by one job.
//! - Runs of different copies are of different CPU sides. Where two copies
//!   hold the same title memory, their runs there are in one job, in the
//!   order they were given ([`cut`]): the later one is merged over the
//!   earlier one, as before. "The same title memory" is the same physical
//!   pages, at whatever address: the title maps a buffer's pages at
//!   several addresses (six views of one 10 MiB buffer in the first room),
//!   binds it by more than one of them, and each address has a copy of its
//!   own with a card side of its own. Two flights that write the buffer by
//!   two of its addresses and land together bring down two versions of the
//!   same bytes. Told apart by address they were merged in the order of
//!   their addresses, in one job or in two beside each other: the older
//!   flight's bytes ended over the newer one's whenever its address was
//!   the higher (the line counts how often it is; and the copy at the
//!   other address takes what title memory holds for the title's at its
//!   next refresh). So pieces are placed by the pages
//!   they lie in ([`Key`]), and the order given is that of the flights
//!   (`fine_runs`: a copy stands behind those of every earlier flight).
//! - What else writes a copy's CPU side in a range shaders write
//!   (`Shadow::merge_under_writes` on the submitting thread) takes the
//!   lock of the piece it writes; a merge holds the locks of the pieces of
//!   its run, in order.
//! - One copy's flights land in order: a landing's jobs are all done
//!   before the landing goes on (the written bits, the telling) and before
//!   the next landing begins, on this one thread.
//!
//! `VICTORY_LAND_SAMPLE=1` (a check; off by default): at the telling, a
//! sample of the stretches the jobs wrote (the last one of every eighth
//! run, up to 256 bytes of it) is read from title memory again and
//! compared with the copy's CPU side under the pieces' lock, at the bytes
//! the stretch landed (written as the card held them; its other bytes are
//! the title's as read, which the CPU side takes only with a refresh). A
//! stretch where the two differ is counted and named, with what tells why
//! (`Shadow::landed_apart`): the title itself wrote there between the
//! merge and the look, or the merge is wrong. A piece that a later run of
//! its job lands over (another copy of the same pages, from a later
//! flight) is not sampled: there title memory is to hold the later one's
//! bytes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use crate::shadow::Shadow;

/// A run to merge: the copy, its offsets `[lo, hi)`, and where its bytes
/// are in the fetched memory.
pub(super) type Run = (Arc<Shadow>, usize, usize, u64);

/// About this many bytes of title memory to a job, and where runs are cut:
/// most landings bring a few megabytes, and jobs of a megabyte left all
/// but one or two threads idle in them (avatarmanz/VIctory-7900XTX cuts
/// at 256 KiB too).
const JOB: usize = 256 << 10;

/// Whether the sample at the telling is taken ([`the module`](self)).
fn sample_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_SAMPLE").is_some_and(|v| v == "1"))
}

/// Since the last line: jobs, the most of them a landing had, and the
/// sample's stretches looked at and those that differed.
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Since the start: pieces a later run of their job landed over (two
/// copies of the same pages, each with bytes from the card); and those of
/// them whose address is the higher of the two, so that merged in the order
/// of addresses, as they were, the earlier flight's bytes would have ended
/// over the later one's.
static OVER: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Where a piece of title memory lies: for a view of the title's physical
/// pages the section and the offset in it (the same for every address the
/// pages are mapped at), for private memory 0 and the address.
type Key = (usize, u64);

/// A piece of a job: the run, its offsets `[lo, hi)`, and whether a later
/// run of the job lands over the same pages (it is then not sampled).
type Piece = (usize, usize, usize, bool);

/// The runs as jobs: each a list of pieces. A run is cut where [`JOB`]
/// bytes of title memory end and where a view of the title's ends (the
/// pages of the next view lie elsewhere); the pieces are taken in the order
/// of where they lie ([`Key`]) and a job ends, once it holds [`JOB`] bytes
/// or more, only where the next piece begins at or after the end of every
/// one before it, so that pieces of pages two copies share are in one job,
/// at whatever addresses the copies have them. Within a job the pieces are
/// in the order given.
fn cut(runs: &[Run]) -> Vec<Vec<Piece>> {
    // (where, run, lo, hi, title address)
    let mut parts: Vec<Part> = Vec::new();
    for (i, (shadow, lo, hi, _)) in runs.iter().enumerate() {
        let start = shadow.span().0 as usize;
        let mut at = *lo;
        while at < *hi {
            let address = start + at;
            let mut end = ((address / JOB + 1) * JOB - start).min(*hi);
            let key = match crate::inplace::view_at(address) {
                Some((base, bytes, section, offset)) => {
                    end = end.min(base + bytes - start);
                    (section, offset + (address - base) as u64)
                }
                None => (0, address as u64),
            };
            parts.push((key, i, at, end, address));
            at = end;
        }
    }
    cut_parts(parts)
}

/// A piece for [`cut_parts`]: where it lies, its run, its offsets `[lo,
/// hi)` in the run's copy, and its title address.
type Part = (Key, usize, usize, usize, usize);

/// [`cut`] of its pieces.
fn cut_parts(mut parts: Vec<Part>) -> Vec<Vec<Piece>> {
    parts.sort_unstable_by_key(|p| (p.0, p.1));
    let mut jobs: Vec<Vec<Part>> = Vec::new();
    let mut job: Vec<Part> = Vec::new();
    // The bytes the job holds, and the end of every piece before the next
    // one (in the section they lie in).
    let (mut bytes, mut reach): (usize, Key) = (0, (0, 0));
    for part in parts {
        let (key, _, lo, hi, _) = part;
        if bytes >= JOB && (key.0 != reach.0 || key.1 >= reach.1) && !job.is_empty() {
            jobs.push(std::mem::take(&mut job));
            bytes = 0;
        }
        job.push(part);
        bytes += hi - lo;
        let end = key.1 + (hi - lo) as u64;
        reach = if key.0 == reach.0 { (key.0, reach.1.max(end)) } else { (key.0, end) };
    }
    if !job.is_empty() {
        jobs.push(job);
    }
    jobs.into_iter()
        .map(|job| {
            // (In the order of where they lie: a piece and the ones that
            // begin before its end.)
            let mut pieces: Vec<Piece> = job.iter().map(|&(_, i, lo, hi, _)| (i, lo, hi, false)).collect();
            for (a, &(key, i, lo, hi, address)) in job.iter().enumerate() {
                let end = key.1 + (hi - lo) as u64;
                for (b, other) in job.iter().enumerate().skip(a + 1) {
                    if other.0.0 != key.0 || other.0.1 >= end {
                        break;
                    }
                    // (The earlier run's piece, and whether its address is
                    // the higher one.)
                    let (earlier, higher) = match other.1.cmp(&i) {
                        std::cmp::Ordering::Greater => (a, address > other.4),
                        std::cmp::Ordering::Less => (b, other.4 > address),
                        std::cmp::Ordering::Equal => continue,
                    };
                    if !pieces[earlier].3 {
                        pieces[earlier].3 = true;
                        OVER[0].fetch_add(1, Relaxed);
                        OVER[1].fetch_add(u64::from(higher), Relaxed);
                    }
                }
            }
            pieces.sort_unstable();
            pieces
        })
        .collect()
}

/// Merges `runs` (their bytes fetched to `fetched` on) into title memory
/// and the copies' CPU sides, as jobs; `after` is the merge's "fetched
/// after" (`Shadow::card_merge_relay`). Returns the bytes written to title
/// memory, when every job is done.
pub(super) fn merge(runs: &[Run], fetched: usize, after: u64) -> usize {
    let jobs = cut(runs);
    COUNTS[0].fetch_add(jobs.len() as u64, Relaxed);
    COUNTS[1].fetch_max(jobs.len() as u64, Relaxed);
    let runs: Arc<Vec<Run>> = Arc::new(runs.to_vec());
    let written = Arc::new(AtomicUsize::new(0));
    type Sample = (Arc<Shadow>, crate::shadow::Merged);
    let samples: Arc<Mutex<Vec<Sample>>> = Arc::new(Mutex::new(Vec::new()));
    let sample = sample_on();
    let work: Vec<Box<dyn FnOnce() + Send>> = jobs
        .into_iter()
        .map(|job| {
            let (runs, written, samples) = (runs.clone(), written.clone(), samples.clone());
            Box::new(move || {
                // (On the timeline: each job, with its runs.)
                let _span = crate::timeline::span("job>", "job<", job.len() as u64, 0);
                let mut title = Vec::new();
                let (mut sum, mut runs_done) = (0, 0usize);
                for (i, lo, hi, over) in job {
                    let (shadow, run_lo, _, at) = &runs[i];
                    // SAFETY: the mapping of the fetched bytes; the copy
                    // into it has run, and the landing waits for this job.
                    let gpu = unsafe { std::slice::from_raw_parts((fetched + *at as usize + (lo - run_lo)) as *const u8, hi - lo) };
                    runs_done += 1;
                    let sampled = sample && !over && runs_done % 8 == 1;
                    if sampled {
                        crate::shadow::sample_next();
                    }
                    let merged = shadow.card_merge_relay(lo, gpu, after, &mut title);
                    if sampled && let Some(merged) = crate::shadow::last_merged() {
                        samples.lock().unwrap_or_else(|e| e.into_inner()).push((shadow.clone(), merged));
                    }
                    shadow.walls_merged(gpu.len(), merged);
                    crate::card_fine::merged(shadow, gpu.len(), merged);
                    sum += merged;
                }
                written.fetch_add(sum, Relaxed);
            }) as Box<dyn FnOnce() + Send>
        })
        .collect();
    crate::shadow::wide_run(work);
    if sample {
        let samples = std::mem::take(&mut *samples.lock().unwrap_or_else(|e| e.into_inner()));
        for (shadow, merged) in samples {
            COUNTS[2].fetch_add(1, Relaxed);
            let (apart, why) = shadow.landed_apart(&merged);
            if apart > 0 {
                COUNTS[3].fetch_add(1, Relaxed);
                static SAID: AtomicU64 = AtomicU64::new(0);
                if SAID.fetch_add(1, Relaxed) < 96 {
                    let (start, len) = shadow.span();
                    tracing::warn!(
                        copy = format_args!("{start:#x}+{len:#x}"),
                        at = format_args!("{:#x}", start as usize + merged.at),
                        bytes = merged.written.len(),
                        apart,
                        "a stretch a landing wrote holds other bytes in title memory than the copy's CPU side at the telling (VICTORY_LAND_SAMPLE=1): {why}"
                    );
                }
            }
        }
    }
    written.load(Relaxed)
}

/// The frame line's part.
pub(super) fn line(frames: f64) {
    let [jobs, most, looked, differed] = [0, 1, 2, 3].map(|i| COUNTS[i].swap(0, Relaxed));
    if jobs == 0 && !sample_on() {
        return;
    }
    let [whole, whole_bytes, whole_stretches, scattered] = [0, 1, 2, 3].map(|i| crate::shadow::WHOLE_RUNS[i].load(Relaxed));
    let [over, higher] = [0, 1].map(|i| OVER[i].load(Relaxed));
    tracing::info!(
        "the landings' merges as jobs: {:.1} jobs a frame, at most {most} in one landing; since the start {whole} runs that \
         changed almost whole merged without reading title memory first ({:.1} MB, {whole_stretches} stretches of changed \
         bytes written), {scattered} such runs changed in more stretches and read first; {over} pieces that a later flight's \
         copy of the same pages (at another address) landed over, {higher} of them at the higher address (merged by address, \
         the earlier flight's bytes would have ended over the later one's){}",
        jobs as f64 / frames,
        whole_bytes as f64 / 1e6,
        if sample_on() {
            format!(
                "; the sample at the telling (VICTORY_LAND_SAMPLE=1) since the last line: {looked} stretches looked at, {differed} \
                 holding other bytes in title memory than the CPU side (0 is what the merge needs, unless the title wrote there)"
            )
        } else {
            String::new()
        }
    );
}

#[cfg(test)]
mod tests {
    /// The cuts of [`super::cut`] over plain numbers: what it does with the
    /// runs' title addresses, without copies (private memory: a piece lies
    /// where its address is). Each piece as (run, lo, hi).
    fn cut_of(runs: &[(usize, usize, usize)]) -> Vec<Vec<(usize, usize, usize)>> {
        // (start, lo, hi) as a run of a copy from `start` on.
        let mut parts: Vec<super::Part> = Vec::new();
        for (i, &(start, lo, hi)) in runs.iter().enumerate() {
            let mut at = lo;
            while at < hi {
                let next = ((start + at) / super::JOB + 1) * super::JOB - start;
                let end = next.min(hi);
                parts.push(((0, (start + at) as u64), i, at, end, start + at));
                at = end;
            }
        }
        super::cut_parts(parts).into_iter().map(|job| job.into_iter().map(|(i, lo, hi, _)| (i, lo, hi)).collect()).collect()
    }

    /// Two copies at two addresses over the same pages of a section: their
    /// pieces lie in the same place, so they are in one job, the later run
    /// behind the earlier one, which is marked as landed over.
    #[test]
    fn copies_of_the_same_pages_at_two_addresses_are_in_one_job_in_the_order_given() {
        const K: usize = 64 << 10;
        // (where, run, lo, hi): section 7, four 64 KiB pages of it; run 1
        // (the later flight's copy) has them from its offset 0, run 0 from
        // 8 * K on; and private memory beside, run 2.
        let mut parts = Vec::new();
        for page in 0..4usize {
            parts.push(((7, (page * K) as u64), 1, page * K, (page + 1) * K, 0x1000_0000 + page * K));
            parts.push(((7, (page * K) as u64), 0, (8 + page) * K, (9 + page) * K, 0x2000_0000 + page * K));
        }
        parts.push(((0, 0x5000_0000), 2, 0, K, 0x5000_0000));
        let jobs = super::cut_parts(parts);
        // (Nine pieces of 64 KiB: two jobs, cut where no page is shared.)
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs.iter().map(Vec::len).sum::<usize>(), 9);
        for page in 0..4usize {
            let (earlier, later) = ((0, (8 + page) * K, (9 + page) * K, true), (1, page * K, (page + 1) * K, false));
            let job = jobs.iter().find(|job| job.contains(&earlier)).expect("the earlier run's piece, marked as landed over");
            let at = |piece| job.iter().position(|p| *p == piece);
            assert!(at(later).is_some_and(|l| at(earlier).is_some_and(|e| e < l)), "the later run's piece of the same page, behind it");
        }
        assert!(jobs.iter().flatten().any(|piece| *piece == (2, 0, K, false)));
        // Pieces of other sections, or apart in one, are landed over by nobody.
        let apart = super::cut_parts(vec![((7, 0), 0, 0, K, K), ((7, K as u64), 1, 0, K, 9 * K), ((8, 0), 2, 0, K, 5 * K)]);
        assert!(apart.iter().flatten().all(|piece| !piece.3));
    }

    #[test]
    fn runs_are_cut_at_job_lines_and_shared_memory_stays_in_one_job() {
        // (`MB`: a job's bytes.)
        const MB: usize = super::JOB;
        // One copy from 0: 3.5 jobs' bytes in one run, cut at each job line.
        let jobs = cut_of(&[(0, 0, 3 * MB + MB / 2)]);
        assert_eq!(jobs, vec![vec![(0, 0, MB)], vec![(0, MB, 2 * MB)], vec![(0, 2 * MB, 3 * MB)], vec![(0, 3 * MB, 3 * MB + MB / 2)]]);
        // Two copies over the same title memory: their pieces there are in
        // one job, in the order given.
        let jobs = cut_of(&[(16 * MB, 0, 2 * MB), (16 * MB + MB / 2, 0, MB)]);
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0], vec![(0, 0, MB), (1, 0, MB / 2)]);
        assert_eq!(jobs[1], vec![(0, MB, 2 * MB), (1, MB / 2, MB)]);
        // Small runs apart go together up to a job's bytes.
        let jobs = cut_of(&[(0, 0, 4096), (0, 8192, 12288), (64 * MB, 0, 4096)]);
        assert_eq!(jobs, vec![vec![(0, 0, 4096), (1, 8192, 12288), (2, 0, 4096)]]);
    }
}
