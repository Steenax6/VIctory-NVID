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
//!   earlier one, as before.
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
//! stretch where the two differ is counted and named: the title itself
//! wrote there between the merge and the look, another copy of the same
//! title memory landed over it, or the merge is wrong.

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

/// The runs as jobs: each a list of (run, offsets `[lo, hi)` of it). A run
/// is cut where [`JOB`] bytes of title memory end; the pieces are taken in
/// the order of their title addresses and a job ends, once it holds a
/// [`JOB`] bytes or more, only where the next piece begins at or after the end
/// of every one before it, so that pieces of title memory two copies share
/// are in one job. Within a job the pieces are in the order given.
fn cut(runs: &[Run]) -> Vec<Vec<(usize, usize, usize)>> {
    // (title address, run, lo, hi)
    let mut parts: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (i, (shadow, lo, hi, _)) in runs.iter().enumerate() {
        let start = shadow.span().0 as usize;
        let mut at = *lo;
        while at < *hi {
            let next = ((start + at) / JOB + 1) * JOB - start;
            let end = next.min(*hi);
            parts.push((start + at, i, at, end));
            at = end;
        }
    }
    cut_parts(parts)
}

/// [`cut`] of its pieces: (title address, run, lo, hi).
fn cut_parts(mut parts: Vec<(usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize, usize)>> {
    parts.sort_unstable_by_key(|p| (p.0, p.1));
    let mut jobs: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    let mut job: Vec<(usize, usize, usize)> = Vec::new();
    let (mut bytes, mut reach) = (0usize, 0usize);
    for (title, i, lo, hi) in parts {
        if bytes >= JOB && title >= reach && !job.is_empty() {
            jobs.push(std::mem::take(&mut job));
            bytes = 0;
        }
        job.push((i, lo, hi));
        bytes += hi - lo;
        reach = reach.max(title + (hi - lo));
    }
    if !job.is_empty() {
        jobs.push(job);
    }
    for job in &mut jobs {
        job.sort_unstable();
    }
    jobs
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
    type Sample = (Arc<Shadow>, usize, Vec<u8>, Vec<u8>);
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
                for (i, lo, hi) in job {
                    let (shadow, run_lo, _, at) = &runs[i];
                    // SAFETY: the mapping of the fetched bytes; the copy
                    // into it has run, and the landing waits for this job.
                    let gpu = unsafe { std::slice::from_raw_parts((fetched + *at as usize + (lo - run_lo)) as *const u8, hi - lo) };
                    runs_done += 1;
                    let sampled = sample && runs_done % 8 == 1;
                    if sampled {
                        crate::shadow::sample_next();
                    }
                    let merged = shadow.card_merge_relay(lo, gpu, after, &mut title);
                    if sampled && let Some((off, written, card)) = crate::shadow::last_merged() {
                        samples.lock().unwrap_or_else(|e| e.into_inner()).push((shadow.clone(), off, written, card));
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
        for (shadow, off, written, card) in samples {
            COUNTS[2].fetch_add(1, Relaxed);
            let apart = shadow.landed_apart(off, &written, &card);
            if apart > 0 {
                COUNTS[3].fetch_add(1, Relaxed);
                static SAID: AtomicU64 = AtomicU64::new(0);
                if SAID.fetch_add(1, Relaxed) < 32 {
                    let (start, len) = shadow.span();
                    tracing::warn!(
                        copy = format_args!("{start:#x}+{len:#x}"),
                        at = format_args!("{:#x}", start as usize + off),
                        bytes = written.len(),
                        apart,
                        "a stretch a landing wrote holds other bytes in title memory than the copy's CPU side at the telling (VICTORY_LAND_SAMPLE=1)"
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
    tracing::info!(
        "the landings' merges as jobs: {:.1} jobs a frame, at most {most} in one landing{}",
        jobs as f64 / frames,
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
    /// runs' title addresses, without copies.
    fn cut_of(runs: &[(usize, usize, usize)]) -> Vec<Vec<(usize, usize, usize)>> {
        // (start, lo, hi) as a run of a copy from `start` on.
        let mut parts: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (i, &(start, lo, hi)) in runs.iter().enumerate() {
            let mut at = lo;
            while at < hi {
                let next = ((start + at) / super::JOB + 1) * super::JOB - start;
                let end = next.min(hi);
                parts.push((start + at, i, at, end));
                at = end;
            }
        }
        super::cut_parts(parts)
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
