//! What the host says when its device is removed: the context said at once
//! (what the submitting thread was doing, the flights, the memory), the
//! lists the report names, and (diagnostics) what the last submissions
//! noted of their lists. Moved out of `host.rs`, which the asset policy
//! keeps under 512 KiB; everything here is the host module's own.

use super::*;

/// What was going on when the host device went, as far as it is known at
/// once: what the submitting thread was doing, how many submissions were
/// sent ahead and had not landed (their lists were on the card, or waiting
/// for it, whatever the submitting thread did meanwhile), and the pipeline
/// library's calls. Atomics and one short lock: nothing here waits.
pub(crate) fn removal_context() -> String {
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = SUBMITTING[0].load(Ordering::Relaxed);
    let doing = match phase {
        0 => "between submissions",
        1 => "preparing a submission",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote",
    };
    let (started, finished) = (STARTED.load(Ordering::Relaxed), FINISHED.load(Ordering::Relaxed));
    // Started and not landed, less the one being prepared: sent ahead.
    let ahead = started.saturating_sub(finished).saturating_sub(u64::from(phase == 1));
    // The lists of those flights, if the line is free to look at (whoever
    // found the removal may hold it).
    let flying = LINE.try_lock().map_or_else(|_| "lists not known: the line was busy".to_owned(), |line| format!("{} lists", line.lists));
    let card = match (phase, ahead) {
        (2, _) => String::new(),
        (_, 0) => " (nothing of ours was running on the card)".to_owned(),
        (_, n) => format!(" ({n} submissions sent ahead had not landed, {flying}: on the card or waiting for it)"),
    };
    // Without device-removed data the runtime says nothing of where the
    // card stood: the switch that asks for it at little cost.
    let data = if dred_any() {
        ""
    } else if cfg!(feature = "diagnostics") {
        "; no device-removed data was asked for: VICTORY_D3D12_DRED=light names the list and command the card stood at"
    } else {
        "; no device-removed data: a diagnostics build with VICTORY_D3D12_DRED=light names the list and command the card stood at"
    };
    // A pass of the title's that is recorded and not run (its host
    // pipeline is not made yet, or it is one the layer drops) leaves what
    // it would have written as it was, and the passes that read it run.
    let (dropped, drawn) = crate::draw::dropped_now();
    format!(
        "the submitting thread was {doing}{card}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {started} finished {finished}; since the last frame line {dropped} draws and dispatches were recorded and not run for want of a host pipeline ({drawn} draws run so far); host pipelines queued and not made yet {} ({} of them no longer waited for){data}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        crate::pipeline::making(),
        crate::pipeline::passed_over(),
    )
}

/// The trail, for the same report: the last things done to the host's
/// objects, newest last.
pub(crate) fn removal_trail() -> String {
    use std::fmt::Write;
    let now = now_us();
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = format!("the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", now.saturating_sub(*at) as f64 / 1000.0);
    }
    text
}

/// The card's memory at every look since [`video_memory_line`] last took
/// them: the most of the card's own in use, the least its budget was, the
/// most of the PC's shared memory in use, and the looks. A look is every
/// asking of [`Host::video_memory`] (the texture queue's, the card copies')
/// and one of our own at most every tenth of a second
/// ([`Host::video_memory_look`]). The memory line is one look in five
/// seconds: what the card held between two of them was not known.
pub(crate) static VIDEO_PEAK: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// [`memory_counts`] as they were when [`video_memory_line`] was last said,
/// and when that was ([`now_ms`]).
pub(crate) static MEMORY_MARK: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// Running totals for [`video_memory_line`] and [`removal_memory`]: the
/// bytes of upload, shared and card buffers made since the start
/// ([`MADE`]), the bytes of host textures alive, the bytes of host copies
/// of textures let go since the start, and host copies of textures made
/// since the start.
pub(crate) fn memory_counts() -> [u64; 6] {
    [
        MADE[1].load(Ordering::Relaxed),
        MADE[3].load(Ordering::Relaxed),
        MADE[5].load(Ordering::Relaxed),
        crate::resource::HOST_TEXTURES[1].load(Ordering::Relaxed),
        crate::resource::KINDS[3].load(Ordering::Relaxed),
        crate::resource::KINDS[6].load(Ordering::Relaxed),
    ]
}

/// With the frame-rate line, about once a second: what the looks at the
/// card's memory saw since the last such line ([`VIDEO_PEAK`], taken), and
/// what was made in that time ([`memory_counts`], noted for the next line
/// and for a removal).
pub(crate) fn video_memory_line() {
    let most = VIDEO_PEAK[0].swap(0, Ordering::Relaxed);
    let least = VIDEO_PEAK[1].swap(u64::MAX, Ordering::Relaxed);
    let shared = VIDEO_PEAK[2].swap(0, Ordering::Relaxed);
    let looks = VIDEO_PEAK[3].swap(0, Ordering::Relaxed);
    let now = memory_counts();
    let then = [0, 1, 2, 3, 4, 5].map(|i| MEMORY_MARK[i].swap(now[i], Ordering::Relaxed));
    let at = now_ms();
    let before = MEMORY_MARK[6].swap(at, Ordering::Relaxed);
    if looks == 0 {
        return;
    }
    tracing::info!(
        "video memory in the {} ms since the last such line, at {looks} looks: the card's own in use at most {} MB, its budget at least {} MB, \
         the PC's shared in use at most {} MB; made in that time: upload buffers {} MB, shared buffers {} MB, card buffers {} MB, \
         {} host copies of textures; host textures alive {} MB (then {} MB), {} MB of filled ones let go",
        at.saturating_sub(before),
        most >> 20,
        least >> 20,
        shared >> 20,
        now[0].saturating_sub(then[0]) >> 20,
        now[1].saturating_sub(then[1]) >> 20,
        now[2].saturating_sub(then[2]) >> 20,
        now[5].saturating_sub(then[5]),
        now[3] >> 20,
        then[3] >> 20,
        now[4].saturating_sub(then[4]) >> 20,
    );
}

/// The memory side of a removal, for [`log_removal_context`]: the process's
/// and the PC's memory at this moment, what the looks at the card's memory
/// saw since the last [`video_memory_line`] ([`VIDEO_PEAK`], not taken),
/// and what was made since that line. Atomics and two plain questions to
/// the system: no lock of ours.
pub(crate) fn removal_memory() -> String {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: plain data the call fills in.
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    // SAFETY: plain data the call fills in.
    let mut p: PERFORMANCE_INFORMATION = unsafe { std::mem::zeroed() };
    p.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
    // SAFETY: a sized structure.
    unsafe { K32GetPerformanceInfo(&mut p, p.cb) };
    // The system counts in pages.
    let pages = |n: usize| (n as u64 * p.PageSize as u64) >> 20;
    let peak = [0, 1, 2, 3].map(|i| VIDEO_PEAK[i].load(Ordering::Relaxed));
    let then = [0, 1, 2, 3, 4, 5, 6].map(|i| MEMORY_MARK[i].load(Ordering::Relaxed));
    let now = memory_counts();
    format!(
        "the process: {} MB private (at most {} MB), {} MB of it in memory (at most {} MB); the PC: {} MB free of {} MB, {} MB committed of a limit of {} MB \
         (at most {} MB), the kernel's own {} MB paged and {} MB not paged, {} handles; the card's memory at the {} looks since the last line of it: \
         in use at most {} MB, the budget at least {} MB, the PC's shared in use at most {} MB; made in the {} ms since that line: upload buffers {} MB, \
         shared buffers {} MB, card buffers {} MB, {} host copies of textures; host textures alive {} MB (then {} MB), {} MB of filled ones let go",
        c.PagefileUsage as u64 >> 20,
        c.PeakPagefileUsage as u64 >> 20,
        c.WorkingSetSize as u64 >> 20,
        c.PeakWorkingSetSize as u64 >> 20,
        pages(p.PhysicalAvailable),
        pages(p.PhysicalTotal),
        pages(p.CommitTotal),
        pages(p.CommitLimit),
        pages(p.CommitPeak),
        pages(p.KernelPaged),
        pages(p.KernelNonpaged),
        p.HandleCount,
        peak[3],
        peak[0] >> 20,
        if peak[1] == u64::MAX { 0 } else { peak[1] >> 20 },
        peak[2] >> 20,
        now_ms().saturating_sub(then[6]),
        now[0].saturating_sub(then[0]) >> 20,
        now[1].saturating_sub(then[1]) >> 20,
        now[2].saturating_sub(then[2]) >> 20,
        now[5].saturating_sub(then[5]),
        now[3] >> 20,
        then[3] >> 20,
        now[4].saturating_sub(then[4]) >> 20,
    )
}

/// Says in the log what was going on when the host device went, once:
/// called by whoever finds it removed. A line for each thing, as soon as it
/// is known, the quick ones first; whoever else finds the removal
/// meanwhile waits here until the first has said it all (five seconds at
/// most), since what follows for both is the end of the run.
///
/// It was one line made whole before any of it was said, and the making
/// read the pipeline library's file for pages of zeros, from the disk if
/// need be, for up to two seconds; a second caller came straight back.
/// At the two losses of 2026-11-26 at the open world's entry the thread
/// that found the removal first was still reading when the submitting
/// thread found it too, reported the failure and ended the run: the log
/// had the failure and not a word of the context, nor the device-removed
/// data that had been asked for.
pub(crate) fn log_removal_context() {
    // 0: not said; 1: being said; 2: said.
    static SAID: Mutex<u8> = Mutex::new(0);
    static DONE: std::sync::Condvar = std::sync::Condvar::new();
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        match *said {
            0 => *said = 1,
            1 => {
                let _ = DONE.wait_timeout_while(said, std::time::Duration::from_secs(5), |said| *said == 1);
                return;
            }
            _ => return,
        }
    }
    // SAFETY: a plain query.
    let reason = match host().map(|host| unsafe { host.device.GetDeviceRemovedReason() }) {
        Some(Err(e)) => format!("{:#010X}", e.code().0),
        Some(Ok(())) => "none: the device is not removed".to_owned(),
        None => "no host device".to_owned(),
    };
    tracing::error!(
        "when the host device went (found by thread {}, removal reason {reason}): {}",
        thread_id(),
        removal_context()
    );
    // The memory at this moment: the memory line is up to five seconds
    // old, and a card or a PC out of memory would not be in it.
    tracing::error!("memory when the host device went: {}", removal_memory());
    if let Some(host) = host() {
        match host.video_memory_groups() {
            Some(_) => host.log_video_memory("where the removal was found"),
            None => tracing::error!(
                "video memory where the removal was found: the system gives no numbers (the adapter does not answer)"
            ),
        }
    }
    // With device-removed data on: the runtime's data too, unless a
    // submission's own report has asked for it already. Only the places
    // that wait for a submission asked; a removal found anywhere else (a
    // view being made, a tile mapped) said none of it. Without the lookups
    // in our own tables: whoever calls may hold their locks.
    if dred_any()
        && !DATA_SAID.load(Ordering::Relaxed)
        && let Some(host) = host()
    {
        tracing::error!(
            "device-removed data, asked where the removal was found{}",
            host.removal_data_of(&[], false)
        );
    }
    if trail_on() {
        tracing::error!("before the host device went: {}", removal_trail());
    }
    // Last, because it may read the file from the disk.
    tracing::error!("when the host device went: {}", crate::pipeline::library_removal_pages());
    *SAID.lock().unwrap_or_else(|e| e.into_inner()) = 2;
    DONE.notify_all();
}

/// A call that makes a buffer or a list for the host failed with `e`: if
/// that is the device being removed, this may be the first to find it, and
/// says the context ([`log_removal_context`]). The failure that ends the
/// run comes later and from whoever cannot go on without what was to be
/// made; at the two losses of 2026-11-26 that was up to 4 ms after the
/// first such call had failed.
pub(crate) fn found_removed(e: &windows::core::Error) {
    // Removed, hung, reset: what a call answers once the device is gone.
    const GONE: [u32; 3] = [0x887A_0005, 0x887A_0006, 0x887A_0007];
    if GONE.contains(&(e.code().0 as u32)) {
        log_removal_context();
    }
}

/// Whether a failure text that names the lists of the submissions on the
/// card has been said: 0 no, 1 it is being made (by the thread
/// [`LISTS_BY`], in [`Host::removal_data`]), 2 said.
pub(crate) static LISTS_SAID: Mutex<u8> = Mutex::new(0);
pub(crate) static LISTS_DONE: std::sync::Condvar = std::sync::Condvar::new();
pub(crate) static LISTS_BY: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Before a failure with the device removed ends the run ([`fail`], its
/// own text said): the thread whose text names the lists that were on the
/// card says so, and any other waits for that text, three seconds at most,
/// while submissions sent ahead have not landed.
///
/// Only the thread that lands submissions has their lists, and with them
/// the device-removed data says which list stood at which command, what
/// that command's marker says and which pipelines the list binds. Run
/// s_we_x5c_full_2 had asked for all of that and got none of it: the
/// submitting thread could not make a list, said so and ended the run
/// while the landing thread had not come to its report yet.
pub(crate) fn lists_said() {
    let mut said = LISTS_SAID.lock().unwrap_or_else(|e| e.into_inner());
    if *said == 1 && LISTS_BY.load(Ordering::Relaxed) == thread_id() {
        *said = 2;
        LISTS_DONE.notify_all();
        return;
    }
    let ahead = STARTED
        .load(Ordering::Relaxed)
        .saturating_sub(FINISHED.load(Ordering::Relaxed))
        .saturating_sub(u64::from(SUBMITTING[0].load(Ordering::Relaxed) == 1));
    if *said == 2 || ahead == 0 {
        return;
    }
    let _ = LISTS_DONE.wait_timeout_while(said, std::time::Duration::from_secs(3), |said| *said != 2);
}

/// For the failure text of a removed device: what the indirect records of
/// each of `runs` asked for, as far as this side knows it. An execution
/// the card stands in ran either a count of work no bound of ours cut, or
/// a program that does not come to its end on what it reads; the clamp's
/// charge tells the two apart.
///
/// Per list with indirect work: the records left to its prelude (how many,
/// in how many passes, of which kinds: clamped ahead of the list from the
/// host copies as they were then, `crate::indirect::hoists`), and the
/// words of its budget the CPU maps (`crate::indirect::BUDGET_WORDS`: the
/// work charged of 2^27 a list, then the clamp's notes). With the budget
/// in the card's memory (the default) those words are what the list's
/// last finished run left; `VICTORY_INDIRECT_CARD_BUDGET=0` keeps the
/// budget in them, and they then say what the list had charged when the
/// card stood.
pub(crate) fn indirect_of(runs: &[&HostList]) -> String {
    let mut text = String::new();
    for (i, run) in runs.iter().enumerate() {
        let hoisted: Vec<(u32, usize)> = {
            let deferred = run.deferred();
            deferred.hoist.runs.iter().map(|r| (r.kind, r.entries.len())).collect()
        };
        let words = match run.budget.get() {
            Some(Some((_, data))) => {
                // SAFETY: the list's own mapped words, mapped for as long
                // as the list lives, whatever became of the device.
                let words: [u32; crate::indirect::BUDGET_WORDS] =
                    std::array::from_fn(|k| unsafe { (*data as *const u32).add(k).read_volatile() });
                Some(words)
            }
            _ => None,
        };
        if hoisted.is_empty() && words.is_none() {
            continue;
        }
        let records: usize = hoisted.iter().map(|h| h.1).sum();
        let mut kinds: Vec<u32> = hoisted.iter().map(|h| h.0).collect();
        kinds.sort_unstable();
        kinds.dedup();
        text.push_str(&format!(
            "; indirect work of list {i} of {}: {records} records left to its prelude in {} passes (kinds {kinds:?})",
            runs.len(),
            hoisted.len()
        ));
        if let Some(words) = words {
            text.push_str(&format!(
                ", budget words {words:?} ({})",
                if card_budget() && run.card_budget.get().is_some_and(|c| c.is_some()) {
                    "as its last finished run left them: the budget is in the card's memory, VICTORY_INDIRECT_CARD_BUDGET=0 shows it as it stood"
                } else {
                    "as they stood: word 0 is the work charged, of 134217728"
                }
            ));
        }
    }
    text
}

/// A range of a host copy a list's draws read (copy, title address, bytes).
#[cfg(feature = "diagnostics")]
pub(crate) type Read = (std::sync::Arc<crate::shadow::Shadow>, usize, usize);

/// The ranges of host copies `list` brings up to date at its submission,
/// each once.
#[cfg(feature = "diagnostics")]
pub(crate) fn refresh_ranges(list: &HostList) -> Vec<Read> {
    let mut ranges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for step in &list.deferred().copies {
        if let Step::Refresh(shadow, address, len) = step
            && seen.insert((std::sync::Arc::as_ptr(shadow) as usize, *address, *len))
        {
            ranges.push((shadow.clone(), *address, *len));
        }
    }
    ranges
}

/// The lists of the last submissions, as each submission had them, while
/// breadcrumbs are on (diagnostics): the command list (by address), the
/// pipelines it binds and the ranges of host copies it reads. The report
/// of a removed device takes a stopped list's from here when the thread
/// that makes it does not have the list: a submission counted as landed
/// before the card stood, or a failure on the submitting thread while
/// nothing was in the air (run s_hang_2: the submitting thread could not
/// make a list for the next submission, and the report had no list to
/// name). Holding the ranges keeps their copies alive a little longer.
#[cfg(feature = "diagnostics")]
pub(crate) static REMEMBERED: Mutex<std::collections::VecDeque<(usize, String, Vec<Read>)>> =
    Mutex::new(std::collections::VecDeque::new());

/// How many lists [`REMEMBERED`] holds.
#[cfg(feature = "diagnostics")]
pub(crate) const REMEMBER_MOST: usize = 512;

/// At a submission: notes its lists in [`REMEMBERED`].
#[cfg(feature = "diagnostics")]
pub(crate) fn remember_lists(runs: &[&HostList]) {
    if !matches!(dred_mode(), Dred::Full | Dred::Light) {
        return;
    }
    let noted: Vec<(usize, String, Vec<Read>)> = runs
        .iter()
        .map(|r| {
            let pipelines = format!("{:?}", *r.pipelines.lock().unwrap_or_else(|e| e.into_inner()));
            (r.list.as_raw() as usize, pipelines, refresh_ranges(r))
        })
        .collect();
    let mut kept = REMEMBERED.lock().unwrap_or_else(|e| e.into_inner());
    for entry in noted {
        // A list submitted again is noted as it is now.
        kept.retain(|k| k.0 != entry.0);
        kept.push_back(entry);
    }
    while kept.len() > REMEMBER_MOST {
        kept.pop_front();
    }
}

/// What [`REMEMBERED`] holds of the command list at `list`.
#[cfg(feature = "diagnostics")]
pub(crate) fn remembered(list: usize) -> Option<(String, Vec<Read>)> {
    let kept = REMEMBERED.lock().unwrap_or_else(|e| e.into_inner());
    kept.iter().rev().find(|k| k.0 == list).map(|k| (k.1.clone(), k.2.clone()))
}

/// For the report of a removed device (diagnostics): the ranges of host
/// copies a list that stopped had brought up to date for its draws (its
/// buffer views, vertices, indices and constants), each against title
/// memory as it is after the removal. A draw of the list may have run for
/// seconds on what its copy held: a range whose copy differs from what the
/// title holds says which bytes the card read that the title did not
/// write, or wrote since (the title runs on while the card stands); one
/// that is the same says the card read the title's own bytes. Ranges that
/// differ are listed (at most 32, the first 3 words of each), the most
/// differing words first.
#[cfg(feature = "diagnostics")]
pub(crate) fn reads_against_title(ranges: &[Read]) -> String {
    let (mut bytes, mut differ) = (0usize, Vec::new());
    for (shadow, address, len) in ranges {
        bytes += len;
        if let Some((words, pending, first)) = shadow.against_title(*address, *len, 3)
            && words > 0
        {
            differ.push((words, *address, *len, shadow.id(), pending, first));
        }
    }
    differ.sort_unstable_by_key(|d| std::cmp::Reverse(d.0));
    let shown: Vec<String> = differ
        .iter()
        .take(32)
        .map(|(words, address, len, copy, pending, first)| {
            let first: Vec<String> = first.iter().map(|(at, title, copy)| format!("{at:#x} title {title:#x} copy {copy:#x}")).collect();
            format!(
                "{address:#x}+{len:#x} (copy {copy}{}): {words} words differ, {}",
                if *pending { ", shaders of a flight may write it" } else { "" },
                first.join(", ")
            )
        })
        .collect();
    format!(
        "; its reads against title memory now: {} ranges ({} MB) of host copies, {} differ{}{}{}",
        ranges.len(),
        bytes >> 20,
        differ.len(),
        if shown.is_empty() { "" } else { ": " },
        shown.join("; "),
        why_behind(ranges, &differ)
    )
}

/// For [`reads_against_title`]: of the first ranges that differ (by words
/// differing), the first 64 KiB block that differs, with what the guards,
/// the journal, Windows and the copy hold of it now: why the copy can be
/// behind title memory there.
#[cfg(feature = "diagnostics")]
fn why_behind(ranges: &[Read], differ: &[(usize, usize, usize, u64, bool, Vec<(usize, u32, u32)>)]) -> String {
    /// Ranges looked at.
    const MOST: usize = 8;
    let mut out = String::new();
    for (k, (_, address, len, copy, _, first)) in differ.iter().take(MOST).enumerate() {
        let Some(&(at, _, _)) = first.first() else {
            continue;
        };
        let Some((shadow, _, _)) = ranges.iter().find(|(s, a, l)| s.id() == *copy && a == address && l == len) else {
            continue;
        };
        out.push_str(&format!(
            "; why {k} ({address:#x}, first differing word {at:#x}): {}; {}; {}; {}",
            crate::watch::block_report(at),
            crate::inplace::view_report(at),
            crate::watch::query_report(at),
            shadow.lag_report(at)
        ));
    }
    out
}

