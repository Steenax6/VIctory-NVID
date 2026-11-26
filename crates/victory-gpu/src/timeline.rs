//! A few frames event by event (diagnostics builds): which thread did what
//! when, so that what a frame waits for is seen and not inferred from idle
//! times that happen to match.
//!
//! `VICTORY_TIMELINE=<file>` writes one line per event of
//! `VICTORY_TIMELINE_FRAMES` frames (8 unless told) once the run is
//! `VICTORY_TIMELINE_AFTER` seconds old (130 unless told): milliseconds
//! since the first of those presents, the thread, the event and two numbers
//! of its own (see the callers). Without the variable an event costs one
//! load of a flag.

#[cfg(feature = "diagnostics")]
mod on {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    /// 0: not yet, 1: events are kept, 2: written, or never asked for.
    static STATE: AtomicU8 = AtomicU8::new(0);
    static LEFT: AtomicU64 = AtomicU64::new(0);
    static EVENTS: Mutex<Vec<(u64, u32, &'static str, u64, u64)>> = Mutex::new(Vec::new());
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

    fn start() -> Instant {
        *START.get_or_init(Instant::now)
    }

    fn number(name: &str, unset: u64) -> u64 {
        std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(unset)
    }

    #[inline]
    pub fn ev(what: &'static str, a: u64, b: u64) {
        if STATE.load(Relaxed) != 1 {
            return;
        }
        let at = start().elapsed().as_nanos() as u64;
        // SAFETY: a plain query.
        let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
        EVENTS.lock().unwrap_or_else(|e| e.into_inner()).push((at, thread, what, a, b));
    }

    /// Whether events are kept now.
    #[inline]
    pub(crate) fn on() -> bool {
        STATE.load(Relaxed) == 1
    }

    /// Runs `take` (the taking of a lock, or another wait) and, while events
    /// are kept, says so if it took 0.2 ms or more: the event `name` with
    /// the microseconds it took.
    #[inline]
    pub(crate) fn waited<T>(name: &'static str, take: impl FnOnce() -> T) -> T {
        if !on() {
            return take();
        }
        let started = Instant::now();
        let taken = take();
        let us = started.elapsed().as_micros() as u64;
        if us >= 200 {
            ev(name, us, 0);
        }
        taken
    }

    /// The steps of one piece of work, each said when it ends with the
    /// microseconds since the one before ([`Steps::mark`]).
    pub(crate) struct Steps(Option<Instant>);

    impl Steps {
        #[inline]
        pub(crate) fn begin() -> Steps {
            Steps(on().then(Instant::now))
        }

        /// The microseconds since the last mark (0 while no events are
        /// kept), and the clock starts again.
        #[inline]
        pub(crate) fn lap(&mut self) -> u64 {
            match &mut self.0 {
                Some(last) => {
                    let now = Instant::now();
                    let us = (now - *last).as_micros() as u64;
                    *last = now;
                    us
                }
                None => 0,
            }
        }

        /// The event `what` with the microseconds since the last mark and
        /// `b`.
        #[inline]
        pub(crate) fn mark(&mut self, what: &'static str, b: u64) {
            if self.0.is_some() {
                let us = self.lap();
                ev(what, us, b);
            }
        }
    }

    thread_local! {
        /// Sums a thread keeps for its next [`took`]: microseconds by slot.
        static SUMS: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
    }

    /// Adds the time since `started` to this thread's sum `slot`, while
    /// events are kept (`started` from [`clock`]).
    #[inline]
    pub(crate) fn sum(slot: usize, started: Option<Instant>) {
        if let Some(started) = started {
            let us = started.elapsed().as_micros() as u64;
            SUMS.with(|s| {
                let mut sums = s.get();
                sums[slot] += us;
                s.set(sums);
            });
        }
    }

    /// The clock for [`sum`]: read only while events are kept.
    #[inline]
    pub(crate) fn clock() -> Option<Instant> {
        on().then(Instant::now)
    }

    /// This thread's sum `slot`, taken.
    #[inline]
    pub(crate) fn took(slot: usize) -> u64 {
        SUMS.with(|s| {
            let mut sums = s.get();
            let us = std::mem::take(&mut sums[slot]);
            s.set(sums);
            us
        })
    }

    /// At every present call, before anything else of it.
    pub(crate) fn present(frame: u64) {
        match STATE.load(Relaxed) {
            0 => {
                let Some(file) = std::env::var_os("VICTORY_TIMELINE") else {
                    STATE.store(2, Relaxed);
                    return;
                };
                if start().elapsed().as_secs() < number("VICTORY_TIMELINE_AFTER", 130) {
                    return;
                }
                let _ = file;
                EVENTS.lock().unwrap_or_else(|e| e.into_inner()).reserve(1 << 16);
                LEFT.store(number("VICTORY_TIMELINE_FRAMES", 8), Relaxed);
                STATE.store(1, Relaxed);
                ev("present>", frame, 0);
            }
            1 => {
                ev("present>", frame, 0);
                if LEFT.fetch_sub(1, Relaxed) <= 1 {
                    STATE.store(2, Relaxed);
                    write();
                }
            }
            _ => {}
        }
    }

    fn write() {
        use std::fmt::Write;
        let Some(file) = std::env::var_os("VICTORY_TIMELINE") else {
            return;
        };
        let events = std::mem::take(&mut *EVENTS.lock().unwrap_or_else(|e| e.into_inner()));
        let Some(&(first, ..)) = events.first() else {
            return;
        };
        let mut text = String::with_capacity(events.len() * 48);
        for (at, thread, what, a, b) in &events {
            let _ = writeln!(text, "{:10.3} {thread:6} {what} {a:#x} {b:#x}", (at - first) as f64 / 1e6);
        }
        match std::fs::write(&file, text) {
            Ok(()) => tracing::info!(events = events.len(), file = %file.to_string_lossy(), "timeline written"),
            Err(e) => tracing::warn!(error = %e, "the timeline was not written"),
        }
    }
}

#[cfg(feature = "diagnostics")]
pub use on::ev;
#[cfg(feature = "diagnostics")]
pub(crate) use on::{Steps, clock, on, present, sum, took, waited};

/// Ends a span begun with [`span`]: its end event when dropped.
pub(crate) struct Span(&'static str, u64);

impl Drop for Span {
    fn drop(&mut self) {
        ev(self.0, self.1, 0);
    }
}

/// The event `begin` now and `end` when the value returned is dropped.
#[inline]
pub(crate) fn span(begin: &'static str, end: &'static str, a: u64, b: u64) -> Span {
    ev(begin, a, b);
    Span(end, a)
}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub fn ev(_what: &'static str, _a: u64, _b: u64) {}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub(crate) fn present(_frame: u64) {}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub(crate) fn on() -> bool {
    false
}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub(crate) fn waited<T>(_name: &'static str, take: impl FnOnce() -> T) -> T {
    take()
}

/// Without diagnostics: nothing is timed.
#[cfg(not(feature = "diagnostics"))]
pub(crate) struct Steps;

#[cfg(not(feature = "diagnostics"))]
impl Steps {
    #[inline(always)]
    pub(crate) fn begin() -> Steps {
        Steps
    }
    #[inline(always)]
    pub(crate) fn lap(&mut self) -> u64 {
        0
    }
    #[inline(always)]
    pub(crate) fn mark(&mut self, _what: &'static str, _b: u64) {}
}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub(crate) fn clock() -> Option<std::time::Instant> {
    None
}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub(crate) fn sum(_slot: usize, _started: Option<std::time::Instant>) {}

#[cfg(not(feature = "diagnostics"))]
#[inline(always)]
pub(crate) fn took(_slot: usize) -> u64 {
    0
}
