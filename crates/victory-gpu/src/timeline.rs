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
pub(crate) use on::present;

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
