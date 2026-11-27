//! What the queue's page-mapping copy costs the thread that calls it: how
//! many calls, batches, ranges and pages, and where the milliseconds of a
//! call go (the queues' waiting work run before it, the flights landed
//! before it, the remap itself). Counting is done only in builds with
//! `diagnostics`; without, [`Call`] is empty and does nothing.

#[cfg(feature = "diagnostics")]
mod on {
    use std::cell::Cell;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::Instant;

    /// Calls, batches, ranges, pages of kinds 0 to 3, microseconds in the
    /// call in all, in the queues' work run before it, in the landings, in
    /// the remap, the longest call's microseconds, calls made by the thread
    /// that presents.
    static COUNTS: [AtomicU64; 14] = [const { AtomicU64::new(0) }; 14];

    /// The thread that presented last (its number from [`id`]), 0 before the first present.
    static PRESENTING: AtomicU64 = AtomicU64::new(0);
    /// Numbers of the threads that called since the last line, and the round
    /// they belong to: a thread says its number once a round.
    static THREADS: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    static ROUND: AtomicU64 = AtomicU64::new(1);

    thread_local! {
        static ID: u64 = {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            NEXT.fetch_add(1, Relaxed)
        };
        static SAID: Cell<u64> = const { Cell::new(0) };
    }

    fn id() -> u64 {
        ID.with(|i| *i)
    }

    /// The calling thread presents: the calls it makes are counted as the presenting thread's.
    pub(crate) fn presenting() {
        PRESENTING.store(id(), Relaxed);
    }

    #[derive(Clone, Copy)]
    pub(crate) enum Part {
        /// The queues' waiting work run before the remap.
        Sync = 8,
        /// Flights landed before the remap.
        Land = 9,
        /// The remap itself.
        Remap = 10,
    }

    /// One call of the page-mapping copy, counted while it lives.
    pub(crate) struct Call {
        began: Instant,
        mark: Instant,
        parts: [u64; 3],
        batches: u64,
        ranges: u64,
        pages: [u64; 4],
    }

    impl Call {
        pub(crate) fn begin() -> Self {
            let now = Instant::now();
            Call { began: now, mark: now, parts: [0; 3], batches: 0, ranges: 0, pages: [0; 4] }
        }

        /// What happened since the last mark (or the start) belongs to `part`.
        pub(crate) fn took(&mut self, part: Part) {
            let now = Instant::now();
            self.parts[part as usize - 8] += now.duration_since(self.mark).as_micros() as u64;
            self.mark = now;
        }

        pub(crate) fn batch(&mut self) {
            self.batches += 1;
        }

        /// A range of `kind` over `pages` pages.
        pub(crate) fn range(&mut self, kind: u32, pages: u32) {
            self.ranges += 1;
            self.pages[(kind as usize).min(3)] += u64::from(pages);
        }
    }

    impl Drop for Call {
        fn drop(&mut self) {
            let all = self.began.elapsed().as_micros() as u64;
            COUNTS[0].fetch_add(1, Relaxed);
            COUNTS[1].fetch_add(self.batches, Relaxed);
            COUNTS[2].fetch_add(self.ranges, Relaxed);
            for k in 0..4 {
                COUNTS[3 + k].fetch_add(self.pages[k], Relaxed);
            }
            COUNTS[7].fetch_add(all, Relaxed);
            for k in 0..3 {
                COUNTS[8 + k].fetch_add(self.parts[k], Relaxed);
            }
            COUNTS[11].fetch_max(all, Relaxed);
            let me = id();
            if PRESENTING.load(Relaxed) == me {
                COUNTS[12].fetch_add(1, Relaxed);
            }
            let round = ROUND.load(Relaxed);
            if SAID.get() != round {
                SAID.set(round);
                if let Ok(mut threads) = THREADS.lock() {
                    threads.push(me);
                }
            }
        }
    }

    /// With the frame line: what the calls did per frame over `frames`.
    pub(crate) fn line(frames: f64) {
        let c: [u64; 14] = std::array::from_fn(|i| COUNTS[i].swap(0, Relaxed));
        let threads = THREADS.lock().map(|mut t| std::mem::take(&mut *t).len()).unwrap_or(0);
        ROUND.fetch_add(1, Relaxed);
        let f = |i: usize| c[i] as f64 / frames;
        let ms = |i: usize| c[i] as f64 / 1e3 / frames;
        tracing::info!(
            "page-mapping copies per frame: {:.2} calls in {:.2} batches and {:.1} ranges, {:.1} pages mapped (kind 0 {:.1}, kind 1 {:.1}), \
             {:.1} unmapped (kind 2), {:.1} of kind 3; {:.2} ms in the calls, of it {:.2} ms in the queues' work run before the remap, \
             {:.2} ms landing flights, {:.2} ms in the remap; the longest call {:.2} ms; called from {threads} threads since the last line, \
             {:.2} calls a frame from the thread that presents",
            f(0),
            f(1),
            f(2),
            f(3) + f(4),
            f(3),
            f(4),
            f(5),
            f(6),
            ms(7),
            ms(8),
            ms(9),
            ms(10),
            c[11] as f64 / 1e3,
            f(12),
        );
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) use on::{Call, Part, line, presenting};

#[cfg(not(feature = "diagnostics"))]
mod off {
    #[derive(Clone, Copy)]
    pub(crate) enum Part {
        Sync,
        Land,
        Remap,
    }

    pub(crate) struct Call;

    impl Call {
        #[inline(always)]
        pub(crate) fn begin() -> Self {
            Call
        }
        #[inline(always)]
        pub(crate) fn took(&mut self, _part: Part) {}
        #[inline(always)]
        pub(crate) fn batch(&mut self) {}
        #[inline(always)]
        pub(crate) fn range(&mut self, _kind: u32, _pages: u32) {}
    }

    #[inline(always)]
    pub(crate) fn presenting() {}
}

#[cfg(not(feature = "diagnostics"))]
pub(crate) use off::{Call, Part, presenting};
