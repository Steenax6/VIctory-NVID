# Architecture

```
 user's files (outside repo)             VIctory
 ───────────────────────────             ─────────────────────────────────────────
 game assets ──────────────────────────► victory-pipeline + victory-game
                                         (victory-formats, victory-shader)
                                                    │
                                                    ▼
                                         cache_dir: converted data
                                                    │
 game executable ─► victory-loader ────► title code (native x86-64)
                    prepare, map, bind      │ imports
                                            ▼
                                         victory-hle ── host Windows (shared APIs)
                                            │ D3D12.X
                                            ▼
                                         victory-gpu ─► host D3D12
                                            │ shaders at pipeline creation
                                            ▼
                                         victory-shader ◄─► cache_dir
```

Shaders are translated when the title creates a pipeline, from the bytes it
passes, wherever it loaded them from (executable, archive, memory).
Converting them ahead of time from files is only a warm-up.

## Crates

| Crate | Role |
|-------|------|
| `victory-core` | config, asset-policy guard, logging |
| `victory-pe` | read-only PE32+ and import-library analysis |
| `victory-loader` | prepare, map, bind and start the title image |
| `victory-hle` | console API implementations, host pass-through policy |
| `victory-gpu` | D3D12.X objects, command lists and GPU memory → host D3D12 |
| `victory-shader` | Xbox shader containers → host shaders |
| `victory-formats` | the GPU's texture layouts (sizes, tiling, detiling), audio codecs; planned: BCPack |
| `victory-pipeline` | converter trait, content-addressed cache |
| `victory-game` | the title-specific formats, build identities, hooks |
| `apps/victory` | command-line front end |
| `xtask` | asset-policy check, git hooks |
| `tests/fixtures/hle-probe` | a stand-in "title" (our own code) for loader tests |

Everything except `victory-game` is platform-level and can be tested with
any Xbox Series title.

## Import resolution

For every `dll!function` the title imports, in order:

1. our implementation in `victory-hle::modules` (always wins);
2. the host's own function, if `HostPolicy` allows that DLL;
3. an export of a DLL the title ships beside its executable, mapped like
   the executable (only DLLs neither of the above serves);
4. a generated stub that names the import when called (`--unimplemented
   trap` breaks into the debugger and aborts; `zero` logs once and returns 0).

`LoadLibrary*` of a console-only DLL we implement (e.g. `xgameruntime.dll`)
returns a pseudo handle; `GetProcAddress` on it finds our functions.
`LoadLibrary*` of a title DLL returns its mapped image, mapping it first if
nothing imports it. By name, as by import, a title gets only host DLLs
`HostPolicy` passes through, and `GetProcAddress` on one returns our
override where import binding would.
`QueryApiImpl` hands out our runtime objects by provider and interface id
(`victory-hle::modules::xgameruntime` lists the areas); online areas answer
as an offline console. Game saves are files under `VICTORY_DATA_DIR`.

Title storage is a drive on the console, and titles build paths on that.
`victory-hle::modules::files` binds every path-taking title import
(kernel32 and the C/C++ runtime) and rewrites paths on a virtual drive to
the host folder behind it; persistent local storage is one such drive over
a folder under `VICTORY_DATA_DIR`. Other paths reach the host unchanged.

An exe's code reaches its static TLS as slot 0; mapped as a DLL it gets
another slot, so the loader redirects those loads (`victory-loader::tls`).
`victory run --execute` installs a crash reporter that names faults as
`module+rva` and unwinds the faulting thread's stack
(`victory-loader::crash`).

`victory census` shows this classification without running anything (it
does not map title DLLs; `victory run` without `--execute` does).

## Shaders

`victory_shader::ir::build` turns a native program into SSA, one
instruction per machine instruction. `victory_shader::structure` then
removes EXEC: lane masks become per-lane booleans, wave-wide mask tests
become per-lane branches, and vector instructions keep a guard only where
a lane can reach them with its EXEC bit clear. `victory_shader::resource`
traces each descriptor and scalar memory address to the root parameter it
comes from (table, heap, root descriptor or constant). A descriptor a
waterfall loop reads from the first lane (`v_readfirstlane_b32`) is traced
to the per-lane load or choice that gives each lane its own (a mask's
`v_cndmask_b32`, or an if/else whose arms each set it: traced on the paths
the lanes take, from structurisation), and accessed with a non-uniform heap
index in one pass. `victory translate`
runs every shader it finds (containers, SGD5 files) through these and
lists what they cannot handle yet.

## The asynchronous pipeline

Everything between the title's submit call and the title seeing the
results: the line of submissions (`queue.rs`), preparing and sending
(`host.rs`, `Host::submit`), the flights on the card, their landing
(`Host::relay_land`, `shadow.rs`) and the write tracking all of it leans
on (`watch.rs`). How it is, what was measured, and the order in which it
is to be rebuilt. The switches named are in `switches.md`.

### As it is

| Stage | Thread | What it does | Held meanwhile |
|---|---|---|---|
| capture | the title's render thread | the lists' host sides are taken from the title's lists, cut at their waits for memory values and put on their queue's line (`Held`): two bursts a frame (83 lists when the present returns, 121 before the next present call), 28 parts | the queue's line |
| prepare and send | `victory-submit` (one) | `release_held_now`, then `Host::submit` part by part: heap sync (descriptor records), write-watch asking, title bytes under flights' shader writes, the lists' steps in recorded order (fills, title buffer copies, declared writes; the refreshes of host copies are flushed before every step that writes title memory or gives a range to shaders), textures, preludes, the card's sync and compare lists, `ExecuteCommandLists`, `Signal` | `RELEASING` (one releaser); `Host::submitted` from the first step to the send |
| flight | the card | our sync list (changed ranges of the copies' CPU sides to their card sides, hashes), the title's lists, our compare list (flags the 4 KiB blocks whose hash changed) | |
| land | `victory-relay` (one) | for the flights the card has run, in the order sent: the flagged blocks fetched on the copy queue (`queue2.rs`), merged into title memory and the copy's CPU side (`Shadow::card_merge_relay`) as jobs of 256 KiB of one copy on the wide threads and this one (`host/land_jobs.rs`), all done before the landing goes on; the ranges opened to refreshes again; then, in order, copies left to the landing, read-backs, bottom-of-pipe writes, fence signals | `LINE` to take and to give back; the side locks of the 64 KiB pieces a merge writes |
| after | `victory-late-wait` | landed lists reset; the submitting thread woken for waits now met | |

Beside them: `victory-wide-N` (four: finds, snapshots, the landing's
merge jobs and the slots of the pass over picked buffer views),
`victory-picked` (the pass over picked buffer views), `victory-thin-N`
(with `VICTORY_THIN=1`: translation of the recordings).

The locks between them: `RELEASING` and the queues' lines (the render
thread's present waits there until the frame is sent, unless the present
is in the line); `Host::submitted`; the journal (`watch.rs`, one lock for
all title memory, held across the kernel's asking and the guards' check,
with one standing asking for the one submission being prepared); the
guards (`watch.rs`: a lock a region of 4 MiB since S3's third step,
taken in the order of addresses where an operation needs several and
never held across a call into the kernel; the fault handler on the
title's threads takes its block's region, and those of the block's other
views; what only asks reads without a lock); per copy the side lock
(`Shadow::side_of`: for a copy in the card's memory one per 64 KiB
piece, held across the merge of a fetched run over the pieces it writes
and across each page of the title's bytes put in under flights; for the
others the snapshot's), `copied`,
`pending` with `flights`, `dirty` with `sent`; `LINE` with its condition
variables, `CARD_WORK`, `QUEUE_ORDER`.

Serial for a reason: the order of sends; the order of the steps that
write title memory (title buffer copies, steps that run code on it)
against the refreshes that read their destination; a range is declared
written by shaders after its last refresh and before its send; one
copy's flights land in order; read-backs, bottom-of-pipe writes and
fences are told in order.

Serial without one: all preparing on one thread under one lock, though
its bulk is per copy and per part; the asking repeated for every part
and flush, though the parts of a burst were submitted in the same
millisecond; the sync list sent as soon as it is recorded, which ties
preparing a part to sending it.

### Measured

2026-11-26, first room of the prologue, Series X profile, diagnostics
build, Windows 11, GeForce RTX 4070 Ti (12 GB), Ryzen 7 9800X3D; twelve
frames of a timeline each (`VICTORY_TIMELINE`, which names the steps of
a send as `p.*`, of a landing as `l.*`, each flight `sent`, `done` and
`landed`, and every wait of 0.2 ms or more for a lock as `L:*`), single
runs on a machine other work ran on: indications.

| ms a frame | before thin recording (run a_tl1) | thin recording, present in the line (run i_thinap1) |
|---|---|---|
| frame | 76.1 | 72.2 |
| submitting thread inside sends; idle | 68.5; 7.3 | 69.9; 0.0 |
| heap sync | 11.4 | 12.1 |
| write-watch asking | 5.5 | 4.1 |
| title bytes under flights' shader writes | 8.1 | 13.0 |
| the lists' steps without their refreshes | 10.3 | 10.4 |
| refreshes of host copies | 22.9 | 19.6 |
| textures; preludes; card lists; `ExecuteCommandLists` with `Signal` | 1.5; 2.3; 2.7; 0.8 | 1.7; 2.9; 2.8; 0.9 |
| the card busy; nothing of ours queued (of it while a send was being prepared) | 41.2; 34.9 (31.8) | 42.2; 30.0 (29.6) |
| landing thread in landings (the merge; the wait for the fetch) | 34.7 (28.5; 4.4) | 42.6 (33.8; 4.7) |
| MB fetched; MB put into title memory | 61.8; 50.5 | 58.6; 46.2 |
| a flight from sent to told, median | 14.7 | 18.5 |
| waits for the guards' lock: the title's recording and job threads, and the translating ones | 13.2 | 50.2 (of it 45.2 in 52 waits of the recording and translating threads, longest 8.8) |
| waits for the guards' lock: submitting; landing | 2.4; 1.2 | 5.4; 5.2 |
| waits for a copy's side lock: submitting; landing | 1.3; 0.1 | 0.8; 0.7 |
| the render thread standing in its present | 54.1 | 29.5 |

What the numbers say:

- The submitting thread is the bound, and it starves the card: the card
  stands in some four gaps of 2 ms and more a frame, each behind one
  long prepare. The five longest sends of a frame are 46.5 of 68.5 ms.
  The longest (7 to 14 ms) is the submission of seven lists that makes
  the heap's buffers ready: one flush that walks about 4,630 copies with
  almost nothing to copy. The pass over picked buffer views feeds it:
  it puts the slots it has just made back on the list of changed ones,
  so the next pass looks at them again (21 to 36 ms a pass on its
  thread; a submission waits 5 ms a frame for it).
- Of a refresh (18.5 ms in a profile at 64.9 ms a frame) the asking is
  9.2 (the guards 6.8) and the copying and comparing 5.7. Of the 86 MB a
  frame the refreshes read, about 21 MB are other bytes than the copy
  holds, and 45 MB are what a landing has just written (23 MB in view
  memory, 22 MB in private memory; the landings' taking has dealt with
  the private half since).
- The same work on more threads was slower, three times that day: the
  heap sync on a thread of its own beside the refreshes (75 to 80
  against 67 to 69 ms a frame; taken out again), `VICTORY_WIDE_REFRESH=1`
  (85 against 69), `VICTORY_WIDE_MERGE=1` (72.1 against 73.6, the merge
  26.6 against 28.5: the wide threads waited 15.4 ms a frame for the
  copy's one side lock). Every thread added queues on the guards' lock:
  with thin recording its waits are 61 ms a frame over all threads. So
  the locks go first and the threads after (see the order below).
- The shared locks cost the submitting thread itself little (3.6 ms a
  frame before thin recording): a send beside a landing is not slower
  for the locks' sake.
- A read census (`VICTORY_READ_CENSUS=16`, run a_rc1) of what landings
  write into private title memory, about 24 MB a frame: of 380 pages
  sampled a frame the title's own code touched 0.0 to 0.3 first, another
  thread of the layer's 17 to 99, and the landing thread's own next
  merge, with nothing between, 292 to 362. View memory and the open
  world are not counted yet.

### How it is to be

```
title's submit - capture: lists detached, the burst stamped
   |- heap lane, per burst: the records written before the burst's submit
   |- tracking lane, per burst: one asking for every range its parts refresh
   `- plan, serial, in order, no bulk bytes: waits met, the steps that write
      title memory in order, ranges declared, the part's jobs made
          `- move, a pool shared by all: refreshes, fills, title bytes under
             flights, textures, preludes of parts k, k+1, ... side by side
COMMIT 1  send, serial, in order: jobs of k and the burst's heap pass done,
          changed ranges taken, sync list, ExecuteCommandLists, Signal
the card  per flight: fetch on the copy queue, merges per copy and piece
          on the same pool
COMMIT 2  tell, serial, in order: read-backs, bottom-of-pipe writes, fences
```

What orders the work, in place of one lock and one thread:

- **R1** A refresh may read title memory at any time after its burst was
  submitted (the card on the console reads it later still; the line of
  submissions already rests on this) and is done before its part is
  sent. The layer's own writes into title memory (title buffer copies,
  steps that run code on it, landings) are made in the order of the
  parts and told to the journal; a refresh planned before such a write
  over the same pages is waited for first.
- **R2** Per copy one lock for "refresh or declare": a range is declared
  written by shaders (`Shadow::writing`) only while no refresh of the
  copy runs, and a refresh reads `pending` under that lock. No byte goes
  into a copy's CPU side inside a pending range except through
  `Shadow::merge_under_writes`. Today one thread keeps this by being
  one.
- **R3** The changed ranges are taken when the part is sent, numbered
  under `QUEUE_ORDER`. Bytes of a later part may go up with an earlier
  send (R1), never the reverse.
- **R4** Merges of one copy in the order of its flights; any order
  across copies and pieces. The side lock is per 64 KiB piece. A flight
  is told after every merge of the flights up to it.
- **R5** The standing asking belongs to the burst, not to the journal;
  the journal is locked by the chunk of its table, the guards by region;
  no call into the kernel and no walk over other regions under a lock
  the fault handler takes.
- **R6** Waits for memory values, the present and page mappings keep
  their places in the line.

What stays serial: the plan (the order of effects on title memory and
of declarations), the send (3.5 ms a frame), the tell (0.1 ms). The
floor is the card's 41 to 44 ms a frame; of that our own passes were 10
to 14 ms in the timing runs, which is the work after this.

### The stages

Order of building, as the measurements above changed it: **S3 and S2
first, side by side, with S1a; then S6a; then S1b, S4, S5.** Until S3
has the guards' lock out of the way no stage adds a thread (S1b stood
first before the heap sync beside the refreshes and the wide refresh
were measured).

| Stage | What changes | Why it is exact | The check (0 in one run) | Touches |
|---|---|---|---|---|
| S1a less on the submitting thread (the pass's part built, the flush's taken out again: see "Built") | a flush leaves out, before it joins, asks or walks, every copy whose counts of events are those it remembered (one look without a lock each); the pass over picked buffer views no longer puts the slots it made itself on the changed list | the refresh by event's own rule, applied at the same moment (the flush), before the walk instead of inside it | `VICTORY_REFRESH_EVENTS=2`: refreshes left out that found bytes; `VICTORY_REFRESH_CHECK=1` not above the base (see "Open") | `host.rs` `Batch::flush`; `shadow.rs` `Shadow::update_free` and what it remembers; `draw.rs` `picked_pass`; `descriptor/host.rs` `mirror_picked` |
| S1b lanes ahead, per burst (not before S3) | the heap sync starts when the burst is queued and is waited for at the send; helper threads bring the copies the burst's parts refresh up to date ahead, and the refresh in the part's turn stays as it is; the lock of R2 | the records are read after the title's submit and before any list of the burst goes; nothing rests on the refresh ahead, since the one in order still runs and asks (R1); R2 keeps bytes out of ranges shaders write | bytes written into a pending range outside `merge_under_writes`, counted under R2's lock; the old heap pass run at the send in a check mode: records it still finds that were written before the burst's submit; `VICTORY_REFRESH_CHECK=1` | `host.rs` `Host::submit`, `Batch`; `shadow.rs` `Shadow::update`, `Shadow::writing`; `descriptor.rs` `sync_heaps`; `queue.rs` (the burst's stamp) |
| S2 landing on several threads (built: see "Built") | the side lock per 64 KiB piece; merges as jobs per copy and megabyte on the pool, as the way (`VICTORY_WIDE_MERGE` goes); a run that changed whole is not read from title memory first where nothing of it waits to go up; telling stays in order | the runs are apart in title memory and in the CPU side; one copy's flights stay in order (R4); `merge_under_writes` takes the same piece's lock | the merge-race count (`VICTORY_MERGE_RACE`); `VICTORY_LAND_BESIDE=2` (writes not at the title's address); new: at the tell, a sample of landed stretches whose title bytes are not the CPU side's | `shadow.rs` `Shadow::side`, `card_merge_relay_fast`, `card_merge_relay_whole`, `relay_merge`, `merge_under_writes_into`; `host.rs` `Host::relay_card_back`; `card_fine.rs` |
| S3 write tracking without global holds (steps 1 to 3 built, see "S3 as built"; the journal's step is open) | first an event for every hold of the guards' lock of 0.2 ms and more, by holder; then: `known_view` and `watchable` answered without the lock, changes of protection and walks over other regions outside it, the fault handler on its region's lock only; the journal by chunk and the standing asking the burst's (asked once a burst; the layer's own writes tell the journal as now); a landing takes its own marks in view memory as it takes the kernel's bits in private memory (the other 23 MB a frame the refreshes read back) | every count of an event is still raised under the lock under which an asking sees it; locks are taken in the order of addresses; the parts of a burst read what was written before the same submit, and a write of the title's after it is found by the next burst's asking (R1); the taking tells every other copy through the journal and passes a page by only where it holds the copy's bytes | `VICTORY_REFRESH_EVENTS=2`; the taking's check as `VICTORY_LAND_TAKEN=2`, for view memory; `VICTORY_REFRESH_CHECK=1` not above the base, by kind; a test of faults against checks in two regions at once | `watch.rs` (`guards`, `guard_fine`, `known_view`, `watchable`, `on_write`, `open_tail`, `mark_with_views`, `title_memory_remapped`, `journal`, `ask_once`, `writes_fine_look`, `take_landed`); `beside.rs`; `inplace.rs` `pages_of`; one call in `shadow.rs` `landed` |
| S4 a part's bulk as jobs | fills, the title's bytes under flights page by page, texture stagings and preludes on the pool, joined where the one thread flushes now | the same order of effects at every flush | those of S1b and S2; the fills' check (`fill_ahead`) | `host.rs` `HostList::run_deferred`, `Batch`, `stale_under_flights` |
| S5 parts in a pipeline | the plan of part k+1 while the jobs of part k run; sends in order; `Host::submitted` no longer over bulk work; the line of all queues as one sequencer (avatarmanz/VIctory-7900XTX `queue/behind.rs` is the shape) | R1 to R6 | jobs of parts up to k not done at the send of k; a refresh not done under a write of ours over its pages; a copy's flights landed out of order | `host.rs`, `queue.rs`, `shadow.rs` |
| S6a transfer heaps on title memory | for buffers in private memory that shaders write and the title's processor side does not (the census names them): a heap opened on the title's pages (`OpenExistingHeapFromAddress1`, `inplace.rs`) is the end of our copies only, card to title memory and title memory to card; shaders keep the copy in the card's memory. A block whose written bit the kernel holds, or with bytes still to go up, lands the old way | the bytes are in title memory before the flight is told, as now; a block the title wrote since goes through the merge byte by byte | in a check mode the same blocks fetched a second time into memory of ours: bytes that differ | `inplace.rs`, `shadow.rs`, `host.rs` `Host::relay_card_back`, `Host::card_before` |

Built (2026-11-26, Package B; single runs on a machine other work ran
on, twelve frames of a timeline each against run i_thinap1: indications):

- S1a, the pass over picked buffer views no longer puts the slots it
  makes itself on its own list (`HostHeap::mirror_passed`): 40,000 slots
  looked at a frame instead of 54,000; no submission waited for the pass,
  before or after. The flush that leaves out copies by their counts was
  built, measured and taken out again: it left out 2,900 of the 4,600
  copies of the seven-list submission and that send went from 9 to 14 ms
  to 6 to 14, but its check counted copies it would have left with old
  bytes in every run (see "Tried and not kept" in `switches.md`). That
  send stays the longest of the frame.
- S2, the side lock per 64 KiB piece (`Shadow::side_of`) and the merge as
  jobs of one copy on the wide threads (`host/land_jobs.rs`), the only
  way; a run that changed almost whole, in eight stretches of changed
  bytes or fewer, written without reading title memory first. With jobs
  of a megabyte (run l_a2): the landing's merge 28.3 against 33.8 ms a
  frame, the landing thread in landings 35.6 against 42.6, waits for
  side locks 0.2 against 1.5 ms a frame over all threads. A landing of
  some 20 jobs merges in 5 to 9 ms of wall time where its jobs sum 25 to
  42; what is left is the guards' lock: the threads that run jobs waited
  12.1 ms a frame for it (5.0 before), the landing thread 3.7, and one
  job that waits there holds the landing. Jobs of 256 KiB and the run
  written by its bytes (not by dwords: a byte the card left as it was in
  a dword it changed is the title's) are not timed apart.
- Checks, prologue (runs l_c1 to l_c5) and the open world's entry (l_w1,
  the final merge): the merge race 0 bytes put back, `VICTORY_LAND_BESIDE=2`
  0 writes not at the title's address, `VICTORY_LAND_TAKEN=2` 0 finds,
  `VICTORY_SEEN_CHECK=1` 0 and 0, the sample at the telling
  (`VICTORY_LAND_SAMPLE=1`) 0 of 63,395 stretches in the prologue and 0 of
  48,726 in the world.

Expected, with what speaks for it: S1a 7 to 14 ms of the submitting
thread (the longest send); S2 the landing from 28 to 34 ms to about 10
(the 15 to 20 ms of side-lock waits are all that stops the wide merge;
avatarmanz reports 45 to 8 ms for 98 MB on four threads); S3 the 17 to
61 ms a frame of waits for the guards' lock over all threads, 5 to 8 ms
of asking on the submitting thread, and it is what S1b, S4 and S5 need
to scale; S6a about 10 ms of the landing thread and the refresh of some
21 MB; S1b, S4 and S5 together the submitting thread's serial rest down
to 20 to 25 ms a frame, below the card's 41.

#### S3 as built

Steps 1 to 3 (2026-11-26; `watch.rs`, `inplace.rs`, one call in
`shadow.rs` `landed`), measured as the table above was: twelve frames of
a timeline of the first room each, single runs, indications.

Who held the guards' lock (step 1: `H:*` and `HS:*` of the timeline; runs
k_hold_plain and k_hold_thin), ms a frame:

| | before thin recording | thin recording, present in the line |
|---|---|---|
| the lock held, all holders (holds a frame) | 11.5 (4,500) | 12.3 (4,600) |
| faults of the title's stores, each across its change of protection | 5.7 (1,190; 4 to 5 us) | 5.7 (1,130; 4 to 5.5 us) |
| checks (the submitting thread) | 2.0 (1,020; 2 us) | 2.3 (1,100; 2 us) |
| the landing opening blocks for its writes | 1.7 (420; 4 us) | 2.1 (430; 5 us) |
| looks (is this a view; two a frame of 0.35 ms sorting the views out of a burst's ranges) | 1.3 (1,120) | 1.4 (1,220) |
| the landing marking what it wrote beside the guards | 0.8 (730) | 0.8 (750) |
| waits of 0.2 ms and more, all threads | 15.2 | 58.2 |
| the title's recording threads; its other threads | 9.4; 4.0 | 44.6; 5.4 |
| submitting; landing | 1.2; 0.6 | 4.4; 3.9 |

No holder kept the lock long (the longest hold was a fault of 3.0 ms, a
thread taken off its processor), and it was waited for four to five times
as long as it was held: a waiter that sleeps is woken late on a machine
whose processors are all busy, and under one lock every thread that
stores into guarded memory meets every other. So step 3 makes the
meetings fewer: a lock a region, what only asks without a lock, nothing
of the kernel's under one.

After steps 2 and 3 (runs k_after_plain and k_after_thin), ms a frame:

| | before thin recording | thin recording, present in the line |
|---|---|---|
| frame | 60.2 (62.2 before) | 57.0 (68.8 before) |
| the guards' locks held, all holders | 0.95 | 1.23 |
| waits of 0.2 ms and more for them, all threads | none (15.2) | none (58.2) |
| the submitting thread inside sends | 40.2 (49.6) | 50.8 (62.8) |
| refreshes of host copies; write-watch asking | 10.8; 2.5 (13.6; 3.6) | 11.7; 2.8 (16.4; 4.4) |

Of the holds that are left, the checks are 0.6 ms a frame (620 to 730 of
0.8 us, each with only the regions locked that have a block to look
at), the faults 0.2 to 0.5 ms (1,320 to 1,370 a frame, two takings of
0.1 to 0.3 us each with the change of protection between them), the
landing's markings 0.1 and its openings 0.06 (those of private memory
take no lock any more). The title's stores fault a tenth to a fifth
more often than before: a view it stores through is made read-only
again by the checks of the other views of its pages, where it stayed
open for good (that was the hole, below). The landing waits for the
journal's lock now and then (0.1 and 0.5 ms a frame; it takes it for
its marks).

The frame did not follow in these runs: the medians of the frame lines
over the first room (115 to 172 s) are 61.0 ms without thin recording
(57.1 to 60.2 in three runs of the tree before) and 60.6 with it (60.4
and 61.8). The waits were the title's recording threads', and the frame
is bound behind them: with thin recording the submitting thread is
never idle (50.8 ms a frame in sends, 62.8 before) and the card busy
38.8 ms; without it the submitting thread idles 18 ms a frame for want
of work (10 before) while the render thread stands in its present. What
the regions free is for the stages that add threads (S1b, S4, S5) and
for the landing on several threads (S2), which met here.

The hole (step 2). The refresh check's blocks that "differed for two
seconds with nothing told" were stores of the title's own, through
another view of the same physical pages that the guards had left
writable while the view a 4 MiB window lies over was noted clean again.
The title shows its streamed pages three times (its own view, the
GPU's, one more), stores through its own and has the shaders read the
GPU's. Its view was writable because a read to a plain address had been
told to the guards, which opens the blocks (96 of the 137 blocks a run
with a trail of every block named), or because an earlier store had
faulted there and opened the block from its kernel page on (41); in both
the other views were counted written once and noted clean by their next
check, and nothing guarded the opened view again, since no copy of ours
lies over it. The rule kept now: **while a block is noted clean, no view
of its pages is writable**; what opens or maps a view notes the guarded
views of its pages as apart, and the check that reports such a block
makes the other views read-only before it notes the block clean. A
block opened whole counts as written anywhere at its other views. The
check read 1,614 such blocks before (2,081 that differed three times in
a row; 1,191,289 differing looks), 11 with the rule and without the
last sentence, and 0 of both with it (7,615 looks, each a store between
a refresh and the look; run k_chk1, all checks on). At the open world's
entry (run k_we_chk, all checks on) it reads 0 for two seconds too; the
35 blocks that differed three times in a row there are windows over
private memory the title was writing, which the kernel watches, and all
matched again. The texture pass
asks the same guards: of the textures a look found changed, 255 of 280
were "guarded and clean (nothing seen)" before and 0 of 457 after (435
"a store the guards saw"). What the picture could show for it: geometry,
tables and texels read through the GPU's view as they were before the
title finished them in place.

A landing takes its own marks in view memory (`take_landed_views`, the
twin of the taking in private memory): some 360 blocks a frame with
1,400 kernel pages (5.7 MB) in the first room, noted in the journal at
the landing and counted taken by the landing's copy where they hold its
bytes (500 blocks and 8.2 MB a frame at the open world's entry). Its
check is `VICTORY_LAND_TAKEN=2`, which counted 0 of 448,385 pages
(29,625 written again since) in k_chk1 and 0 of 586,657 (19,601) in
k_we_chk. The other checks in those two runs: the events' 0 finds in
149 and 156 lines; the seen check 0 and 0; no subresource copied from
title memory over what the card had written last, no level a view may
show left with nothing. The thin check is not 0, and in one line of each
run above the tree before: draws that would have drawn otherwise in the
call, a frame, 0.10 and 4.49 in two of the prologue's 31 lines (0.04 in
one of 32 before) and 0.02 to 0.07 in four and 63.86 in one of the
world's 31 (0.04 to 0.08 in six of 28 before). They are of the kind the
tree before has: a table the title wrote again between a call and its
translation. Whether the two lines come of the refresh check, which was
on in these runs and not in those and holds the frame at 140 ms, or of
the title's threads no longer waiting for the guards, is not told apart
here: it wants the thin check alone on this tree.

Open of S3, its fourth step: the journal by chunk of its table, and the
standing asking the burst's. The journal's lock is still one, held
around each asking with the guards' check inside it, and checks do not
interleave for that reason alone: the exactness of the check's four
steps and of the landing's taking is argued with it held. Taken apart,
a check and a taking lock the chunks of their pages in the order of
addresses, the numbers of the askings come from one counter, and the
standing asking (`ask_once`, with what the guards learn for a
submission) becomes an object of the burst that `Host::submit` hands
down: that is in `host.rs` (`Batch`, where the asking is begun and
ended) and wants the burst's stamp of S1b. Also left where it is: the
landing still opens guards for the pieces no one view of ours holds
whole (some 100 a frame; `shadow.rs` `card_merge_relay_fast`), and so do
the title's buffer copies (`host.rs`); both could go beside the guards.

Settled, not to be tried again as they are: shaders reading title memory
in place (`VICTORY_INPLACE`: over the bus, slower); a copy queue for the
title's bytes going up (card time, not the processor's); large pages for
the title's sections (the guards need 4 and 64 KiB protections). Windows
has no written bit to read beside `GetWriteWatch`, and none for mapped
views: what is written there is known by guards or told by the writer.

Decisions not taken, and left out of the plan: leaving shader-written
pages on the card and landing them when something touches them (the
census says almost nothing does; a call into the kernel on such a page
fails instead of faulting, and the title's memory then holds
inaccessible pages); and telling a fence when its work is sent instead
of when it has landed (Xeltax/VIcky `e434676d1`: the title may then read
results a frame old; with the present in the line the frame-begin signal
is set 72 to 100 ms after its call, 0 to 19 without, and whether the
title's next frame waits for it is not measured).

Open: what of the bytes sent up the card never reads cannot be seen from
here. (The blocks the refresh check found differing with nothing told,
some 1,500 a run of the prologue, were a hole of the write tracking: see
"S3 as built". The check reads 0 of them since, so the check of every
stage is 0 there, not "not above the base". Who held the guards' lock is
in the same place.)
