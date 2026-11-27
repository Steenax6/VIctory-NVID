# Switches and measurements

Every switch is an environment variable. What is on by default has a switch
that puts the old way back (`=0`); what is opt-in says so. The figures are
from Windows 11 with an NVIDIA GeForce RTX 4070 Ti (12 GB), an AMD Ryzen 7
9800X3D and 64 GB of memory, on the diagnostics build, as medians of the
frame line over a scene. State: 2026-11-26.

| Scene | before 2026-11-25 | 2026-11-25 | 2026-11-26 |
|---|---|---|---|
| Opening cutscene | 1-5 fps | 132-135 ms a frame | about 100 ms (10 fps) |
| First room | 182 ms (5.5 fps) | 89-90 ms | 76 ms (13 fps) |
| Open world | | | 77 to 97 ms, by scene (77 ms by boat and by car, 63 to 94 over 22 minutes) |

## What `scripts\play.ps1` sets

Six opt-in switches, unless you set them yourself (`-Plain` leaves them as
the environment has them; `victory.exe` run by hand has them off):
`VICTORY_ASYNC_SUBMIT=1`, `VICTORY_QUEUE2=1`, `VICTORY_INDIRECT_WINDOW=1`,
`VICTORY_TEXTURE_NO_WAIT=1`, `VICTORY_PLACED_TEX=1`, `VICTORY_NO_WATCHDOG=1`.
Do not combine `VICTORY_INDIRECT_WINDOW=1` with
`VICTORY_CARD_FETCH_INFLIGHT=1` (a device loss upstream).

`VICTORY_NO_WATCHDOG=1` keeps the title from ending itself when its main
thread has stood for about 30 s (a long pipeline wait is enough): other
forks' runs ended at `+0x23cf683` six to eight minutes into the world. It
patches two places of the 1.0.2.1 executable, and only when the build's
hash and the bytes are the expected ones; the log line `title watchdog
hooks` says how many applied (2). Every run of 2026-11-26 had it, the 22
minutes in the open world among them: both places applied, and no run was
ended by the title.

`scripts\play.ps1 -SkipIntro` (`VICTORY_AUTO_SKIP=1`; a number skips that
many scenes) holds A through the opening scene, which the title skips while
A is held for about three seconds: the first room is there some 55 s
sooner. The layer knows a scene by its draws (500 and more a frame for four
seconds), holds A for four seconds and takes the cut to black for the skip;
holds that change nothing end after three.

`VICTORY_AUTO_START=1` answers the start screen's "press A" for you: A is
pressed briefly every 2 s from the moment the game adds its user until it
moves on (at most 30 s). The first-start menus come later and are left to
you.

`VICTORY_INPUT_LATCH=0` turns off the latch that keeps a tap shorter than a
frame pressed for the game's next read.

## On by default

| Switch (`=0`: as before) | What it does |
|---|---|
| `VICTORY_INDIRECT_HOIST`, `VICTORY_INDIRECT_HOIST_ALL` | single indirect records are clamped in their list's prelude, one pass for all records of a buffer, where no shader of the list has declared a write over them (some 1,900 of 4,700 a frame; each used to be a compute pass of ours between two draws) |
| `VICTORY_INDIRECT_WINDOW_SIZES=16,64`, `VICTORY_INDIRECT_WINDOW_KEEP` | windows of records clamped ahead grow 64, 256, 1024 and outlive direct draws whose shaders store nothing |
| `VICTORY_FILL_HOT`, `VICTORY_FILL_DIRECT`, `VICTORY_LATE_COPY` | only reads of the 64 KiB of title memory where shaders were seen to change bytes under a fill wait for flights (one such fill used to make every fill wait); constants there are bound from the buffer's host copy; a title buffer copy out of such memory is done when its submission lands |
| `VICTORY_LATE_COPY_ORDER` (2026-11-26, not measured in the game yet; `0`: as before) | a title buffer copy done when its submission lands (`VICTORY_LATE_COPY`) keeps its place: if a later step of its submission touches its destination (a refresh of a host copy, a fill, a texture, another copy, a range given to shaders) it is made at the submission, behind the wait for the submissions its source needs, and a later submission that reads where such a copy is still to land waits for that landing. Before, those steps read the bytes from before the copy (Marvisak/VIctory pull request #2810 names the order). Both are waits `VICTORY_LATE_COPY` had removed; the line "title buffer copies out of memory shaders write" counts them, and `0` is there to measure what the order costs |
| `VICTORY_CARD_COPY` (2026-11-27; one run each way in the first room) | a title buffer copy out of memory shaders were seen to write (`VICTORY_LATE_COPY`'s case) is made on the card when the title records it, between the two buffers' host copies, in the list's order, and its destination counts as written by the list's shaders: the landing brings its bytes to title memory, and nothing waits for it. Before, the copy was left to its submission's landing, and the next refresh of its destination waited for that landing: 0.43 waits a frame, 18 ms each, 7.05 ms a frame of the submitting thread in the first room; with the switch none. A later step of the same submission that reads the destination on the CPU (a fill, a texture, a title copy, a step of the title's own) has the copy made on the CPU as well, as before. The frame line "made on the card" counts the copies and why others are not; `VICTORY_CARD_COPY_CHECK=1` compares them at their landing |
| `VICTORY_CARD_CONSTANTS` (2026-11-26; in every run since, not measured on its own) | constant buffers of fresh descriptor tables that lie in a host copy in the card's memory are bound there, not filled into upload memory behind the flights (theinkbit/victory-lot9-next ffd04e130) |
| `VICTORY_HEAP_SYNC_FAST`, `VICTORY_DESCRIPTOR_STRIPES` | descriptor records the title copies into its heap are found by the write watch's 4 KiB pages and mirrored in runs; slots have striped locks, so recording threads no longer stand behind that pass |
| `VICTORY_SHADOW_CACHE` | recording threads remember which host copy an address range has |
| `VICTORY_USER_COPY`, `VICTORY_REFRESH_PARTS`, `VICTORY_REFRESH_4K`, `VICTORY_GUARD_4K`, `VICTORY_VIEW_KNOWN`, `VICTORY_ABSENT_ONCE`, `VICTORY_ASK_SKIP_VIEWS`, `VICTORY_SEND_CHANGED` | title memory is copied by the processor (a fault falls back to the kernel's copy), by 4 KiB pages and only where written; guards are not re-armed over other views; unmapped runs are remembered (`VICTORY_ABSENT_FRAMES`, 30); bytes the copy-back itself wrote are not sent up to the card again |
| `VICTORY_TEXTURE_DROP_LEVELS` (`0`, `auto`, `force1`, `force2`), `VICTORY_TEXTURE_PROMOTE_MB`, `VICTORY_RESERVED_REUSE`, `VICTORY_STAGE_LEFT_OUT`, `VICTORY_TILED_IDLE_MS`, `VICTORY_STAGING_POOL_MB` | large block-compressed title textures (1024 a side and more) are made as reserved textures and, past a line of host-copy bytes, without their one or two most detailed levels; their views clamp the level of detail, so the sizes shaders read stay the title's. On a 12 GB card the title's textures did not all fit: some 22,000 waited for room in every run, and which were there differed from run to run (the first room came up lit in one run, dark in the next). Now none is held back there. Reserved textures and staging buffers are reused, not made and released |
| `VICTORY_PLACED_TEX=1` (opt-in; `play.ps1` sets it) | the other host textures of placed textures come out of 16 MiB heaps instead of an allocation each (fxceless/victory-fork pull request #903 with smaller heaps, filled fullest first): making one takes 9 us; about 1 GB of the card stays free inside the heaps after the cutscene |
| `VICTORY_OUTPUT=<w>x<h>` (opt-in, off by default; 2026-11-27, not run in the game yet) | the size the title outputs at, one of its five modes: 1280x720, 1664x936, 1920x1080, 2560x1440, 3840x2160 (the default, and what it renders to without the switch). It writes the size into the title's live settings where it applies them, so the device and the render size follow; for any size but 3840x2160 it also jumps over the check that the back buffers' pool is nearly used up, which that pool's size (made for 4K) fails. The title's own upscaler renders at 2/3 of that size, so the picture gets softer below the title's own sizes. A value that is not one of the five is said so in the log and does nothing; so does a build other than 1.0.2.1 or a place whose bytes differ (the large block is compared by length and BLAKE3, not stored). Log line when it applied: `VICTORY_OUTPUT: output size 1920x1080` (two `hook applied` lines for the two places). Theirs by origin: avatarmanz (the AMD fork), `resolution.rs` |
| `VICTORY_RENDER_SCALE=<0.5..1.0>` (opt-in, off by default; 2026-11-27, not run in the game yet) | the fraction of the output size the title's own upscaler renders at: 2/3 without the switch (2560x1440 for 4K), 0.5 half the width and height, 1.0 the output size itself. It replaces the one constant the upscaler stores its scale from; the picture gets softer below 2/3 (the title's largest render size stays at 2/3 there, so memory is not saved). A value outside 0.5 to 1.0 is said so in the log and does nothing. Log line: `VICTORY_RENDER_SCALE: render scale 0.5`. Theirs by origin: avatarmanz, who ran it at 1.0 only |
| `VICTORY_GROUP_SPLIT`, `VICTORY_GROUP_PIECES=2` | a dispatch that asks for more than 65,535 thread groups in x runs in pieces, each told where its groups start (pull request #976), indirect records too: one of the title's indirect lighting passes asks for about 75,000 and lost its tail, a band of stale blocks along the bottom of the picture |
| `VICTORY_SECTION_MB=2` (`0`: one section a request, as before) | the title's physical pages come from sections of 2 MiB (fxceless/victory-fork issue #1089): with one section of 6.1 GiB mapped a 64 KiB view at a time, every map and unmap of the title waited for the worker of Windows that trades such a section's pages (48 ms a frame of the title's threads in the opening cutscene) |
| `VICTORY_PIPELINE_FIX`, `VICTORY_VIEW_RAW_OWN_COPY` | a pipeline the host refuses as the title described it is made again with its state put right (blending on integer targets and the like: the one mesh pipeline refused in the first room); a raw buffer view whose bytes do not start on 16 gets a copy of its own bytes instead of reading zeros |
| `VICTORY_CARD_FINE`, `VICTORY_CARD_WIDE_HASH`, `VICTORY_CARD_SYNC_BATCH`, `VICTORY_MERGE_PAGES`, `VICTORY_FAST_MERGE`, `VICTORY_WIDE_COMPARE`, `VICTORY_BURY_ASIDE` | shader writes to copies in the card's memory are found per 4 KiB block (was 64 KiB), fetched as runs and merged where they differ; title bytes under flights go up a page at a time where shaders never wrote |
| `VICTORY_EXCHANGE_PARITY` (2026-11-26; not yet measured in the game) | a 64-lane compute program run as two 32-lane host waves exchanges values through two pairs of group-memory dwords per console wave, taken in turn, so each exchange costs one group barrier instead of two (the second only kept the dwords from being rewritten before every thread had read them). Two more dwords of group memory per console wave. Changes translated shaders: the shader cache keys on it |
| `VICTORY_MERGE_BLOCK_RUNS=32` (2026-11-26, not measured in the game yet; `0`: as before) | a 4 KiB block the title rewrites in more than 32 places while a submission sent ahead may write its buffer goes up as one run, and where a page still has more changed runs than a submission sends (`VICTORY_MERGE_RUNS`, 512) its fullest blocks do too: the submissions in the air are no longer waited for over it. That wait came once a frame in the first room, 12.5 ms of the submitting thread: the title's per-draw constants, rewritten in blocks a shader had once written, differ from the old ones in 42,700 runs of some six bytes a frame. Raising `VICTORY_MERGE_RUNS` instead turned the wait into 42,000 copies a frame and was no faster (74.6 against 78.6 ms a frame). The line "census, title bytes under flights' shader writes" of a diagnostics build counts the blocks sent so that a submission then in the air turned out to have written: 0 says the rule lost nothing |
| `VICTORY_SYNC_BARRIER` (2026-11-26; in every run since, not measured on its own) | a pipeline sync in the title's list waits for every shader write before it (one unordered access barrier), where it did nothing: dispatches that read what the one before wrote no longer overlap it (avatarmanz/VIctory-7900XTX `0b5f191b8`, Marvisak/VIctory pull request #1228). The log names the first eight syncs; `=0` is there to measure what the barriers cost on the card |
| `VICTORY_WAVE_LOCAL_FIRST` (2026-11-26; not yet measured in the game) | a 64-lane compute program whose only wave operations are the allocation idiom (the first lane reserves the wave's count on a counter, each lane takes the base plus the lanes below) runs each 32-lane host wave on its own, without group barriers, before the forms whose two host waves meet at barriers to exchange: each half reserving its own slots is exact. Not where a first-lane read takes a value that differs between lanes. A light-probe allocation program had 33 barriers and 3.7 ms of card time a frame for 272 instructions. `VICTORY_WAVE_WHY=1` prints `WAVE64-MEETINGS` and `WAVE64-LOCAL-FIRST` lines. Changes translated shaders: the shader cache keys on it |
| `VICTORY_KEEP_SHOWN=3`, `VICTORY_TEXTURE_AWAY_MS=3000` (2026-11-27; three picture runs a side with scripted movement in the open world) | a slot of the view heap that showed a texture and is written with a view of another texture (or of the same in other memory or another size) goes on showing the old one until the texture pass or a draw makes the new one, instead of nothing; `1` keeps only a texture moved, `2` also one in another size, `3` any texture of the same dimension. A slot whose texture's memory is found unmapped is made null only once it has stayed so for 3 s. The texture pass runs once a frame and the title writes views after it: in motion 1,000 to 4,000 slots a second read zeros until the next frame or later, and a glass case flipped between white and its contents from frame to frame. Pictures that differ from both neighbours for one picture, of the quiet ones: 13, 5 and 14 without, 2, 1 and 3 with, none of those an object's look. The frame line "slots that showed a texture made to show nothing" counts them by why and how many frames they stayed null |
| `VICTORY_EARLY_DEPTH` (2026-11-26; not yet measured in the game) | a pixel program the console tests depth and stencil before (`[earlydepthstencil]`, `DEPTH_BEFORE_SHADER` in its `DB_SHADER_CONTROL`) gets the same order on the host (`ForceEarlyDepthStencil`): before, a host ran such a program for hidden pixels too where it writes views or discards, and kept those writes (Marvisak/VIctory pull request #2085). Part of the translation: a change re-translates every program |

## Title buffers in the card's memory

The host copy of every title buffer is in the card's memory
(`VICTORY_CARD_BUFFERS_MIN_KB=64`, `VICTORY_CARD_BUFFERS_MB=1280`; before,
only those of 1 MiB and more, `=1024` and `=1024`, and a shader's every
load and store on the others crossed the bus), and the card's hashes of a
copy cover the title buffer's own bytes, not its whole 64 KiB
(`VICTORY_CARD_HASH_LEN`, on; `=0`: the whole copy). The first room went
from 87.8-89.0 to 77.2 ms a frame, the card busy 56 % of the time instead
of 74-78, the frame line's copies from 17.5 to 26 ms, the picture the same
(fxceless/victory-fork c36971b measured 68.0 to 49.8 ms of card time a
frame on an RTX 4070). A line with the frame-rate line says how many copies
went where, by size.

`VICTORY_LAND_BESIDE` (2026-11-26; on, `=0`: as before; the first room
75.5 to 68.7 ms a frame, preparing 55.1 to 42.2 ms, copying back 36.0 to
29.8 ms, the faces clean, and with `=2` no write of 2,580 a frame there
and 3,856 a frame in the open world's bedroom that was not at the title's
address):
what shaders wrote goes into title memory through a view of the layer's
own of the same pages, beside the guards, wherever the title's address is
a view of physical pages; the guarded blocks under it count as written
and stay read-only. Until now a landing opened every guarded block it
wrote into and the block's next check guarded it again: in the first room
all 781 guarded blocks opened a frame were opened by landings, none by
the title, with 3.8 ms a frame of the landing thread and 4.6 ms of the
submitting thread in changes of protection (census and profile of
2026-11-26). `=2` also reads every such write back through the title's
address and counts those that are not there. With it the line "landings
beside the guards per frame" says how many writes went that way, and the
census's "guarded blocks opened by the layer's own writes" should fall to
nearly nothing (avatarmanz/VIctory-7900XTX `0d5e4c8cb`, Marvisak/VIctory
pull request 3069; this tree's own code).

The refresh by event (2026-11-26; the way since its check counted 0; the
old way, asking every time, is gone; `VICTORY_REFRESH_EVENTS=2` is the
check): a
copy in view memory is not asked about again while nothing happened in
its memory since it was last brought up to date. What can make a guarded
block count as written (a write that faults, one of ours that opens or
marks it, a new mapping, a page the journal notes) raises a count for
its 4 MiB of title memory, with the lock held under which an asking
sees it, and a copy remembers the counts of the range it last brought up
to date. A range that held blocks not committed is asked once in every 30
frames all the same (`VICTORY_ABSENT_FRAMES`): a block committed with
nothing told is found by looking, as before. `=2` is the check: every
refresh is asked as before, and each one that `=1` would have left out
and that found bytes to copy is counted and named ("a refresh found bytes
to copy where no event was counted", with the block the guards reported
and what it was to them); the line "refreshes by event" must say 0 of
those, and it did in the prologue and at the open world's entry. Copies
in private memory are asked as before: only the kernel knows what was
written there.

The check's first form found 7 such refreshes in the prologue's 205 s and
2 in the open world's first 130 s. It decided that nothing had happened
before the asking took its locks and counted whatever the asking then
found; a write whose fault or mark was counted in that moment was a find,
though `=1` asks for it at the next refresh. It decides with the locks
held now, and the line counts the others apart ("whose event was counted
between the refresh's look and its asking"). Reading the code for a
writer the events miss found two places, both put right: a mapping's
count was raised before the guards' lock was taken (a refresh between
the two remembered the count and not the mapping), and a copy whose
memory turned out not to be watched could be left out. The runs after
that (s_y2_p_check, 205 s, and s_we_y2_check, 130 s) had 0 finds in
every line, and 0 of the others too: so the 9 were not the check's own
moment, as first supposed, but went with the event side as it is now
(the mapping's count, or the ranges with absent blocks, which are asked
once in 30 frames and whose finds at that asking were not told apart
then). The line now counts those askings as well: the ones that found
blocks that had been absent, which is what they are for, and the ones
that found other bytes no event had announced, which must be 0 too.

Measured with both this and the landings' taking below against the old
way, same build, back to back in the first room: preparing 36.4 against
40.6 ms a frame, the copies read 70 against 89 MB a frame, the frame 67.8
against 66.6 ms. The work comes off the submitting thread and the frame
did not follow then, because the title's render thread and the
submitting thread take turns (see "Where the time goes"); it is work not
done, with checks that found nothing, so both are the way now.

The landings' taking (2026-11-26; the way; `VICTORY_LAND_TAKEN=2` is its
check, which counted 0 of 743,252 pages in the prologue and 0 of 181,911
at the open world's entry): what shaders wrote into private title memory (which the kernel
write-watches; no guard there) set the written bits of its pages, the
next refresh took them for the title's, and the landing's own copy read
the pages it had just been given: 22.8 MB a frame in the first room,
read and found equal on the submitting thread. With the switch a landing
that fetched 512 KiB or more of such memory takes the bits right behind
its write (one kernel asking for the pieces that lie together, the
journal's lock held) and notes the pages in the journal at once, so
every other copy over them learns of the landing as before; its own copy
compares each noted 4 KiB page with its CPU side on the landing thread,
and a 64 KiB page whose noted pages all hold the copy's bytes counts as
taken by that copy. A write of the title's behind the taking sets its
bit again, one before it is in the bytes compared. `=2` takes and notes
and marks nothing: the refresh reads as before and first compares each
page `=1` would have passed by; one with other bytes, no bit and no
later note is a find (0 is what `=1` needs; run it with
`VICTORY_REFRESH_CHECK=1`, which samples every copy's refreshes against
title memory). The line "landings' written bits taken in private title
memory" says how many pages a frame the copy no longer reads, and the
guards line's "equal to the CPU side, not sent up" should fall by about
that. Each taking costs a kernel call of 30 to 70 microseconds on the
landing thread.

The same for view memory (2026-11-26; no switch of its own,
`VICTORY_LAND_TAKEN=2` is its check too): there the kernel keeps no
bits, the landing writes beside the guards and tells them, and the next
refresh's check reported the blocks to the landing's own copy as to
every other. The landing takes the guards' marks of its blocks right
behind the telling (the journal's lock held; noted in the journal at
once), and its copy compares and counts as above. Only blocks a check
could note clean without a call are taken: one a store of the title's
opened meanwhile, or a view of whose pages became writable, is its next
check's. The line "landings' marks taken in view memory" counts the
landings, blocks and kernel pages (some 360 blocks and 5.7 MB a frame in
the first room); what the copies count taken is in the line above it,
with the pages of private memory. For the check, "written again since"
is asked of the guards where the kernel's bit is asked for private
memory.

To check both in a run: `VICTORY_REFRESH_EVENTS=2 VICTORY_LAND_TAKEN=2
VICTORY_REFRESH_CHECK=1`; the lines "refreshes by event", "landings'
written bits taken" and "refresh check, since the start" must count no
finds.

The journal's answers by the copy's stamps (2026-11-27; the way, no
switch; `VICTORY_JOURNAL_CHECK=1` is its check, diagnostics builds only):
a refresh asked the journal for every noted page of its range and threw
away nearly all of them: in the first room 103,000 entries a frame, 1,500
acted on, and 4,400 of 5,100 answers with nothing to act on (the open
world's entry alike). The copy now hands the journal what it has of each
page (its stamp, a part's, or nothing to ask for a page it never took),
and the journal lists only the pages noted later than that or than the
answer's own number; each group of 64 pages (4 MiB) of its table keeps
the highest number noted there, and a group none of whose pages is newer
for the copy is passed by. An answer with nothing newer allocates
nothing. The check walks every entry of each answer's range as before and
counts the answers that differ (line "journal answers to refreshes", "0
differed" is what it needs): 0 in 11.2 million answers of the prologue,
10.6 million of the opening cutscene and 4.0 million at the open world's
entry.

The landing's merge as jobs (2026-11-26; the way; `VICTORY_WIDE_MERGE` and
the merge one run after the other are gone; `VICTORY_LAND_SAMPLE=1` is
its check): what shaders wrote to copies in the card's memory is merged
into title memory as jobs of 256 KiB of one copy each, on the wide threads
and the landing thread, all done before the landing goes on; the copy's
side lock is one per 64 KiB piece. A run that changed almost whole with
nothing of it waiting to go up, in eight stretches of changed bytes or
fewer, is written by those stretches without reading title memory first.
The timeline names each job (`job>`, `job<`). `=1` samples, at the
telling, the last stretch of every eighth run: the bytes it landed in
title memory and the copy's CPU side must be the same; the line "the
landings' merges as jobs" counts those that are not (0 is what the merge
needs).

`VICTORY_REFRESH_CHECK=1` (a diagnostic, slow): after every refresh the
first 256 bytes of each 64 KiB block of it are read from title memory
and compared with the copy. A block DirectStorage is filling differs for
as long as the read takes (it writes the pages through a view of the
runtime's own and tells the guards when it is done), and that is as it
should be: the open world's entry had 1,024 blocks "stale three times in
a row" in 130 s, all in 4 MiB windows, while 14,000 such reads came in.
So the line "refresh check, since the start" tells the looks apart:
those under a read the title has submitted and that is not done; those
with none; blocks that differed three times in a row with none (named,
with what the guards hold of the block); of them, those that matched
again once a write behind the guards was told there and those that
matched again with none told; and blocks that differed for two seconds
with nothing told. Only the last are copies that keep old bytes; each is
named a second time, with what the guards hold of every other view of
the block's pages and whether that view is writable. Until 2026-11-26 a
run of the prologue had some 1,600 of them, every one a block with a
writable view of its pages elsewhere (the title's own, opened for a read
or by a fault and never guarded again); a check that notes a block clean
now makes every view of its pages read-only first, and the count is 0
(`docs/architecture.md`, "S3 as built").

A read of the kernel's into title memory (`ReadFile`) now says when it is
done, and blocks a check guarded again while the kernel was writing count
as written once more: a file read without buffering goes on into such a
block with no fault, and the host copies there kept the bytes of the
moment they looked. The line "reads of the kernel's into title memory
that finished" counts those blocks; no switch.

## Pipelines

`VICTORY_PIPELINE_EARLY_WORKERS=<n>` sets how many workers may make the
pipelines the run before drew with, before anything asks for them (default:
an eighth of the logical processors, at least one); pipelines a draw asks
for take every worker. Each translation keeps its pipeline library in
`pipelines\<16 hex digits>\`, filled at first from the libraries of the two
translations before and of the one file earlier builds kept in
`pipelines\` (left as it is; it can be deleted once a run has passed the
loading screens).

`VICTORY_PIPELINE_WORKERS=<n>` sets how many threads build pipelines that
are not cached yet (default: half the logical processors, at most 8). With
16 threads, 14 cut the intro's time below 1 fps on an empty cache from 55 s
to 6-12 s; 16 was no better than the default.

`VICTORY_PIPELINE_WAIT_MS=24`: a shorter per-frame wait for pipelines still
being built (objects can go missing meanwhile). It bounds draws only: a
dispatch stands for its compute pipeline until it is made, since a
compute pass the title runs once is not run again (fxceless/victory-fork
pull request #1312); the line "compute pipelines a dispatch stood for"
says how many and how long.

## Textures the game reads back

`VICTORY_TEXTURE_BACK` (on by default): textures the game's shaders write
and its own code then reads on the processor are copied back into the
game's memory once the list that wrote them has run, in the console's
layout. The water's wave passes write such maps (256x256 R32_FLOAT and
R32G32_FLOAT, 64x64 R32G32_FLOAT, 128x128 R16G16_UINT), and boats float
on what the processor reads there; without the copy they sat in a pit in
the water and the first boat mission failed (Geckomacabre/vx-layer
`eee5667d7`, fxceless/victory-fork pull requests #999 and #1152). Only the
levels and layers the shaders' views expose are copied; a list that writes
none of these textures costs nothing.

The value is the list of kinds, `<width>x<height>f<format number>`: unset
or `1` is the built-in `256x256f41,256x256f16,64x64f16,128x128f36`, `0`
none, a list of its own replaces it, and entries with `+` or `-` add to or
take from it (`+512x512f41,-64x64f16`). The log line "textures shaders
wrote, copied back into title memory per frame" counts them, and the first
texture of each kind says its swizzle mode. Textures in the pipe/bank XOR
swizzle modes are copied back too (the 256x256 R32_FLOAT map is one, mode
27); one in a mode `victory-formats` has no address pattern for is counted
as passed over (the log says "could not be tiled e=Unsupported" once).

In 22 minutes from a save at the first boat mission: 287,148 textures
copied back, about 18 a frame (1.4 MB), none dropped and none passed over;
the boat floats, casts off and steers.

## When the card's memory is full

Levels are left out first (`VICTORY_TEXTURE_DROP_LEVELS`, above).

`VICTORY_TEXTURE_SPILL` (on by default; `=0`: they wait) is the last step:
a texture that finds no room on the card even so is made whole in the PC's
memory that the system lets the card read over the bus, instead of waiting
for room (and being missing: black skin and walls in the open world). A
texture there is read more slowly, but it is there (fxceless/victory-fork
pull request #601, NitricAcidDev/VIctory-Omega-Fork `4237574084`). It goes
back to the card when it is let go as idle and next made while the card has
room. `VICTORY_TEXTURE_SPILL_PERCENT` (50; 10 to 90) is the share of the
system's budget for that memory the textures may fill; 4 GB of the PC's
memory are always left free. The log line "textures that found no room on
the card" counts those made there and the passes that found no room in
either.

Spilling before leaving levels out was measured and dropped: the opening
cutscene took 170 ms a frame against 106, with the same picture. In the
prologue (205 s, to the first room) 11,356 textures were made there instead
of waiting, 2,391 of them alive at the end (606 MB), beside 2,346 alive
without their most detailed levels (2.8 GB of the card not taken); none was
held back. In the open world (eight minutes from a save at the first boat
mission) 22,389 textures were made there, 4,368 of them alive at the end
(1.4 GB), and none waited for room; the frame time was that of the
session without it (78.9 against 77.0 ms).

`VICTORY_WAIT_RESIDENT` (on by default): new heaps for textures and
committed host textures are made resident, and waited for, before anything
names them. Lists already on the card read the descriptor heaps as they
are, and a slot showing a texture in memory the system had not mapped yet
was a page fault for them (avatarmanz/VIctory-7900XTX `0322b39ab`, a device
loss about every 15 minutes there). The log line "new texture memory made
resident" says how often and how long it waited (1,491 times and 1.2 s in
all in the prologue run).

`VICTORY_RELOOK_MS=20000`: how often a texture picked by index whose memory
was not there is looked at again when nothing tells that it came (the
runtime's mappings and the writes behind the guards do tell); slots looked
at before and fresh ones are served in turns (fxceless/victory-fork pull
request #1246).

## Textures written by DirectStorage

The game maps physical pages that held other data, writes a texture's
view, has DirectStorage fill the pages and waits for the read's status.
A texture picked by index that was uploaded in between holds what the
pages held before, page by page. Every write that no guard sees
(DirectStorage into physical pages, at each address showing them, and its
reads to plain addresses), and every copy of the game's that the layer does
in the game's memory itself (buffer copies at their submission, or when a
submission sent ahead lands, and `CopyMemoryToMemoryX`), is numbered when
its bytes are there, and every
upload of the texture pass keeps the number its read began at: one that
read before a write under its texture is stale until uploaded again.

The stale slots are counted, in all and at each draw of the game's
decoration bake (the log line "bake sources, since the start": bake draws,
the ones with stale slots, those slots by what wrote them, and slots under
a read still running at a bake). For two runs the texture pass uploaded
them again ahead of every bake draw (`VICTORY_BAKE_WRITES`, some 25,000
uploads a run): the head was baked with noise as before, and a character's
body came apart in the run that did it, so that is gone.

`VICTORY_PAGE_WRITES` (unset, `1` or `all`: the guarded blocks under such a
write are opened, so host copies of buffers and of textures read through
tables take its bytes, and the stale slots are due at once at every pass;
`guards`: the first alone, as it was by default while every upload brought
a whole texture; `slots`: the second alone; `0`: neither). A write, and a
mapping, is taken for the slots the spans of the views name (a texture
ends where the next view starts) and also for a slot whose view holds its
bytes though another view starts between: the first bytes of the texture,
up to the end of the level the view starts at. The line "writes behind the
guards" counts the writes, the slots under them, the stale ones, the stale
ones now, the slots that took a tell by what their view holds, and the
slots the spans named whose view held none of the tell.

What the texture pass does is counted (no switch; lines every five
seconds). "texture uploads by reason" says for every upload what made its
slot due (the first upload of the memory, the slot's view written again,
more of the memory mapped, a look that found it changed, and one row for
each who told of the memory: DirectStorage into pages or to plain addresses,
the layer's copies, texels and shader writes copied back, reads through the
kernel, mappings) and what its read found against the memory's last upload,
level by level: megabytes of levels that changed, that were the same and
that were all zero, and the uploads none of whose levels had changed.
"texture uploads as recorded" has the host copies made anew, the uploads of
textures another texture's view starts inside of, the memories by how many
times they went up and the addresses with several descriptions. "textures
found changed by a look" splits those by what was known of their memory
since the upload's read (a numbered write that made the slot stale, a tell
that lay in the texture's span and was taken for another texture's, a store
the guards saw, blocks nothing guards, nothing). "told to the texture pass"
counts the tells by who told, "the texture pass per frame" is the pass's
time by part, "the texture queue by what its slots wait for" the queue
(and how many slots its index by due frame did not hold when it was
compared with a walk over all of them: 0, or a pass was handed too few).

A texture goes up level by level. Its memory is told apart at its levels
(the block of the smallest ones, then each level), and of each part the
upload asks three things: is it the game's (every level a view shows; a
more detailed one only where no other texture starts before its end), are
its pages there, and does the host copy hold it as it is (what the host
copy holds is noted with it). Only a part that is the game's, is there and
is other than what the host copy holds is detiled and copied; one that is
not the game's is not read at all, and one whose pages are not there is
zeros on the host, once. "texture uploads level by level" has the
megabytes read, not read for either reason, detiled, not detiled because
the host copy held them, and copied, and the uploads that copied nothing.
It also has what would show a level gone black: the levels that went up as
zeros because nothing of them was there and how many of those got their
bytes since (a level with only some pages there takes the bytes that are
there), and the times a level a view may show was left with nothing on the
host when an upload was decided though other levels of its texture were
brought (0: none may be; the first eight are said in a warning with the
texture's memory).
The megabytes in "texture uploads by reason" are those read.

A texture draws read through tables keeps its staging buffer and is
checked at every submission that samples it. The check reads its memory
when the guards saw a write or the runtime told of a mapping; at its first
three checks; on its turn every eighth frame, once in that frame (the hash
for what the guards do not see); and at every check while it is taken for
one written behind its guards, which ends after 64 checks that found it
unchanged. A texture with pages missing, and one made level by level of
which the game does not hold every level, is read again 30 frames after a
read at the earliest unless a write is seen or a mapping told. "textures
read through tables, MB of title memory their checks read per frame" has
those megabytes by why the check read.

Whether a write would have been known without the read is counted too, to
decide if the reads of unchanged textures can go: "what the journal, the
guards and the write watch knew of the parts of textures an upload of the
pass read again" has, by the upload's reason, the parts changed with
nothing seen (must be 0 before anything is left unread on their word), the
parts changed and seen, and for one upload in sixteen the megabytes of
unchanged parts with nothing seen (what leaving them unread would save).

For a texture both the game and the card write (the game's bake compresses
its result on the card and copies it, level by level, into textures it also
has views of), what was written last is what the texture holds, by the
order the lists are submitted in: the copies of an upload from the game's
memory are decided and recorded when its list is submitted, into the list's
prelude, and left out for the levels the card wrote last. They were
recorded while the game's list was, and an upload recorded before such a
card copy ran after it when its list was submitted later: 27 times in one
run, whose textures then showed what their memory had held before (noise
baked onto a head in about half the runs). The same holds for every view
of that memory: a view the game placed where a texture the card writes
starts (another format of the same family, the same texels and levels) is
a texture of its own here and is brought what the card wrote by a copy on
the card, in the next list's prelude, and no upload from memory goes over
it afterwards. The line "textures the title fills that the card copied
into" counts the levels copied from memory over what the card had written
last (0: none can be), the levels uploaded ahead of a card copy of their
own list (the copy runs after, nothing is lost), the uploads withheld, the
times the game was found to have written such memory after the card (its
bytes then go up again), and the placed views in such memory: how many
start where the texture does, have a host copy, are shown by a slot, were
fed from the card, or do not fit it.

At the game's Submit the addresses a read into physical pages will write
are known; until the read is done the texture pass leaves a texture there
for the next frame instead of uploading the bytes from before and again
after the read ("due slots left for after a read" in the same line).

## Light where no rays run

`VICTORY_AMBIENT` (on by default; `=off`: the game's own output): the game
fills its indirect light by tracing rays, which do not run while the
structure builder is off, so whatever the sun does not reach directly is
black. A small pass of ours replaces the screen probes' resolved light
with a stand-in: a level times a warm white, shaded in creases and under
objects from the scene's depth, which the game's lighting then applies
with each surface's colours. The reflections, which without rays mirror
the open sky indoors, are held to half of that light, and the program that
asks the probes for rays is left out. The value is the level: unset 0.3
(the level that came within 2 to 4 display-luma steps of console footage
of the prison, Marvisak/VIctory pull request #2619; 0.6 looked balanced by
eye on an RTX 4070, fxceless/victory-fork pull request #1093), a number of
its own, or `sky` for a level that follows the sky the game draws (night
to noon; Belobug/VIctory-Belobug-AMD). `VICTORY_AMBIENT_TINT=r,g,b`,
`VICTORY_AMBIENT_SHADE=strength,radius,near,tan` (strength 0: no shading),
`VICTORY_AMBIENT_REFLECT=most,least` (parts of the level; `0`: the
reflections as the game has them) and `VICTORY_AMBIENT_SKY=gain,least,power`
tune it; `VICTORY_AMBIENT_PROGRAMS=<hash>=resolve,...` names the programs
for another build of the game. It is off whenever `VICTORY_BVH_BUILDER=1`.
The log says "ambient light: stands in for the indirect light while no
rays run" once with the level, then the reflections' and the left-out
program's lines.

## List calls written down (`VICTORY_THIN=1`, opt-in)

`VICTORY_THIN=1` (with `VICTORY_ASYNC_SUBMIT=1`, whose line of
submissions it uses; off by default until it is verified) makes the title's
list calls write their arguments into a stream of words, with copies of what
D3D12 reads at the call (arrays and structures passed, the records of render
target and depth views). Threads of ours translate the streams through the
same list methods as before, on lists of ours, starting while the title
still records (every 1,024 words) and several lists at once. A submission
stands in its queue's line until its lists are translated and is then cut at
its waits and sent as before. Ported from avatarmanz/VIctory-7900XTX
`crates/victory-gpu/src/list/thin.rs`. Why: the title records a chain of 19
lists in one job, 36 to 50 ms of our translation on one thread, and its
render thread waits 22 to 42 ms for that job before it submits and presents.

What is read later than before and why that is what the console's card sees
is listed call by call at the head of `crates/victory-gpu/src/list/thin.rs`.
A title object whose last reference goes while recordings are not translated
is freed once every recording begun before it is (at most 2 s later). The
present, and a submission of ours that is waited for, first wait until every
submitted recording is translated.

- `VICTORY_THIN_THREADS=<n>` (6 unless told, 1 to 16): the translating
  threads. `VICTORY_THIN_PRIORITY=<-2..2>` (0, normal, unless told): their
  Windows thread priority.
- `VICTORY_THIN_CHECK=1` (diagnostics build): for every draw and dispatch the
  first 32 bytes of each bound descriptor table and root view are kept at
  the call and compared at translation. The line "thin check" counts as a
  difference only a table whose first descriptor changed: a table's records
  are what the path in the call reads at the draw. A root view's bytes are
  read by both paths only at the submission (its host copy is refreshed, or
  it is filled into upload memory, then), so its changes are counted apart
  as none. The log names the first 16 tables and 4 views with the words at
  the call and at translation. Run t_thinchk4 (2026-11-26), loading and first
  room: tables 0 in every line, root views 2.2 to 7.7 a frame, each a
  block of the title's own constants that was still zeros at the call and
  filled by the title afterwards (the old path would have read zeros had it
  read them at the draw; it never did).

The line "list calls translated by threads of ours per frame" gives the calls
and the threads' time, lists, pieces, recordings let go of unfinished, the
title objects whose freeing waited and those freed past a recording held open,
and how long presents stood for translation (the timeline's `thinwait`
events; each translated piece is a `thin>`/`thin<` pair).

First room, 2026-11-26 (indications: other work ran on the machine): the
render thread no longer waits for the recording job (no wait of 3 ms or more
in the timeline, 22 to 42 ms before); the title's job threads write their
calls in a few milliseconds and six threads of ours translate about 42,000
calls a frame in 200 ms summed (about 30 ms each). The frame is now the
submitting thread's: it sends 60 to 64 ms of a 61 to 66 ms frame and idles
under a millisecond, and the present stands for translation under a
millisecond. So the frame did not get shorter (71.5 and 66.8 to 72.3 ms
against 70.9 without), and its sending got slower (preparing 51 to 54 ms
against 45). What shortens the frame from here is the submitting thread's
work, or the present no longer waiting for it.

With thin recording more uploads are "withheld at submission because the
card wrote last" (13 to 43 a run against 1 to 11) and fewer "not asked for"
(both counts of the line "textures the title fills that the card copied
into"). The title's lists are now translated while the submitting thread,
which notes the card's copies into such textures at their submission, is
still behind with the lists before them: a list asks for an upload where
before it found the card's copy noted and asked none. Whether to copy is
decided at the submission in either case, in the order of submissions, with
title memory compared for writes since the card's (`CardState::decide`),
so a withheld upload is one the path in the call did not make either, and
none that was needed: the line's count of subresources copied over what the
card wrote last stays 0, and of the title writing after the card 0.

To become the default and the path in the call to be removed: the picture of
the first room and of the open world as without it; the always-on invariant
lines at 0 and the warnings of a run of the same kinds and counts; the thin
check at 0 over the first room and an open-world drive; no "freed past a
recording held open"; a measured gain on a quiet machine.

## Where the time goes

On 2026-11-25 the first room (90-100 ms a frame) was bound by the card,
busy 73 % of the time: a timing run (`VICTORY_GPU_TIMING=1`) put the
title's own commands at 62 ms a frame on the card (3,850 dispatches, 5,100
indirect draws) and the passes of ours at 14 ms. The card's time is not
spread evenly: about 60 single passes of the title take some 34 ms of it
and the other 8,700 commands share 27 ms. Why those passes are slow as
translated is open, and it is where the next gain would be.

Since the title's buffers are in the card's memory the same room takes
78.6 ms with the card busy 54 % of the time: the card alone no longer sets
the pace there, and the title's recording threads and our submitting
thread share the rest.

On 2026-11-26 the first room takes 66 to 69 ms. A profile of it (35 s,
1000 looks a second) shows no thread saturated: the submitting thread is
on a processor 72 % of the frame and waits 26 %, each of the title's five
recording threads is on a processor 48 % and waits 46 %, the thread that
lands what shaders wrote 36 %. The title's render thread and the
submitting thread take turns: about 30 ms the render thread waits for its
recording (of the recording threads' 171 ms summed, 144 are our
`State::bind`: what a draw may read 41, views made at the draw 36, clock
reads 12) while the submitting thread idles 15 ms for want of work; then
about 36 ms it stands in its present, blocked on the queue's lock until
the submitting thread has sent the frame, while each recording thread
idles 28 ms. The order is inferred from idle times that match, not seen
event by event. So work taken off the submitting thread no longer
shortens the frame by itself (4 ms less preparing left it where it was);
what would is letting the two turns overlap (the present in the line of
submissions; `VICTORY_ASYNC_PRESENT` was measured no faster when the
submitting thread was the whole frame, and it faulted), the heap sync
started at the title's submit instead of in the sending turn (9 ms), and
less work in `State::bind` on the recording threads.

The same day a timeline (`VICTORY_TIMELINE`) put events to that order,
and the recording was taken up first. What the runs of that evening
showed (single runs of the first room with a profile or a timeline, on
a machine other work ran on: indications, to be timed side by side):

- The list that took 15 ms in one frame and 30 in the next was the list
  the pass over the heap's picked buffer views happened to run in: the
  pass ran inside the first list call that bound a shader picking by
  index after a frame began, 12 to 45 ms of that call, and in every
  other frame that call was in the chain of 19 lists the title's render
  thread waits for. The passes run on a thread of their own now (what
  waits for them: below).
- A draw made ready again what its list had made ready before: 140,000
  views a frame asked for, 122,000 of them known. A list remembers
  (`Seen` in `draw.rs`; `VICTORY_SEEN_CHECK=1` makes everything ready
  as before and counts the differences, none in 100 million views), and
  makes each buffer view of its tables once and copies it into the
  table slots (`KeptViews`).
- Smaller things on the same path: a moment is the tick count and not
  the performance counter (12 ms a frame of the recording threads), a
  slot made from the record that is there is answered without its lock,
  what a draw's slots showed before is retired once a draw, a
  recording's own maps hash with a multiply, a draw allocates nothing
  for its tables.

The chain is recorded in 16 to 18.5 ms where it took 33 to 52, its
longest list in 6 to 7 ms (15, or 30 with the pass); the recording
threads are on a processor 129 ms a frame summed where they were 170,
of which `State::bind` 84 where it was 145. The frame followed only a
little (60 to 62 ms a frame in the last seconds of that run, with no
profile running, against 65 to 68): the title's present now stands
through most of the frame, and the submitting thread is on a processor
78 % of it. What the recording still waits for: the guards' lock in the
title's own write faults (18 ms a frame of the recording threads in
those runs, 8 in the profile of the afternoon).

### Shaders that pick buffers by index: when their views are made

A shader that picks a buffer from the view heap by an index it computes
reads the slot when its list runs on the card, and nothing says which
slots it names. So a list may go to the card only with every buffer view
the title wrote before it submitted the list made, and the bytes behind
it brought up to date; and no view may be made under a list that is
prepared or on the card, unless that list cannot name it.

The pass that makes those views ran once a frame, started by the frame's
first bind of such a shader, its refreshes steps of the list that
started it. Counted at one entry into the open world (150 s, 2026-11-27):
945 lists with such binds were prepared before the list that carried
their pass's refreshes; 19,609 while the next frame's pass was making
the views their own frame had written (a slot they had read as nothing
became a view over bytes not copied yet, under a list in the line or on
the card); and every one of 50,669 came up with views written and not
made. The evening before, one world entry in 16 had lost the device with
the card standing in a draw; that this was the cause is not shown. What
is shown is the first room: one frame every few seconds drew single
objects with another's material data (hair light grey, the bedding with
the window's wire fence): 6 such pictures in 66 s on the tree of
2026-11-26 (none before that day's work), 3 with every list of a pass's
own frame behind its refreshes and the change of the next section, none
in 1,988 pictures with what follows. One run each.

Now every submit call of the title's whose lists bind such a shader asks
for a pass of its own, its submission waits for that pass where it is
prepared, and what the passes found is brought up to date ahead of its
lists. Passes are also started ahead, where a list first binds such a
shader and where such a list is closed; they run one after the other on
their own thread. Why no list in the line or on the card can name a view
a later pass makes is written at `Passes` in `draw.rs`. A diagnostics
build says with the frame line how many submissions stood for their
pass, whether any came with no pass asked for, how many buffer views
were written and not made by then (those written after the submit call),
and how many buffer views that read title memory a pass found no host
copy for (what a list can still name and read nothing of).

What it costs, in the open world (single runs of 150 s from one save):
some 30 passes a frame take 25 to 30 ms of their thread, submissions
wait 3 to 9 ms a frame for them, and the frame was 67 to 76 ms (72 on
average) where it was 69 to 79 (mostly 72) with the pass once a frame;
73 to 78 in two runs a build ran beside. In the first room: 10 to 20
passes, 22 to 27 ms of their thread, 4 to 8 ms waited, the frame 59 to
66 ms as before. In 22,797 submissions of one world run none came with
no pass asked for, and the passes found a host copy for every buffer
view that reads title memory. With the passes made on the submitting
thread itself, where a submission is prepared, they took 18 to 23 ms of
it and the frame in the open world was 88 to 96 ms: not kept.

What stays as it was: the bytes behind views made earlier are looked at
once a frame, so what the title writes into such a buffer between two
submissions of one frame is seen a frame late; and a view whose memory
had no host copy when its pass looked is made when all slots are looked
at again, every 16 frames.

### What is let go while a submission stands in a line

`retire` kept a host object for the submissions started when it was let
go, and dropped it at once if none was running. With
`VICTORY_ASYNC_SUBMIT=1` a submission the title has made stands in its
queue's line until the submitting thread takes it, and has no number
until then: what was let go meanwhile did not wait for it. Such an
object now waits until every submission that stood has begun, and then
for the submissions started by that moment (`AWAITED` in `host.rs`). In
the first room that is, per 5 s, some 90,000 objects that were dropped
at once before and 600,000 that were kept only for the submissions
begun (about one a draw: by that number, a draw's table slots letting go
of what they showed last), and 15 to 50 let go by the idle sweeps.
Whether a list that stood ever named one of them is not known; the first
room's single wrong frames were still there with this (3 pictures in
66 s; see above for what took them away).

## Tried and not kept

- The landing in the order of the flights for copies of the same pages
  (2026-11-27; branch `cpu/landfix`, merged and taken out again the same
  night). `VICTORY_LAND_SAMPLE=1` set alone reads a few stretches apart:
  63 of 278,218 in 125 s of the prologue. The title maps one buffer at
  several addresses and binds it by more than one; each address has a
  host copy of its own; two flights that wrote the buffer by two of its
  addresses and land together are merged into title memory in the order
  of the addresses, so the earlier flight's bytes can end over the later
  one's. With the landing's pieces placed by the pages they lie in and a
  copy merged behind the copies of every earlier flight the sample read 0
  (of 644,220 stretches in the prologue, of 419,030 at the world's entry)
  and the frame was as long as before (56.2 against 57.4 ms), but the
  first room again showed single frames with objects in another's
  material data: in both runs with it that were watched, in none without.
  Why is not known (the order was by a copy's last flight; a copy two
  flights wrote stood behind as a whole). Whether the bytes the sample
  finds apart ever show, in the picture or in what the title does, is not
  known either: nothing has been traced to them. Not pursued.
- `VICTORY_ASYNC_PRESENT=1` (opt-in; the title records the next frame
  while the last is sent): its first form passed every wait the queues
  held at the present call, and the title faulted within two minutes. A
  present in the line now keeps those waits in order: it is shown only
  once every wait before it is met, and the title stands at its next
  present until then; a stand that reaches two seconds passes the waits
  from the title's thread, as a present there does, and is counted. Run
  that way (2026-11-26, `VICTORY_WAIT_FORCE=0` then) it was 175 s clean
  in the first room with the picture right, 63.6 against 65.2 ms a frame.
  The line "presents in the line per frame" says how many went into a
  line, how long the title stood at its present, and the two-second
  stands since the start. For it to become the default: a run of the
  prologue into the open world and some minutes there with that count at
  0, no fault, no device loss, and the picture and the boat as without
  it.
- The heap sync on a thread of its own beside the refreshes (2026-11-26,
  after Xeltax/VIcky `d2d40ca6e`; its askings first, the walk beside, the
  join before the send): the submitting thread still waited 2.5 to 4 ms a
  frame for the askings and 4 to 5 ms at the joins, the sync itself took
  10.5 to 12.5 ms a frame instead of 8.5 to 9, and the refreshes beside it
  slowed ("copies" 28 against 20 ms): the first room 75 to 80 against 67
  to 69 ms a frame, back to back on the same build (s_plain1, s_plain0).
  Taken out again.
- `VICTORY_WIDE_REFRESH=1` (opt-in; a flush's copies refreshed on the
  wide threads) with the refresh by event: the first room 85 against 69
  ms a frame, "copies" 29 against 20 ms (s_wide1 against s_plain0, 2026-11-26).
  The longest send of a frame is the list that makes the heap's buffers
  ready: one flush of about 4,600 copies, almost all with nothing to
  copy, 7 to 13 ms; spread over threads they cost more than on one.
- A flush that leaves out, before it joins, asks or walks, every copy whose
  counts of events are those it remembered (2026-11-26, the first step of
  stage S1a in `architecture.md`; one look at the counts, without a lock:
  copies in view memory before the flush's asking, copies the kernel's
  write watch answers for after it, where the asking that stands held
  their range). It left out some 2,900 copies a frame (340 in view
  memory) of the 4,600 the submission that makes the heap's buffers ready
  walks, and that send went from 9 to 14 ms to 6 to 14. Its check (the old
  walk run for every copy; a copy the look would have left out whose CPU
  side the walk changed, outside the ranges flights may be writing, with
  nothing counted in the range remembered since the look) counted 108 to
  142 such copies in each prologue run and 132 at the open world's entry:
  most of them the heap's copy (174 MB in view memory), some hundred to
  seven thousand bytes each, and twice a copy of 90 MB of private memory
  with 4 to 6 MB. Not explained: a range remembered by its two ends in
  view memory may hold memory that is no view, which no event and no
  asking of the submission covers (the submission's asking leaves out a
  range whose first block is a view); a landing may change the CPU side
  next to a declared range (a flagged 4 KiB block reaches past it) while
  the check compares. Taken out again; the gain is not worth a copy that
  keeps old bytes.
- `VICTORY_NAMED_WRITES=1`: not yet tried in a run. (`VICTORY_HOT_BLOCKS`,
  blocks written often left open, was never run and is gone, and with it
  the guards' changes of protection under their lock,
  `VICTORY_GUARD_UNLOCKED=0` and `VICTORY_ALIAS_RUNS=0`; and
  `VICTORY_MARK_ONLY=0`, which opened every other view of a written
  block's pages: what keeps a clean block's other views read-only goes by
  the marks.)
  (`VICTORY_MERGE_FORGET` is gone: it dropped the kernel's written bits a
  landing set, and with them the title's of before the landing;
  `VICTORY_LAND_TAKEN` takes them and notes them.)
- The argument clamp between enhanced barriers (Zettrox/ztrx pull request
  #139): the same time on the card and in the title; not in this tree.
- Page marks set by shaders for copy-backs (Marvisak/VIctory `cd94d35`):
  its gain is the CPU compare this tree no longer does; not ported.
- Copy-backs written without the kernel's call (fxceless/victory-fork pull
  request #1032): no time goes into that call here.
- Upstream's `VICTORY_JOIN_PARTS` and `VICTORY_BACK_OWN_VIEW`, and
  DirectStorage workers by physical cores: ported and measured, no gain on
  this card; not in this tree.
- `VICTORY_INPLACE=1`: slower here (the first room 99.6 ms a frame against
  87).
- `VICTORY_CARD_WINDOWS_MB=1024`: the cutscene within the spread of runs,
  the card nearly full.
- `VICTORY_BVH_BUILDER=1` with `VICTORY_BVH_QUARANTINE=1`: the builder
  started and the title faulted within seconds of the first 3D frame; the
  likely causes are fixed since, not yet tried again (see
  `fork-ray-tracing.md`).
- `--console series-s`: a quarter faster, but the picture is black (the
  title upscales with FidelityFX there, whose programs do not run yet).
- Filling the levels left out on the processor: 683 of 736 ms a frame went
  into it while loading; the views clamp the level of detail instead.
- Spilling before leaving levels out: slower, see "When the card's memory
  is full".

## Debugging switches

Diagnostics build: `cargo build --profile diag -p victory --features diagnostics`

| Switch | Effect |
|---|---|
| `VICTORY_CAPTURE=<dir>` + `VICTORY_CAPTURE_FRAMES=<list>` | save those frames as PNG (diagnostics build) |
| `VICTORY_TIMELINE=<file>` (+ `VICTORY_TIMELINE_AFTER=<seconds>`, 130 unless told; `VICTORY_TIMELINE_FRAMES=<n>`, 8 unless told) | a few frames event by event (diagnostics build): one line per event with the milliseconds since the first of those presents, the thread, the event and two numbers of its own. Events: the present call and its return, each submission of the title and its signals, what the submitting thread sends and when it idles, what the landing thread lands, bottom-of-pipe writes and fences set, every list's reset and close (the close with what the list holds: draws, indexed draws, indirect executions and dispatches, sixteen bits each), the pass over picked buffer views on its thread (`pass>`, `pass<`) and a submission that waits for it (`pwait>`, `pwait<`), the early pass over textures picked by index (`early>`, `early<`), and the title's own waits and wake-ups (`WaitForSingleObject`, `SetEvent`, `ReleaseSemaphore`, served by the layer only with this switch). It shows what a frame waits for where the profile only shows who idles. Also in it: the steps of every send with their microseconds (`p.blocked`, `p.heap`, `p.sweep`, `p.ask`, `p.stale`, `p.steps`, `p.refresh`, `p.texture`, `p.prelude`, `p.snap`, `p.card`, `p.exec`), each submission sent (`sent`: its number and the value signalled behind it), the card reaching that value (`done`, from a thread that only waits for it), the landing's steps (`l.fetch`, `l.merge`, `l.card`, `l.hand`, `l.done`, `l.tell`, `l.bury`; each job of the merge on whichever thread runs it, `job>`, `job<`, with its runs) and the submissions landed (`landed`), a wait of the submitting thread for landings (`rwait>`, `rwait<`), and every wait of 0.2 ms or more for one of the pipeline's locks, named (`L:submitted`, `L:line`, `L:cardwork`, `L:order`, `L:journal`, `L:guards`, `L:mapping.r`, `L:mapping.w`, `L:side`, `L:queue`) with the microseconds waited. For the guards' locks (one a region of 4 MiB; `L:guards` is a wait for those an operation takes together) the holds are in it too, by holder: every hold of 0.2 ms or more (`H:check` a check's walk over its blocks, with the blocks of its range; `H:settle`; `H:fault` a write of the title's that faulted; `H:opening`, `H:marking` writes of the layer's and the runtime's; `H:mapped`, `H:remapped` the runtime's mappings), and each thread's holds summed (`HS:` and the same names: microseconds and holds, said whenever a thread has a millisecond or 256 holds together), since many short holds keep the others waiting as one long hold does |
| `VICTORY_READ_CENSUS=<n>` | a count, not for frame rate: one 4 KiB page in `n` of what landings write into private title memory is made inaccessible until its next access, and the line "read census" says per frame who touched such pages first: other code than the layer's on a title thread (the title's own reads and writes), the layer's code on a title thread, another thread of the layer's, or the landing thread's own next access with nothing between (nobody on the processor's side looked at what was brought back). Run it with `VICTORY_LAND_TAKEN=1`, so that the copy's own refresh does not read the pages again |
| `VICTORY_READ_WAIT=1` or `2` | a count, in a build with the diagnostics feature: when a submission stands for flights because its reads need them, which steps would wait (a buffer copy's or a fill's source, a refresh, a texture), which flights' late copies or shader ranges they overlap, and whether the bytes of the overlap are other ones after the wait; the lines start with "read wait:" and come with the frame line (`2` also gives one line for every wait). It changes no wait, copy, landing or refresh; the snapshots of the overlapped bytes take time of their own, outside the time counted |
| (no switch; diagnostics build) the line "page-mapping copies per frame" with the frame line | what the queue's page-mapping copy (`CopyPageMappingsBatchX`, which the title calls nearly every frame while the player moves and it streams) costs the thread that calls it: calls, batches and ranges a frame, the pages mapped (kinds 0 and 1), unmapped (kind 2) and of kind 3, the milliseconds in the calls in all and of them in the queues' work run before the remap (`before_sync`), in landing the flights whose shaders may write the pages, and in the remap itself, the longest single call since the last line, how many distinct threads called since the last line, and the calls a frame made by the thread that presents. The timeline has a span around each call (`pmap>`, `pmap<`, the batches in it) |
| `VICTORY_CARD_COPY_CHECK=1` | diagnostics build: when a submission with title buffer copies made on the card (`VICTORY_CARD_COPY`) has landed, each copy's destination in title memory is compared with its source there (what the copy on the CPU at the landing would have written); the frame line "made on the card" adds the copies compared and those that differed, and the first ones that differ are logged with their offsets and whether a shader of the same submission was declared to write the source after the copy (then the card's bytes are the ones in the list's order, and the CPU's would not be). 0 of 2,023 differed in the first room |
| `VICTORY_SEEN_CHECK=1` | diagnostics build: a draw makes ready every view of its tables as it did before a list remembered what it had made ready (`draw.rs`, `Seen`), and makes every buffer view again that its list would copy from the ones it kept (`descriptor/host.rs`, `KeptViews`); the frame lines count the views a list knew ready that were not (the recording then holds another step, upload, state or kept object) and the kept views that are not the descriptor made now. Both stayed 0 over 100 million views and 29 million copies in the first room (2026-11-26). It costs what the remembering saves |
| `VICTORY_KEYS=30:Enter,35:X` | press keys at those seconds (diagnostics build); the key bindings make them pad input |
| `VICTORY_KBM=30:W*2,31:MouseX=400*0.5,33:MouseLeft` | play keys, mouse motion (counts over the hold) and wheel steps into the keyboard and mouse pad (diagnostics build) |
| `VICTORY_LOG=victory_hle::modules::gameinput=debug` | log each pad state the game reads |
| `VICTORY_TITLE_ASSERTS=<n>` | continue past up to `n` `int3` asserts in the game's code, where the first one ends the process; the check counts as passed, so what follows is undefined and the run is not one to trust |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, report the commands in flight (diagnostics build; about 28 ms a frame here) |
| `VICTORY_D3D12_DRED=light` / `=faults` | cheaper forms for long runs (diagnostics build; their cost is not measured yet, the 22 minutes in the open world ran with `faults`): `light` keeps the breadcrumbs (which list stopped at which command, the pipelines it binds) without a marker text per command; `faults` keeps only the faulting address and the allocations named there. The report says the removal reason and whether the runtime gave a fault address and breadcrumbs. With breadcrumbs, each list that stopped after starting also gets "its reads against title memory now": the ranges of host copies its draws were given, each compared with title memory after the removal; those that differ are listed with their first differing words (title's value, copy's value) and whether a flight's shaders may write them. The lists of the last submissions are noted as they were submitted, so a list of a submission that counted as landed is named too, and a failure on a thread that has no list asks for the data itself ("device-removed data, asked by the failing thread") |
| `VICTORY_TITLE_DEBUGINFO=<seconds>` | the title writes the debug info block its crash handler attaches (frame, streaming requests, tile heap, entity pools, camera, mission) every that many seconds into the log, as `title debug info:` lines; its builder is known for the 1.0.2.1 executable, `VICTORY_TITLE_DEBUGINFO_RVA=<hex>` names it for another (diagnostics build; fxceless/victory-fork pull request #643) |
| `VICTORY_TABLED_SKIP_FRAMES=off` | upstream's 240-frame window for table-read descriptor slots; steenax uses 2 frames, since the window crashed at world entry (upstream pull request #827) |
| `VICTORY_BACK_RUN_GAP=0` | every run of a copy-back is a write of its own; by default runs under 64 bytes apart are one write (upstream pull request #857) |
| `VICTORY_WAVE_PAIRS=0` | no paired form for 64-lane programs on 32-lane cards (upstream pull request #854) |
| `VICTORY_MESH_LINK=0` | mesh programs over 32 KiB are not linked with their pixel program (upstream pull request #854) |
| `VICTORY_CODE_IN_USE=0` | no measure of the shader code the host pipelines hold in the driver (on by default, release builds too, and cheap: one `GetCachedBlob` per pipeline where it is made). The periodic line `shader code of pipelines in use` gives the sum of the cached blobs' sizes of the pipelines bound at least once and not yet released, the most so far, and the sum of those made and never bound; a warning is logged once at 300, 400 and 450 MB. The blob is a proxy for the driver's machine code (it holds other state as well). NVIDIA drivers were measured elsewhere to remove the device at 512 MB of code in use (about 448 MB of blob in 262 pipelines at the point of removal) |
| `VICTORY_GPU_TIMING=1` + `VICTORY_D3D12_DRED=1` | the card's time per command and per pass of ours (diagnostics build; submissions are waited for, so not for frame rate); `VICTORY_TIMING_LINES=<n>` lists that many commands |
| `VICTORY_D3D12_ADAPTER=<n or text>` | the card the host device is made on. By default the one the system ranks first for performance (never a software adapter), where the runtime used to take the first one it lists (a processor's own graphics could come first); the log says at start `host adapter: <name> (vendor ..., .. MiB dedicated video memory)`. A number is the place in the system's ranking (0 is the default), text is part of an adapter's name (`rtx`); a value that names none is logged and ignored (Marvisak/VIctory pull request #3073) |
| `VICTORY_TITLE_PRIORITIES=0` | the title's threads keep the priorities it sets. By default its idle and lowest threads (its streaming pager's among them) run below normal: at idle they ran once in four seconds or so while every processor was busy, and the loading that waits for them with them (Marvisak/VIctory pull request #2903; not measured here yet, so this is the switch for a comparison). The log names the first 16 such calls (`a title thread's idle or lowest priority is set below normal`) |
| `VICTORY_CRASH_DUMP_DIR=<dir>` | on an unhandled exception, a full-memory minidump of the process (every thread, the exception named in it) is written to `<dir>\crash-<time>-<pid>.dmp`, and the crash log says where and how big. Opt-in: the dump is the size of the process (20 to 26 GB here) and holds the title's code and data, so it stays on this machine and takes room and time before the process ends (the log line gives both). A directory below the source tree, or inside any copy of it (a `.victory-root` above it), is refused and the crash log says so. dbghelp is loaded only then (Geckomacabre/vx-layer 707b75be0) |
| `VICTORY_MAP_LEAN=1` | an experiment, off until measured in the world: mapping the title's physical pages asks the host only for what it needs. A view mapped where an unmap of exactly that range left its placeholder is mapped over it at once, without looking at or splitting placeholders (the host refuses and the range is split as before if the title joined it with others since); section pages already committed through another view are not committed again. Both save calls that wait for the process's address-space lock behind the GPU side's guards and the driver; 6.5 to 5.0 us a page without contention on the author's machine, the gain under the lock in the world is the fork's claim (200 to 400 us a page there), not measured here. Compare with and without in the frame line and the streaming of the first room (Marvisak/VIctory pull request #2916 from avatarmanz/VIctory-7900XTX 49d7126) |
| `VICTORY_DSTORAGE_STAGING_MB=<n>` | at least `n` MiB of compressed bytes of DirectStorage requests in flight, where the title asks for the console's 32 MiB (the limit only counts, nothing is allocated for it; a request larger than it still runs alone). Unset: as the title asks. fxceless/victory-fork issue #1268 ran with 256 for loading (a cold world entry at 94.6 s, together with other changes, so its own share is not known; not measured here). The log says `dstorage: staging limit at least this` when it applies |
| `VICTORY_WINDOW_MAX=1920x1080` | largest size of the game window; unset, it is fitted to the screen's work area. The game's render targets keep their size |
| `VICTORY_EFFICIENCY_MODE=1` | Windows' choice about running the process at reduced speed (its "efficiency mode": lower clock, the slow cores of a hybrid processor, while the window is not in front) is left alone. By default the process opts out of that throttling, as it does of the timer-resolution one. Taken from avatarmanz |
| `VICTORY_PROCESS_PRIORITY=1` | opt-in, not measured yet: the process runs above normal priority, so that a program that keeps the processors busy does not starve the title's loading thread (it runs below normal). Taken from avatarmanz, where it is on by default |
| `VICTORY_CARD_AWAKE=1` (opt-in; `=turns`: on and off by turns of 20 s, the log says when, to compare in one run) | a small compute program (one group of 64 threads counting, about 2 ms a send, never more than two sent) runs on a compute queue of its own while title submissions go out, so the driver does not take the card's idle stretches between submissions for a light load and lower its clocks. Nothing is sent once no submission has gone out for a quarter of a second. Not measured here yet. Taken from avatarmanz (`host/awake.rs`) |
| `VICTORY_AB=<seconds>,<switch>[,<switch>..]` | diagnostics build: the named switches change while the title runs, so one run measures both sides in the same scene. Every `<seconds>` the next phase begins (at a present); switch n is on in the phases whose number has bit n, so all combinations come one after another and again. At a phase's end one line: frames, mean, median and p90 ms, draws a frame (the phase's first second is left out). Only the switches read through `toggled!` follow it, which are `VICTORY_JOIN_COPIES` and `VICTORY_CARD_AWAKE`; a switch named here does not go by its own variable any more. Without the feature it does nothing. Taken from avatarmanz's `VICTORY_AB` |

`cargo xtask refusals <log>` says what a run lost and why, from its log (`%TEMP%\steenax.log` after `scripts\play.ps1`): the pipelines the host refused, grouped by shaders; the debug layer's reasons for them (only in a run with `VICTORY_D3D12_DEBUG=1`, diagnostics build); the shaders that do not translate; and, in a diagnostics build's log, the draws and dispatches dropped a frame for want of a host pipeline, by reason (Marvisak/VIctory pull request #2892).

Three more lines come with the frame-rate line, without a switch (counters only, nothing waits):

- `lists submitted with a host texture let go before (by the title or the sweep): ...` once any was found: lists submitted holding a host texture the title released or the sweep let go (each named with its path), counted per list and texture, as held by a state of the list (a texture a draw or copy of the list uses: held before the keep of bound targets too) or by a keep alone (without it the texture would have been freed before the list ran), since the start and since the last line, with the first eight named (size, format, frame released, frame submitted). On 2026-11-26 it never came in 25 entries into the open world; in the prologue the first count reached 6 in 205 s (textures of the decoration bake released in the frame their list was submitted, each held by a state of the list) and the second, a texture only the keep held, stayed 0 in every run.
- `pipelines made by the driver: ...` while pipelines were made in the interval: the driver calls (create, library load, library store) of the pipeline workers, their summed time over the wall time (the average number at once) and the most in flight at once. A most near 2 with an average near 2 on many workers says the driver builds about two at a time.
- `title lists submitted again in a later submission ...`: lists the title submits again without a reset, by the fence of the earlier submission (not reached, no signal behind it yet, reached). The first of the first two is logged once with the frame number.
