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
| `VICTORY_CARD_CONSTANTS` (2026-11-26; in every run since, not measured on its own) | constant buffers of fresh descriptor tables that lie in a host copy in the card's memory are bound there, not filled into upload memory behind the flights (theinkbit/victory-lot9-next ffd04e130) |
| `VICTORY_HEAP_SYNC_FAST`, `VICTORY_DESCRIPTOR_STRIPES` | descriptor records the title copies into its heap are found by the write watch's 4 KiB pages and mirrored in runs; slots have striped locks, so recording threads no longer stand behind that pass |
| `VICTORY_SHADOW_CACHE` | recording threads remember which host copy an address range has |
| `VICTORY_USER_COPY`, `VICTORY_REFRESH_PARTS`, `VICTORY_REFRESH_4K`, `VICTORY_GUARD_4K`, `VICTORY_MARK_ONLY`, `VICTORY_VIEW_KNOWN`, `VICTORY_ABSENT_ONCE`, `VICTORY_ASK_SKIP_VIEWS`, `VICTORY_ALIAS_RUNS`, `VICTORY_SEND_CHANGED` | title memory is copied by the processor (a fault falls back to the kernel's copy), by 4 KiB pages and only where written; guards are not re-armed over other views; unmapped runs are remembered (`VICTORY_ABSENT_FRAMES`, 30); bytes the copy-back itself wrote are not sent up to the card again |
| `VICTORY_TEXTURE_DROP_LEVELS` (`0`, `auto`, `force1`, `force2`), `VICTORY_TEXTURE_PROMOTE_MB`, `VICTORY_RESERVED_REUSE`, `VICTORY_STAGE_LEFT_OUT`, `VICTORY_TILED_IDLE_MS`, `VICTORY_STAGING_POOL_MB` | large block-compressed title textures (1024 a side and more) are made as reserved textures and, past a line of host-copy bytes, without their one or two most detailed levels; their views clamp the level of detail, so the sizes shaders read stay the title's. On a 12 GB card the title's textures did not all fit: some 22,000 waited for room in every run, and which were there differed from run to run (the first room came up lit in one run, dark in the next). Now none is held back there. Reserved textures and staging buffers are reused, not made and released |
| `VICTORY_PLACED_TEX=1` (opt-in; `play.ps1` sets it) | the other host textures of placed textures come out of 16 MiB heaps instead of an allocation each (fxceless/victory-fork pull request #903 with smaller heaps, filled fullest first): making one takes 9 us; about 1 GB of the card stays free inside the heaps after the cutscene |
| `VICTORY_GROUP_SPLIT`, `VICTORY_GROUP_PIECES=2` | a dispatch that asks for more than 65,535 thread groups in x runs in pieces, each told where its groups start (pull request #976), indirect records too: one of the title's indirect lighting passes asks for about 75,000 and lost its tail, a band of stale blocks along the bottom of the picture |
| `VICTORY_SECTION_MB=2` (`0`: one section a request, as before) | the title's physical pages come from sections of 2 MiB (fxceless/victory-fork issue #1089): with one section of 6.1 GiB mapped a 64 KiB view at a time, every map and unmap of the title waited for the worker of Windows that trades such a section's pages (48 ms a frame of the title's threads in the opening cutscene) |
| `VICTORY_PIPELINE_FIX`, `VICTORY_VIEW_RAW_OWN_COPY` | a pipeline the host refuses as the title described it is made again with its state put right (blending on integer targets and the like: the one mesh pipeline refused in the first room); a raw buffer view whose bytes do not start on 16 gets a copy of its own bytes instead of reading zeros |
| `VICTORY_CARD_FINE`, `VICTORY_CARD_WIDE_HASH`, `VICTORY_CARD_SYNC_BATCH`, `VICTORY_MERGE_PAGES`, `VICTORY_FAST_MERGE`, `VICTORY_WIDE_COMPARE`, `VICTORY_BURY_ASIDE` | shader writes to copies in the card's memory are found per 4 KiB block (was 64 KiB), fetched as runs and merged where they differ; title bytes under flights go up a page at a time where shaders never wrote |
| `VICTORY_SYNC_BARRIER` (2026-11-26; in every run since, not measured on its own) | a pipeline sync in the title's list waits for every shader write before it (one unordered access barrier), where it did nothing: dispatches that read what the one before wrote no longer overlap it (avatarmanz/VIctory-7900XTX `0b5f191b8`, Marvisak/VIctory pull request #1228). The log names the first eight syncs; `=0` is there to measure what the barriers cost on the card |

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
being built (objects can go missing meanwhile).

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

`VICTORY_PAGE_WRITES` (unset or `guards`: the guarded blocks under such a
write are opened, so host copies of buffers and of textures read through
tables take its bytes; `0`: not; `slots`, `1` or `all`: the stale slots are
also due at once at every pass, about 120 uploads a second more in the
opening cutscene). The line "writes behind
the guards" counts the writes, the slots under them, the stale ones and
the stale ones now.

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
time by part, "the texture queue by what its slots wait for" the queue.

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

## Tried and not kept

- `VICTORY_ASYNC_PRESENT=1` (the title records the next frame while the
  last is sent): as it stands the title faults with it within two minutes,
  because waits between its queues are passed at the present; a variant
  that met them in order ran without a fault and no faster.
- `VICTORY_HOT_BLOCKS=1`, `VICTORY_MERGE_FORGET=1`, `VICTORY_NAMED_WRITES=1`:
  not yet tried in a run.
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
| `VICTORY_KEYS=30:Enter,35:X` | press keys at those seconds (diagnostics build); the key bindings make them pad input |
| `VICTORY_KBM=30:W*2,31:MouseX=400*0.5,33:MouseLeft` | play keys, mouse motion (counts over the hold) and wheel steps into the keyboard and mouse pad (diagnostics build) |
| `VICTORY_LOG=victory_hle::modules::gameinput=debug` | log each pad state the game reads |
| `VICTORY_TITLE_ASSERTS=<n>` | continue past up to `n` `int3` asserts in the game's code, where the first one ends the process; the check counts as passed, so what follows is undefined and the run is not one to trust |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, report the commands in flight (diagnostics build; about 28 ms a frame here) |
| `VICTORY_D3D12_DRED=light` / `=faults` | cheaper forms for long runs (diagnostics build; their cost is not measured yet, the 22 minutes in the open world ran with `faults`): `light` keeps the breadcrumbs (which list stopped at which command, the pipelines it binds) without a marker text per command; `faults` keeps only the faulting address and the allocations named there. The report says the removal reason and whether the runtime gave a fault address and breadcrumbs |
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

`cargo xtask refusals <log>` says what a run lost and why, from its log (`%TEMP%\steenax.log` after `scripts\play.ps1`): the pipelines the host refused, grouped by shaders; the debug layer's reasons for them (only in a run with `VICTORY_D3D12_DEBUG=1`, diagnostics build); the shaders that do not translate; and, in a diagnostics build's log, the draws and dispatches dropped a frame for want of a host pipeline, by reason (Marvisak/VIctory pull request #2892).

Two more lines come with the frame-rate line, without a switch (counters only, nothing waits):

- `pipelines made by the driver: ...` while pipelines were made in the interval: the driver calls (create, library load, library store) of the pipeline workers, their summed time over the wall time (the average number at once) and the most in flight at once. A most near 2 with an average near 2 on many workers says the driver builds about two at a time.
- `title lists submitted again in a later submission ...`: lists the title submits again without a reset, by the fence of the earlier submission (not reached, no signal behind it yet, reached). The first of the first two is logged once with the frame number.
