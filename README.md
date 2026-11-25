# ztrx

**ztrx** runs the Xbox Series build of the game natively on Windows PCs with
NVIDIA graphics cards, from game files you supply yourself.

**This repository contains no game files or assets.** You need your own copy
of the game. Not affiliated with or endorsed by its developer or publisher.

## Status (2026-11-25)

Tested on Windows 11 with an **NVIDIA GeForce RTX 3050** (8 GB):

| | |
|---|---|
| Boot, first-boot screens, intro, opening credits | works |
| Prologue | playable from the cell to the boat chase: cell, cafeteria, robbery, police chase (wanted level, minimap), boat, binoculars |
| Stability | 1 h 30 min+ in one session without a crash |
| Frame rate | 1-3 fps in the world on that card (the first run is slower while shaders translate; they are cached after that). On an RTX 4070 Ti with a Ryzen 7 9800X3D, with `scripts\play.ps1`: 6-9 fps in the opening cutscene and about 11 fps in the first room (they were 1-5 and about 4 before the changes of 2026-11-25, see "Faster settings"); the first room is lit and complete there in every run, and the band of stale blocks along the bottom of the picture is gone |
| Known problems | anything in shadow is black and bright scenes overexpose (the game's ray-traced lighting does not run yet); some blocky artifacts; boats sit low in the water |

## Requirements

- Windows 10/11 x64, a CPU with AVX2
- A Direct3D 12 GPU with shader model 6.6 and mesh shaders (NVIDIA RTX 20 series or newer)
- Rust (the version in `rust-toolchain.toml`)
- Several GB of free Windows commit memory (raise the paging file if `crashes.log` reports error 1455)

## Build

```powershell
cargo build --release -p victory
```

The release profile builds with thin LTO and one codegen unit (about a
minute; 10-25 % less frame time than without, measured on the opening
cutscene and the first room).

## Play

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<your game folder>\<the game's exe>"
```

`scripts\play.ps1` keeps the shader cache and saves in folders you choose
(`-Cache`, `-Data`) and presses no buttons for you. Controller: XInput pads
(a DualShock/DualSense through DS4Windows works). With no pad connected the
keyboard acts as one while the game window has focus (Enter is A).

`VICTORY_AUTO_START=1` answers the start screen's "press A" for you: A is
pressed briefly every 2 s from the moment the game adds its user until it
moves on (at most 30 s). The first-start menus come later and are left to
you.

## Faster settings

`scripts\play.ps1` turns five opt-in switches on unless you set them
yourself: `VICTORY_ASYNC_SUBMIT=1`, `VICTORY_QUEUE2=1`,
`VICTORY_INDIRECT_WINDOW=1`, `VICTORY_TEXTURE_NO_WAIT=1`,
`VICTORY_PLACED_TEX=1` (`-Plain` leaves them as the environment has them;
`victory.exe` run by hand has them off).
Do not combine `VICTORY_INDIRECT_WINDOW=1` with `VICTORY_CARD_FETCH_INFLIGHT=1`
(a device loss upstream).

What 2026-11-25 added is on by default; each has a switch that puts the old
way back (`=0`). Measured on an RTX 4070 Ti in the first room, with the four
switches above: from 5.5 to 10-11 frames a second (182 to about 95 ms a
frame); the thread that prepares and sends submissions went from about 195
to about 47 ms of work a frame.

| Switch (`=0`: as before) | What it does |
|---|---|
| `VICTORY_INDIRECT_HOIST`, `VICTORY_INDIRECT_HOIST_ALL` | single indirect records are clamped in their list's prelude, one pass for all records of a buffer, where no shader of the list has declared a write over them (some 1,900 of 4,700 a frame; each used to be a compute pass of ours between two draws) |
| `VICTORY_INDIRECT_WINDOW_SIZES=16,64`, `VICTORY_INDIRECT_WINDOW_KEEP` | windows of records clamped ahead grow 64, 256, 1024 and outlive direct draws whose shaders store nothing |
| `VICTORY_FILL_HOT`, `VICTORY_FILL_DIRECT`, `VICTORY_LATE_COPY` | only reads of the 64 KiB of title memory where shaders were seen to change bytes under a fill wait for flights (one such fill used to make every fill wait); constants there are bound from the buffer's host copy; a title buffer copy out of such memory is done when its submission lands |
| `VICTORY_HEAP_SYNC_FAST`, `VICTORY_DESCRIPTOR_STRIPES` | descriptor records the title copies into its heap are found by the write watch's 4 KiB pages and mirrored in runs; slots have striped locks, so recording threads no longer stand behind that pass |
| `VICTORY_SHADOW_CACHE` | recording threads remember which host copy an address range has |
| `VICTORY_USER_COPY`, `VICTORY_REFRESH_PARTS`, `VICTORY_REFRESH_4K`, `VICTORY_GUARD_4K`, `VICTORY_MARK_ONLY`, `VICTORY_VIEW_KNOWN`, `VICTORY_ABSENT_ONCE`, `VICTORY_ASK_SKIP_VIEWS`, `VICTORY_ALIAS_RUNS`, `VICTORY_SEND_CHANGED` | title memory is copied by the processor (a fault falls back to the kernel's copy), by 4 KiB pages and only where written; guards are not re-armed over other views; unmapped runs are remembered (`VICTORY_ABSENT_FRAMES`, 30); bytes the copy-back itself wrote are not sent up to the card again |
| `VICTORY_TEXTURE_DROP_LEVELS` (`0`, `auto`, `force1`, `force2`), `VICTORY_TEXTURE_PROMOTE_MB`, `VICTORY_RESERVED_REUSE`, `VICTORY_STAGE_LEFT_OUT`, `VICTORY_TILED_IDLE_MS`, `VICTORY_STAGING_POOL_MB` | large block-compressed title textures (1024 a side and more) are made as reserved textures and, past a line of host-copy bytes, without their one or two most detailed levels; their views clamp the level of detail, so the sizes shaders read stay the title's. On a 12 GB card the title's textures did not all fit: some 22,000 waited for room in every run, and which were there differed from run to run (the first room came up lit in one run, dark in the next). Now none is held back there: 17,400 title textures alive in 2.9 GB where 9,500 took 3.9 GB. Reserved textures and staging buffers are reused, not made and released |
| `VICTORY_PLACED_TEX=1` (opt-in; `play.ps1` sets it) | the other host textures of placed textures come out of 16 MiB heaps instead of an allocation each (fxceless/victory-fork pull request #903 with smaller heaps, filled fullest first): making one takes 9 us, the opening cutscene 158 -> about 125 ms a frame with the levels switch on; about 1 GB of the card stays free inside the heaps after the cutscene |
| `VICTORY_GROUP_SPLIT`, `VICTORY_GROUP_PIECES=2` | a dispatch that asks for more than 65,535 thread groups in x runs in pieces, each told where its groups start (pull request #976), indirect records too: one of the title's indirect lighting passes asks for about 75,000 and lost its tail, a band of stale blocks along the bottom of the picture |
| `VICTORY_PIPELINE_FIX`, `VICTORY_VIEW_RAW_OWN_COPY` | a pipeline the host refuses as the title described it is made again with its state put right (blending on integer targets and the like: the one mesh pipeline refused in the first room); a raw buffer view whose bytes do not start on 16 gets a copy of its own bytes instead of reading zeros |
| `VICTORY_CARD_FINE`, `VICTORY_CARD_WIDE_HASH`, `VICTORY_CARD_SYNC_BATCH`, `VICTORY_MERGE_PAGES`, `VICTORY_FAST_MERGE`, `VICTORY_WIDE_COMPARE`, `VICTORY_BURY_ASIDE` | shader writes to copies in the card's memory are found per 4 KiB block (was 64 KiB), fetched as runs and merged where they differ; title bytes under flights go up a page at a time where shaders never wrote |

Experiments, off: `VICTORY_ASYNC_PRESENT=1` (the title records the next
frame while the last is sent; as it stands the title faults with it within
two minutes, because waits between its queues are passed at the present; a
variant that met them in order ran four minutes without a fault, at the same
frame rate, since the card sets the pace by then, and was not kept),
`VICTORY_HOT_BLOCKS=1`, `VICTORY_MERGE_FORGET=1`, `VICTORY_NAMED_WRITES=1`
(not yet tried in a run), the argument clamp between enhanced barriers
(Zettrox/ztrx pull request #139; on this card the same time on the card and
in the title, left on the branch `narrow-waits`), page marks set by shaders
for copy-backs (Marvisak/VIctory `cd94d35`: its gain is the CPU compare this
tree no longer does; not ported), `--console series-s` (a quarter faster, but the
picture is black: the title upscales with FidelityFX there, whose programs
do not run yet).

Where the time goes now (first room, 90-100 ms a frame): the card is busy
about 73 % of the time. A timing run (`VICTORY_GPU_TIMING=1`) puts the
title's own commands at 62 ms a frame on the card (3,850 dispatches, 5,100
indirect draws; they were 113 ms while each indirect record had a clamp pass
of ours beside it) and the passes of ours at 14 ms (23 before the wide
hash). The title's recording threads and our submitting thread take turns
for the rest. 30 frames a second is out of reach of this design on this
card: the title's commands alone take twice a frame's time there.

`VICTORY_PIPELINE_WORKERS=<n>` sets how many threads build pipelines that
are not cached yet (default: half the logical processors, at most 8). On a
Ryzen 7 9800X3D (16 threads) with an RTX 4070 Ti, 14 cut the intro's time
below 1 fps on an empty cache from 55 s to 6-12 s; 16 was no better than the
default.

## Shader warm-up (optional)

A build that changes the shader translator starts with an empty shader
cache, and the first run is slow while the shaders translate. They can be
translated beforehand, on the processor only (4.5 to 6.5 minutes upstream):

```powershell
$env:VICTORY_CACHE_DIR = "$env:LOCALAPPDATA\ztrx\cache"; $env:VICTORY_HOST_WAVE = 32; $env:VICTORY_VIEW_LIMIT = 999999
target\release\victory.exe translate "<your game folder>\shaders.rpf" --warm all --jobs 14
```

The cache folder is the one `scripts\play.ps1` uses (`-Cache`). The two
numbers must match the card: a run's log has `host wave lanes ... max=32`
and `host view heap capped at 1000000` (the limit is that number minus 1);
these are an NVIDIA card's.

## Debugging switches

| Switch | Effect |
|---|---|
| `VICTORY_CAPTURE=<dir>` + `VICTORY_CAPTURE_FRAMES=<list>` | save those frames as PNG (diagnostics build) |
| `VICTORY_KEYS=30:Enter,35:X` | press keys at those seconds (diagnostics build) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, report the commands in flight |
| `VICTORY_PIPELINE_WAIT_MS=24` | shorter per-frame wait for pipelines still being built (objects can go missing meanwhile) |
| `VICTORY_TABLED_SKIP_FRAMES=off` | upstream's 240-frame window for table-read descriptor slots; ztrx uses 2 frames, since the window crashed at world entry (upstream pull request #827) |
| `VICTORY_BACK_RUN_GAP=0` | every run of a copy-back is a write of its own; by default runs under 64 bytes apart are one write (upstream pull request #857) |
| `VICTORY_WAVE_PAIRS=0` | no paired form for 64-lane programs on 32-lane cards (upstream pull request #854) |
| `VICTORY_MESH_LINK=0` | mesh programs over 32 KiB are not linked with their pixel program (upstream pull request #854) |
| `VICTORY_GPU_TIMING=1` + `VICTORY_D3D12_DRED=1` | the card's time per command and per pass of ours (diagnostics build; submissions are waited for, so not for frame rate); `VICTORY_TIMING_LINES=<n>` lists that many commands |

Diagnostics build: `cargo build --profile diag -p victory --features diagnostics`

## License

GPL-3.0-or-later (see `LICENSE`). Based on VIctory (Marvisak) and victory-fork (fxceless).
