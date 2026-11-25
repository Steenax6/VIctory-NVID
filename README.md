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
| Frame rate | 1-3 fps in the world (the first run is slower while shaders translate; they are cached after that) |
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

## Faster settings (optional, off by default)

What this section listed before (`VICTORY_RELAY=2`, `VICTORY_WIDE_BACK=1`,
card copies of title buffers from 1 MB to 1024 MB) is the default since
upstream `c96b4f2` (about 230 ms a frame where it was about 400 there).

Newer switches from upstream (`f85101b`), off by default. Together they took
the first room from about 3.5-4 to 6-7 frames a second on an RTX 4070:
`VICTORY_ASYNC_SUBMIT=1`, `VICTORY_QUEUE2=1`, `VICTORY_INDIRECT_WINDOW=1`,
`VICTORY_TEXTURE_NO_WAIT=1`. Do not combine `VICTORY_INDIRECT_WINDOW=1` with
`VICTORY_CARD_FETCH_INFLIGHT=1` (a device loss upstream). Set them in
PowerShell before running the script.

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

Diagnostics build: `cargo build --profile diag -p victory --features diagnostics`

## License

GPL-3.0-or-later (see `LICENSE`). Based on VIctory (Marvisak) and victory-fork (fxceless).
