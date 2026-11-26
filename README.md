# steenax

**steenax** runs the Xbox Series build of the game natively on Windows PCs with
NVIDIA graphics cards, from game files you supply yourself.

**This repository contains no game files or assets.** You need your own copy
of the game. Not affiliated with or endorsed by its developer or publisher.

This tree lives on the branch `steenax` of
[Steenax6/VIctory-NVID](https://github.com/Steenax6/VIctory-NVID), a fork of
[Marvisak/VIctory](https://github.com/Marvisak/VIctory). It left upstream at
`ffe310e` (2026-11-24) and is not merged with it since: changes from
upstream and from the other forks are taken one at a time, where they bring
a gain or a fix on the machine below or are of use on any (keyboard and
mouse, for one). Each commit says where its change is from. The fork's
`main` is upstream's, untouched.

## Status (2026-11-26)

Measured on Windows 11 with an **NVIDIA GeForce RTX 4070 Ti** (12 GB), an
AMD Ryzen 7 9800X3D and 64 GB of memory, with the settings of
`scripts\play.ps1`. Frame times are medians over a scene.

| | |
|---|---|
| Boot, first-boot screens, intro, opening credits | works |
| Prologue | playable from the cell to the boat chase: cell, cafeteria, robbery, police chase, boat, binoculars |
| Open world | played from a save at the first boat mission: the boat floats, casts off and steers; on foot; a car is entered and driven |
| Saves | the game's autosave is written and continued at the next start; a save made with another fork loads |
| Frame rate | opening cutscene about 10 fps (100 ms a frame), first room 13 fps (76 ms), open world 10 to 13 fps (77 to 97 ms, by scene; 77 ms by boat and by car over 22 minutes) |
| Stability | 22 minutes in the open world in one session, ended by hand: no device loss, no error in the log |
| Memory | the process peaks at 20 to 26 GB. The card's 12 GB are full: textures that do not fit lose their most detailed levels first and go to the PC's memory last; none waits for room, in the prologue or in the open world |
| Known problems | the game's ray-traced lighting does not run yet: a stand-in fills in for its indirect light (`VICTORY_AMBIENT`), so what the sun does not reach is evenly lit, not black, and bright scenes overexpose; black blocky patches on the water, most of them in the shallows |

In the first boat mission the "cast off" prompt comes back until the map
has been opened once (View; Tab on the keyboard): a tutorial step of the
game, not a fault.

## Requirements

- Windows 10/11 x64, a CPU with AVX2
- A Direct3D 12 GPU with shader model 6.6 and mesh shaders (NVIDIA RTX 20 series or newer)
- Rust (the version in `rust-toolchain.toml`)
- Memory for a process of 20 to 24 GB (raise the paging file if `crashes.log` reports error 1455)

## Build

```powershell
cargo build --release -p victory
```

## Play

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<your game folder>\<the game's exe>"
```

`scripts\play.ps1` keeps the shader cache and saves in folders you choose
(`-Cache`, `-Data`; by default under `%LOCALAPPDATA%\steenax`), turns the
faster settings on and presses no buttons for you. `-SkipIntro` holds A
through the opening scene. The first run is slow while shaders translate
and pipelines are made; both are cached after that.

Controller: XInput pads (a DualShock/DualSense through DS4Windows works).

### Keyboard and mouse

The game reads pads only, so the keyboard and mouse act as a pad: on the
first connected pad (both work at once), or on a stand-in pad when none is.
Click into the game window to capture the mouse: its motion is the right
stick. Esc frees the cursor (and opens the menu) and captures it again;
Alt-Tab, losing focus or sizing the window frees it.

| Keys | Pad |
|------|-----|
| W A S D / I J K L, mouse | left stick / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E | LB / RB |
| Z / C, right / left mouse button | LT / RT |
| F / H, middle mouse button | stick clicks (LS / RS, RS) |
| Esc, Tab | Menu, View |

Rebind everything in `[controls]` of `victory.toml` (see
`victory.example.toml`): sensitivity, inversion, dead zone, letters by
their place on a US keyboard, walk speed, mouse buttons and wheel, and a
second layout a key can switch to.

## Faster settings

`scripts\play.ps1` turns six opt-in switches on unless you set them
yourself (`-Plain` leaves them alone; `victory.exe` run by hand has them
off): `VICTORY_ASYNC_SUBMIT=1`, `VICTORY_QUEUE2=1`,
`VICTORY_INDIRECT_WINDOW=1`, `VICTORY_TEXTURE_NO_WAIT=1`,
`VICTORY_PLACED_TEX=1`, `VICTORY_NO_WATCHDOG=1`. Everything else that makes
this tree faster is on by default.

The switches a player may want:

| Switch | Effect |
|---|---|
| `VICTORY_AUTO_START=1` | answers the start screen's "press A" |
| `VICTORY_AUTO_SKIP=1` | what `-SkipIntro` sets |
| `VICTORY_TEXTURE_DROP_LEVELS=0` | large textures keep every level of detail (for cards with more memory; on 12 GB many then go to the PC's memory) |
| `VICTORY_TEXTURE_SPILL=0` | textures that find no room on the card wait for it instead of going to the PC's memory |
| `VICTORY_AMBIENT=<level>` / `=off` | the stand-in for the game's indirect light while its rays do not run (0.3 by default; `off`: what the sun does not reach is black) |
| `VICTORY_PIPELINE_WORKERS=<n>` | threads that build pipelines not cached yet (default: half the logical processors, at most 8) |
| `VICTORY_WINDOW_MAX=1920x1080` | largest size of the game window |

[docs/switches.md](docs/switches.md) has every switch, what each change
measured, what was tried and not kept, and the debugging switches.

## Shader warm-up (optional)

A build that changes the shader translator starts with an empty shader
cache. The shaders can be translated beforehand, on the processor only
(about a minute with 14 jobs on the machine above):

```powershell
$env:VICTORY_CACHE_DIR = "$env:LOCALAPPDATA\steenax\cache"; $env:VICTORY_HOST_WAVE = 32; $env:VICTORY_VIEW_LIMIT = 999999
target\release\victory.exe translate "<your game folder>\shaders.rpf" --warm all --jobs 14
```

The cache folder is the one `scripts\play.ps1` uses (`-Cache`). The two
numbers must match the card: a run's log has `host wave lanes ... max=32`
and `host view heap capped at 1000000` (the limit is that number minus 1);
these are an NVIDIA card's.

## Debugging

The diagnostics build (`cargo build --profile diag -p victory --features
diagnostics`) can save frames, play scripted input, report the commands in
flight when the card gives up and time each command on the card: see
"Debugging switches" in [docs/switches.md](docs/switches.md).

## License

GPL-3.0-or-later (see `LICENSE`). Based on VIctory (Marvisak) and victory-fork (fxceless).
