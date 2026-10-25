# WINDOWS_AGENT — instructions for the agent on the Windows PC

You are a Claude Code agent running on the project owner's Windows PC
(RTX 3060, Chrome). Another agent works on the same repo from a Mac.
The owner reports the web demo runs slowly here and "can't handle so
many sound sources". Your job is to find out exactly why on this
machine, fix what is Windows-specific, and report numbers the Mac side
can design against.

To start: open Claude Code in the repo root and say
"Read WINDOWS_AGENT.md and follow it."

## Read first

1. `README.md`: what the engine is.
2. `WINDOWS_PLAN.md`: the Windows fix ladder. W0, W1 and the W2 frame
   limiter have shipped. The rest is yours.
3. `SOURCES_PLAN.md`: the many-sources plan. The Mac agent builds the
   engine side of it. You measure it.
4. `TESTS.md`: how tests are organised. Follow its conventions.

## Who owns what

To avoid merge conflicts between the two machines:

| Area | Owner |
|---|---|
| Rust engine (`crates/omg-dsp`, `omg-core`, `omg-scene`, `omg-web/src`) and the SOURCES_PLAN steps | Mac agent |
| Web host: `web/main.js` render loop and frame pacing, `web/gpu.js` and the WebGPU dispatch in `web/worker.js`, audio context setup, `tools/serve.py` | you |
| `web/worklet.js` governor | shared: change it only for a Windows-specific reason, and say so in your report |
| `reports/` | you write, the Mac agent reads |

If you find an engine bug or a needed engine change, write it up in
your report instead of changing the crate. Small, obviously correct
fixes are fine if you add a test for them.

## Setup (PowerShell)

```powershell
winget install -e --id Rustlang.Rustup
winget install -e --id OpenJS.NodeJS.LTS
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
# open a new terminal so PATH picks them up, then:
rustup default stable
rustup target add wasm32-unknown-unknown
```

`tools/build_web.sh` is a POSIX shell script. Run it from Git Bash, or
use the PowerShell equivalent:

```powershell
$env:RUSTFLAGS = "-C target-feature=+simd128"
cargo build --release -p omg-web --target wasm32-unknown-unknown
Copy-Item target\wasm32-unknown-unknown\release\omg_web.wasm web\omg_web.wasm
Remove-Item Env:RUSTFLAGS
```

Skip `omg-godot` (needs rustc ≥ 1.94 and Godot) and expect `omg-gpu`
tests to run for real here, because this machine has a GPU.

## Step 1 — measure before touching anything

Work on a branch: `git checkout -b windows/<topic>`.

1. Tests:
   ```powershell
   cargo test --workspace --exclude omg-godot --release
   node tools/web_smoke.mjs
   ```
   Note any failure. `crates/omg-scene/tests/world_traced_cost.rs` is
   a wall-clock test and may be flaky under load.
2. Audio-thread scaling, no browser involved:
   ```powershell
   node tools/bench_sources.mjs
   node tools/bench_web.mjs
   ```
   `bench_sources.mjs` sweeps the number of sources and prints the
   realtime factor per configuration plus the largest source count that
   keeps a safe margin. This is the number SOURCES_PLAN needs from this
   machine.
3. The browser. Start `python tools/serve.py`, open
   `http://localhost:8000/web/` in Chrome, open DevTools → Console.
   Record:
   - the `[sys]` and `[sys-audio]` lines (the `gpu` field names the
     adapter Chrome actually got; if it is not the 3060, that alone
     explains a lot)
   - 30 s of the stats panel while walking from the living room into
     the club doorway (sim tick ms, render load, gaps, gpu dispatch ms,
     the governor's point budget and tap ceiling)
   - Windows audio output device and whether it is Bluetooth
4. A/B the existing flags, 30 s each at the same spot:

   | URL | Tests |
   |---|---|
   | `/web/` | baseline |
   | `/web/?px=8` | pixel budget lifted |
   | `/web/?fps=0` and `?fps=60` | frame limiter off / lower |
   | `/web/?latency=playback` | bigger audio buffers |
   | `/web/?q=low`, `?q=high` | sim quality tier pinned |
   | `/web/?early=ism`, `?early=traced` | early-reflection backend |
   | GPU switch in the quality panel on vs off | WebGPU compute |

## Step 2 — report

Write `reports/windows-3060.md`, commit it on your branch and push:

```markdown
# Windows 3060 report — <date>
## Machine
CPU, GPU, RAM, Windows build, Chrome version, display resolution and Hz,
audio device (wired/USB/Bluetooth), [sys] and [sys-audio] lines.
## Tests
Pass/fail summary, failures verbatim.
## bench_sources.mjs
The full printed tables, verbatim.
## Browser A/B
| Config | sim tick ms | render load | gaps/min | gpu ms | points | ceiling | notes |
## Diagnosis
Which loop saturates (render, sim, audio), with the numbers that show it.
## Changes made
Per change: what, why, before/after numbers, flag name if gated.
## Asks for the Mac agent
Engine-side problems or numbers that should change SOURCES_PLAN defaults.
```

## Step 3 — fix what the numbers point at

Work down WINDOWS_PLAN in order, but only where your Step 1 numbers
show the problem:

- **W2 governor hook:** shed the frame cap before audio quality when
  GPU dispatch time climbs.
- **W3:** audio auto-fallback to `balanced` latency on the next session
  after heavy gaps, and the Bluetooth hint.
- **W4:** WebGPU dispatch hygiene on D3D12 (deeper buffer pool, one
  submit per tick, staging ring).
- Anything new you find. Gate risky changes behind a URL flag first,
  A/B them, then make the winner the default.

For each change: before/after numbers in the report, no regressions on
`node tools/web_smoke.mjs`, and keep the Mac path unchanged unless the
change is clearly better everywhere.

## Rules

- Never push to `main`. Push `windows/<topic>` branches; the owner
  merges.
- Rebase on `main` before pushing. If `web/omg_web.wasm` conflicts,
  take `main`'s Rust, rebuild the wasm, and commit the rebuilt file.
- Commit the wasm only when Rust sources changed.
- Do not commit audio files as mp3 or wav; `.gitignore` drops them and
  the deployed page then 404s. Use Ogg FLAC like the other assets.
- Re-run Step 1 after every change set and update the report.
