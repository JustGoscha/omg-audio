# Test runs — 2026-10-29: v5 → v5.5

The engine measured headless with **impulses only** (single-sample
clicks; no music, no loops), through the whole engine as the Godot host
runs it: the demo town's geometry, a 20 Hz tick with exact frames at
120 Hz between, kernels on their own thread, the render governor, the
mixer with the built-in HRIRs. The ear is off (`EAR=0`) so the numbers
are the engine's own. Tool: `demo/web/wasm/examples/assess.rs`.

- **before** — engine `6ecb9f6` (v5, before this session's work), with
  only `Mixer::set_ear` added to measure; raw outputs in `before/`.
- **v5.5** — engine at the commit that adds this file; raw outputs in
  `after/`. "v3 today" is v3 mode on the same engine.

Reproduce (each run paces itself in real time; `PACE=0` runs as fast as
it renders and builds kernels in the tick):

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release -p omg-web --example assess
MODE=v5 EAR=0 DELTA=0.1 target/release/examples/assess door
MODE=v5 EAR=0 SRC=club SPOT=square target/release/examples/assess still
MODE=v5 EAR=0 target/release/examples/assess walkout   # moving, teleport, ringout
```

The before-runs' click levels are lower than v5.5's at some places for a
reason of their own: the old play log lost part of a one-shot's first
sample at a fractional delay (fixed tonight) — compare within a run, not
absolute levels across.

### Door: the club door shuts, the club clicks δ later

Listener in the entrance; each click's response against the settled closed one (dB, low / mid / high; early = first 80 ms after the free-field arrival).

| δ | v5 before: early | v5 before: late | **v5.5: early** | **v5.5: late** | v3 today: early |
|---|---|---|---|---|---|
| 0.05 s | +12.0 / +18.1 / +27.3 | -2.3 / -1.9 / +1.1 | +0.2 / +0.7 / +1.8 | +0.8 / +0.8 / +0.5 | +19.9 / +29.3 / +36.9 |
| 0.1 s | +4.5 / +9.3 / +18.6 | -3.1 / -2.7 / -2.1 | -1.4 / -1.5 / -2.0 | +1.4 / +1.2 / +0.8 | +18.4 / +27.7 / +35.3 |
| 0.2 s | -2.5 / -1.8 / +2.7 | -1.3 / -0.8 / -1.0 | -1.9 / -2.2 / -2.6 | +2.0 / +2.0 / +1.2 | +15.6 / +24.8 / +32.4 |
| 0.4 s | -1.3 / -1.4 / +0.0 | -0.2 / +0.2 / -0.2 | -2.0 / -1.7 / -0.5 | +1.7 / +1.6 / +0.8 | +11.5 / +20.4 / +27.9 |
| 0.8 s | -0.3 / -1.3 / -0.2 | -0.6 / -0.3 / -0.5 | -2.4 / -2.0 / -1.5 | +0.1 / +0.2 / -0.3 | +7.0 / +15.5 / +22.8 |
| 1.6 s | +1.0 / +0.9 / +1.3 | -0.1 / +0.5 / +0.4 | -1.5 / -1.1 / -1.2 | -0.4 / -0.3 / -0.6 | +1.8 / +8.9 / +15.7 |

### Standing still: identical clicks, nothing moves

Spread of the clicks' energy (sd / span, dB, mid band) — ideally 0.

| source, listener | v5 before: early | v5 before: late | **v5.5: early** | **v5.5: late** | v3 today: late |
|---|---|---|---|---|---|
| club, from the square | 3.23 / 11.8 | 3.36 / 12.4 | 0.72 / 2.7 | 0.64 / 2.2 | 3.49 / 11.4 |
| club, from the street north | 5.89 / 21.2 | 3.66 / 12.4 | 0.20 / 0.6 | 0.35 / 1.1 | 7.45 / 23.2 |
| bell, on the square | 3.26 / 10.7 | 3.94 / 13.4 | 0.37 / 1.3 | 0.46 / 1.1 | 5.52 / 18.1 |
| bell, open field | 0.02 / 0.1 | 2.17 / 7.5 | 0.00 / 0.0 | 0.01 / 0.0 | 5.15 / 18.0 |
| a clap in the hall | 0.59 / 2.0 | 0.01 / 0.0 | 0.58 / 2.0 | 0.01 / 0.0 | 1.32 / 4.8 |

#### moving

v5 before:
```
moving (v5): a car at 100 km/h, 300 m off; click arrival − physical arrival (ms)
   -42 -40 -38 -37 -35 -34 -32 -31 -29 -27 -26 -24 -23 -21 -20 -18 -16 -15 -13 -12 -10 -8 -7 -5 -4 -2 -0 +1 +3 +4 +6 +8 +9 +11 +13 +14 +16 +18 +19 +21
   worst 41.6 ms
```
v5.5:
```
moving (v5): a car at 100 km/h, 300 m off; click arrival − physical arrival (ms)
   +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 +0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0 -0
   direct dB: -50 -51 -50 -51 -51 -51 -51 -51 -51 -50 -50 -51 -50 -51 -51 -50 -51 -50 -51 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50 -50
   worst 0.4 ms
```

#### walkout

v5 before:
```
walkout (v5): clap at 3 s in the hall, the listener walks out of its door 3.2–7.5 s
   tail: 1.90 s to −60 dB; 10 ms windows more than 4 dB off its course: 0
   envelope re peak every 0.1 s: 0.0s -11  0.1s -16  0.2s -17  0.3s -22  0.4s -25  0.5s -25  0.6s -30  0.7s -31  0.8s -38  0.9s -39  1.0s -42  1.1s -48  1.2s -43  1.3s -46  1.4s -54  1.5s -48  1.6s -51  1.7s -53  1.8s -53
```
v5.5:
```
walkout (v5): clap at 3 s in the hall, the listener walks out of its door 3.2–7.5 s
   tail: 2.02 s to −60 dB; 10 ms windows more than 4 dB off its course: 0
   envelope re peak every 0.1 s: 0.0s -9  0.1s -16  0.2s -19  0.3s -22  0.4s -25  0.5s -27  0.6s -31  0.7s -35  0.8s -36  0.9s -39  1.0s -39  1.1s -43  1.2s -44  1.3s -44  1.4s -45  1.5s -47  1.6s -49  1.7s -54  1.8s -59  1.9s -57  2.0s -57
```

#### teleport

v5 before:
```
(scene added tonight — it did not exist)
```
v5.5:
```
teleport (v5): a click from 100 m west in flight while the source jumps 4 m north
   click 1 (west, due 3.292 s): L -28.3 dB  R -42.2 dB  (L−R +13.9); before it -300.0 dB
   click 2 (north, due 3.212 s): L -4.7 dB  R -7.1 dB  (L−R +2.4)
   click 1 − click 2: -25.4 dB (spreading alone: -28.0 dB)
```

#### ringout

v5 before:
```
(scene added tonight — it did not exist)
```
v5.5:
```
ringout (v5): a click from 100 m; the source switched off 0.05 s after the second
   direct: on -34.4 dB, switched off in flight -34.4 dB
   after it (1 s): on -64.0 dB, switched off -63.0 dB
```


## Listening

`audio/*_before.m4a` / `*_after.m4a`: the same run on both engines, each
pair raised by the same gain (its loudest peak at −1 dBFS) so the pair
stays comparable. Clicks, so expect clicks and what the town makes of
them:

| file | what to listen for |
|---|---|
| `door_v5_0.05_*` | the click 50 ms after the club door shuts (6.05 s): before, the open door's paths still sound in it; v5.5, the shut door |
| `door_v5_0.1_*` | the same, 100 ms after |
| `still_club_square_v5_*` | ten identical clicks of the club heard from the square, nothing moving: before, they come back louder and softer at random; v5.5, the same click ten times |
| `still_club_north_v5_*` | the same from the street north of the club |
| `still_bell_square_v5_*` | the bell on the square: before, a few random rays (and ~30 dB too quiet); v5.5, the bell bent around the cathedral |
| `walkout_v5_*` | a clap in the hall, then walking out of its door: the tail rings from where the clap was |
| `moving_v5_*` | a car at 100 km/h, 300 m off, clicking 4× a second: before, the clicks bunch and spread (−42…+21 ms); v5.5, on time |
| `teleport_v5_after` | a click from 100 m west in flight while the source jumps next to you: it still arrives from the west, then the near one |
| `ringout_v5_after` | a click from 100 m, the source switched off while it is on its way: it arrives, and rings |

## What did it

See `git log 6ecb9f6..` — in order: path glides at 8 ms and warm reverb
hand-overs (no kernel cuts); emission positions on the timeline (moving
sources, ghosts for jumps, ring-out for switched-off sources); exact
two-edge diffraction (the bell); the estimator steady when nothing moves
and fresh when something does (splitting every ray at partitions, no
statistical restarts when still, every window restarting on a world
change with a ray burst, valid echo chains kept); the ear's comfortable
band; a channel that isn't rendering settles (no stale paths replayed);
ghosts inherit their place's response; a pathless reverb is heard.

## Not fixed (measured)

- v3 mode is still slow and noisy (its exponential averages): the door
  takes seconds to land, still clicks spread 3–7 dB.
- The ear (when on) ducks a club-loud place and lifts it back over ~4 s
  after the door shuts — by design for places above 80 dB SPL.
- Rare strong next-event samples (bounces right by a source) slow the
  running mean's convergence after a restart (~1–2 dB for ~1 s).
- Town tick ~48 ms native High with every source traced (+5 ms tonight:
  two-edge paths and splits).
