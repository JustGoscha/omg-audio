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
- A loud room heard through its walls: its tail converges slowly (see
  the next-event section below; OPEN.md).
- Town tick ~48 ms native High with every source traced (+5 ms tonight:
  two-edge paths and splits; 56 ms after the next-event draws below).

## Later the same night: next-event draws

Every bounce connects to sources (next-event estimation). With more
sources than a tier's `nee`, each bounce used to draw `nee` of them
with replacement, ∝ loudness / distance², and weight each draw by
1/(nee·p). Two faults, measured with the club clicking while the town's
other sources click around it (`assess crowd`, its own channel, `RUNG=4`
so the mixer's budgets take nothing):

- A rare draw weighed the sum of all the sources: beside the fountain,
  the club seen through its door came in hundreds of times over. At
  the street north, the third click came back +5 dB and stayed raised
  for 20 s while the running mean diluted it.
- Sources behind the surface were drawn and then skipped: a quarter of
  the connections were wasted.

Now each source the surface faces is connected at most once, with
probability π ∝ what it would bring there with nothing in the way
(capped at 1: the near loud ones for sure), weighted 1/π (`v3.rs`
`pick_systematic`, the GPU walk alike). Unbiased; no connection weighs
more than the share of the sources not connected for sure. Raw outputs
in `nee/`.

Crowd, ten identical clicks (mid band, dB):

| | before: first 0.25 s | before: after | **after: first 0.25 s** | **after: after** |
|---|---|---|---|---|
| street north | −34.9, sd 1.78 (a +5 dB jump) | −43.4, sd 0.56 | −36.9, sd 0.56 | −40.9, sd 1.33 |
| square | −39.9, sd 1.15 | −42.7, sd 1.52 | −37.7, sd 0.51 | −41.1, sd 1.90 |

Every source in the town at five places, 2 s after a standing start
(40 ticks, 8 seeds), against a reference of 8× the rays, 16× the late
rays, every source connected, 100 ticks (`examples/nee_ab.rs`); the
sources within 20 dB of the loudest at each place (`audible_*.txt`;
all within 60 dB: `all_*.txt`):

| error vs reference | before: mean / >6 dB | **after (High, nee 4)**: mean / >6 dB | after at nee 3 (Low, Medium) |
|---|---|---|---|
| early low | 3.27 dB / 15.9 % | 2.79 dB / 11.7 % | 3.17 dB / 12.9 % |
| early mid | 2.50 dB / 10.5 % | 1.53 dB / 3.3 % | 2.51 dB / 13.2 % |
| early high | 4.57 dB / 23.6 % | 4.35 dB / 21.5 % | 5.20 dB / 25.0 % |
| late low | 5.30 dB / 26.9 % | 4.30 dB / 23.1 % | 4.13 dB / 21.2 % |
| late mid | 6.28 dB / 41.4 % | 4.69 dB / 24.2 % | 4.92 dB / 31.2 % |
| late high | 6.60 dB / 41.1 % | 5.26 dB / 32.1 % | 5.71 dB / 33.0 % |

Nee 3 traces as many connections as the old nee 4 did (2.32 a bounce
against 2.31): the tiers below High keep their cost, High spends more
on correctness. The connections now reach farther sources (more wall
crossings to trace); a branch-free slab test in the BVH (exactly the
old one, NaN cases included) gives 6 ms of it back. Tick
(`examples/tick_ms.rs`, the town, every source traced):

| | before | after |
|---|---|---|
| native Low | 24.1 ms | 23.8 ms |
| native Medium | 35.9 ms | 35.3 ms |
| native High | 49.4 ms | 56.1 ms |
| web worker (wasm), Low | 32.4 ms | 33.4 ms |
| web worker (wasm), Medium | 47.7 ms | 51.6 ms |

(The web page's governor sheds a tier above 40 ms: the web runs Low.)

Not fixed: the tail of a loud room heard through its walls still
converges slowly (above, "after" sd 1.3–1.9 dB; the club's tail at the
square lands within ±2 dB of the reference only after ~20 s; the same
click's tail differs by up to 8 dB between seeds) — see OPEN.md, "A
loud room's field through its walls".

## Later still: the hand-over between convolution and its stand-in

The club's level in the clicking town depended on the mixer's rung:
at the default (96 paths, 4 convolved sources) its own channel was
2.2–3.3 dB under what it plays with every source convolved — and it
moves between the two whenever the convolution budget re-ranks the
sources, which is heard as the club popping. Separated (`TAPS=96` vs
`CONV=4` on the top rung, `standin/crowd_*_before.txt`): the path
budget costs nothing; the whole loss was the parametric reverb that
stands in for a source's convolution. At the ears it was off by up to
8 dB per band against the convolution it replaced
(`standin/ears_*`; `means.sh` averages a run):

| stand-in − convolved (dB, low / mid / high) | early, before | late, before | **early, after** | **late, after** |
|---|---|---|---|---|
| club from the square | +4.1 / +0.7 / −4.2 | +2.3 / −0.2 / −4.3 | +0.9 / +3.5 / +2.9 | +1.2 / +2.2 / +0.3 |
| club from the street north | −0.3 / −5.8 / −8.0 | +2.4 / −4.1 / −7.5 | +0.9 / +1.6 / +3.7 | +1.4 / +0.3 / +0.1 |
| club from the entrance | −0.1 / −2.3 / −2.1 | −2.2 / −4.6 / −6.0 | −0.2 / +2.4 / +4.6 | +0.8 / +0.3 / −2.2 |

Mean |error| 3.4 → 1.6 dB, worst 8.0 → 4.6 dB. What is left is time:
one decay per band from the free-field arrival can match a response's
energy and its centre time, not its shape — a field heard through
walls builds up first, or starts strong and rings long. Four faults,
each isolated in a DSP test:

- **Band shaping** (`fx_fdn_bands`): the reverb's send was two
  shelves relative to the mid (capped at ±12 dB; the one-sided part a
  low-pass fitted through 800 Hz): with the bass 20 dB over the mids,
  the mids came out +7 dB, the highs +14…+25 dB. Now the bands as the
  kernel synthesizes them (`BandSplit::shape`), and the three sends
  solved together so each band plays the energy it is given through a
  network whose treble decays faster (`Fdn::band_make_up`): within
  0.2 dB at equal decays, 1.7 dB otherwise.
- **The one-sided share** (`fx_reverb_handover`): the diffuse and the
  one-sided networks were identical, fed the same input — copies,
  adding coherently in W: +3 dB. The second network now has its own
  line lengths and orthogonal injection signs (`Fdn::second`).
- **When it arrives**: the stand-in decayed at the response's late
  slope (T20) from the free-field arrival; a field through a door is
  front-loaded, so 50 ms after the club door shut its first 100 ms
  came 3.5 dB short (`town_system`
  `a_door_shut_between_sounds_is_shut_for_the_next_one`). Now the decay
  that gives the network's output the response's centre time — its
  energy arrives on average as late (`DenseResponse::centre_time`,
  `fdn::rt60_for_centre_time`).
- **Several sources of one signal** (the club's four speakers):
  identical networks in every renderer, so four speakers' stand-ins
  added coherently (up to +6 dB) where their kernels — of different
  noise seeds — add as energies. Every renderer the mixer makes now
  has networks of its own (`Renderer::decorrelate`).

Crowd, the club's own channel (dB, first 0.25 s / after), default rung
against every source convolved:

| | before: rung 1 | before: rung 4 | **after: rung 1** | **after: rung 4** |
|---|---|---|---|---|
| square | −39.8 / −44.1 | −37.6 / −41.6 | −36.7 / −43.3 | −37.6 / −41.6 |
| street north | −39.4 / −44.1 | −36.8 / −40.8 | −35.8 / −41.9 | −36.8 / −40.8 |
| entrance | −19.6 / −26.6 | −19.4 / −26.2 | −19.3 / −26.7 | −19.4 / −26.2 |

The club door shut, the click δ later against settled (from the
entrance, `PACE=0`: kernels rebuilt in the tick, one a tick — the
stand-ins play meanwhile; `standin/door_*`):

| δ | before: early | before: late | **after: early** | **after: late** |
|---|---|---|---|---|
| 0.05 s | −3.6 / −1.6 / −1.2 | −5.8 / −4.9 / −5.0 | −2.1 / −0.6 / +0.1 | −4.5 / −4.2 / −4.0 |
| 0.1 s | −2.6 / −1.6 / −0.8 | −5.3 / −4.5 / −4.3 | −1.5 / +1.0 / +3.5 | −4.9 / −4.0 / −3.1 |

Render cost unchanged (web 18.5 % of real time; Godot stress 2.93×).

## Claps: the tail's grain

"Claps have a crackling late reverb": crackle is a tail made of a few
isolated clicks over a quieter field. Measured as the kurtosis of the
output (E[x⁴]/E[x²]², 20 ms windows, energy-weighted) over the 0.5 s
after each click's early part — a dense reverberant tail is noise-like
(≈ 3), sparse clicks read far higher (`assess still`, the tail kurtosis
line; `crackle/`):

| tail kurtosis, low / mid / high | v5.5 | v3 mode, same engine |
|---|---|---|
| a clap in the hall | 2.9 / 3.1 / 3.1 | 3.6 / 9.9 / 64.7 |
| the club from the entrance | 2.6 / 3.1 / 3.2 | 2.9 / 7.1 / 44.1 |
| a clap on the open square | 5.3 / 17.6 / 38.2 | 28.5 / 172.8 / 388.8 |

v5.5's tails indoors are noise-like. On the open square a clap's tail
is a sparse train of facade echoes — physically so; v3 makes it ten
times spikier.

## A closed door sounds closed: the bands

The tracer gives energies in three bands (centres 125 / 790 / 5600 Hz);
the renderer plays them by splitting the sound into bands and summing
them at their gains. The kernel's noise, its impulses, the stand-in's
send and the beds seeping through walls split with one-pole crossovers
(6 dB/octave): the loudest band's skirt covered the others. A response
shaped like a wall's transmission — bass 0, mids −14, treble −34 dB —
played its mids at −7 and its treble at −18 dB (`fx_band_colour`): the
field behind a closed door far brighter than traced. Now
Linkwitz–Riley crossovers (24 dB/octave; `bands::BandSplit`, the low
band through the upper crossover's all-pass so equal gains stay flat):
−13.1 and −32.9 dB. The impulse rig splits its output the same way
(kept in the rig, so engines of any age read alike — numbers from here
on are in these bands; earlier tables in one-pole bands).

The club from the entrance, door shut against open, settled (dB,
low / mid / high; `bands/door_*`):

| | before | after |
|---|---|---|
| early | −25.4 / −30.0 / −37.9 | −25.9 / −34.6 / −41.1 |
| late | −23.8 / −27.7 / −33.9 | −24.0 / −32.8 / −38.6 |

The bass leaks as it did; the mids and treble are 3–5 dB further down,
as the door's transmission says. Right after a shut the field through
the door is a young estimate and its treble reads low (the click 50 ms
after: −2.6 / −2.6 / −5.2 dB against settled; one-pole bands hid it
under the bass's skirt). The stand-in still follows the convolution
(`bands/ears_*`: mean |error| 1.4 dB). Render cost: the split costs nine
second-order sections where it ran two one-poles — web render 18.7 →
19.2 % of real time, Godot stress 2.85 → 2.60× (per-path taps keep their
own shaping: OPEN.md).

## Taps: each path's own band shape

Every path (the direct sound, the solved echoes, the edge bends) carries
three band gains. They were played by two first-order shelves relative
to the mid, capped at ±12 dB — a −20 dB band near −8 (`fx_tap_eq`'s
`deep_band_cuts_reach_their_band_gain`, ignored as a known bug) — or,
bass-heavy, by a one-pole low-pass fitted through 800 Hz that ignored
the treble gain (a wall's −34 dB treble played 28 dB high). Now two
second-order shelves at the band edges (`filter::Shelves`, their
coefficients following the gliding gains every 16 samples): each band
centre within 1.7 dB for steps up to 20 dB between neighbouring bands,
a wall's whole 62 dB span within 2–4.3 dB; the deep-cut test passes.

The bell across the open field — its direct sound's treble carrying
261 m of air absorption — early treble −18.1 → −21.0 dB (`taps/`).
Render cost: web 19.2 → 20.1 % of real time; Godot's 480-path rung
7.3–7.6× → 6.4× real time.

## Moving paths: no warble

A path reads its source at a fractional delay. Read linearly, the
treble falls by how far between two samples the read lands (at 10 kHz
−2 dB at half a sample, none on one), so a path whose delay slides —
every path while you walk — had its treble modulated as the fraction
cycled: at a walk ~200 times a second, a 10 kHz tone's level swinging
2.2 dB (`fx_moving_treble`: five-cycle windows). The play log and the
live delay line now read with a cubic (Hermite, through two samples
either side; exact on whole samples): 0.96 dB at a walk, 0.47 dB at a
car's 30 m/s. A one-shot's interpolation reaches two samples before its
start and one past its end. The car at 100 km/h still arrives within
0.4 ms. Render cost: web 20.1 → 20.5 % of real time, Godot's 480-path
rung 6.4 → 5.8×.

