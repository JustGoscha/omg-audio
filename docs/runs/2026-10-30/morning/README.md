# The morning's build in Godot and headless (Oct 30)

The engine at 2879a2d (main): every echo's twin off the ground (06c6110)
and a reflection bent at an edge keeping its energy (83b53b8) — see
ENGINE_V6.md, "Third order, and every chain's twin off the ground" and
"A reflection bent at an edge keeps its energy". Recorded as the night's
final takes were (`../final/`, df7f9d5): Godot's defaults, the walk on
the GPU, quality auto, from the audio thread, 0 frames lost in all 25
takes. `sh tools/record_runs.sh docs/runs/2026-10-30/morning v6.2 "fly
car door bunker curtains"`.

- `<scenario>_v6.2.m4a`: the demo's own sounds, to listen to.
- `<scenario>_v6.2_tone.m4a`: one source playing a steady 500 Hz tone,
  everything else off and the ear stage off — whatever the output holds
  above 3 kHz, the engine made (fly: the flute, car: the car, door: the
  club). bunker and curtains have no tone mode: their `_tone` files are
  a second take with the demo's sounds.
- `.txt`: the analysis (`analyze_rec`), `.frames.csv`: per game frame.
- `fly_takes/`: 8 tone takes of the flight on this build and 8 on
  ee8994b (the build before the morning's engine changes), the same
  session.
- `headless/`: the A/B runs behind the morning's commits (below).

## The tone runs

Above 3 kHz under the tone, 5 ms windows; level jumps: 10 ms off the
90 ms course. The night's final take (df7f9d5) beside it.

| scenario | worst window | over −40 dB | jumps > 4 dB | tick | night: worst / over −40 / jumps |
|---|---|---|---|---|---|
| fly | −31.6 dB | 13 | 31 | 3.7–19.2 ms | −35.2 / 3 / 55 |
| car | −44.2 dB | 0 | 85 | 2.7–9.0 ms | −45.3 / 0 / 65 |
| door | −44.9 dB | 0 | 1 | 6.8–22.8 ms | −36.3 / 5 / 2 |

bunker (walking along it): tick 8.1–18.0 ms; curtains (rolled up, then
down): 5.0–15.6 ms; no frame late in any take.

**The flight's 13 windows are a take, not the build.** Eight tone takes
of the flight on each build, the same session, the windows over −40 dB
(crossing and inside):

| build | takes | median |
|---|---|---|
| ee8994b (before the morning) | 2, 0, 1, 4, 7, 1, 11, 5 | 3 |
| 2879a2d (now) | 13, 4, 0, 3, 0, 0, 8, 4 | 3.5 |

Headless runs on the dumped Godot level meant to use the GPU walk
(`assess fly SIGNAL=tone COMPUTE=gpu GEOMETRY=…`) read alike on all three
builds — **but they walked on the CPU**: `assess` was built without the
`gpu` feature, and `COMPUTE=gpu` falls back to the CPU walk without
failing (correction, Oct 30 midday; `demo/web/wasm/examples/cellar.rs`
says which walk ran). On the GPU walk the walks land when they land; the night's four takes of df7f9d5 were 3 / 0 / 5 /
2. The car's level jumps (85 against the night's 65) are one take each;
headless (`headless/motion/*cartone*`, three seeds) the car's tone is
the same on every build to 0.1 dB rms.

## headless/

Deterministic (`PACE=0`) unless noted; `<commit>_…` per build: ee8994b
(before the morning), 06c6110 (the twins), 83b53b8 (the edge gate).

- `suite/2879a2d.txt`: the whole suite on main — 421 passed, 0 failed;
  the GPU tests 3 / 3; the CPU–GPU parity 34 / 34.
- `shoebox/`: `examples/shoebox_ref` (v6.2 against the image sources).
  The first 80 ms: −0.6 / −0.8 / −1.3 → +0.2 / +0.1 / −0.1 dB (default),
  −1.4 / −1.4 / −1.8 → −0.6 / −0.4 / −0.5 (a corner source), −0.6 /
  −0.7 / −0.6 → +0.1 / +0.1 / +0.3 (scattering 0.5); the late part the
  same on both builds (the near-surface bias, OPEN.md).
- `wall_echo/`: `examples/wall_echo`, one mirror wall, the echo against
  its image source over the sphere's size k (8192 rays, three seeds). No
  ground, 10 m wall: −1.2 … −1.6 dB at k 0.06 before, +0.1 … +0.5 now;
  with the edge bends off −0.1 on both. With the ground, a 3.1 m wall:
  −0.6 → −0.2 dB at k 0.06.
- `still/`: `assess still` at nine places of the town (v6.2), the early
  and late energy of each click. Most places within 0.3 dB; where edges
  stand near the paths more plays: the clap in the hall +1.3 dB early,
  the bell from the north +1.7 … +2.3 early and +1.1 … +2.8 late, the
  clap on the square +0.5 … +0.9 early.
- `motion/`: `assess fly SIGNAL=tone` (the town; `godotfly`: the Godot
  level, CPU walk), `doorswing`, `cartone`, seeds 1–3, every build: no
  window over −40 dB added anywhere; the door swing climbing back 21.4
  dB on average (ee8994b and 06c6110 alike) and 21.0 on 83b53b8.
- `club/`: `examples/club_levels` (v4), the club heard at 13 places with
  the doors open and shut: +0.05 dB on average, +0.6 at most.
- `echo_check/`: the square's late echoes of third order and their
  ground twins (`demo/web/wasm/examples/echo_check`): their twins appear on
  06c6110 (+182.9 ms −22.0 dB under the flutter's −20.6; +97.6 ms −15.3
  under −13.8), not on ee8994b.
- `third_order/`: `echo_ref3` over 25 places, the doors as the town has
  them: the wall-only third-order echoes over −30 dB.
- `cost/`: `examples/tick_ms` (the town) and `bench_sources` (64 / 256
  sources, v6.2) per build. v6.2 native 19.8 → 20.3 → 20.2 ms, web 12.4
  → 12.8 → 12.9; v5 web low 11.5 → 13.4 → 13.5 (the twins: every
  coherent chain is v5's solver's); bench 64 sources 16.3 → 17.3 → 17.1.
