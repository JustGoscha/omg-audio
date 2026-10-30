# The final build in Godot: v6.1 against v6.2 (Oct 30, night)

Every scenario of the Godot demo recorded with the engine as committed
(df7f9d5: v6.2, edge leaks fixed, echoes between two faces and their
ground twins, the wall split into rooms, the GPU walk through a
crossing — the CPU walking until a walk launched since lands — and the
dense early field played, arriving when the trace says). Godot's
defaults: the walk on the GPU, quality auto, recorded from the audio
thread (0 frames lost in all 20). `sh tools/record_runs.sh <dir> "v6.1
v6.2" "fly car door bunker curtains"`.

- `<scenario>_<engine>.m4a`: the demo's own sounds, to listen to.
- `<scenario>_<engine>_tone.m4a`: one source playing a steady 500 Hz
  tone, everything else off and the ear stage off — whatever the output
  holds above 3 kHz, the engine made (fly: the flute, car: the car,
  door: the club). **bunker and curtains have no tone mode**: their
  `_tone` files are a second take with the demo's sounds, and their
  treble numbers are the music's, not artifacts.
- `.txt`: the analysis (`analyze_rec`), `.frames.csv`: per game frame
  (where the ears were, the tick's cost, the rung).

## The tone runs

Above 3 kHz under the tone, 5 ms windows (a pure tone through a linear
engine has nothing there); level jumps: 10 ms off the 90 ms course.

| scenario | engine | worst window | over −40 dB | jumps > 4 dB | tick |
|---|---|---|---|---|---|
| fly | v6.1 | −31.9 dB | 5 | 42 | 2.1–17.4 ms |
| fly | v6.2 | −35.2 dB | 3 | 55 | 3.0–20.6 ms |
| car | v6.1 | −49.7 dB | 0 | 50 | 2.3–8.0 ms |
| car | v6.2 | −45.3 dB | 0 | 65 | 2.5–8.2 ms |
| door | v6.1 | −35.5 dB | 7 | 0 | 5.0–20.0 ms |
| door | v6.2 | −36.3 dB | 5 | 2 | 5.8–21.7 ms |

Single windows of −35 … −40 dB come and go from take to take: on the
GPU walk the walks land when they land, and no two takes are alike.
Four v6.2 takes of the same build: fly 3 / 0 / 5 / 2 windows over
−40 dB, the door 5 / 4 / 0 / 0 (the take before the dense early field's
timing followed the trace, 3310dca: 0 and 0; v6.1 between takes 1 → 5
in the fly). The headless runs, deterministic, are clean
(`../early/motion_follow/`: the flight's tone none over −40 dB, worst
−50.8 dB). door: the windows at 7.1–7.5 s come with the leaf sealing (a
fast drop of the level), in both engines. car: both engines alike (its
level jumps as many in each; their cause is not looked into here).

bunker (walking along it): the tick 6.0–14.7 ms (v6.1), 7.3–19.9 ms
(v6.2); curtains (rolled up 6 s, then down): 4.6–18.5 ms, 4.5–16.7 ms;
no frame late.

## What the first take showed, and what came of it

The first take of the night (at 913a97b, before df90f4f) had two things
that looked like v6.2 regressions:

**Treble flying into the cathedral (v6.2: 8 windows over −40 dB inside,
worst −35.2; v6.1: 1).** Headless on the town (CPU walk) the flight is
clean in every engine. On the dumped Godot level with the GPU walk
(`assess fly SIGNAL=tone COMPUTE=gpu GEOMETRY=godot_level.geo`, three
seeds) it reproduced — and in v6.1 as much as v6.2: 45–79 windows over
−40 dB inside, none on the CPU walk (`headless/fly_godotlevel_before_*`).
The flight's line runs through the nave's pillars; each crossing dropped
the averages and the GPU walk in flight, and that walk lands 3–6 ticks
late on the Godot level: none landed until the listener stopped. The
cathedral played its direct sound alone — reverb −180 dB where the CPU's
walk reads −20.6 dB — then all of it at once. Fixed (df90f4f, ENGINE_V6.md
"The GPU walk through a crossing"): the CPU walks the ticks just after a
crossing. After (`headless/fly_godotlevel_after_*`): 0 windows over −40
dB, three seeds, both engines; reverb −20.5 dB all the way.

The tick while flying, GPU walk, Godot level, v6.2
(`headless/fly_godotlevel_cost_*`):

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| before (no field) | mean 7.4, max 20.8 ms | 8.2, 19.5 | 5.0, 14.8 |
| after | mean 11.1, max 32.8 ms | 10.1, 29.7 | 10.4, 28.9 |
| CPU walk alone | mean 14.1, max 26.6 ms | 14.1, 18.9 | 14.0, 20.1 |

**Level jumps opening the door (v6.2: 8 over 2 dB).** Headless at the
Godot spot (`assess doorswing OPENING=1 LIS=21,31,1.6 RUNG=3`, three
seeds), the engine before the night's later commits (45452ce) against
913a97b: the same seed-dependent dips of 3–8 dB mid-swing in both
(`headless/dooropening_*`) — the swing's estimate, not a regression;
this take has none. With the ear stage on, the opened door's level sinks
~7 dB over ~2 s: the ear adapting to the louder sound. With it off
(`EAR=0`) the level steps to −6/−7 dB and stays, as the door standing
open reads (`headless/doortruth_ear0_*`: −5.8 … −7.3 dB).

## Reproduce

    export PATH="$HOME/.cargo/bin:$PATH"
    sh tools/record_runs.sh "$PWD/docs/runs/<dir>" "v6.1 v6.2" "fly car door bunker curtains"
    LOUDEST=-40 ./target/release/examples/analyze_rec docs/runs/<dir>/fly_v6.2_tone
    cargo build --release -p omg-web --example assess --features omg-audio/gpu
    EAR=0 SIGNAL=tone MODE=v6.2 COMPUTE=gpu SEED=1 GEOMETRY=<dumped level> \
      ./target/release/examples/assess fly

(The Godot level dumps with `OMG_DUMP_GEOMETRY=<file>` set when Godot runs.)
