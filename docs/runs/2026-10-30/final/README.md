# The final build in Godot: v6.1 against v6.2 (Oct 30, night)

Every scenario of the Godot demo recorded with the engine as committed
(df90f4f: v6.2, edge leaks fixed, echoes between two faces and their
ground twins, the wall split into rooms, the GPU walk through a
crossing). Godot's defaults: the walk on the GPU, quality auto, recorded
from the audio thread (0 frames lost in all 20). `sh tools/record_runs.sh
<dir> "v6.1 v6.2" "fly car door bunker curtains"`.

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
| fly | v6.1 | −37.5 dB | 2 | 34 | 2.2–15.9 ms |
| fly | v6.2 | −36.9 dB | 2 | 32 | 2.8–17.4 ms |
| car | v6.1 | −46.3 dB | 0 | 42 | 2.3–11.1 ms |
| car | v6.2 | −44.3 dB | 0 | 54 | 2.4–8.7 ms |
| door | v6.1 | −37.9 dB | 3 | 0 | 5.0–19.9 ms |
| door | v6.2 | −35.0 dB | 4 | 0 | 6.7–25.5 ms |

fly: v6.2's two windows are the wall crossing itself (2.57 s, where
outside becomes inside at once), none inside the cathedral; v6.1's two
are inside (2.89 s). car: both engines alike. door: the windows at
7.10 s (both engines) come with the last fast drop as the leaf seals
(v6.2: −27 → −34 dB within ~20 ms; v6.1: −36 → −41 dB).

bunker (walking along it): the tick 6.3–15.2 ms (v6.1), 6.1–16.5 ms
(v6.2); curtains (rolled up 6 s, then down): 5.3–23.6 ms, 4.1–16.3 ms;
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
