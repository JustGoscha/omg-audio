# Godot recordings — 2026-10-29: importance budgeting, the quality ladder, the threaded walk

What the Godot demo **plays**, recorded in real time from its audio
thread (`OmgWorld.start_recording`): the real host — the simulation
thread, the output thread, Godot's audio callbacks, the window — with
quality on **auto** (the level governor at a 16 ms tick target) and the
walk on 4 threads (M1 Max). Engine at the commit that adds this folder.

```sh
sh tools/build_godot.sh
sh tools/record_runs.sh <dir> "v3 v6" "fly car door"   # records, analyses, makes .m4a
```

Scenarios (`demo/godot/demo/record.gd`):

- **fly** — flying at 8 m/s from the field north through the cathedral's
  south wall to the flute, then hovering 3 s.
- **car** — standing 4 m from the west road while a car passes at
  100 km/h (from 150 m south to 150 m north).
- **door** — 1 m outside the club's closed inner door; it opens at 2 s and
  shuts again at 6 s while the club plays.

Each runs twice per engine: `_tone` — its source plays a steady 500 Hz
alone (every other source and the beds off), so whatever comes out above
3 kHz the engine made (the analyses, `*_tone.txt`); and without — the
demo's own sounds, to listen to (`*.m4a`). `*.frames.csv`: per game
frame, the ears' position, the tick's cost by part, the rung, the phase.

## Results (tone runs)

Worst 5 ms window above 3 kHz under the whole, per phase (dB); the tick's
range (ms); where quality auto went; the audio thread (callbacks with
frames rendered late / slowest take).

| scenario | engine | worst artifact by phase | tick ms | quality auto | audio thread |
|---|---|---|---|---|---|
| fly | v3 | outside -48.0; crossing -44.0; inside -39.9 | 1.0–14.3 | rung 1 at the start; 0.1s → 2, 2.1s → 3 | 0 / 8 µs |
| fly | v6 | outside -55.6; crossing -41.0; inside -41.7 | 1.2–16.1 | rung 2 at the start; 0.5s → 3, 6.3s → 4 | 0 / 3 µs |
| car | v3 | approach -50.3; pass -41.4; away -30.3 | 1.0–7.0 | rung 2 at the start; 0.4s → 3, 2.4s → 4, 4.5s → 5, 6.5s → 6, 8.5s → 7, 10.5s → 8 | 0 / 10 µs |
| car | v6 | approach -39.3; pass -51.1; away -28.0 | 1.0–5.8 | rung 2 at the start; 0.5s → 3, 2.5s → 4, 4.5s → 5, 6.5s → 6, 8.6s → 7, 10.6s → 8 | 0 / 6 µs |
| door | v3 | closed -38.5; opening -47.4; open -48.5; shutting -48.6; shut -38.4 | 4.3–25.0 | rung 2 at the start; 0.6s → 3, 6.0s → 4 | 0 / 15 µs |
| door | v6 | closed -61.6; opening -47.7; open -48.3; shutting -42.6; shut -32.2 | 4.2–23.7 | rung 2 at the start; 0.5s → 3, 8.9s → 4 | 0 / 3 µs |

v6 here has its refinements on v3 (ENGINE_V6.md): a door restarts the
averages, standing still they converge; both engines play the reverb
through the diffused network and walk on 4 threads. The tone runs turn
the ear off (its tinnitus is the ear model's) and silence the demo's own
one-shots (owls and bells on timers: in the first recordings they played
over the tone in some runs and read as artifacts).

**The door, as played** — the tone's level after the shut command (6 s),
re the closed level before it opened (dB):

| | 6.5 s | 6.75 s | 7.0 s | 7.5 s | 8.0 s | 8.5 s |
|---|---|---|---|---|---|---|
| v3 | +36.2 | +33.2 | +29.3 | +24.0 | +20.0 | +16.6 |
| v6 | +30.8 | +16.3 | −2.8 | −0.9 | +5.5 | +4.2 |

(the leaf swings ~0.5 s; v6's ±5 dB after is the low rungs' estimate
noise — the closed level itself wanders as much.)

**What changed today, as the game plays it** (fly, before → after):
the tick 4.5–83.5 ms → 1.2–18 ms (auto now holds rung 3–4 in the furnished
cathedral instead of a fixed low tier whose ticks overran the 50 ms
period); every artifact window inside the cathedral under −38 dB (was
−32.5); no audio callback late (a 23 ms stall ~13 s after load, from
levelling the ambience bed on the render path, is gone).

**Open** (OPEN.md): the car's bursts once it is ~120 m north (an
occlusion switching at once — the morning's engine does the same); the
estimate's ±2–5 dB wobble at the low rungs while moving.
