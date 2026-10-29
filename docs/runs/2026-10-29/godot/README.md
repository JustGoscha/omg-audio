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
| fly | v3 | outside -58.2; crossing -40.7; inside -38.0 | 1.2–17.4 | rung 2 at the start; 0.4s → 3, 2.4s → 4, 2.9s → 3 | 0 / 5 µs |
| fly | v6 | outside -58.0; crossing -45.5; inside -39.2 | 1.2–18.4 | rung 2 at the start; 0.5s → 3, 2.5s → 4, 3.0s → 3 | 0 / 3 µs |
| car | v3 | approach -50.9; pass -51.6; away -21.1 | 1.0–6.6 | rung 2 at the start; 0.3s → 3, 2.3s → 4, 4.4s → 5, 6.4s → 6, 8.4s → 7, 10.4s → 8 | 0 / 13 µs |
| car | v6 | approach -52.5; pass -47.5; away -19.9 | 1.0–6.6 | rung 2 at the start; 0.3s → 3, 2.3s → 4, 4.3s → 5, 6.3s → 6, 8.4s → 7, 10.4s → 8 | 0 / 11 µs |
| door | v3 | closed -45.7; opening -47.7; open -47.4; shutting -39.1; shut -13.5 | 4.0–17.5 | rung 2 at the start; 0.5s → 3, 2.3s → 2, 5.2s → 3, 6.1s → 2 | 0 / 5 µs |
| door | v6 | closed -41.4; opening -46.9; open -46.0; shutting -54.8; shut -34.3 | 3.8–17.2 | rung 2 at the start; 0.5s → 3, 2.2s → 2 | 0 / 3 µs |

v6 runs v3's estimate today (its refinements come next): v3 and v6
differ by run-to-run variance only — the real-time host isn't
deterministic.

**What changed today, as the game plays it** (fly, before → after):
the tick 4.5–83.5 ms → 1.2–18 ms (auto now holds rung 3–4 in the furnished
cathedral instead of a fixed low tier whose ticks overran the 50 ms
period); every artifact window inside the cathedral under −38 dB (was
−32.5); no audio callback late (a 23 ms stall ~13 s after load, from
levelling the ambience bed on the render path, is gone).

**Open** (OPEN.md): the car's bursts once it is ~120 m north (an
occlusion switching at once — the morning's engine does the same); the
door's lag (after the leaf shuts the tone keeps falling for 2+ s) and a
treble floor after it that varies run to run; the flute's level wobbling
±2 dB over 10 ms windows while hovering (the estimate's noise at the low
rungs).
