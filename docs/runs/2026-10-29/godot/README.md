# Godot recordings — 2026-10-29: importance budgeting, the quality ladder, the threaded walk

What the Godot demo **plays**, recorded in real time from its audio
thread (`OmgWorld.start_recording`): the real host — the simulation
thread, the output thread, Godot's audio callbacks, the window — with
quality on **auto** (the level governor at a 16 ms tick target) and the
walk on 4 threads (M1 Max).

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

Recorded again in the evening with the afternoon's fixes (engine at
b16bcfa: the ambience fader, a door's averages, the corner leak; the
car at aa02443: a source's place stamped when it was first sent), the
machine otherwise idle. Worst 5 ms window above 3 kHz under the whole,
per phase (dB); the tick's range (ms); where quality auto went; the
audio thread (callbacks with frames rendered late / slowest take).

| scenario | engine | worst artifact by phase | tick ms | quality auto | audio thread |
|---|---|---|---|---|---|
| fly | v3 | outside -53.4; crossing -41.8; inside -38.7 | 1.2–16.5 | rung 2 at the start; 0.7s → 3, 6.3s → 4 | 0 / 3 µs |
| fly | v6 | outside -54.8; crossing -40.8; inside -42.0 | 1.2–15.7 | rung 2 at the start; 0.6s → 3 | 0 / 3 µs |
| car | v3 | approach -51.6; pass -54.0; away -41.1 | 1.0–7.0 | rung 2 at the start; 0.6s → 3, 2.6s → 4, 4.7s → 5, 6.7s → 6, 8.7s → 7, 10.7s → 8 | 0 / 7 µs |
| car | v6 | approach -53.6; pass -53.1; away -44.9 | 1.0–6.0 | rung 2 at the start; 0.6s → 3, 2.7s → 4, 4.7s → 5, 6.7s → 6, 8.7s → 7, 10.7s → 8 | 0 / 5 µs |
| door | v3 | closed -42.4; opening -47.5; open -48.5; shutting -49.3; shut -61.7 | 4.2–19.2 | rung 2 at the start; 0.6s → 3 | 0 / 3 µs |
| door | v6 | closed -61.3; opening -47.4; open -48.5; shutting -49.2; shut -41.4 | 4.4–26.5 | rung 2 at the start; 0.7s → 3 | 0 / 3 µs |

The morning's recordings, for comparison: the car away −30.3 (v3) /
−28.0 (v6) dB — the ambience bed bursting in (below) — and the door
shut −38.4 / −32.2 dB. v6 here has its refinements on v3 (ENGINE_V6.md);
both engines play the reverb through the diffused network and walk on
4 threads. The tone runs turn the ear off (its tinnitus is the ear
model's) and silence the demo's own one-shots and beds.

**The door, as played** — the tone's level after the shut command (6 s),
re the closed level before it opened (dB):

| | +0.5 s | +0.75 s | +1.0 s | +1.5 s | +2.0 s | +2.5 s | +2.9 s |
|---|---|---|---|---|---|---|---|
| v3 | +32.8 | +30.4 | +27.2 | +22.0 | +16.8 | +11.8 | +8.8 |
| v6 | +37.6 | +30.1 | +8.8 | +0.3 | +0.1 | +2.2 | +1.3 |

(the leaf eases shut over ~1 s; the closed level before it opened is
itself a young estimate, a few dB low — OPEN.md.) In the morning v6
restarted at every tick of the swing (lurching), at noon only as it
started (+9 … +11 dB still 2.9 s after the command, depending on
whether the leaf happened to pause before it closed).

**What changed today, as the game plays it**

- Morning (fly, before → after): the tick 4.5–83.5 ms → 1.2–18 ms (auto
  holds rung 3–4 in the furnished cathedral instead of a fixed low tier
  whose ticks overran the 50 ms period); no audio callback late (a 23 ms
  stall ~13 s after load, from levelling the ambience bed on the render
  path, is gone).
- Afternoon: the car's bursts at ~120 m were the ambience bed — decoded
  ~13 s after load, muted, it burst in at full level because its fader
  only moved while a bed played (fixed in the mixer: a silent channel's
  fader takes its value at once). The door: a leaf easing shut now
  restarts the averages as it starts and where it stops, the late
  average following quickly on its way. A bounce in a corner no longer
  sees a speaker through the wall beside it.

- Evening: every moving source's pitch jittered at the frame rate — the
  demo moves its cars on 60 Hz physics and sends their places every
  120 Hz frame, each send stamped with its own time (the track stood
  still a frame and jumped at twice the speed the next). A place sent
  again now keeps the time it was first sent there. The car's closest
  pass had a one-sample step in both engines (−38 … −45 dB above 3
  kHz); now −53 / −54 dB.

**Open** (OPEN.md): a shut door's field is a heavy-tailed
estimate (it reads low right after a restart and steps up as rare
strong paths come); the estimate's wobble at the low rungs while moving.
