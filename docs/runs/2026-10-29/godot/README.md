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

Recorded again at night with the facades' echoes named (engine at
aec9e83: one instant per frame since 83687bd — a frame carries the clock
it was traced at, the node stamps a frame's poses and places with the
last physics step, the mixer's windows begin where the previous one
ends, `../doppler/`; and first-order mirror chains named from the
world's faces, a rejection held only while nothing moves), the machine
otherwise idle; the fly scenario moves the ears on physics steps, as
the player does. Worst 5 ms window above 3 kHz under the whole, per
phase (dB); the tick's range (ms); where quality auto went; the audio
thread (callbacks with frames rendered late / slowest take).

| scenario | engine | worst artifact by phase | tick ms | quality auto | audio thread |
|---|---|---|---|---|---|
| fly | v3 | outside -51.0; crossing -37.0; inside -39.8 | 1.5–17.2 | rung 2 at the start; 0.6s → 3, 6.7s → 4 | 0 / 3 µs |
| fly | v6 | outside -55.1; crossing -40.8; inside -41.2 | 1.6–16.2 | rung 2 at the start; 0.5s → 3, 6.3s → 4 | 0 / 6 µs |
| car | v3 | approach -55.8; pass -54.3; away -52.0 | 1.5–10.4 | rung 2 at the start; 0.6s → 3, 2.6s → 4, 4.6s → 5, 6.6s → 6, 8.7s → 7 | 0 / 12 µs |
| car | v6 | approach -55.0; pass -56.1; away -45.6 | 1.5–8.0 | rung 2 at the start; 0.6s → 3, 2.6s → 4, 4.6s → 5, 6.6s → 6, 8.7s → 7, 10.7s → 8 | 0 / 6 µs |
| door | v3 | closed -45.6; opening -47.1; open -48.5; shutting -48.5; shut -50.6 | 4.2–18.8 | rung 2 at the start; 0.6s → 3, 5.5s → 4 | 0 / 10 µs |
| door | v6 | closed -60.7; opening -45.6; open -48.6; shutting -48.8; shut -44.5 | 4.3–23.9 | rung 2 at the start; 0.6s → 3, 5.3s → 4 | 0 / 7 µs |

v6 (the default) under −40 dB in every phase. v3's crossing here is one
window at −37.0 dB; two more v3 fly runs read −41.9 and −42.0 there —
single worst windows move a few dB from run to run (the evening's sets:
fly v3 crossing −41.8 at b16bcfa, −40.4 at 83687bd; v6 −40.8, −41.9).

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
| v3 | +33.8 | +31.5 | +27.5 | +23.3 | +19.0 | +13.3 | +10.2 |
| v6 | +32.7 | +28.2 | +19.4 | +0.8 | −2.9 | −2.2 | −3.2 |

(the leaf eases shut over ~1 s and lands between +1.0 and +1.15 s from
run to run — v6 at +1.0 s read +8.8, +9.6, +15.4, +18.5 and +19.4 dB
in five runs, all near the shut level by +1.25 s; later it sits within
±3 dB of the shut level, a young estimate's spread; the closed level before it
opened is itself a young estimate, a few dB low — OPEN.md.) In the morning v6
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

- Late evening: a frame is the world at one instant (`../doppler/`): in
  the game a car 30 m off warbled 9.0 Hz rms on a 500 Hz tone, the ears
  walking at 6 m/s 0.74 Hz; now 0.27 and 0.04 Hz, as headless.
- Night: the facades' echoes are named from the world's faces — a car
  passing the town's west face 30 m off is echoed ~0.45 s later, 10–13
  dB under, at every quality rung (it was heard at two rungs of eleven).

**Open** (OPEN.md): a shut door's field is a heavy-tailed
estimate (it reads low right after a restart and steps up as rare
strong paths come); the estimate's wobble at the low rungs while moving.
