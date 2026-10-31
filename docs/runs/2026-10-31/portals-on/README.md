# The tour, Oct 31 (morning): portals on

The whole level walked by the recorder's `tour` scenario —
`sh tools/record_tour.sh docs/runs/2026-10-31/portals-on v6.2 10` — on
this morning's build: Godot's defaults (quality auto, portals on),
recorded from the audio thread, 0 frames lost, 306 s, the same seeded
route as the night's takes (`../tour/`: before tonight; `../earth/`:
nothing on the ground bending sound).

What changed since `../earth/`:

- **Rays cast four boxes at a time, triangles whole** (0c095ff): a cast
  570 → ~395 ns on the Godot level.
- **Portals on by default** (40af170 and after): an enclosed source's
  rays record what leaves each opening and bring their diffuse bounces
  past it straight to the ears (light tracing); a still source's bounces
  are kept once its records hold 32 ticks and drawn each tick by what
  they carry, eight a traced ray; no portal from a record younger than
  8 ticks or from another world; records that would start over are
  traced only once the source and the world stand a tick, two sources a
  tick at most, their rays split over the walk's threads.

The governor: tick 11.8 ms (11.5 in `../earth/`), portals 0.94 ms a tick
on average (0.5–0.6 ms steady, ~2 ms for a few seconds after a door or a
curtain), rungs 1–10 (`../earth/`: 2–10; the first portals-on take,
before the refills were split: 0–10, 2,228 frames on rungs 0–1, now 712
on rung 1).

## What the take shows, against the night's

`tour_scan` flags a level 10 dB over the 2 s before for 150 ms or more,
no scripted event in the 0.6 s before. Leaving out what is expected
(stepping into the club's lobby, the cars passing at 241–249 s, walking
out of the cathedral's silence after its clap, the stage):

| take | moments | unexplained | of them +20 dB or more | largest |
|---|---|---|---|---|
| `../tour/` (before tonight) | 22 | 16 | 9 (8.9 s) | +46.8 dB (south field, 3 s) |
| `../earth/` | 17 | 10 | 2 (0.8 s) | +42.7 dB (south, 0.5 s) |
| portals on | 20 | 13 | 4 (11.5 s) | +23.1 dB |

The club's bursts — +40 to +47 dB in the field, the bunker path, the
cathedral after its clap, the lane — are gone: nothing unexplained rises
more than 23 dB. Of what is left, the longest is not the engine's: at
209 s the tour stops on the road at (−13, −45) and the bass steps from
−52 to −32 dB, then holds near −35 — a car appearing 900 m along the
road (the demo spawns cars there). Recorded with every source stopped, a
car comes in from silence at −47 dB as its motor starts, portals on or
off; the club alone shows no step at that stop, in Godot or headless. A
95 dB motor 900 m off is ~30 dB SPL, inside the engine's audible window
on a quiet road: the engine hears what is there — the demo shouldn't make
a car appear within earshot (`OPEN.md`). The rest: a few +15…+21 dB
moments of 0.3–1 s in the field south of the town and on the bunker path
(the walk's heavy tail, smaller), and the road toward the dome.

## Files

- `tour_v6.2.m4a`: the take; `tour_v6.2.mp4` (local, not in git): the
  window at 10 frames a second with the sound.
- `tour_v6.2_spectrogram.png`: the whole walk.
- `tour_v6.2.scan.txt` (`examples/tour_scan`), `tour_v6.2.txt`
  (`analyze_rec`), `.tour.csv`, `.frames.csv` (the `portals` column: the
  tick's portal part in ms).
