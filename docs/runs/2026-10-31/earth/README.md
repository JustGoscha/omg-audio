# The tour, Oct 31 (late night): nothing on the ground bends sound

The whole level walked again by the recorder's `tour` scenario —
`sh tools/record_tour.sh docs/runs/2026-10-31/earth v6.2 10` — on 0163e6f:
Godot's defaults (quality auto), recorded from the audio thread, 0 frames
lost, 306 s, the same seeded route as the night's take
(`docs/runs/2026-10-31/tour/`, the baseline here). What changed in the
engine since that take:

- **Nothing lying on the ground bends sound.** The edge set kept an edge
  where open air wraps round it by more than ~205°, probed in 32
  directions — and a level's ground is one sheet with nothing under it:
  the probes going down found nothing, so the foot of every wall and every
  paving slab's border read as an outside corner and bent every ray
  passing within metres (each bend weighing the bass up to 3×). A probe
  in the earth now counts as blocked: Godot level edges 3384 → 3128 (on
  the ground 229 → 25).
- **Edge strings merged where they are priced**, not filtered while
  searching (the bell over the cathedral found both its corner strings
  again); **a jamb's paths fade in** over 8–16 cm of opening.
- (Portals, off by default here, got light tracing.)

The governor ran the same: tick 11.5 ms (11.4), rungs 2–10.

## What the take shows, against the night's

| | night (baseline) | late night |
|---|---|---|
| moments standing out (scan) | 22 | 17 |
| dropouts (scan) | 3 | 3 |
| loudest unexplained | 120.0 s, south field: bass +46.8 dB for 3 s | 114.4 s, south: +42.7 dB for 0.5 s |

**Gone:** the club's bass bursts in the cathedral 1.3 s after the clap
(the night's take: bass −12 dB at 66.25 s, held at −27…−31 dB for 5 s;
now the clap's tail dies into a quiet nave, mid band −30 → −77 dB over
4 s, the take's third "dropout" — expected), in the lane (82.2 s, +22.8 dB
for 1.25 s), on the bunker path (112.9 s, +28.5 dB for 2.2 s), in the
south field (120.0 s, +46.8 dB for 3 s) and at the dome's door (284.9 s,
+32.4 dB).

**Left:** the same heavy tail, shorter. At 114.4 s, standing on the bunker
path at (26, 7), the club's bass reads −58 dB; the moment the walk sets
off (a moving listener's estimate is a young one) one tick brings it to
−12 dB, gone in half a second. In the square (77.5 s) and the lane
(85.5 s) single 250 ms bass spikes, +12 to +23 dB. Replayed headless
(`examples/tour_replay`, the Godot level, the club alone, rung 3, three
seeds, 76–86 s), against 6643f06:

| | low band mean | median | 50 ms steps ≥ 6 dB | largest |
|---|---|---|---|---|
| 6643f06 | −8.9 / −11.5 / −18.1 dB | −25 / −28 / −28 dB | 9 / 8 / 8 | +26 / +22 / +16 dB |
| 0163e6f | −8.8 / −11.3 / −13.0 dB | −21 / −22 / −20 dB | 4 / 5 / 4 | +25 / +14 / +13 dB |
| 0163e6f, portals | −9.6 / −10.1 / −12.3 dB | −16 / −14 / −16 dB | 2 / 2 / 5 | +15 / +16 / +8 dB |

The mean (what the estimate converges to) stays; the typical tick moves
up toward it and the jumps halve — more of the club's field arrives
steadily, less in rare bursts. With portals (off by default: their cost)
more so. The largest left (+25 dB at 85.65 s, seed 1, the same on both
builds) is one ray from the lane through the club's lobby door and its
open inner door, skimming its floor to the speakers (`PATHS=1`).

## Files

- `tour_v6.2.m4a`: the take; `tour_v6.2.mp4` (local, not in git): the
  window at 10 frames a second with the sound.
- `tour_v6.2_spectrogram.png`: the whole walk.
- `tour_v6.2.scan.txt` (`examples/tour_scan`), `tour_v6.2.txt`
  (`analyze_rec`), `.tour.csv`, `.frames.csv`.
