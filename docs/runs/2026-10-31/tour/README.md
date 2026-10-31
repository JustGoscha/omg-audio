# The tour, Oct 31 (night): the final render

The whole level walked by the recorder's `tour` scenario —
`sh tools/record_tour.sh docs/runs/2026-10-31/tour v6.2 10` — on the
night's build: Godot's defaults (quality auto), recorded from the audio
thread, 0 frames lost, 306 s. The route is the evening's
(`docs/runs/2026-10-30/tour/README.md`); what changed since that take:

- **The ears at 1.6 m.** The tour floated the feet 0.4 m over the
  floors: the ears rode at 2.0 m — the doorways' lintels. Walking out of
  the club they grazed one, which the engine takes for passing through a
  wall (every average restarts): the club's bass fell ~10 dB for half a
  second at its door. Gone (2c948e5).
- **The radio loops without a gap.** The clip faded to silence at both
  ends; it is a seamless loop now (f297c1b). The evening's take dropped
  out by 25 dB in the bunker every 15.6 s; this one doesn't.
- **Quality where it fits.** The evening's take ran on the quality
  ladder's rungs 0–2 (128–256 rays): the governor took the whole tick for
  scaling while 6–7 ms of it were mirror chains that cost the same on
  every rung. It now prices the next rung from the part that scales, and
  the chains cost about half (9881b40): this take ran on rungs 2–10, most
  of it 3–7 (384–768 rays), the tick 11.4 ms (target 16).

  | | tick | chains | rungs (frames) |
  |---|---|---|---|
  | Oct 30 | 10.8 ms | 6.7 ms | 0: 4176, 1: 2055, 2: 5284, 3: 1122, 4: 2548, 5–7: 1326 |
  | Oct 31 | 11.4 ms | 4.0 ms | 1: 10, 2: 2096, 3: 4225, 4: 2264, 5: 807, 6: 3081, 7: 2130, 8–10: 3759 |

- **Cars without a distance fade.** Both demos faded a car's motor out
  between 140 and 320 m ("perceptual envelope", against the manifesto):
  a car now plays at its own level at any distance — distance, air and
  the mix's audible window decide what is heard — and its slot frees once
  it is as far past as cars start out (900 m). The tour's two cars start
  900 m out (they popped in at 200 m), so the road stop waits 40 s.

## Files

- `tour_v6.2.m4a`: the take, to listen to; `tour_v6.2.mp4` (local, not in
  git): the window at 10 frames a second with the sound.
- `tour_v6.2_spectrogram.png`: the whole walk.
- `tour_v6.2.scan.txt` (`examples/tour_scan`): every step's level and
  three bands; what stands out (10 dB over the 2 s before, 150 ms or
  more, no event in the 0.6 s before) and what drops out (15 dB under).
- `tour_v6.2.txt` (`analyze_rec`), `.tour.csv` (when each step began),
  `.frames.csv` (each game frame: the ears, the tick by part, the rung).

## What the take shows

Expected, among the scan's 22 moments and 3 dropouts: the bells 0.7 s
after they are struck; stepping into the club's lobby (+35 dB, its inner
door then opened); the club heard straight through its two open doors
from the lane; walking out of the cathedral towards the open portal; the
cars passing at 242 and 247 s and receding; the stage's voice in view
through the dome's door, and at 1 m on the stage; the claps' tails
ending (the dropouts); single footsteps in the quiet.

Not expected — **bass bursts from the club** (its inner door left open),
the heavy tail of the field estimate: in the field south of the town
(120.0 s, the bass +41 dB to −9 dB for 3 s, the loudest moment of the
walk), in the cathedral 1.3 s after the clap (66.4 s, +37 dB), on the
bunker path (112.9 s), along the west road (281 s). A rare path — out
through the doors, bent round corners (each bend weighs the bass up to
3×), next-event connections near the speakers — lands in one tick's
estimate with a weight thousands of times the typical and decays as the
running mean dilutes it. Headless replays of the same stretches don't
burst at the same places: it is chance, not the place. The cure is the
open item in `OPEN.md` ("A loud room's field through its walls
converges slowly"): the field estimated from the sources' side too.
