# The tour: the whole level, one predictable walk (Oct 30, evening)

The final render of an engine change from now on: the Godot demo walked
through by a script — `sh tools/record_tour.sh <dir> v6.2 10` (the
recorder's scenario `tour`, `demo/godot/demo/record.gd`) — on 9a36901
(the cellar's fix, the ear's start, the pane fix). Godot's defaults (the
walk on the GPU, quality auto), recorded from the audio thread, 0 frames
lost; 224 s.

The route: beside the piano (the bells struck once, 0.7 s later they
arrive) → the corridor → the seminar (its door shut behind, the curtains
down, a clap) → out to the square → the club's entrance (its inner door
opened) → into the club → out across the square → the cathedral (its
portal opened, an owl, the curtains down, a clap) → back across the
square → down the lane to the bunker path → the blockhouse, down the
stair, through the hatch to the radio → back up → south to the echo
canyon, a clap at three places → the west road, two cars. The RNG is
seeded and every random event of the demo (owls, bells, cars, the clap's
and the steps' variation) is off its timer and played at a fixed point of
the walk: the same tour every take. The engine traces in real time on its
own threads, so its paths — not the tour — vary a little from take to
take.

- `tour_v6.2.m4a`: the take, to listen to; `tour_v6.2.mp4` (local, not in
  git): the window at 10 frames a second with the sound.
- `tour_v6.2_spectrogram.png`: the whole walk, 20 Hz – 22 kHz.
- `tour_v6.2.scan.txt` (`demo/web/wasm/examples/tour_scan`): every step's
  level and its three bands; what stands out (10 dB over the 2 s before,
  150 ms or more, no event in the 0.6 s before it) and what drops out
  (15 dB under).
- `tour_v6.2.txt` (`analyze_rec`): the audio thread and the engine per
  phase (its treble check is for tone takes; here it reads the music).
- `.tour.csv`: when each step began; `.frames.csv`: each game frame.

## What the take shows

- **Expected**: the bells arriving 0.7 s after they are struck (300 m),
  the club's 23 dB as its inner door opens, the claps' tails dying in the
  seminar and the cathedral, the radio up close, the cars.
- **The radio's loop**: down in the bunker the radio drops ~26 dB for
  0.4 s — the clip (`radio48.ogg`, 15.6 s) fades in over its first ~0.3 s,
  a gap every time it loops. Content, not the engine.
- **Bursts with the club open** (its inner door left open): across the
  square and the path 0.2–2.9 s of +12 … +25 dB, and in the field south of
  the town a 0.65 s burst +31 dB under 150 Hz — the club's bass. The map
  (`examples/club_map OPEN=4`) shows the square loud where the doors line
  up (−2 … −13 dB) and, further off, neighbours a metre apart 20 dB apart:
  the field reaching out through openings estimated from rare heavy
  paths (OPEN.md, "A loud room's field through its walls") — the
  "pockets of loudness" heard around the club. The cure is that field
  estimated from the sources' side too (bidirectional).
- **A dip at the club's entrance door**: stepping out of the entrance
  (19.4, 31) the bass falls ~20 dB for 0.7 s and returns. Not seen headless
  (the town's open door has no leaf box); in Godot the open leaf is an
  object of its own — next to look into.
- **The first take** walked through the kiosk (a straight leg across its
  corner): half a second of near silence inside it, rightly. The route goes
  round it now; the scan's dropouts list catches such legs.
