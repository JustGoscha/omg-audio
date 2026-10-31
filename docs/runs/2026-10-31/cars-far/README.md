# The tour, Oct 31 (late morning): cars from out of earshot

The same seeded tour (`sh tools/record_tour.sh docs/runs/2026-10-31/cars-far
v6.2 10`), the engine as in `../portals-on/` (portals on), and one change in
both demos and the recorder: **a car starts where its motor is out of
earshot**, not 900 m out.

The manifesto's rule for dropping a source — only when it is too quiet to
matter — holds for starting one too. From 900 m a 95 dB motor arrives at
~30 dB SPL: on a quiet road, a car out of nothing (`../portals-on/`: at the
road stop the bass stepped from −52 to −32 dB the moment the tour's car
started; recorded with every other source stopped, a car comes in from
silence at −47 dB). Now a car starts at the distance where its motor falls
under the threshold of hearing (~22 dB SPL at 125 Hz, ISO 226, the ground's
+3 dB): 2.3–6.3 km for the demo's motors (86–95 dB at 1 m), and drives in.
The tour's two cars set out at its start, 6.7 km away, to pass the road stop
when it stands there; the road's `car` steps are markers now.

The road, bass, dB (1 s steps):

| | 205–210 s (the stop) | 210–230 s | 231–236 s | the passes |
|---|---|---|---|---|
| `../portals-on/` | −51 → **−34** (the car starts) | −35 … −42 | −42 … −45 | −22 (242 s), −21 (247 s) |
| here | −51 … −60 | −55 … −60 | −47 … −45 (the cars come round into view, ~200–330 m) | −30 (238 s), −27 (243 s) |

The cars pass ~4 s earlier than the tour's steps assumed (its own pace),
and the scan lists the passes (237.8, 242.7 s) among its moments: its car
events are the road's markers, 28 s before.

Files: `tour_v6.2.m4a`, `tour_v6.2_spectrogram.png`, `tour_v6.2.scan.txt`,
`tour_v6.2.txt`, `.tour.csv`, `.frames.csv` (the mp4 local only).
