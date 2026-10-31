# Portals A/B in Godot (Oct 31, night)

The tour from the square to the walk south (`--from=12 --until=56`: the
club's lobby and the club, the square with the club open, the
cathedral, the bunker, the bunker path), recorded in Godot twice on
the same build: `tour_v6.2` the engine as it ships, `tour_v6.2_portals`
with `--portals` (`Config::portals`: a room heard through its openings
from its sources' side — `crates/omg-core/src/v3/portal.rs`, OPEN.md).
No video (the governor free to climb). Each with its `.m4a` to listen to,
`.scan.txt` (`examples/tour_scan`), `.frames.csv`, `.tour.csv`.

| | tick | rungs (frames) | flagged moments | worst |
|---|---|---|---|---|
| shipped | 11.0 ms | 2: 1842, 3: 3221, 4: 834, 5–7: 1745 | 7 | +43 dB to −7 (path, 52 s); +38.5 to −10.6 (bunker path, 71 s) |
| portals | 13.7 ms (portals 6.1) | 0: 5486, 1: 683, 2: 1472 | 12 | +49 dB to −6 (path, 52 s, 16 s); +42 to −11 (south, 104 s) |

A first portals take (its cost then counted as scaling: the governor at
rung 0 throughout) had its worst moments milder — the path +25 dB, the
bunker path +17, the lane +12 — and more small ones; this one, its cost
counted as fixed, has them as large as the shipped engine's. Takes
differ from run to run (the engine traces in real time); one pair is weak
evidence either way, but nothing here says portals are ready: they cost
6 ms a tick in Godot (the governor sheds rays for them), and far from
the club the bursts stay. Headless (`examples/portal_ab RAW=1`) they are
steadier near the club (six-dB jumps 366 → 78 in 790 ticks at its
entrance) and converge where the walk doesn't — OPEN.md. Off by default.
