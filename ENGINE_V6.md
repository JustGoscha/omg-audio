# Engine v6 — steady in motion, quick at a door, cheap enough to run

v4 and v5.5 rendered the traced response as traced, and measured well
in single events (a click, one door), yet in the game they popped flying
into a room, the moving car jittered, and the ticks ran long. v3 —
discrete early arrivals and a parametric reverb gliding with the trace —
was steadier in motion. v6 (`Config { v6: true }`, the default in both
demos) starts from v3's estimate and fixes what was weak in it, one
measured step at a time (`demo/web/wasm/examples/assess.rs`,
`tick_move.rs`; the Godot recorder, `docs/runs/2026-10-29/godot/`).

Every step below was judged in motion against v3 (flying through the
cathedral wall into the flute, a car passing with a steady tone, the
club door opening and shutting) as well as on its own measurement.

## What a tick spends, by what the ear hears

- **Edge paths tracked every tick, searched by importance.** Each source
  hidden from the listener keeps the edges its last search found, and
  every tick solves them again for where both ends are now — bend
  points, legs, prices: exact and continuous however things move. The
  search for *which* edges carry it (every exposed edge for one bend,
  pairs for two) is rationed per tick (`Budget::edge_searches`,
  `edge2_searches`): the sources whose stale search costs the ear most
  first — arriving loudness (energy^0.3: twice as loud per 10 dB) × how
  far either end moved (in 20 cm steps). Standing still nothing is
  searched. A search looks at no more than 1024 edges for whether the
  ear (or the source) sees them, the most promising first.
- **The bend point in closed form**: both legs unfolded about the edge's
  line into one plane, where the shortest path is straight (it was a
  24-step golden-section search).
- **The quality ladder** (`sim::LADDER`): eleven rungs, cheapest first;
  the named tiers are rungs 4, 6, 8, 10. Every rung simulates the same
  physics — nothing is cut, every path reaches the ears — a lower rung
  only takes fewer samples of what matters least, shed first and
  hardest: two-edge searches (off below rung 3), then the early family's
  high bounces and the reverb's late rays, then rays and one-bend
  searches. Never shed: direct paths, their occlusion, the edges found.
- **Quality auto** (`governor::LevelGovernor`, shared by Godot and the
  web worker) targets 16 ms a tick — a 60 Hz frame; the tick shares the
  machine with the game. It sheds at once when two ticks in a row run
  over (as many rungs as the overrun asks, each ~1.45× the one below),
  climbs one rung after 2 s under 60 % of the target, and leaves out the
  extra-ray ticks after a door moved.
- **The walk on threads** (`Config::walk_threads`: half the hardware
  threads, at most 4; wasm one; `OMG_WALK_THREADS`). Each ray draws its
  own random numbers (seeded from the tick's, its family, its index) and
  what the rays leave is replayed in ray order: the same field to the
  last bit on any number of threads (`fx_walk_threads`). The tracer's
  maps hash with fixed keys, so the same seed and inputs repeat exactly.

Flying through the cathedral with all 19 town sources (M1 Max), the tick
per rung — one thread: 4.0 · 6.1 · 8.9 · 12.4 · 17.9 · 24.8 · 31.7 ·
46.0 · 57.9 · 92.3 · 130 ms; four: 2.0 · – · 3.8 · 5.8 · 7.5 · 9.8 ·
12.2 · 17.7 · 21.1 · – · 44.0 ms. Before: low 42.5 ms (two-edge searches
22.9 of it), high 80 ms. In the Godot demo, as played: the tick 4.5–83.5
ms → 1.2–18 ms; auto holds rung 3–4 in the furnished cathedral.

## A door moves: the averages start over

v3 averaged the open door's field into the shut one: 50 ms after the
club door shut its click still sounded +19 / +28 / +37 dB (low / mid /
high, early) above where it settles, +4 / +15 / +23 dB after 0.8 s. v6,
when a door or pane starts moving (by a millimetre or more), restarts
every source's averages as a running mean — the n-th tick after the
change weighs 1/n — until it is as steady as the EMA, which takes over.
The next 5 ticks trace 4× the late rays (≈ the 19 ticks' worth the late
EMA holds) and the first 3 twice the early rays: a mean of a few ticks
of a heavy-tailed estimate otherwise sits under the truth and steps up
as the rare strong paths come. 50 ms after the shut: within −4.5 … +3.9
dB early, ±3 dB late (`a_door_shut_between_sounds_is_shut_for_the_next_one_v6`;
+24.8 dB without the restart).

A door on its way — the Godot demo eases a leaf shut over a second —
restarts as it starts and again where it comes to rest; in between,
every tick is a moving tick, and the late average follows as quickly as
the early one (a door changes every order of the field at once, not the
slow drift of the tail as a listener walks). A change after a pause
shorter than the early average's memory, (2 − α)/α ≈ 5 ticks, is the
same motion going on (a leaf easing into its frame moves under the
millimetre for a tick or two before it closes). Each average counts the
ticks it holds (1/Σw² of its weights), and standing still a tick joins
it as the next tick of a running mean of that many.

Two designs before this one failed, both in the game. Restarting at
every tick of the swing played each tick's own heavy-tailed estimate:
the club's tone through its closing door climbed back 55–110 dB on the
way and lurched into the treble (−28 … −42 dB under the tone; `assess
doorswing EAR=0`, three seeds, two spots). Restarting only as the door
started kept the swing's field in the averages for seconds (+15 dB
1.5–3 s after the swing, `a_door_swinging_shut_takes_the_level_down_without_bursts_v6`;
in Godot +9 … +11 dB 3 s after the shut command). Now: 3–13 dB of
climbing back, the treble −44 … −50 dB (the last gap closing drops the
level ~25 dB in ~30 ms — physical); −1.9 dB from the door shut all
along 1.5–3 s after the swing (the test); in Godot within 6 dB 2 s
after the shut command, 0 … +5 dB at 3 s. Right after the restart where it stops, the young
estimate reads low (the heavy tail: OPEN.md).

(Measure with the ear off — `EAR=0`: the club plays at PA level, the
ear's adaptation recovers for seconds once the door shuts, and its
tinnitus after loud sound reads as treble.)

## Standing still, the averages converge

v3's averages keep a fixed share of every tick's noise (35 % early, 10 %
late a tick): standing still the estimate never settles past it. v6:
while neither the listener nor a source has moved a centimetre from
where the still spell began, each tick joins a running mean (the n-th
weighs 1/(n₀ + n), n₀ the EMA's own count; down to 0.005, ~10 s of
memory); moving, the EMA as before. The late part's click-to-click
spread (`assess still`, rung 3): the club heard from the square through
its walls 10.3 / 9.1 / 9.5 → 3.6 / 3.1 / 3.7 dB; the club inside 1.75 /
1.05 / 1.78 → 0.34 / 0.65 / 0.41 dB; the bell in the square 4.1 / 4.0 /
6.2 → 2.0 / 2.2 / 1.35 dB (`standing_still_the_reverb_settles_v6`: v3
9.84, v6 1.59 dB).

## The reverb's input diffused (every engine)

The delay network that plays the reverb took its input bare: an impulse
came out of its eight lines as eight impulses over the first 80 ms, 64
over the next pass — far sparser than a room's diffuse field at that
time, heard in the treble as a spiky tail. Four Schroeder allpasses in
series before the lines (Dattorro's input diffusion: lossless, flat in
spectrum) spread each arrival into a dense burst: the club's treble tail
kurtosis 16 → 8 (noise is 3; the traced response convolved has ~3). A
clap in the cathedral stays spiky by this measure for a physical reason:
a flutter echo between its walls 16 m apart (spikes at 81, 94, 100, 149
ms).

## Found by recording the game

The Godot host records exactly what its audio thread plays (`OmgWorld.
start_recording`, `demo/godot/demo/record.gd`, `tools/record_runs.sh`):
- a 23 ms audio-callback stall ~13 s after load in every scenario: the
  late-decoded ambience bed was levelled (`flatten_slow_loudness`, ~45
  ms) inside the output thread's render, holding the lock the audio
  callback takes. Now levelled on its own thread; slowest callback 3–5
  µs.
- edge searches from inside the furnished cathedral cost 5–11 ms each
  (looking down the whole edge list for edges the ear sees): capped.
- the receding car's bursts at 8.7 s: the ambience bed decoded ~13 s
  after load, muted before it arrived, burst in at full level — its
  fader ticked only while a bed played, so it glided down from 1. A
  fader whose channel is silent now takes its value at once.
- every moving source's pitch jittered at the frame rate: the demo
  moves its cars on 60 Hz physics and sends their places every 120 Hz
  frame, each send stamped with its own time — the track stood still for
  a frame and jumped at twice the speed the next. A place sent again is
  now stamped with when it was first sent there (as the listener's pose
  already was), and the simulation's clock stays the present. The town's
  passing car (`assess cartone PHYS=60`): pitch jitter 16.5 → 4.7 Hz rms
  (4.3 moving smoothly); in Godot its treble artifacts −50.3 / −47.2 /
  −50.6 → −53.5 / −54.5 / −57.4 dB (approach / pass / away).

## Still open (OPEN.md)

A loud room heard through its walls or a shut door is a heavy-tailed
estimate: young (right after a restart) it reads low and steps up as
rare strong paths come — paths from the sources too (bidirectional) or
learnt splitting at partitions would fix it. Also: the estimate's ±2–4
dB wobble moving at the lowest rungs; a cheaper transmission query (half
the walk); wasm walks on one thread; the web page's real-time stamps not
yet measured in a browser; whether the town has facade echoes the chain
finder misses (a deterministic image-source reference).
