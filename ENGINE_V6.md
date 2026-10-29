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

## A fast turn fades every HRIR change

The strongest paths play through their nearest measured HRIR, re-chosen
every 128 samples and crossfaded over 10 ms. A change asked for during
a fade started over at once, dropping the half-faded filter: a click
whenever a turn changed the HRIR within 10 ms — a flick of the view in a
first-person game, a source swept past close by. Now a change waits for
the fade under way (the latest wins): a 500 Hz tone through a path
turning 1500°/s, its treble −27.2 → −67.6 dB under the tone
(`hrtf::tests::a_fast_turn_crossfades_every_hrir_change`).

## One instant per frame (every host)

A frame is the world at one instant, and the mixer moves every path's
delay over the time between two frames' instants. Measured in each host
with a 500 Hz tone (`docs/runs/2026-10-29/doppler/`), a car passing 30 m
off warbled 6.8 Hz rms in the browser and 9.0 in Godot (headless 0.26),
and the ears walking at 6 m/s fluttered 4.8 / 0.7 Hz (headless 0.03):
frames carried the listener's stamp rather than the clock their paths
were traced at (a still listener in Godot kept its first send's), the
page posted its pose at 20 Hz stamped when posted, Godot sent the ears
every 120 Hz frame while they move on 60 Hz physics steps, and the
mixer's update windows began the new motion after the previous one
began instead of where it ended — torn at unequal steps. Now
`Frame::time` is the simulation's clock at the trace; an exact frame of
an instant already traced is not made, and a tick of that instant is
that moment traced again (its paths glide on); the page posts every
rendered frame with its time and the worker makes one frame of each;
Godot stamps a frame's poses and places with the last physics step's
game time (`Engine::set_listener_at`, `set_source_at`). Browser 0.58 /
0.04 Hz, Godot 0.27 / 0.04 Hz — as headless.

## Echoes found, not chanced upon (every engine)

A mirror chain (an echo off one or more flat surfaces) is the exact
solver's: the walk leaves it out, so a chain nobody names is not heard
at all. The town's west facade mirrors a car passing 30 m off for ~1.8 s,
10–13 dB under it (`examples/echo_ref`, every triangle as a mirror); the
engine played it at two rungs of eleven. The finders' rays come across a
facade 100 m off rarely, a chain rejected while the car came up was held
out for 2 s whatever moved, and a moving source's chain was solved from
where the source is now (near the facade's end its place mirrored past
it). Now first-order chains are named from the world's flat faces every
tick (the mesh's coplanar triangles and the panels' sides: the mirror
point of where the source emitted inside the face), a rejection holds
only while nothing moved, and the light cone runs on the chain's image
with only the emission point solved: the echo in every frame at every
rung, at the image source's delay and level — with its twins off the
ground (the ground first or last), as the direct sound has its own. A
tick over the Godot level's 3109 faces costs no more than before
(`tick_move GEOMETRY=`).

## v6.1: the reverb as long and as loud as traced

v6 is v3's reverb, and v3's reverb carried two shortcuts. The decay time
fitted to the late echogram was clamped at 5 s — the cathedral read
exactly 5.00 — and the fit ran on an echogram cut off at 3 s, which pulls
long decays short. And the late level was taken in the echogram's legacy
units (energy / 16): the reverberant field played 12 dB under what the
same rays trace (ENGINE_V4.md, "Level"); it was also held under 0.7.
v6.1 (`Config::v61`, the demos' default; v6 stays selectable) fits the
decay as a least-squares line over the Schroeder curve from −5 dB down
to −25 dB (or as deep as it goes, 5 dB at least), with what would still
arrive after the echogram's end — the decay carried on from its last
0.2 s — added and the line fitted again until it holds (Lundeby), no
clamp; and plays the late level in the taps' units, uncapped
(`tracer::fit_reverb`, `v3::reverb_v61`). Exponential tails of 0.3 … 15
s read back within 0.1 % (`tracer::tests::a_long_decay_reads_its_own_
reverberation_time`; v6 read 8 and 15 s as 5.00). A closed 30×20×15 m
hall, walls absorbing 6 % (`fx_simulation::a_hall_rings_and_sounds_as_
long_and_loud_as_theory_says`): Sabine with the air's absorption 7.25 s,
diffuse theory's reflected-over-direct +16.9 dB — v6 5.00 s, +6.5 dB;
v6.1 6.50 s, +14.7 dB; v5's traced response 6.32 s, +15.1 dB. In the
cathedral by the flute (`examples/probe`): v6 5.00 s, reverb −21.0 dB;
v6.1 5.61 s, −8.9 dB, the late part of v5's response −8.9 dB.

## A curtain is cloth that reflects nothing

`Material::DRAPE`: a heavy drape hanging free, the cloth itself per pass
— it reflects nothing; sound passes through it or is absorbed in it, and
what passes, the wall behind sends back through it again (through, off
the wall, back: a velour drape's installed absorption, 0.14 / ~0.58 /
~0.65). A surface that reflects nothing only passes rays on (the walk
spends none reflecting nothing: `pure` in `walk_ray` and the GPU
kernel). The Godot demo hangs them 10 cm off the seminar room's walls
and between the cathedral's lancets, rolled up until a cord (E) lowers
them — an OmgDoor on its side, its covered share growing from the top,
priced continuously as it comes down. The seminar room's RT60 4.8 → 1.0
s, the cathedral's ~7 → 3.2 s (`record.gd` curtains, quality pinned).

## v6.2: a door heard where it is

v6 and v6.1 average each tick's traced field over the last few ticks
(α 0.35 a tick while a door moves): a door swinging shut was heard where
it had been ~0.15–0.3 s before — the reverb and the early sound behind
it trailed the leaf ("a closing door seems to cut the sound later").
Without the average a tick's estimate is too noisy (55–110 dB of
climbing back, above). v6.2 (`Config::v62`, on v6.1; the demos'
default) keeps the average and prices it again: every walked path that
crossed a door's opening or bounced off its leaf keeps where and which
way (`v3::Event`), and at every tick and every frame between, the
average's door paths are priced with the leaf where it is now — a path
that crossed where the leaf now stands passes t² (its transmission), a
bounce off where the leaf no longer stands is gone. The host says what
an object fills (`Simulation::set_object_aperture`: a doorway for its
leaf, a window for its curtain); Godot sends every OmgDoor's whole box,
the web town every door's hole.

What it took to be exact, each found by measuring:

- **Not by area.** The share of the opening covered is an assumption,
  and wrong: through a gap a few millimetres wide in a 4 cm leaf only
  near-straight rays pass — the walk's energy through a 5 mm gap sat 14
  dB under its area (a slot's view factor). Each path priced by its own
  crossing needs no model.
- **A slot passes what enters it.** A path entering the leaf's slab
  through the gap that runs into the leaf's edge reflects along the slot
  and through: a crossing is priced by where it enters the slab.
- **Reflections a closing leaf would give.** A closing door keeps in the
  room what it no longer lets out — reflections never walked, as the
  leaf wasn't there. Where a path passes an opening's open part, a
  quarter of the time it also walks on as the leaf would reflect it
  there: recorded for the pricing, nothing deposited (`VIRTUAL_P`).
- **A ratio, not a difference.** Priced as the average as walked less
  what the door now removes — two large noisy sums — near shut the noise
  was all that was left, under the shut door's level. Now: what the door
  paths deposited (exact) × what the kept paths carry now against as
  walked.
- **Split at the leaf.** What passes a shut leaf was one ray in thirty
  (the draw's floor, × 34): a heavy-tailed few that a short average reads
  low — v6.2 without it was worse than v6.1 (5.5 against 4.8 dB rms,
  below). At a door's leaf a ray now splits (v4's rule, once a ray): what
  passes and what reflects are both walked. A curtain (it reflects
  nothing) neither splits nor reflects virtually.
- **Kept, not dropped.** Per tick 256 paths a part (early, late), by
  energy over a 4 × 4 grid of where they met the opening (pre-thinned
  while walking against the last tick's threshold, unbiased); the six
  newest ticks apart, older ones joined — a path walked by an open door
  weighs in long after its tick's weight is small (its share is tens of
  dB over the shut door's), so none is dropped by weight. A door
  standing still costs no pricing (its paths' signature unchanged).

Measured against the door standing still at each openness (averaged
long — what the swing should sound like at every instant):

- Isolated (`fx_simulation::a_door_swinging_shut_is_heard_where_it_is`,
  a room's doorway, the ears at four places before and beside it): rms
  v6.1 4.4 dB, v6.2 2.9 dB; over three seeds 4.80 → 3.40 dB (better in 10
  of 12).
- The town's club door, the ears 1 m before it (`assess doorswing
  EAR=0`, three seeds, against `assess doortruth DUR=8`): v6.1 5.85 dB
  rms, v6.2 4.08 dB, better in every seed; the treble under the tone
  unchanged (−38 … −45 dB worst).
- Godot, the club's inner door shutting while you stand before it
  (`record.gd door`, quality auto): 0.7 s after the shut command v6.2 is
  22 dB down where v6.1 still holds at −12 dB until the leaf settles;
  within 3 dB of shut at 6.94 against 7.11 s; no level jumps, nothing
  over −40 dB above 3 kHz (`docs/runs/2026-10-29/door`).
- Standing doors gain from the split too: the club heard from the
  square through its walls and shut door (`assess still`, click to
  click) spreads 3.0 / 2.4 / 2.6 → 1.1 / 0.9 / 1.2 dB late, 1.1 / 1.1 /
  2.6 → 0.5 / 0.5 / 0.5 dB early, where v6.1's drifted 7 dB down over
  30 s; a door slammed shut (`assess door`), 0.1 s on, the highs +8 /
  +11 dB over where they settle → −5 / −3 dB. Walking out of the hall and
  flying through the cathedral's wall: unchanged.
- Cost (the town, every source sounding, the ears by the club door,
  `examples/door_ms`): a tick 15.4 → 17.8 ms standing (the split at
  leaves and the paths' records), 18.8 → 23.1 ms while the door moves; a
  frame between ticks 0.18 → 0.38 ms while it moves.

Left: a door seen at a slant through a centimetre-wide gap is still a
heavy-tailed estimate (one position of the four: v6.1 2.2, v6.2 2.7 dB);
the GPU's walk records no door paths and doesn't split at leaves (with
the walk on the GPU — Godot's default — only the CPU's late family, the
reverb, is priced); a path meeting apertures more than six
times is left to the averages; the virtual reflections are first order;
where a door comes to rest the averages still start over (young).

## Still open (OPEN.md)

A loud room heard through its walls or a shut door is a heavy-tailed
estimate: young (right after a restart) it reads low and steps up as
rare strong paths come — paths from the sources too (bidirectional) or
learnt splitting at partitions would fix it. Also: the estimate's ±2–4
dB wobble moving at the lowest rungs; a cheaper transmission query (half
the walk); wasm walks on one thread; second-order echoes between two
faces (a street's two facades) are still left to the finders' rays —
first order and its ground twins are named from the faces.
