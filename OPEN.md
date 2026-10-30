# Open items

What is still open after the Oct 29 2026 session (engine v5.5, see
ENGINE_V5_5.md; measured in docs/runs/2026-10-29/). Each item says what
is wrong or missing, how it was measured, and what would fix it. Develop
against CLAUDE.md; measure before and after with impulses
(`demo/web/wasm/examples/assess.rs`, `probe.rs`), test on the live engine
(`demo/web/wasm/tests/town_system.rs`).

## Known issues (measured, not fixed)

### Tests that guard the legacy engine
About half the test files drive the legacy engine (WorldSim, AutoPaths,
the PT tracer — only `demo/app` uses it; TESTS.md lists which). They
stayed green while the live engine lost the bell's path over the
cathedral. Port each claim that matters to the live engine (town_system
did so for the bells, and for the shadow boundaries), then retire the
legacy engine and its tests (`omg-scene` world/sim/early/late/dome,
`omg-core` pt/pt_mesh/paths/tracer) — "consolidate duplicated code".
`world::tests::shadow_boundaries_are_continuous` failed on the legacy
engine; ported (`town_system::shadow_boundaries_are_continuous`), the
live engine failed it too — a 31 dB spike of a two-edge path at one
spot — and fixed (ENGINE_V6.md, "Edge paths that leaked"); the legacy
copy retired.

### v3 mode is slow and noisy — v6 answers both
Exponential averages (EMA 0.35 / 0.1): the club door took seconds to
land (+19…+37 dB 50 ms after the shut), identical clicks standing still
spread up to 10 dB sd from the square. v6 (ENGINE_V6.md, the demos'
default) restarts the averages at a world change and makes a still spell
one running mean: within ±4.5 dB 50 ms after the shut; the square's
club 3–4 dB sd. v3 stays as the comparison.

### A loud room's field through its walls converges slowly
(v6 standing still averages it as one running mean — 3–4 dB sd click to
click at rung 3 instead of 9–10 — but each tick's estimate is still
this heavy-tailed: the cure below is what makes it cheap and quick.)
The club heard from the square: its tail (past 0.25 s) lands within
±2 dB of a many-ray reference only after ~20 s standing still, single
ticks bring samples ~1000× the typical (the bass +10 dB, diluting over
15 s), and the late level of the same click differs by up to 8 dB
between seeds after 30 s (`docs/runs/2026-10-29/nee/`,
`examples/dense_track.rs`, `assess crowd`). A listener's ray reaches the
club's inside rarely: its one split goes to the first partition it meets
(usually another building), so it enters through the 2 % transmission
draw carrying 50× — into a room ~60 dB louder than outside. Raising the
draw's floor makes it worse (the reflected rays' 1/(1−p) compounds over
a late ray's ~100 partition bounces). Also near-source bounces (1/r² up
to `MIN_DIST`) are rare and strong. Fix: paths from the sources too —
a first bounce from each source stored on its room's surfaces and
connected through walls from the listener's vertices, weighted against
the walk's own paths by multiple importance sampling (bidirectional) —
or split at partitions by a learnt value of what lies beyond (path
guiding). Large. v6.2 splits at a door's leaf (once a ray): the club from
the square, its late part click to click 1.33 / 0.99 / 2.69 → 0.56 /
0.42 / 0.56 dB (`assess still`, three seeds), over 30 s the mid band's
range 2.4–3.6 → 1.3–1.5 dB, the first click within 1 dB of the last. A
room heard through its walls alone: v6.2 also walks on through a wall a
sounding source is enclosed behind, a quarter of the time — its seeds
within 0.9 dB (4.4 before), the one-second sd 1.3–1.9 → 0.3–0.7 dB. The
bidirectional estimate stays the cure for what remains (rooms reached
only over several walls).
The same shows right after a door shuts: the field through the shut
door restarts as a young estimate. Across four seeds the click 50 ms to
0.8 s after the shut lands on average 2.3 dB (early part, per band) and
1.5 dB (tail) from settled, one band in five more than 3 dB off (from
−9 to +6 dB); the default seed's run reads 2–8 dB low for the first
0.2 s and +4 dB in the mids at 0.8 s (`assess door`; `SEED=n`).
One-pole bands and the stand-ins' coherent sum (+6 dB, fixed) had hidden
it. More rays right after the change did not move the average (six times
the late rays for eight ticks: 2.1 / 1.8 dB) — the heavy tail is the
estimator's, not the sample count's.

### Walking out of a room, its sound lingered (v6) — solved Oct 30
Back up from the bunker the radio played 10–15 dB over the street's
level for ten seconds (the GPU walk; the CPU walk 12 dB on arrival, 3
dB ten seconds on): a still spell's running mean began from the moving
average of the way here, and the GPU's late walks were taken for
estimates of here. Now a walk carries where it was launched, a fold
weighs the ticks it stands for, and a still spell counts the way here as
`STILL_PRIOR` estimates (ENGINE_V6.md, "Walking out of a room, its sound
goes"). Left: the trade itself — fewer and a young estimate of a field
reached through openings jumps from place to place, more and the way
lingers; the cure for both is that field without the heavy tail ("A
loud room's field through its walls", above).

### Moving sources on the GPU walk
The GPU walk connects to where a source emitted its direct sound; its
echoes reach that point too (exact only on the CPU, which solves each
connection's own emission time). Pass the track to the kernel.

### The mixer's tap budget drops paths with their energy
Over its budget a quiet source keeps its strongest paths only; the rest
go silent. Measured in the clicking town (`assess crowd`, `TAPS=96` on
the top rung): no level change for the club — the loss that was there
came from the convolution hand-over (fixed). Not yet measured for many
loud sources at once (rung 0: 48 paths for everything).

### A tap's band shape at a wall's whole span
Paths play their band gains through two second-order shelves at the
band edges: within 1.7 dB of each band centre for steps up to 20 dB
between neighbouring bands, but a wall's whole span (bass 28 dB over the
mids, treble 20 under) only within 2–4.3 dB — the shelves haven't
finished turning by the next centre, and no gains of two second-order
shelves fit all three (the solve diverges). The kernel's Linkwitz–Riley
split gets 1.2 dB at nine sections a sample. Steeper shelves would hit
the centres but step between them: against a curve through the three
centres, straight in dB over log frequency (between the bands a wall's
response is smooth — the mass law slopes), symmetric Butterworth shelves
of order 2 (as now) / 4 are off by rms 0.6 / 2.0 dB (max 1.3 / 4.4) for a
wall (0, −14, −34 dB) and 1.5 / 2.6 dB (max 4.4 / 5.3) for the whole
span — order 4 fits only the centres (0.2 dB). Kept: second order (Oct
29; the magnitudes computed in closed form). A fit for the extreme span
would move the corners too.

### Tick cost
Measured flying through the cathedral with all 19 town sources traced
(`tick_move`, M1 Max), per rung of the quality ladder (`sim::LADDER`),
the walk on one thread: 4.0 · 6.1 · 8.9 · 12.4 · 17.9 (low) · 24.8 ·
31.7 (medium) · 46.0 · 57.9 (high) · 92.3 · 130 (ultra) ms; on four
(`Config::walk_threads`, the default here): 2.0 · – · 3.8 · 5.8 · 7.5 ·
9.8 · 12.2 · 17.7 · 21.1 · – · 44.0 ms. Quality auto targets 16 ms a
tick (`LevelGovernor`). In the Godot demo (more detailed geometry: the
furnished cathedral's late rays walk ~100 segments each) it holds rung
4 flying into the cathedral, 10–16 ms a tick. A 2020 mid-range desktop
core is ~1.5× slower than an M1 Max's; wasm workers walk on one thread
(the web page's auto settles lower, or the walk goes to the GPU).

What a tick spends now, by part: the walk (early and late rays, half
of it next-event connections pricing their line to the source through
the BVH: `segment_hits` collects every crossing, sorted), then the edge
searches (rationed by importance), then chains. A cheaper transmission
query (front-to-back, stopping once nothing audible gets through) is
the next lever; the late family's rays could take Russian roulette past
what the reverb is fitted from, but measured no cheaper in either world
(Oct 29: the rays end at 3 s or escape before they fade 35 dB).

Oct 31: the tour of the Godot town (`tools/record_tour.sh`) ran on rungs
0–2 — 128–256 rays — at ~11 ms a tick, 6–7 ms of it the mirror chains,
which cost the same on every rung; the governor took the whole tick for
scaling and never climbed. It now prices the next rung as the fixed part
(`governor::fixed_ms`: direct paths, chains) plus 1.45× the rest, and
the chains cost about half (a moving source's track searched from the
last knot found; trains hashed without building them; the solves on the
walk's threads): Godot, the tour to the club and out, chains 6.4 → 3.7
ms, rungs 2–4. Most of what remains is churn: a moving listener re-tries
every chain rejected for want of a path (held only while both ends
stand), 500–1000 solves a tick of which a few hundred stay; and the
demo's feet and hands, walking with the ears, name a mirror off nearly
every face near them (`examples/tour_replay COST=1 BODY=1`: +1.5 ms).

### A car receding on the west road: bursts at ~120 m — solved Oct 29
Not the car, not the engine's paths: the ambience bed. The Godot demo
decodes its bed ~13 s after load — 8.7 s into the car scenario, every
run — and the tone runs mute it (−200 dB) before it arrives. The bed's
fader ticked only while a bed played: it still stood at 1 and glided
down from there, so the bed burst in at full level and faded over
~100 ms (above 3 kHz, both ears alike). The rain's fader the same. A
fader whose channel is silent now takes its value at once
(`fx_mixer::a_bed_muted_before_it_arrives_stays_muted`). The car's away
phase above 3 kHz: worst −28.0 → −46.2 dB, windows over −40 dB 8 → 0
(approach −39.3 → −49.9 dB). Headless `assess` loads no bed: why the
replay in the demo's own level never burst.

### A door shut: the young estimate sits low, then steps up (early taps: high where an eased door rests)
After the Godot demo's club door closes (`record.gd --scenario=door
--tone`), the tone sits ~20 dB under where it settles for ~0.5 s, then
steps up within 30 ms as a rare strong path through the door arrives —
the heavy tail of "a loud room's field through its walls" above, heard.
Looked into (Oct 29): among the loudest samples were leaks — bounces in
a corner (on the floor or a door's edge, a fraction of a millimetre
from the wall beside them) whose connection to a speaker started inside
the transmission query's blind margin and passed the wall unpriced:
fixed (`World::transmission_free`), but the level converges as slowly
without them. What remains is physical sampling: few rays get into the
club (the 2 % draw), and bounces 0.3–0.5 m under a speaker (on its
cabinet) bring 1/r². Splitting v6's rays at their first partition (as
v4 does) did not help: from the square the split mostly goes into some
other building, the club's late spread 5.1/4.2/3.5 → 6.7/6.1/7.3 dB.
Headless, `assess door` (instant shut, PACE=0): the click 50 ms after
the shut, its late treble against settled: +13.9 dB with the default
seed, −12.9 … +0.3 dB with seeds 1–4 — the same heavy tail (the treble
has the fewest paths through a heavy door), not a bug of its own.
v6.2 (the split at the door's leaf): the click 0.1 s after the shut,
early −5.3 dB and late −2.6 dB in the treble from settled (v6.1 +8.0 /
+11.4 dB, `assess door`); on the GPU walk (Godot's default) its early
rays pass a leaf half the time: +12.4 / +13.5 / +13.4 → −0.3 / −1.2 /
−0.2 dB, and the shut door no longer reads nearly open in the treble
(−16 → −45 dB under open).
Where an eased door comes to rest (Oct 30, `assess doorswing EAR=0`,
three seeds, the field's own energies per tick against the door shut
all along): from 0.25 s on the reverb is within ±1.8 dB, but the early
taps read high — v6.2 −0.8 … +4.7 dB, v6.1 −0.2 … +6.5 dB, in two
seeds still +3 … +4 dB 3 s on. Not the heavy tail's low reading: the
field plays the strongest 24 of ~2000 early bins, and the strongest
bins of a younger, noisier average carry more (a selection's bias). It
shrinks as the average ages; the whole early field played removes it —
the sum over all bins is unbiased at any age (v6.2 plays it since: "The
dense early field", below). (The tone through the shut door, 6 s on,
2–4 dB over the door shut all along in all three seeds: the tone's
level is its taps' phasor sum, and wanders by as much standing still;
the field's energies there agree within 0.2 dB in two seeds.)

### The car's closest pass: a one-sample step — solved Oct 29
The Godot car scenario showed a single-sample step as the car passed
4 m away (v3 and v6, −38 … −45 dB above 3 kHz). The car's places were
stamped with each 120 Hz send while its physics moved it at 60 Hz: its
track stood still a frame and jumped at twice the speed the next, and
at the closest pass the delay moves fastest. A place sent again now
keeps the time it was first sent there: the pass −53 / −54 dB.

### A closing door is heard where it is — v6.2, Oct 29
Reported while playing (Oct 29, night): with v6.1 a door shutting
muffles what comes through it later than with v6. Measured: not the
reverb's length — the averages trail a moving door (α 0.35 a tick: ~0.15
s), and v6.1's louder reverb made the trailing audible. v6.2
(ENGINE_V6.md, "v6.2") prices the averaged paths through a door with
its leaf where it is now, every tick and frame: against the door
standing at each openness, rms 5.85 → 4.08 dB (the town's club, three
seeds), 4.4 → 2.9 dB (an isolated doorway, four places); in Godot 22 dB
down 0.7 s after the shut command where v6.1 still held at −12 dB.
Left over: a door seen at a slant through a centimetre-wide gap (a
heavy tail: through a narrow gap a few paths carry all); the GPU walk
(Godot's default) records no door paths and doesn't split at leaves —
while a door moves the CPU walks the tick instead (on the GPU, against
the door standing: 12.0 → 3.6 dB rms), which costs a CPU tick (~23 ms
in the town) for that second; a standing door there is steady now (the
GPU's early rays pass a leaf half the time); the restart where a door
comes to rest (young estimate, below). Next: door events in the kernel
(a moving door priced without the CPU tick).

### The dense early field — played since Oct 30 (v6.2)
Inside 80 ms the field played the strongest 24 of ~1000–2500 early bins
and dropped the rest (v3's split, kept by v6): the taps carried 26 % of
the early energy inside the club, 38 % for a clap in the hall, 82–88 %
outdoors, and against v5 (the traced response convolved) the club's
first 80 ms played 6.4 / 4.7 / 4.1 dB under. Adding what was dropped to
the late network's level didn't place it early (it spread over the
decay: early +0.6 … +1.4 dB, late over by up to 4.8 dB). Now a short
network of its own plays it from the first arrival on, as late on
average as the trace says (ENGINE_V6.md, "The dense early field"):
inside the club −3.4 / −1.3 / −1.1 dB early, the whole response −2.6 /
−0.7 / −0.6 dB from v5; in motion as before. Left: its envelope is an
exponential from the first arrival — a field gathered late in the
window spills past 80 ms (inside the club the treble 1.3 dB over v5's
late part); a second network, or an envelope built from the bins' own
course, would follow its shape. Inside the club the low band still reads
3.4 dB under v5's (which a shoebox finds 1.3 dB over its image sources
early), and the network costs the mixer 7–18 %. Also seen:
a clap in the cathedral's late part 3.7 / 2.9 / 1.0 dB over v5's in the
first 3 s. A shoebox against its image sources (ENGINE_V6.md) finds v5
1.4 dB over in the early part and both v5 and v6.2 within 1 dB of each
other late: v5 isn't the yardstick; the cathedral itself is untested.

### Mirror paths near a surface read low (every engine) — fixed on a branch
Against a shoebox's image sources (`examples/shoebox_ref`, ENGINE_V6.md
"Against the image sources") the late part reads low, more the larger
the walls' mirror share (scattering 1: 0 dB; 0: −1.8 … −0.6 dB) and the
nearer the source's surfaces (α 0.1: −1.0 dB mid-room, −4.5 dB 0.5 m off
the floor); in time it grows with the path (`PROFILE=1`: −1.1 dB at
100–200 ms, −6.4 at 300 ms, −12 by 0.7 s). Found (Oct 30): a sphere hit
counts a ray whose closest approach lies on the disk through the source
across the ray, weighted as if it could lie anywhere on it; where the
disk reaches past a surface no ray's can. The source over a floor alone
reads exactly that (h/r 0.12 / 0.25 / 0.49: −2.1 / −1.4 / −0.7 dB).
Not the exact solver's claim on mirror chains (off: the same numbers).
A correction by the whole sphere's reachable share overshot in a room
(+0.7 … +2.3 dB); by each hit's own disk it holds: the branch
`sphere-clearance` (d97f849) divides each hit's weight by its disk's
free share (the clearance along ±x, ±y, ±z as a box; the sphere stops
growing where less than half is reachable, no weight more than
doubled), on the CPU and the GPU walk alike — the shoebox late +0.4 /
+0.5 dB, the corner source +0.4 / +0.2 / +0.1, the floor +0.3. Not on
main: in motion a door swinging shut jitters more (climbing back 17 /
28 / 34 dB against 14 / 19 / 20, a treble window over −40 dB in two
seeds of three). Six seeds, `assess doorswing EAR=0` (v6.2, the club
door): climbing back 18.1 dB on average on main, 22.3 with the branch
(23.7 uncapped); windows over −40 dB 4 / 6 / 11 in all. v6.1: 11.8 →
14.7 dB, windows 10 → 6. Not the tick's sampling noise: a higher
quality rung moves neither (main 18.1 / 17.8 dB at rungs 8 / 10, the
branch 22.3 / 25.2 / 19.2 at 8 / 9 / 10) — a few rare heavy events per
swing, the club's speakers reached through the doorway by rarer, heavier
hits. To judge by ear. Considered, not built: counting the straight
continuation of a segment that ends at a surface near the source (and
the backward one of a segment leaving it) as passes through the
sphere's cut-away part — samples at the ordinary weight instead of
rarer ones weighted up. It, and the branch's share, assume what lies
behind the surface is dead space; behind a thin wall it is the next
room, whose own rays already pass there (through the wall): both would
count it twice. Knowing which cut-away parts are solid is the missing
piece — or an estimator for mirror paths that needn't find the
source's sphere (bidirectional) takes the bias and the jitter at once.
Rebased on main after the edge gate (branch `clearance-rebase`, Oct 30
morning): the shoebox within 0.5 dB everywhere — late +0.5 / +0.4 / +0.2
(main −1.8 / −0.9 / −0.6), the corner source late +0.3 / +0.3 / 0.0
(main −4.4 / −3.4 / −3.1), early +0.2 … −0.1 on both. The door swing
the same trade as before: six seeds (`EAR=0`), climbing back 18.1 dB on
average on main, 22.1 on the branch (33.2 and 26.9 in two seeds),
windows over −40 dB 3 against 11. Still to judge by ear.

### A reflection whose last leg passes an edge was dropped — solved Oct 30
A wall's early echo on the square read ~1.8 dB under the image sources;
the sphere-clearance branch moved it 0.3 dB, so not that bias. A mirror
wall alone (`examples/wall_echo`) read 0 dB at k 0.02, −1.5 at 0.06 (the
default), −2.4 at 0.12 — and −0.1 at every k with diffraction off. Every
segment passing within `EDGE_REACH` of an edge bends at its closest
approach, and a sphere hit on a bent last leg was gated by what the
straight line from the last bounce to the source doesn't carry: zero
wherever the source is in view. Right for the direct sound and for a
chain the solver owns, whose straight wave is counted exactly elsewhere;
any other path's bent rays are all there is of it near an edge. Now a
ray keeps whether its chain was all mirrors, bends aside, and the gate
applies only where that chain is the solver's (CPU walk and GPU kernel).
`fx_simulation::a_walls_echo_passing_its_edge_keeps_its_energy`: the echo
with the bends −1.5 → +0.45 dB (without +0.06). The shoebox's first 80
ms −0.6 / −0.8 / −1.3 → +0.2 / +0.1 / −0.1 dB; a corner source −1.4 …
−1.8 → −0.4 … −0.6; the late part unchanged. Left: without ground the
echo now reads +1.3 dB at k 0.02 — a ray bends only where its closest
approach lies inside the segment, not at its launch from the wall, so
deflected rays flow into the directions that don't bend.

### The GPU walk fell silent through pillars — solved Oct 30
Flying through the cathedral on the GPU walk (Godot's default), the
Godot level's nave pillars on the line: each crossing dropped the
averages and the walk in flight, and with the GPU's walk 3–6 ticks late
on that level none landed until the listener stopped — the reverb gone
for 1.6 s, then back at once (a treble burst, −34 dB under the tone).
The CPU walks the ticks just after a crossing now (ENGINE_V6.md, "The
GPU walk through a crossing"). Left: those ticks cost a CPU walk (the
flight's tick max 29–33 ms on the Godot level); a crossing that leaves
the room the same (a pillar) still drops everything — a test of which
side a surface separates would keep the averages there.

### A facade's echo found by chance — first order solved Oct 29, third order and the twins Oct 30
A car passing the field (`assess cartone`) played the town's facade
echoes at some rungs and runs and not others. The reference
(`examples/echo_ref`: every triangle a mirror, the mirror point inside
it, both legs clear) says they are there: the west facade (x = 0)
mirrors the car for ~1.8 s, 10–13 dB under it, ~0.45 s late, and two
eastern faces (x = 14.5, 22) earlier, ~7 dB under. Unnamed, an echo is
not heard at all (the walk leaves the solver's paths to it). Three
causes: a chain rejected for want of a path was held out for 2 s however
the ends moved (named as the car came up, it was held through the
echo's life); the finders' rays come across a facade 100 m off rarely
(~0.1 % of their sphere); and a moving source's chain was solved from
where the source is now at every step of its light cone (the echo went
8 m before the facade's end). Now a rejection holds only while neither
end moves, first-order chains are named from the world's flat faces
every tick, and only the emission point is solved — the echo in 281 of
281 frames at every rung, at the image source's delay and level
(`fx_simulation::a_facades_echo_of_a_passing_car_plays_at_every_rung`;
before, 18 at rung 0). Each first-order chain's twins off the ground
are named with it (the ground first or last: the echo's own ground
reflection, as the direct sound has its): in the test the facade, then
the ground on the car's side, 0.03 ms later at the grass's mirror share
— 281 of 281 frames, 0 before. Second order between two large faces (a
street's facades, an inside corner) is named from pairs of faces now (Oct
30: ENGINE_V6.md, "Echoes between two faces"; an inside corner 100 m off
a passing car 226 of 226 frames, 0 before), with its ground twins (third
order: 226 of 226, 0 before). Third order (Oct 30): the finders name it
at once — the town's two late wall-only echoes over −30 dB against
`examples/echo_ref3` at 25 places, a street's flutter off a passing car
in every frame; no sweep of its own. Missing were the twins off the
ground: of every chain the finders name, and of every chain at all where
the ground is the level's own faces (both demos pass no ground plane).
Named now, off the face under the listener and the source; and v6's
exact frames play a merged tap whole (ENGINE_V6.md, "Third order, and
every chain's twin off the ground"). Open: a twin off sloped ground, or
off a floor under neither end (both on balconies over a street).

### Doppler in the hosts: one instant per frame — solved Oct 29
Measured in the browser (headless Chrome, `tools/web_doppler.mjs`) and
in Godot (`record.gd` carfar, walk) with a 500 Hz tone: a car passing
30 m off warbled 6.8 Hz rms in the web and 9.0 in Godot, the ears
walking at 6 m/s 4.8 and 0.7 — headless 0.26 and 0.03. Frames carried
the listener's stamp, not the instant their paths were traced at (in
Godot a still listener kept the stamp of its first send); the page
posted at 20 Hz stamped when posted; Godot sent the ears every 120 Hz
frame while they move on 60 Hz physics steps; and the mixer's update
windows tore at unequal steps. Now a frame is the world at the clock it
was traced at, the hosts stamp with when their state was current (the
page every rendered frame, Godot the last physics step) and each window
begins where the previous one ends: web 0.58 / 0.04 Hz, Godot 0.27 /
0.04 (`docs/runs/2026-10-29/doppler/`, `fx_doppler_host::doppler_stays_
steady_under_the_threaded_engines_timing`). In some Godot runs the car
beats with the west facade's echo (1.2–1.8 Hz): the chain finder found
it — the item above.

### Interpolation on moving paths
Paths read with a cubic (Hermite): a 10 kHz tone on a sliding delay
swings 0.96 dB at a walk, 0.47 dB at 30 m/s (was 2.2 dB with lines). A
windowed-sinc read would flatten it further at ~4× the cost.

### The club too quiet far away (5–13 dB in the bass)
Measured before this session against a building-acoustics estimate
(`cargo run --release -p omg-web --example club_levels`); the splits at
every partition change what reaches through walls — re-measure.

### v5 reuses only diffuse-first late paths
Only late rays whose first bounce scattered diffusely are reconnected
(~10 % on concrete, ~35 % on stone). Reconnect at the first diffuse
vertex behind a specular prefix, or reuse the early family.

### Chain order is capped at 24 reflections
`MAX_COHERENT_REFL = 24`: a 4 m flutter is exact for ~280 ms, then only
the scattered estimate carries it. Cost cap; lift with a cheaper solver.

## Not done yet (from the research, ranked)

1. **Shared late reverb (common slopes)** — the late field's decay rates
   are the space's, shared by all sources (MoD-ART; Götz et al. common-
   slope model): fit K decay modes per band jointly from all sources' late
   responses, render K shared reverberators, each source sending its
   amplitudes; per-source kernels keep only the early residual. Tail cost
   stops growing with source count; lifts the web's 8 convolved sources ×
   2.5 s cap; less tail noise. Large. A/B the modal approximation against
   full per-source kernels.
2. **GPU late rays** — the GPU walk is latency-bound (~0.7 ms per bounce
   per thread), so late rays (hundreds of bounces) run on the CPU. Remedies:
   wavefront NEE (shadow rays as their own dispatch), many more rays per
   dispatch, path reuse to make per-tick work shallow. The web's sim tick
   is ~32–40 ms of its 50 ms budget. Medium.
3. **Low-frequency wave solver** — below the Schroeder frequency
   (~150–250 Hz in rooms) geometric acoustics is wrong: room modes, bass
   diffraction, bass through doors. FDTD on the GPU (or 2D on the CPU,
   Planeverb) around the listener as the renderer below a crossover set by
   the grid. Large (v5 core).
4. **Outdoor propagation** — ground reflection with a complex impedance
   (Delany–Bazley/Miki: the 200–600 Hz ground dip, phase); refraction from
   wind/temperature gradients (curved rays: upwind shadow zones, night
   inversions carrying sound kilometres); turbulence fluctuations. A
   timeline ≥ 30 s for km-scale sounds. Medium–large.
5. **Per-band scattering** — `Material::scattering` is one scalar; real
   surfaces are nearly specular for bass, rough for treble (ISO 17497-1).
   Touches the walk (per-band weights on the diffuse/specular draw), the
   chain solver, GPU material/panel layouts (walk layout bump), Godot
   material parsing; needs sourced per-band values. Medium.
6. **Octave bands** — 3 bands → 8 (63 Hz–8 kHz, one f32x8): material data
   is published in octave bands; the high band (2.1–16 kHz) hides most of
   air absorption's and the HRTF's range. Medium–large refactor.
7. **Diffraction** — BTM/VDaT/BEDRF instead of the UTD-like knife edge at
   low frequencies; keep occlusion paths solved while the source is still
   visible (no gap at corners). Medium.
8. **Convolution cost** — partition-level kernel updates (only stages whose
   part changed); a shared inverse FFT across sources (needs one block
   clock for all convolvers); frequency-domain crossfades. Small–medium.
9. **Ear model** — the adaptation's time constants and its ½ share are
   literature ranges and a display choice; an ISO 532-3 loudness model
   would measure what arrives. Medium.
10. **HRTF** — interpolated minimum-phase + ITD for point paths (no
    nearest-neighbour switching), near-field correction below ~1 m
    (weapons, hands, whispers), Bilateral Ambisonics if the bus stays at
    order 2. Medium.

## Smaller items

- **A listener grazing a surface counts as passing through it.** Ears
  exactly at a lintel's height (the tour floated them at 2.0 m under
  2.0 m doorways) cross the wall's edge: `Tracer::crossed` drops every
  average, and the restart plays as a dip as deep as a fresh estimate's
  (the club's bass ~10 dB for half a second at its door). A person's ears
  never ride a lintel; the tour now walks at 1.6 m. A crossing could be
  counted only past a millimetre into the solid.
- **A source switched on starts its early field low.** Hands switched
  on at a clap read 7 dB under hands traced all along, ~1 s to settle
  (`examples/body_onset`): the running mean of a few heavy-tailed ticks
  (the same as a door's shut, above). A new sound's first reflections —
  a shot from a new place, a car spawned — are thin until the
  bidirectional estimate.

- **Body**: the full-sphere head-and-torso HRIRs (SADIE II KEMAR) already
  carry the torso's shadow; legs add a ~11° cone straight down. Only a
  crouching listener's knees remain (design in the session's notes:
  a box under the listener, knife-edge priced, only where the posture
  differs from the HRIR's).
- **Ear window floor**: the audible window sits under the adapted level
  or the 0.4 s masker; judged per band now, but masking itself spreads
  upward in frequency (a flute doesn't mask a club's bass) — not modelled.
- **Material values not yet sourced**: drywall, carpet, wood panel,
  metal, grass transmission; the church's stained glass is visual only
  (the walls are solid stone to the engine).
- **Club loop**: mids lifted ~10 dB, still ~24 dB under the bass (a dance
  mix sits ~15 dB under); a fuller source recording would do better than
  more EQ.
- **Web kernels**: capped at 2.5 s, 8 sources convolved (the rest play
  the parametric reverb); see the shared late reverb above.
