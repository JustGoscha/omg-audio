# SOURCES_PLAN — from 14 sources to 1000

Goal: as many simultaneous sources as possible, with cost that grows
far slower than source count. The few sources that matter get the
full path-traced treatment. Everything else still plays, still sits in
the right place and still feeds the room, at a fraction of the cost.

This is the concrete build plan for GPU_PLAN.md §C7 ("exact for the
few, statistical for the many"). C7a, the batched GPU solve, has
shipped. This file replaces the C7b–C7d sketches with measured numbers
and an order of work.

## Can 1000 sources really be logarithmic?

Not entirely, and the plan should say so honestly.

- **Every audible source costs at least one read and one add per
  sample.** Its own signal has to reach the mix. That floor is linear
  in N but tiny: 1000 sources × 48 kHz is about 48 M adds per second,
  roughly 1–2 % of one core.
- **Everything above that floor can be made sublinear.** Propagation,
  spatialisation and reverb are where the cost lives today. They can be
  paid per *cluster*, per *room* and per *listener* instead of per
  source:
  - Picking the top K out of N is a partial sort, O(N + K log K). A
    spatial tree over the sources makes cluster formation
    O(K log N), the "lightcuts" result from graphics.
  - Late reverb is a property of the room, so it costs O(rooms).
  - The listener-side trace gathers everyone's energy in one pass,
    O(1) in N with reservoir sampling.
- **Many copies of one sound become one texture.** A crowd, rain on
  leaves or a field of cicadas, where every member plays the same kind
  of sample, can be synthesised as one granular stream per cluster.
  That is the only route to genuinely O(log N) or O(1) per sample, and
  it only applies to such textures.

So the target is: per-sample cost ≈ K_exact × (full cost) + K_light ×
(small cost) + N × (one add), with K_exact and K_light fixed by the
device budget, and sim cost ≈ O(K log N) per tick.

## Where the cost goes today

One `Renderer` per source on the audio thread (`crates/omg-dsp/src/renderer.rs`):

| Per-source item | Work per sample | Scales with |
|---|---|---|
| Early taps (≤160 kept) | delay read, two shelves, lowpass, 9 SH gains, ~14 smoothers | taps × sources |
| Point HRIR convolutions (≤24) | 128 taps × 2 ears each, ≈ 6 k multiply-adds for 24 | points × sources |
| Own-room FDN | 8 lines, mixing matrix, damping | sources |
| Remote (coupled-room) FDN | 8 more lines, EQ, 9 SH gains | sources |

Shared once for the mix: the order-2 bus decode, ambience, rain, the
output stage.

On the sim side (`crates/omg-scene/src/world.rs`): the early solve is
batched (C7a), quiet sources refresh every 4th tick, muted sources are
skipped. But every source still runs its own late-field trace
(`world_late[si].reverb_world`), its own routing and its own bend
refresh.

The point budget and the tap ceiling are both **per source**. With 14
sources the worst case is 14 × 24 point convolutions, which is why the
worklet governor sheds to a 2-point floor so quickly.

**Measured numbers:** see "Measurements" below; `tools/bench_sources.mjs`
reproduces them on any machine.

## The tiers

Every source carries a **priority** recomputed each sim tick:

    priority = estimated level at the listener      (cheap propagation estimate)
             × signal activity                       (short RMS envelope; silence = 0)
             × importance hint                       (dialogue, gameplay cues > ambience)

The estimate comes from what is already cheap: free-field 1/d and air
absorption, times the room-to-room transmission the acoustics graph
already holds (`environment.rs`), times the source's cached bend or
occlusion factor. No per-source ray work is needed to rank.

| Tier | Who | Sim | Audio | Rough audio cost |
|---|---|---|---|---|
| **Exact** | top K_exact by priority | full exact early solve, bends, routing | full tap set, point HRIRs from a **mix-wide** point budget | today's per-source cost |
| **Light** | next K_light | direct path + order-1 reflections from the cached route; no bends refresh beyond the 5 Hz LOD | ≤ 6 taps, SH bus only, no point convolutions | ~5 % of exact |
| **Bed** | everyone else that is audible | priority estimate only; position, level and spectrum tilt per tick | signal summed into one of ~32 direction cells; each cell encoded once to SH | one add per sample |
| **Off** | inaudible (masked, silent or out of range) | nothing | nothing | 0 |

Rules that hold across tiers:

- **Late reverb is per room, not per source.** One FDN per room that
  the listener can hear, plus one remote FDN per audible doorway. Every
  source, whatever its tier, sends into its own room's FDN with a
  level from the priority estimate. This turns the 2 × N reverbs into
  O(rooms) and is the single biggest structural saving after the point
  budget.
- **Point budget and tap ceiling become mix-wide.** Taps from all
  exact-tier sources are ranked together by loudness; the strongest
  P_total get point convolutions and the strongest T_total get taps at
  all. A quiet source behind the club keeps two taps while the voice in
  front keeps sixty (GPU_PLAN C7d).
- **Promotion and demotion are hysteretic and crossfaded.** A source
  must beat the weakest member of the tier above by ~3 dB for ~250 ms to
  move up, and fall ~3 dB below for ~1 s to move down. Tap keys stay
  stable through a promotion, so the renderer crossfades instead of
  clicking.
- **Bed sources still move.** Cell assignment updates per tick, and a
  source crossing a cell edge is panned across the two cells for one
  tick. Direction resolution is about 20°, which is below what the ear
  resolves for quiet, background sources.
- **Textures.** A cluster of bed sources that share one sample bank
  (crowd, birds, rain) can switch to granular synthesis: one stream per
  cluster whose grain rate tracks the member count. Optional, last.

## Build order

Each step ships on its own, keeps the demo sounding the same within
the tolerances stated, and adds isolated tests in the TESTS.md
convention (`fx_<effect>.rs`).

1. **S1 — mix-wide point budget and tap ceiling.** Move the budgets
   from each `Renderer` into the engine: rank all live taps across
   sources once per 128-sample block, hand each renderer its share.
   Biggest win for the least code; the worklet governor keeps working
   but now trades one global number.
   *Gate:* with 14 sources at the heaviest spot, the same total point
   count costs the same, and at the global ceiling muting the loudest
   source returns its points to the next one within one governor
   window.
2. **S2 — shared room reverbs.** Replace the per-renderer FDN and
   remote FDN with per-room ones owned by the engine. Sim emits
   per-room RT60 and per-source send levels instead of a reverb block
   per source; the per-source late trace becomes one listener-side
   trace per room (C7c) on the sim side.
   *Gate:* a single source in a room renders the same RT60 and wet
   level as today within 1 dB / 10 %; 64 sources in one room cost the
   same reverb work as one.
3. **S3 — priority and tiers in the sim.** Compute priority per tick,
   assign tiers with hysteresis, give light-tier sources the reduced
   path set, skip everything else for them. Expose K_exact and K_light
   in `quality.rs` next to the existing tier budgets.
   *Gate:* 128 sources tick under the doorway budget; a source walked
   past climbs to exact and back without a click (fx test drives
   promotion on a synthetic scene).
4. **S4 — the bed.** Direction cells, per-source delay read and gain,
   one SH encode per cell. Source count limit becomes memory, not CPU.
   *Gate:* 1000 bed sources cost less than one exact source on the
   audio thread; their summed level matches the closed-form sum within
   1.5 dB.
5. **S5 — the scalable scene.** `WorldSim::from_scene(...)` (see
   TESTS.md) so scenes with hundreds of sources can be built and tested
   without the demo. A stress scene ("market square": 300 voices, 40
   birds, a fountain) becomes the benchmark.
6. **S6 — textures (optional).** Granular cluster synthesis for
   same-bank crowds.

## Device budgets

The governor already measures the audio thread's load. With S1–S4 it
steers three numbers: K_exact, the mix-wide point budget P_total, and
the mix-wide tap ceiling T_total. K_light and the bed are cheap enough
to leave fixed. Defaults per machine class come from
`tools/bench_sources.mjs` runs on each class (Mac M1 reference, the
RTX 3060 Windows box, a mid phone).

## Measurements

Filled in from `tools/bench_sources.mjs`. Add a row set per machine.
