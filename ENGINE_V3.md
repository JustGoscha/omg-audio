# ENGINE_V3 — one listener tracer for everything

Engine v1 simulated rooms: image sources in shoeboxes, portals between
them, hand-routed doorway paths. Engine v2 ("traced", C6) moved early
reflections onto one world mesh but kept a zoo of special cases next to
it: per-source late traces, door-routed wet fields, bend strings over
and around buildings, occlusion floors, room blending. Every bell bug
fixed on 2026-10-25 lived in that zoo.

Engine v3 replaces all of it with one mechanism, scaled by compute:

> **n** rays leave the listener. Each ray bounces up to **x** times.
> At every bounce the surface looks for the sources it can see directly
> (next-event estimation), priced by how diffuse the surface is, what
> it absorbs, and every wall the connection passes through. The ray
> itself keeps travelling, and when a specular chain runs into a source
> it counts too. Sources grow with distance so far ones are still found;
> every hit is then computed from the source's real position.

`n` and `x` are the quality knobs. Everything else is geometry and
materials.

## The estimator

Units: a unit source gives free-field energy 1/d² at distance d (tap
amplitude 1/d). All energies are per band.

1. **Direct (bounce 0).** One deterministic segment listener → source:
   `amp = T(segment) · air(d) / d`, delay d/c. Exact and noise-free, so
   localization never flickers. `T` multiplies the per-band
   transmission of every surface crossed (paired faces give the real
   wall thickness; door leaves and panes are overlay boxes).
2. **Next-event estimation (every bounce).** At hit point P with
   material (α absorption, τ transmission, s scattering), for each
   source S with distance r from P and incidence cosine cos θ:
   `E = throughput · s·(1 − α − τ²) · cos θ · T(P→S)² · air(L + r)² · 4 / (n · r²)`
   arriving at delay (L + r)/c from the ray's first direction. This is
   Lambertian radiance (`ρ·E_irr/π`) seen over the ray's solid angle
   `4π/n`. It is what makes a lit facade send the bell back at you.
3. **Source spheres (specular chains).** A ray segment whose last
   interaction was a specular reflection counts a hit when it passes
   within `r = k · L_s` of source S, where `L_s` is the unfolded path
   length to S. Every source then covers the same solid angle `π k²`
   seen from the listener, near or far, so a bell 300 m out is found as
   often as a voice 3 m away. The hit is then computed from the real
   source position: `E = throughput · T(P→S)² · air(L_s)² · 4 / (n · k² · L_s²)`,
   with `T` priced along the real segment from the last bounce to the
   source (a ray that grazes the outside of the bell tower still pays
   for whatever stands between that bounce and the bell itself).
   Segments after a diffuse bounce don't count: NEE already covered
   that path. Primary segments don't count: that is the direct path.
4. **Continuation.** The ray transmits with an importance-floored
   probability (re-weighted, as in the v2 late tracer), else reflects
   keeping `1 − α − τ²`: specular with probability `1 − s`, else
   cosine-weighted diffuse.

Checks this must pass, each an isolated test (`crates/omg-core/tests/fx_v3_*.rs`):
- free field: only the direct tap, exact;
- a wall between: the direct pays the wall's transmission exactly;
- a specular mirror: the reflected energy converges to the image
  source's `R/(2h)²`, independent of `k`;
- a Lambertian plane under a coincident source/listener at height h:
  the reflected energy converges to the analytic `1/(2h²)`;
- a bell behind a building in a lit street: the shadow is filled by
  facade reflections (NEE) instead of silence;
- the late level and RT60 in a shoebox match the v2 tracer.

## From paths to the renderer

The renderer contract does not change (`ParamBlock`: taps + reverb +
remote). Contributions are binned per source:

- **Early** (within 80 ms of the free-field arrival): binned by first
  direction (64 cells) × delay (2 ms). Each bin becomes one tap with a
  stable key, its energy-weighted delay and direction. The strongest
  K bins per source are kept. Bins are averaged over ticks (EMA), so
  stochastic noise becomes slowly-changing tap gains, not churn.
- **Late** (after 80 ms): into an echogram relative to the free-field
  arrival, EMA'd over ticks, fitted for RT60 and level exactly like v2
  (`estimate_reverb`). A strongly anisotropic late field becomes the
  directional `remote` send, as in v2.

## What v3 deletes (once it is the default)

Image sources and shoeboxes at runtime (kept as the test oracle),
portals and door routing, routed wet fields, room blending, per-source
late traces, the bend floor and bend caches, the over/around strings,
the straight-ray tap, LOD by staleness. Diffraction is the one physics
v3 does not get for free (step V3.4).

## Build order

- **V3.0 — core tracer** (`omg_core::v3`): direct, NEE, source spheres,
  continuation, binning to `ParamBlock`. Isolated tests above.
- **V3.1 — in the sim** (`early = 3`): one listener trace per tick for
  all active sources; door leaves and panes as overlay boxes; the club
  rig as N sub-sources. Selectable in the quality panel next to ism and
  traced. Bells probe: open field loudest, shadows filled, indoors
  quieter.
- **V3.2 — many sources:** NEE samples M sources per hit by importance
  instead of all of them; sources in a BVH for the sphere tests. Cost
  per source becomes a lookup.
- **V3.3 — GPU:** the same loop as a WebGPU kernel (the v2 late tracer
  and PT discovery already run there).
- **V3.4 — diffraction by sampling:** rays passing near a mesh edge
  get deflected by the uncertainty-relation rule (Stephenson), so
  bending around corners and over roofs emerges from tracing. Until
  then v3 has no diffraction: shadows get reflections and
  transmission only.
- **V3.5 — default:** once the probes and the full suite agree, v3
  becomes the default and v1/v2 machinery leaves the runtime.

## Budgets (first cut)

| Tier | rays n | bounces x | NEE sources per hit |
|---|---|---|---|
| High | 2048 | 6 | all (≤ 16) |
| Med | 1024 | 4 | all (≤ 16) |
| Low | 512 | 3 | all (≤ 16) |

Source sphere angle: k = 0.06 (≈ 3.4°).
