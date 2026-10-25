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
5. **Two ray families, one estimate.** EARLY rays (many, `x` bounces)
   count paths with 1..=x reflections. LATE rays (fewer) keep bouncing
   until the path is 3 s long or has lost 60 dB, and count only paths
   with more than x reflections. Every path is counted by exactly one
   family, so the sum is unbiased — and the reverb tail is traced to
   its end (no bounce cap: a live 8×6×3 m room traces 3.0 s against
   Eyring's 3.2 s, where the v2 tracer's 64-bounce cap stopped at
   1.1 s). The tail is a slow statistic: few late rays per tick,
   one sampled source per bounce, averaged over ~10 ticks.

Checks this must pass, each an isolated test (`crates/omg-core/tests/fx_v3_*.rs`):
- free field: only the direct tap, exact;
- a wall between: the direct pays the wall's transmission exactly;
- a specular mirror: the reflected energy converges to the image
  source's `R/(2h)²`, independent of `k`;
- a Lambertian plane under a coincident source/listener at height h:
  the reflected energy converges to the analytic `1/(2h²)`;
- a bell behind a building in a lit street: the shadow is filled by
  facade reflections (NEE) instead of silence;
- the late level and RT60 in a shoebox match the v2 tracer;
- a live room's tail is traced to the end (Eyring, not a bounce cap).

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
- **V3.3 — GPU (done):** the walk as a WebGPU kernel — see *V3.3 — the
  walk on the GPU* below.
- **V3.4 — diffraction (done):** two parts, disjoint so nothing counts
  twice.
  - *In the walk:* a ray passing within 3 m of an exposed edge bends
    around it at its closest approach, by an angle drawn from a spread
    σ = λ / (4π·a) (uncertainty relation: the narrower the gap, the wider
    the spread; bass spreads more). The bent ray keeps its angle to the
    edge (Keller's cone); per-band weights keep all three bands unbiased.
    Multi-edge paths and bends after reflections emerge from this.
  - *Edge next-event estimation:* for every source whose straight line
    is blocked, each edge the listener can see gets one exact bent path
    (bend point, legs priced through whatever they cross, knife-edge per
    band); the strongest three distinct ones are kept. Random walks with
    exactly one bend and no reflection are left to this.
  - *Gating:* a bent path only carries what the straight line from where
    it bent does not — on the lit side nothing is added.
  - *Edges:* sharp creases and free borders of the mesh that are truly
    exposed: > 225° of open space around them (a wall's foot on the
    ground is not), not a coplanar seam between wall strips, not filled
    this tick by a pane or door leaf, not at ground level.
  Tests: a thin screen follows Kurze–Anderson within 5 dB from the
  shadow boundary to a 30° bend, bass bends further than treble, the lit
  side is unchanged, and only exposed edges diffract.
- **V3.5 — default:** once the probes and the full suite agree, v3
  becomes the default and v1/v2 machinery leaves the runtime.

## V3.3 — the walk on the GPU

What moves: the ray walk, both families, everything in it — raycasts,
next-event segments (importance-sampled sources), source-sphere tests on
specular and bent chains, segment transmission (mesh crossings deduped
and paired into walls of real thickness, overlay panels, earth and air
pockets), the importance-floored transmission branch, the diffraction
bends (V3.4: nearest usable edge, truncated-Cauchy mixture, Keller's
cone, visibility gating) and the sky escapes. One WGSL kernel,
`crates/omg-gpu/shaders/v3_walk.wgsl` (K5), one thread per ray, late
rays scheduled first (they are the long ones). The same file runs
natively (wgpu, `omg-gpu::GpuV3Tracer`) and in the browser (`web/gpu.js`).

What stays on the CPU: the exact direct segment per source, the direct
visibility and the edge next-event paths (`Tracer::prepare`, the very
code `trace` runs), the EMA, the tap/reverb construction. They refresh
every tick, so a walk landing a tick late never delays a direct path.

**Contract** (`omg_core::v3::fold`, the only v3.rs hook is the module
line): `Tracer::trace_walked(world, listener, sources, budget, walk)`
hands the walker a flat `WalkJob` — a 160-byte header (rotation, seed,
budget, listener, counts, grade, edge bounds) plus scene words (sources
with free-field arrival, visibility and fixed-point norm; panels;
pockets; this tick's usable edges) — and folds back a `Walked`: per
source the early slots (64 direction cells × 40 two-ms bins) and the
300-bin echogram, each slot (e₀, e₁, e₂, Σe₁·first-direction), plus the
sky cells — binned on the device exactly as `deposit` bins them.
`Walk::Pending` (async walker, nothing finished) holds the smoothed
state; `Walk::Refused` (no device, caps: 64 sources, 64 panels,
8 pockets) traces that tick on the CPU at the CPU budget.

**Accumulation.** WGSL has no float atomics. Float CAS loops worked but
were 5–25× slower: every deposit of one source hits the same few hot
words and CAS retries go quadratic under contention. The kernel adds
into 64-bit fixed point instead (integer `atomicAdd` on the low word,
carry from the returned old value into the high word), each source at
its own scale — 2^44 × max(d², 1), d its distance — so a bell 300 m out
resolves as finely as a voice at 1 m: −132 dB of the free-field energy
per contribution, 5·10⁵× it of signed headroom. A second pass
(`resolve`) turns the bins into f32 true units and sums each source's
echogram into its `reflected` total. Deviations from the CPU walk, both
unobservable: echogram direction sums start at 80 ms (all
`late_direction` reads), and the edge query tests every usable edge
(cached in workgroup memory) instead of walking the CPU's 4 m grid —
the probe test finds them identical.

**Selection.** `omg_scene::v3walk`: a registered `V3Walker` walks while
the compute switch is on (`quality::set_gpu_backend`); `quality::v3_gpu()`
then raises the v3 tier budgets 8× (early and late rays) — only then,
so a CPU walk never runs at a GPU budget. Native: `OMG_GPU=1` registers
`GpuV3Walker` (async: one walk in flight, collected next tick without
blocking; `OMG_EARLY=v3` selects the engine). Offline `--render` walks
synchronously instead — its ticks run back to back with no 50 ms for
the GPU to hide in, so there the GPU walk (~60 ms/tick wall, latency-
bound) is slower than the CPU's (~29 ms); the 8 s walkthrough renders
within 0.3 dB RMS of the CPU walk. Web: `sim_v3_enable` after
gpu.js built the pipelines; the worker pumps one job per tick and
injects the resolved bins (`sim_v3_inject`), version-checked through
`sim_v3_version`. The quality panel's compute switch applies to v3 and
the early stat reads `v3 · gpu Xms`.

**Parity** (`crates/omg-gpu/tests/v3_probe.rs`, `v3_parity.rs`; Apple
M-series, Metal):
- deterministic: transmission, raycast and nearest edge identical to the
  CPU on 14 000 random segments (demo world, thick wall + leaf, bunker);
- mirror floor ×0.998 of the image source (CPU ×0.998), Lambertian floor
  ×0.9994 of 1/(2h²) (CPU ×0.9994), free field: direct only, sky = 1;
- shoebox: RT60 within 0.5 %, late level within 0.1 dB, early taps
  within 0.2 dB; thick wall within 0.05 dB per band; facade fill
  identical; doorway open/leaf within 0.05 dB; bunker within 0.4 dB;
  sampled NEE unbiased within 15 %;
- diffraction: knife-edge shadow identical to 0.01 dB at three depths,
  bends over a barrier within 0.75 dB; demo world, 6 sources at the
  tier budget with diffraction: tap energy within 0.1 dB for the strong
  ones and within the CPU's own seed-to-seed spread (≤ 2.5 dB) for the
  far, walled-off ones, late level within 1.5 dB.
Means mislead here: the estimator is heavy-tailed (one ray that
transmits on the importance-floored branch and grazes a near source
sphere carries 10× a tick), so the tests compare medians (lower
quartiles for bimodal fields) over independent ticks.

**Cost** (native, demo world, listener at the club doorway, 16 sources,
High tier, diffraction on):

| | rays (early + late) | sim thread per tick |
|---|---|---|
| CPU walk | 1024 + 128 | 20.8 ms |
| GPU walk, async | 8192 + 1024 | 3.1 ms (direct + edge paths + fold) |

The GPU itself takes ~27 ms per 8× walk (dispatch to readback): the
early family ~10 ms, the rest is the late family's longest rays — one
thread walking a 600-segment tail is latency-bound (~20 µs per segment
against ~1 µs on a CPU core), so even a 1× walk takes ~20 ms of GPU
time. Asynchronous, that latency sits inside the 50 ms tick. (Measured
with the sim thread kept hot: macOS demotes a thread that sleeps 80 % of
the time to an efficiency core, which stretches any 2 ms burst to
~11 ms, GPU or not.)

**Open:** the edge query scans all usable edges (300 in the demo world;
a world with thousands wants the CPU's grid uploaded); the late family's
tail latency (a wavefront/persistent-thread scheme would spread long
rays across threads); and the browser path is unexercised in this
environment — the same WGSL validates under wgpu's WebGPU default
limits, and the JS driver mirrors `GpuV3Tracer::submit` buffer for buffer.

## Budgets (first cut)

| Tier | early rays n | bounces x | late rays | native ms/tick indoors |
|---|---|---|---|---|
| High | 1024 | 6 | 128 | 25–31 |
| Med | 768 | 4 | 96 | — |
| Low | 512 | 3 | 48 | 8–10 |

With the GPU walker (compute = gpu): 8× the early and late rays of the
tier (High 8192 + 1024), same bounces.

Next-event connections: 4 sampled sources per early bounce, 1 per late
bounce (all of them when there are fewer). Source sphere angle:
k = 0.06 (≈ 3.4°). Early bins average over ticks with weight 0.35,
the late echogram with 0.1.
