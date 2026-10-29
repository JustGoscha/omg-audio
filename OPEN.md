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
did so for the bells), then retire the legacy engine and its tests
(`omg-scene` world/sim/early/late/dome, `omg-core` pt/pt_mesh/paths/
tracer) — "consolidate duplicated code". `world::tests::shadow_
boundaries_are_continuous` (legacy) still fails.

### v3 mode is slow and noisy
Exponential averages (EMA 0.35 / 0.1): the club door takes seconds to
land (+20…+37 dB 50 ms after the shut, +2…+26 dB at 1.6 s), identical
clicks standing still spread 3–7 dB sd. v5.5 is the default; v3 stays as
the legacy comparison.

### A loud room's field through its walls converges slowly
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
guiding). Large.

### Moving sources on the GPU walk
The GPU walk connects to where a source emitted its direct sound; its
echoes reach that point too (exact only on the CPU, which solves each
connection's own emission time). Pass the track to the kernel.

### The mixer's tap budget drops paths with their energy
Over its budget (rung 0: 48 taps for every source) a quiet source keeps
its strongest path only; the rest go silent instead of joining its
convolution. Fold dropped paths' energy into the dense response, or give
the simulation the budget so it does.

### Tick cost
The town with every source traced: ~56 ms per tick native High on the
CPU (50 ms budget at 20 Hz; 49 ms before the next-event draws reached
farther sources, a faster BVH box test gave 6 ms back), the web
worker's wasm 52 ms at Medium, 33 ms at Low (its governor sheds a tier
over 40 ms). The tier governors shed quality; a lighter two-edge search
(cache the string per listener cell), fewer chain solves at exact
frames, and a cheaper transmission query (half the tick: every hit
along a segment, sorted) would buy it back.

### Linear interpolation on moving paths
A path's delay read with linear interpolation: a full-band click's
energy depends on the fractional delay (±1–1.5 dB), and moving sources'
treble is modulated with it. A 4-point interpolator (Hermite) costs ~2×
per path read.

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
