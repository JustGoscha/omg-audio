# GPU compute port — implementation plan

Move the ray workloads of the **simulation clock** to GPU compute (wgpu
native, WebGPU on the web), keeping the audio clock untouched. This is
the "wgpu compute port" milestone from the README, written as a
step-by-step plan with acceptance criteria so it can be implemented
incrementally by anyone (or any model) without re-deriving the design.

Two independent tracks, both behind runtime settings:
- **Track A (Phases 0–5)**: the GPU port. The CPU tracer is kept
  forever as a switchable backend (`OMG_QUALITY` picks its budget,
  `OMG_GPU` / WebGPU feature-detect picks the backend).
- **Track B**: a CPU quality ladder — one setting that scales rays,
  refresh rates, ISM order and the audio-side tap ceiling so the CPU
  path stops overspending its budget on perceptually unimportant
  work. Track B has no GPU dependency, fixes the current breakup
  directly, and should land FIRST.

## Why (and why not)

What GPU buys:
- **Ray budget headroom**: 4096 rays/trace and 384 dome rays are sized
  for one CPU worker thread at 20 Hz. On GPU these are trivially small;
  10–100× more rays enables mesh-everywhere acoustics (not just shoebox
  rooms) and the **directional late field** (the other open milestone —
  `Echogram.dirs` per bin is already computed, GPU makes it cheap to
  keep at higher ray counts).
- **CPU contention**: on weak machines the 20 Hz sim thread competes
  with the AudioWorklet for cores. Sim ticks cost up to ~15 ms at
  doorways (`tools/` sim probe, July 2026); moving the ray work to GPU
  frees that CPU.

What GPU does NOT buy — be honest about this:
- It does **not** fix audio-thread deadline misses. `eng_process` (the
  per-sample DSP in `omg-dsp`) stays on CPU forever: a 128-frame
  quantum is ~2.9 ms and GPU dispatch+readback jitter cannot live
  inside it. Do not attempt to port anything under `crates/omg-dsp/`.

## Load-bearing constraints (read before writing any code)

1. **The web wasm is freestanding.** `web/worker.js:10` and
   `web/worklet.js` instantiate `omg_web.wasm` with `{}` imports — no
   wasm-bindgen, no web-sys, no JS glue. **Do not add wgpu (or any
   bindgen dependency) to `omg-web`.** On the web, the GPU driver is
   plain JavaScript in `worker.js` using the WebGPU API directly; it
   reads inputs from and writes results into the wasm linear memory
   through new flat-buffer exports (Phase 3). Native uses the `wgpu`
   crate. Both hosts execute the **same WGSL files**.
2. **Two clocks stay decoupled.** GPU work is dispatched during sim
   tick N and consumed at tick N+1 (pipelined, never blocking).
   Staleness of one tick (50 ms) is already absorbed by the design:
   the late field is a slowly-varying statistic under EMA
   (`crates/omg-scene/src/sim.rs` — `TraceGate` doc comment).
3. **CPU path remains, always.** It is the fallback (no adapter, old
   browser, node harnesses) and the test oracle. Runtime selection:
   native env var `OMG_GPU=1|0`, web feature-detect `navigator.gpu`
   with CPU fallback. All existing headless tools (`web_smoke.mjs`,
   `env_probe.mjs`, `bench_web.mjs`) run in node without GPU — they
   must keep passing unchanged on the CPU path.
4. **Statistical parity, not bit parity.** GPU float order and a
   different RNG make bins differ; that is fine. Parity is asserted on
   derived quantities (Phase 0 tolerances). Do not chase bit-exactness.
5. **Do not touch**: `crates/omg-dsp/**`, `web/worklet.js`, the
   ParamBlock format (`crates/omg-core/src/params.rs`), the existing
   CPU tracer (`crates/omg-core/src/tracer.rs` stays as-is — the GPU
   path is an *alternative implementation*, not a replacement).

## What gets ported (ranked)

| # | Workload | Where | Shape | GPU fit |
|---|----------|-------|-------|---------|
| K1 | Stochastic shoebox tracer | `omg-core/src/tracer.rs:80` `trace()` | 4096 rays × ≤64 bounces, analytic box raycast | Perfect — the file's own header says so |
| K2 | Ambient dome fan | `omg-scene/src/dome.rs:290` | 384 rays × ≤6 events vs mesh BVH + panels | Good — needs BVH traversal in WGSL |
| — | AutoPaths diffraction search | `omg-core/src/paths.rs` | branch-and-bound, ~166 µs/query, cached | Poor fit — **stays on CPU** |
| — | ISM early reflections | `omg-core/src/ism.rs` | order ≤3, deterministic, cheap | Not worth it — stays on CPU |

## Repository layout after the port

```
crates/omg-gpu/            new crate: native wgpu host
  src/lib.rs               device init, buffer mgmt, LateBackend impl
  src/layout.rs            #[repr(C)] mirror structs + byte-layout tests
  shaders/trace_box.wgsl   K1 (single source of truth, shared with web)
  shaders/dome.wgsl        K2
web/gpu.js                 new: WebGPU driver used by worker.js
                           (fetches the same .wgsl files; serve.py
                           already serves the repo root)
```

`omg-gpu` is native-only: add it to the workspace but NOT as a
dependency of `omg-web`. `omg-app` gains an optional dep on it.

---

## Phase 0 — golden baselines and the parity harness

Goal: lock in what "correct" means before touching anything.

1. Add `crates/omg-core/tests/trace_golden.rs`:
   - Three fixed configurations (small live room, large dead room,
     source near an open doorway — reuse scene setup from existing
     tests in `omg-scene`), fixed `Rng` seeds.
   - For each: run `trace()` + `estimate_reverb()`
     (`tracer.rs:168`), assert and RECORD in the test as constants:
     RT60 per band, total energy level (dB), per-bin-aggregate
     anisotropy from `Echogram::agg_dir`.
   - These constants become the GPU acceptance targets with
     tolerances: **RT60 ±7 % per band, level ±0.5 dB, anisotropy
     ±0.05**. (Rationale: separate CPU seeds already differ by a few
     percent; the EMA in `Sim` absorbs this.)
2. Verify: `cargo test -p omg-core` green; run twice with different
   seeds to confirm the tolerances are realistic before committing.

Acceptance: test file exists, passes, tolerances documented in it.

## Phase 1 — K1 kernel + native unit parity

Goal: the shoebox tracer runs on wgpu, validated against Phase 0
goldens, not yet wired into the app.

1. Create `crates/omg-gpu` (workspace member). Pin the latest stable
   `wgpu` (exact version in Cargo.toml, no `*`).
2. `src/layout.rs` — flat GPU structs, all `#[repr(C)]`, explicit
   padding, with a unit test asserting `std::mem::size_of` for each:
   - `GpuTraceJob { source: [f32;3], _p0: f32, listener: [f32;3], _p1: f32, source_energy: [f32;3], seed: u32 }`
   - `GpuShoebox`: mirror EXACTLY the fields `Shoebox::raycast_hit`
     reads (`crates/omg-core/src/scene.rs:64` — read that function
     first and copy its semantics; six faces, per-face material:
     `absorption: [f32;3]`, `scattering: f32`).
   - Output: `900 × u32` energy bins (NBINS=300 × NBANDS=3, layout
     `bin*3+band`) + `900 × u32` direction accumulators (bin × xyz),
     fixed-point (see gotcha G2).
3. `shaders/trace_box.wgsl` — port `trace()` (`tracer.rs:80-166`)
   line by line. One thread = one ray; workgroup size 64;
   `n_rays/64` workgroups. Preserve EXACTLY:
   - the receiver-sphere check BEFORE advancing to the hit, including
     the escaping segment (`tracer.rs:104-128` — subtle and
     load-bearing: that segment is how sound exits a doorway);
   - no Russian roulette (`tracer.rs:98-102` comment — measured
     decision, do not "optimize" it back in);
   - 64-bounce cap, `MAX_TIME` 3.0 s cutoff, `1e-7 * per_ray`
     energy floor, `WALL_EPS = 1e-3` surface nudge;
   - Lambertian vs specular branch on `mat.scattering`
     (`tracer.rs:152-163`) including the grazing-direction guard.
   - RNG: PCG32 in WGSL, per-thread state seeded
     `hash(job.seed, thread_id)`. Distribution parity only. Needed
     samples per bounce: unit sphere (2), scatter decision (1).
4. Native parity test `crates/omg-gpu/tests/parity.rs`: run the three
   Phase 0 configurations through the kernel
   (`pollster::block_on`), decode fixed-point, run the SAME
   `estimate_reverb` on the result, assert within Phase 0 tolerances.
   The test must `return` (skip, with a printed notice) when
   `wgpu::Instance::request_adapter` yields none — CI may be headless.

Acceptance: `cargo test -p omg-gpu` green on a machine with a GPU;
`cargo test --workspace` green everywhere else (test self-skips).

## Phase 2 — native integration behind `LateBackend`

Goal: `cargo run --release` uses the GPU when `OMG_GPU=1`.

1. In `omg-scene`, define the seam where `sim.rs` calls `trace()`
   today (three call sites: `sim.rs:172`, `sim.rs:209`,
   `sim.rs:318`):
   ```rust
   pub trait LateBackend {
       /// Queue a trace; result arrives at a later collect().
       fn submit(&mut self, job_id: u32, job: TraceJob);
       /// Non-blocking: echograms finished since the last call.
       fn collect(&mut self) -> Vec<(u32, Echogram)>;
   }
   ```
   The CPU impl computes synchronously in `submit` and returns it on
   the next `collect` — behavior identical to today (same-tick), no
   pipelining needed on CPU.
2. Rework `Sim` so a trace result is consumed when it *arrives*
   rather than inline: keep the last `Echogram` per (source, room)
   job id and feed the EMA (`echo_avg.ema(...)`) on arrival. The
   `TraceGate` logic is unchanged — gate on submit. One-tick-late
   results on GPU are within design (constraint 2).
3. `omg-gpu` implements `LateBackend` with a pipelined ring: submit →
   encode dispatch + `copy_buffer_to_buffer` to a staging buffer →
   `map_async`; `collect()` polls `device.poll(Maintain::Poll)` and
   drains ready maps. NEVER block in either call.
4. `omg-app`: select backend from `OMG_GPU` env (default CPU for
   now); log the choice once at startup.
5. Regression: `cargo run --release -- --render demo.wav --secs 12`
   with `OMG_GPU=0` vs `OMG_GPU=1` — compare the printed level
   report: within 1 dB everywhere. Run the walkthrough regression
   (`cargo test --workspace`) both ways.

Acceptance: level reports match within tolerance; no frame-time
regression in the sim tick (log tick ms both ways at the doorway
position (22.5, 31), the known worst case).

## Phase 3 — web host (JS WebGPU driver)

Goal: the browser demo uses WebGPU for K1, with automatic CPU
fallback; node harnesses unaffected.

1. New wasm exports in `crates/omg-web/src/lib.rs` (flat buffers,
   same style as `eng_param_buf_ptr`):
   - `sim_gpu_jobs() -> *const u8` / `sim_gpu_jobs_len() -> u32`:
     after `sim_tick`, the trace jobs the gate opened this tick,
     serialized as `GpuTraceJob` array + one `GpuShoebox` each
     (byte layout documented in `layout.rs`; add a shared
     `LAYOUT_VERSION: u32` export bumped on any change).
   - `sim_gpu_inject(job_id: u32, len: u32)`: worker writes the
     decoded f32 echogram (bins then dirs) into
     `sim_gpu_buf_ptr()`, then calls this; the sim routes it into
     the EMA exactly like a CPU result.
   - Behind a runtime flag `sim_gpu_enable(1)`: when off (default),
     `sim_tick` traces on CPU as today. **Default off** keeps
     `web_smoke.mjs` / `bench_web.mjs` / `env_probe.mjs` valid
     without any change.
2. `web/gpu.js`: feature-detect (`navigator.gpu` +
   `requestAdapter()`), fetch `crates/omg-gpu/shaders/trace_box.wgsl`
   (served by `serve.py` — repo root), create pipeline once, then per
   tick: read job bytes from wasm memory → write GPU buffers →
   dispatch → async readback → decode fixed-point → `sim_gpu_inject`.
   If init fails at any point: never call `sim_gpu_enable(1)`, log
   one console line, demo runs exactly as today.
3. `worker.js`: after `sim_tick` (line ~30), hand the jobs to gpu.js;
   inject results before the NEXT tick (pipelined, same as native).
4. HUD: surface `gpu: on/off` + last dispatch ms in the existing
   debug panel (`web/main.js` field-debug section) so a tester can
   see which path is live.
5. Regression: `node tools/web_smoke.mjs` and `node
   tools/env_probe.mjs` green (they exercise the CPU path); manual
   browser A/B at the club doorway with the debug panel open.

Acceptance: demo runs with `gpu: on` in Chrome, identical-sounding at
the doorway; hard fallback verified by testing once in a browser
without WebGPU (or with `navigator.gpu` stubbed out).

## Phase 4 — K2: the dome fan on GPU

Goal: `dome.rs` ray fan (384 rays × ≤6 events, mesh BVH + panels) on
both hosts.

1. Flatten the existing BVH (`omg-core/src/mesh.rs` — `BvhNode`,
   `PackedTri`; built once per scene, static) into two GPU buffers.
   Add the flattening + a layout size test to `omg-gpu/src/layout.rs`.
   Panels (`dome.rs` `Panel` — glass/door leaves, the only per-tick
   dynamic geometry) are a small separate buffer rewritten each tick.
2. `shaders/dome.wgsl`: port `trace_escape` + `trace_through`
   (`dome.rs:303`, `dome.rs:361`). BVH traversal with a fixed
   32-entry local stack. Output is per-ray (escape direction bin id +
   3 band transmissions) — only 384 small records, so NO atomics:
   plain per-thread output slots, binned on the host (CPU) exactly as
   `dome.rs:394-416` does today.
3. Same host plumbing as K1 on both platforms; same pipelined
   consumption (dome output already goes through a per-bin EMA,
   `dome.rs` `EMA = 0.35`).
4. Regression: `node tools/env_probe.mjs` (CPU path) green; browser
   walk through the vestibule and a door swing with GPU on — the
   probe's assertion (no level step) checked by ear + HUD.

Acceptance: parity test for one fixed pose set (indoor, outdoor,
no-opening room) within ±0.5 dB per dome bin vs CPU.

## Phase 5 (stretch) — spend the headroom

Only after 1–4 are stable:
- **Batch all jobs in one dispatch** (job id = `workgroup_id.z`) —
  one submit for all sources instead of N.
- **Raise ray budgets on GPU**: N_RAYS 4096 → 32768, dome 384 → 4096,
  gated on measured dispatch time (<10 ms). Expect audibly smoother
  RT60 at doorways (less Monte Carlo flutter for the EMA to hide).
- **Directional late field**: `Echogram.dirs` is already computed per
  bin; with the bigger budget, feed per-bin direction + anisotropy
  into the late-field rendering (design TBD — this is the README's
  other open milestone, unblocked by this plan but not part of it).
- **Mesh-everywhere in-room acoustics**: K2's BVH traversal makes
  `trace()` over `Mesh` affordable → non-shoebox rooms.

---

## Track C — PT-early: path-traced early reflections (replace ISM)

Goal: a real-time path-traced early-reflection engine over arbitrary
geometry — rooms with things IN them — that keeps every invariant the
tap pipeline depends on (stable identities, exact delays, Doppler by
glide). ISM stays only as the test oracle: in an empty shoebox the two
must agree, and that equivalence is the acceptance gate.

Why ISM loses once rooms have contents: each image source needs a
visibility test against every occluder (combinatorial in order), and
occluded images don't just attenuate — they need diffraction handling
per image. A traced path's cost is CONSTANT in scene clutter (BVH
depth), constant in source count for the walk itself, and occlusion
falls out of the tracing instead of being patched in.

### The algorithm

1. **Listener-launched** rays (reciprocity): one fan from the head,
   NOT per-source mirroring. Deterministic golden-spiral base
   directions with a per-tick low-discrepancy rotation, so coverage
   accumulates across ticks instead of re-sampling the same set.
2. **Bounce over the world mesh** (the dome's BVH — walls, slabs,
   roofs, door leaves, furniture when it arrives) up to ~4 events,
   specular with material scattering (same rules as the late tracer).
3. **Next-event estimation at every vertex**: connect each bounce
   vertex to each audible source with a shadow ray. A clear connection
   yields a path listener→v1→…→vk→source whose length is the exact
   segment sum — sample-accurate delay, NO receiver-sphere
   quantization (this kills the classic precision objection to MC
   early reflections). Direction at the ear = the first segment;
   per-band gain = product of surface reflections × air × 1/d.
4. **Path-space hashing** — the stability trick that makes it a tap
   engine instead of noise: key = hash(source id, ordered chain of hit
   surface ids). Every ray that finds the same surface chain dedupes
   into ONE path record (keep the energy-weighted representative).
   The key survives motion while the geometry glides → the tap keeps
   its identity, the delay line keeps gliding, Doppler and click-free
   transitions survive by construction.
5. **Temporal path cache with hysteresis**: the path table persists
   across ticks. A sighted path refreshes its TTL (~0.5 s); unsighted
   paths decay and fade out through the normal slot release — a path
   missed by this tick's rays does NOT flicker. The deterministic
   rotating fan + cache turns temporal coverage into spatial
   completeness.
6. **Validation before emission**: a candidate path re-checks with
   exact segment intersections (no ray epsilon slop) before becoming
   a tap; grazing false positives die here, not in the ear.
7. Direct paths and diffraction stay as they are (straight-tap logic,
   AutoPaths knife-edge floors) — PT-early replaces the ISM taps only.

### GPU mapping (kernel K3)

Thread-per-ray mega-kernel: bounce loop + NEE, emitting compact path
records (source id, packed surface chain, length, band gains, first
segment direction — ~32 B) into an append buffer. Host (or worker JS)
folds records into the hash table and emits ParamBlock taps. Budget:
4–8 k rays × 4 events × NEE to ~10 sources — trivially realtime on
the measured M-series numbers; the CPU fallback runs the same code at
1–2 k rays and simply converges the cache slower, same correctness.

### GPU-first, CPU-capable

K3 is designed FOR the GPU: bounce loop + NEE is the same shape as
the late tracer that already runs at 1.5 ms for 32 k rays, and the
append-buffer output (~32 B/record) is a fraction of an echogram
readback. The CPU fallback is the same algorithm at 1–2 k rays —
correctness identical, cache convergence slower. Web host reuses the
phase 3 bridge pattern (flat job/record buffers, versioned, JS driver,
silent CPU fallback).

### The early-reflections backend is a SETTING, not a migration

`early = ism | traced | hybrid` (env var native, quality panel web —
same pattern as the late-field cpu/gpu toggle, A/B-able live):

- `ism` — today's engine, kept permanently. It is the exact solution
  for empty shoeboxes, the cheapest path on GPU-less devices, and the
  oracle the traced backend is validated against. Never deleted.
- `traced` — the full PT-early path set from K3.
- `hybrid` — probably the end state (see below).

Both backends emit the SAME tap structure through the same pipeline;
switching mid-walk crossfades through the normal slot release, exactly
like the tap-ceiling ladder.

### The endgame: one mesh, one tracer — rooms stop being simulated

The full win of tracing is not (only) speed: traced over the WORLD
mesh, the portal apparatus stops existing. No door graph BFS, no
virtual sources at apertures, no routed-source pre-door paths, no
coupled-reverb door emitters, no Sabine seep balance — a ray that
threads a doorway or a window costs one BVH traversal like any other
ray, and inter-room sound is EMERGENT. The repo already proves this
twice in miniature: the ambient dome (no room graph, openings localize
the world outside by construction) and the mesh tracer's
emergent-portal test (doorway hole, listener outside, zero authoring).
Track C's end state is that architecture for every path: sources and
listener in one world mesh, K3 finding whatever transport exists.

What straight rays cannot deliver stays explicit, by physics not by
architecture:
- **Diffraction** — rays don't bend; without the knife-edge floors the
  shadow zone behind a jamb goes silent. AutoPaths stays, priced over
  the same mesh.
- **Transmission** — rays must continue THROUGH surfaces with
  mass-law loss (the dome's trace_through already does this), or the
  club stops rumbling through its walls.
- **Late field** — remains the statistical tracer (K1), eventually
  also over the world mesh instead of per-room shoeboxes; the per-room
  Sabine/FDN machinery then simplifies to per-region decay estimates.

### C6 — the endgame build: world-mesh PT, portals dissolve

What C1–C5 did NOT touch: portal routing, doorway virtual sources and
the crossing blend (both rooms fully simulated while you walk a door —
the measured 2.5× doorway tick). C6 removes them:

- **C6a — surface ids on the world mesh** *(SHIPPED)* (the deferred C0): every
  authored face one stable u16; patches inherit via the existing
  tri back-ref. Gate: id uniqueness/coverage test.
- **C6b — mesh chains.** *(SHIPPED: pt_mesh.rs — box equivalence 25/25, doorway threads at free-space level and masonry costs 45 dB with zero portal code)* pt discovery over the dome's world-mesh BVH
  (surface-id chains instead of wall indices); exact solve becomes
  mirror-across-plane per hit surface + segment revalidation, plus
  TRANSMISSION segments (a chain may cross glass/walls with mass-law
  loss — trace_through semantics inside path records). Gate: in a
  mesh-built empty box, identical output to the analytic solver.
- **C6c — one listener context.** *(SHIPPED: early_world.rs — one
  listener-launched chain cache over the world mesh shared by all
  sources; portal routing/virtual sources/aperture re-radiation/blend
  gone behind `early=traced` (ism keeps them); door leaves, glass
  panes and furniture are transmissive `extras` boxes in the solve;
  AutoPaths supplies ONE dominant-edge diffraction tap per source,
  scaled by the complement of the solved direct; the late field is a
  per-ROOM probe per tick — the diffuse level is a room statistic —
  so a doorway blend probes two rooms, not two rooms × N sources.
  Gates: walk-in continuity, walk-past aperture contrast 7.1 dB with
  a step-free profile, door-leaf muffle/restore, and doorway tick
  0.98×/0.90× open-square — the door spike is structurally gone.)*
  Sources and listener in world
  coordinates, one PathCache for the world (not per room): doorways
  and windows are just holes the chains thread. Portal routing,
  virtual sources, aperture re-radiation and the crossing blend are
  DELETED behind `early=traced`; ism keeps the old machinery. Jamb
  diffraction: blocked directs already hand off to knife-edge bends
  (C5) — jamb edges come from the same mesh edges AutoPaths extracts.
  Gate: env_probe-style walkthrough — level continuity through every
  doorway with the portal code off; doorway tick ≤ 1.2× open-square.
  Deferred to C6d: world discovery is CPU `mesh_chains` for now (the
  box-bitmap GPU kernel can't express surface-id chains — the BVH
  kernel that replaces it is the same work item as the dome/late
  port), and the coupled-room wet send still uses the door-chain
  route for its excitation heuristic.
- **C6d — late field + dome follow.** *(LATE FIELD SHIPPED: the
  stochastic tracer was already geometry-generic, so `early=traced`
  now measures ONE echogram per source over the world mesh — actual
  source, actual listener, door leaves overlaid as panel boxes
  (`WithPanels`). Coupled-room decay, doorway wet and its DIRECTION
  come out of the measurement: the anisotropic share of the tail
  plays on the directional wet bus via `Echogram::late_direction`,
  the diffuse remainder feeds the FDN — the routed radiator/remote
  machinery is gone from the traced path entirely. Budget: one world
  trace per tick, round-robin across source gates (reverb is a slow
  statistic; the EMA absorbs staleness) at 512/384/256 rays by tier.
  Gates: mesh↔shoebox tracer parity (level parity to 4 decimals),
  corridor listener measures living-room wet through the door at
  rt60 0.82 s with ~29% directional share, leaf close/reopen drops/
  restores it, doorway tick now 0.83×/0.60× open-square.)*
  Remaining in C6d: the GPU side — BVH flatten + a world-trace
  kernel (K2) claiming the same one-trace-per-tick slot behind the
  late seam, PT discovery (`mesh_chains`) as a bitmap-free chain
  kernel over the same buffers, and phase 4's dome kernel sharing
  them. The per-room FDN reduces to region estimates once regions
  replace rooms in authoring.

### Hybrid: where the combination beats either

Each method is strongest at a different order:

- **Order-1 and the direct set**: ISM/analytic — a handful of exact,
  never-flickering, zero-variance paths that carry localization. Keep
  them analytic even in mesh rooms (single-reflection solve per major
  planar surface + occlusion shadow ray = trivially cheap, and the
  perceptually dominant paths get determinism for free).
- **Order 2+ and everything occluded/cluttered**: traced — where
  image mirroring explodes and tracing's constant budget wins.
- **Empty shoebox rooms at Low tier**: pure ISM (no rays spent).

The hybrid dedupes by path key: an analytic path and a traced path
with the same surface chain are the same tap (analytic wins — exact
beats sampled). This is likely the shipping default; `ism` and
`traced` remain as pure modes for A/B and regression.

### Acceptance

- **Shoebox equivalence**: empty golden rooms — PT-early must
  reproduce the ISM ≤3-order path set (delay ±0.5 ms, level ±1 dB,
  per path) after cache convergence. ISM stays as the oracle and as
  a selectable backend (see above) — it never retires.

### Implementation phases (C0–C5)

Same discipline as Track A: each phase lands behind the previous one
as oracle, nothing ships without its gate, `bench_web.mjs` (the audio
clock) must read unchanged after every phase.

**C0 — surface identity on the world mesh.** Path keys hash chains of
SURFACE ids, so surfaces need stable identities. `MeshBuilder` gains a
`surface_id: u16` per original triangle (coplanar authored face =
one id: each wall, slab, roof plane, door leaf, facade). The dome's
`build_world_mesh` (dome.rs:124) assigns them during construction;
tessellated patches inherit their original's id (the `tri` back-ref
already exists, mesh.rs). Gate: a unit test walks every triangle and
asserts patches of one authored face share one id, distinct faces
never collide.

**C1 — CPU reference tracer (`omg-core/src/pt.rs`).** Pure function:
`pt_trace(mesh, sources, listener, budget, rng) -> Vec<PathRecord>`
with `PathRecord { source: u16, chain: [u16; 4], order: u8,
length_m: f32, gains: [f32; 3], dir: [f32; 3] }`. Listener-launched
golden-spiral fan + per-call rotation; specular bounces over the BVH;
NEE shadow ray to every source at every vertex (and order-0 from the
listener itself = the direct path); transmission continuation through
surfaces with mass-law loss; exact-segment revalidation before a
record is emitted. No cache here — pure, testable.
Gate (`tests/pt_golden.rs`): in the Phase 0 golden shoeboxes (walls
get 6 surface ids), the union of records over 8 rotated calls
contains every ISM path of order ≤ 2 with delay error ≤ 0.5 ms and
level error ≤ 1 dB, and no path that ISM proves impossible (chain
replay must mirror-reconstruct within epsilon).

**C2 — path cache + tap emission + the setting (`omg-scene`).**
`PathTable`: `HashMap<u64 /*src<<48 | chain hash*/, CachedPath>` with
TTL (~10 ticks), energy-weighted representative, glide (length/dir
update on re-sighting), fade-out on expiry (emit a last ParamBlock
without the key; the renderer's slot release does the rest). Tap key
= low 32 bits of the path hash namespaced above the ISM key range.
`early = ism | traced` setting: `quality::set_early(u32)` + native
`OMG_EARLY=ism|traced` + a quality-panel selector next to the
late-field toggle (worker message, same pattern). In `traced`, Sim
skips `image_source_taps` for in-room sources and splices the cache's
taps for that (source, room) instead; everything else (straight taps,
portals, diffraction floors, late field) is untouched at this phase.
Gate: walkthrough regression under `traced` — level trajectory within
1 dB of `ism` everywhere in the empty-scene demo; stationary listener
shows zero tap churn per tick (add a churn counter to eng_debug).

**C3 — GPU kernel (`omg-gpu/shaders/pt_early.wgsl`).** Thread-per-ray
mega-kernel: bounce loop + NEE, records appended via an atomic cursor
into a fixed buffer (`PathRecord` = 32 B packed; cap 4096
records/dispatch, cursor overflow drops and logs — the cache absorbs
misses). Inputs: flattened BVH + surface ids (one static buffer,
rebuilt only on geometry change), per-dispatch uniform (listener,
rotation seed, source array ≤ 16). Host dedupes into the same
PathTable. Layout tests + LAYOUT_VERSION bump as in phase 1.
Gate: same C1 golden gate, run through the kernel (self-skips without
adapter); plus the speed probe — target ≥ 8k rays × 4 bounces × NEE
under 3 ms.

**C4 — web host.** Extend the phase 3 bridge: `sim_pt_jobs` /
`sim_pt_inject` flat buffers (records in, versioned), gpu.js gains the
second pipeline (same fetch/init/fallback pattern), worker pumps both
kernels per tick. CPU fallback = C1 at reduced budget inside the wasm.
Gate: node harnesses green on the CPU path; browser A/B ism↔traced at
the club doorway with the churn counter visible.

**C5 — hybrid + clutter (the payoff).** *(Engine layer shipped:
Aabb occluders validate every unfolded segment, the blocked direct
path hands off to a knife-edge bend over the nearest silhouette —
single-edge and face-hugging double bends, lit-side proximity keeps
the boundary continuous — gated by the shadow-walk test. Demo-scene
furniture authoring (visuals + colliders + per-room occluder lists
into Sim) and occluder-face reflection chains are the remaining
content/extension work.)* `hybrid` mode: analytic
order-1 (per major planar surface: one mirror solve + occlusion
shadow ray) + traced order-2+, deduped by chain key, analytic wins.
Then put THINGS in rooms — furniture boxes into the world mesh with
surface ids — and add the occluder regression: a pillar between
source and listener, walk through its shadow, assert no level step
(the PT direct path hands off to the diffraction floor). This is the
phase where ISM stops being reachable and `traced/hybrid` becomes the
default; `ism` remains selectable for empty-shoebox A/B forever.

Budgets (from the measured kernel numbers): 8k rays × 4 events ≈ the
late tracer's 32k×64 work — ~1.5 ms native; CPU fallback 1–2k rays.
Cache TTL turns the 20 Hz fan into effective 160k rays/s of coverage.
- **Occluder test**: a pillar between source and listener — direct
  tap hands off to the diffraction floor with no level step (extend
  the existing shadow-walk regression).
- **Stability**: stationary listener → zero tap birth/death per tick;
  walking pace → order-1 paths never flicker (identity churn counter
  in the debug panel).
- `bench_web.mjs` unchanged (audio clock untouched, as ever).

## Track B — the CPU quality ladder (independent of GPU, do first)

The GPU port keeps the CPU tracer as a switchable backend forever
(constraint 3). Track B makes that CPU path *survivable on weak
machines*: one quality setting that scales every budgeted workload,
shedding perceptually unimportant work before anything audibly breaks.
It needs no GPU knowledge and can land before Phase 1 — it also
becomes the fallback tier the web demo drops to when WebGPU is absent.

### The one knob

`Quality = High | Medium | Low | Auto` (default Auto).
- Native: `OMG_QUALITY=high|med|low|auto`.
- Web: URL param `?q=` + a selector in the debug panel; new wasm
  exports `sim_set_quality(u32)` and `eng_set_tap_ceiling(u32)`.
- Auto: a governor walks tiers from measured load — worklet `load`
  ratio and `gaps` (audio clock, already metered in `worklet.js`) and
  `tickMs` (sim clock, already sent by `worker.js:39`). Shed
  immediately on a gap or `load > 0.85`; recover one tier after ~10 s
  clean. Same hysteresis pattern as the existing point-budget
  governor — extend it, don't build a second one beside it.

### What each tier scales

Every value is already a constant in exactly one place — the tier
table turns them into fields on a `QualityTier` struct threaded to
where the constant lives today:

| lever | file | High | Med | Low | why it's perceptually cheap |
|---|---|---|---|---|---|
| trace rays `N_RAYS` | `omg-scene/src/sim.rs:15` | 4096 | 2048 | 1024 | variance only, never bias — the EMA absorbs it (`tracer.rs:98` comment). RT60 flutters slightly more; nobody localizes flutter. |
| trace refresh (gate max age) | `sim.rs` `TraceGate` `age >= 8` | 8 | 12 | 16 | the late field is a slow statistic; staler refresh ≠ different sound, just slower response to room changes |
| dome rays | `omg-scene/src/dome.rs:34` | 384 | 256 | 160 | dome bins are 9 coarse sectors under a 0.35 EMA — extra rays only reduce bin noise |
| dome max events | `dome.rs:35` | 6 | 5 | 4 | events ≥5 carry little energy (each bounce/pane multiplies transmission down) |
| ISM order | `sim.rs:14` `ISM_ORDER` | 3 | 3 | 2 | order-3 images are the quietest taps; the late field covers the gap. Do NOT drop below 2 — order-2 carries audible early pattern. |
| audio tap ceiling (per source) | `omg-dsp/src/renderer.rs:281` `MAX_INCOMING` | 160 | 112 | 64 | measured July 2026: a doorway carries ~720 live taps (5 src × ~145) vs 95 in the open — the weakest ~half sit far below the masking threshold of the strongest. **This is the lever that actually stops audio-thread breakup**; taps that fall out release with the existing fade (`renderer.rs:288`), so shedding is click-free by construction. |
| point budget cap | `web/worklet.js` `BUDGET_MAX` | 32 | 16 | 8 | already adaptive; the tier just lowers the ceiling the governor may climb to |

Bounce cap (64) is deliberately NOT in the table: truncating bounces
biases the RT60 fit in low-absorption rooms (the no-Russian-roulette
comment, `tracer.rs:98` — measured, not theory). Rays and refresh are
the honest levers; leave bounces alone.

### Implementation order

1. `eng_set_tap_ceiling(n)`: make `MAX_INCOMING` a renderer field
   (default 160), export it like `eng_set_point_budget`
   (`omg-web/src/lib.rs:313` is the pattern), wire the worklet
   governor to shed it alongside the point budget (gap or
   `load > 0.85` → drop a tier immediately). This alone is the
   biggest breakup fix in the whole document — land it first and
   measure at the club doorway (22.5, 31).
2. `QualityTier` struct in `omg-scene`, threaded into `Sim::new` /
   `WorldSim`; `sim_set_quality` export; native env parsing in
   `omg-app`.
3. Auto-governor: extend the existing worklet governor to walk tiers;
   sim-side tier changes ride the existing `postMessage` channel
   (worker already sends `tickMs`, main already routes settings).
4. Debug panel: show current tier + which meter forced it.

### Acceptance

- `node tools/bench_web.mjs` at Low: rtf at the worst-case position
  ≥ 2× at budget 8 (vs ~1.1× today).
- `node tools/env_probe.mjs` passes at every fixed tier (continuity
  through doors must not depend on quality).
- Tier switches while walking through the club doorway produce no
  audible step (slot fades + EMA cover both clocks) — verify by ear
  and with the blackbox recorder armed.
- `cargo test --workspace` green at every tier (tests that pin
  levels/RT60 run at High — set it explicitly in test setup).

## Gotchas (G-notes for the implementer)

- **G1 — no f32 atomics in WGSL.** Echogram accumulation uses
  `atomic<u32>` fixed-point. Scale: per-ray energy starts at
  `source_energy/n_rays ≤ 1/4096` and only decays; the total across
  all rays is ≤ max(source_energy) ≈ 1. Use scale `2^30` (headroom
  4× above the ≤1 sum; u32 max ≈ 4.29e9 = 4 × 2^30). Decode:
  `f = u as f32 / 2^30`. Direction accumulators hold signed values:
  store `(x * 2^28) + 2^31` offset-binary per component, decode
  accordingly — document both scales as constants in layout.rs AND
  at the top of the WGSL file, with the same names.
- **G2 — two-level accumulation.** 4096 threads × ~hundreds of
  receiver crossings hammering 900 global atomics is fine at this
  size — do the simple global-atomics version FIRST, measure, and
  only add workgroup-local staging if a dispatch exceeds ~2 ms.
  Do not prematurely build the reduction.
- **G3 — `map_async` never blocks the tick.** Native: `collect()`
  polls; web: the readback promise resolves between ticks. If a
  result hasn't arrived by the next tick, skip — the gate will
  resubmit; the EMA holds the last estimate meanwhile.
- **G4 — WebGPU in a dedicated Worker** is supported in current
  Chrome/Edge; Safari and Firefox vary. Feature-detect inside the
  worker itself, not the page. Fallback must be the no-op path
  (constraint 3), tested by stubbing `navigator.gpu = undefined`.
- **G5 — buffer layout drift** is the classic silent killer. Every
  `#[repr(C)]` struct gets a `size_of`/`offset_of` unit test, the
  WGSL structs carry a comment with the expected byte offsets, and
  `LAYOUT_VERSION` (exported from wasm, checked by gpu.js) hard-fails
  the GPU path into CPU fallback on mismatch instead of decoding
  garbage.
- **G6 — the trace-skip gate stays.** GPU makes tracing cheap, not
  free; `TraceGate` (sim.rs:38) also bounds *result churn* feeding
  the EMA. Leave its thresholds alone in this project.
- **G7 — `estimate_reverb` stays on CPU.** It's a tiny Schroeder fit
  over 300 bins; porting it buys nothing and costs a readback format.
  GPU returns raw echograms only.

## Verification ladder (run after every phase)

```sh
cargo test --workspace                 # incl. goldens + layout tests
cargo run --release -- --render a.wav --secs 12   # native, OMG_GPU=0
OMG_GPU=1 cargo run --release -- --render b.wav --secs 12  # then diff reports
sh tools/build_web.sh                  # wasm still freestanding & building
node tools/web_smoke.mjs               # CPU path in node
node tools/env_probe.mjs               # ambience/rain continuity
node tools/bench_web.mjs               # audio-thread rtf UNCHANGED (GPU must not touch it)
```

The last line is the sentinel for the whole plan's discipline: the
GPU port lives entirely on the sim clock, so `bench_web.mjs` (which
measures `eng_process`) must report the same numbers before and after
every phase.
