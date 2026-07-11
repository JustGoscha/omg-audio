# omg-audio

**Live demo: <https://justgoscha.github.io/omg-audio/web/>** — headphones on,
works on Android Chrome (turn your body to look around). Desktop: WASD +
mouse, Space throws a whistling, exploding ball.


A path-traced sound propagation engine targeting the web (wasm + WebGPU +
AudioWorklet) with binaural output. This is the native Rust skeleton — the
same core compiles to both targets; native is the fast-iteration reference
build.

## Hear it

```sh
cargo run --release              # live playback, Ctrl+C to quit
cargo run --release -- --render demo.wav --secs 12   # offline render + level report

# use any WAV as the source signal instead of the synthetic clapper
cargo run --release -- --input assets/aria48.wav --render out.wav

# scripted walkthrough (living room → corridor → great hall → open air) with
# two FIXED sources: music in the living room, a narrator in the great hall.
# Cross-room audibility uses a portal model: a source in another room renders
# as a virtual source at the doorway, per-band muffled per door crossed.
# Within 1.5 m of any doorway BOTH connected rooms are simulated and
# equal-power crossfaded (that includes fading the world open at the exit:
# outdoors = direct + grass-ground reflection, no walls, no reverb).
# The renderer crossfades taps on path-identity change and glides them on
# motion (Doppler) — room changes cannot click or chirp.
# Coupled rooms: a cross-room source also carries its OWN room's reverb,
# rendered as a directional wet emitter at the doorway (second FDN per
# source) — you hear the hall being a hall from the corridor.
cargo run --release -- --walkthrough --render walk.wav --json walk.json \
    [--music m.wav --voice v.wav]
python3 tools/render_viz.py walk.json frames 30   # top-down schematic frames
ffmpeg -framerate 30 -i frames/%05d.png -i walk.wav \
    -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest walkthrough.mp4
```

Pre-rendered examples in the repo root: `walkthrough-two-sources.mp4`
(fixed sources + doorway transitions — start here), `walkthrough-voice.mp4`,
`walkthrough-music.mp4` (older: source follows the listener),
`demo-music-orbit.wav`. Source material is
public-domain/CC0: LibriVox *Alice in Wonderland* (voice) and Kimiko
Ishizaka's *Open Goldberg Variations* (music, CC0).

The demo: a percussive source orbiting the listener (8 s period) in an
8×6×3 m room with mixed materials (concrete, drywall, carpet, wood,
acoustic tile). You should hear the source circle around you, early
reflections coloring with material absorption, and a ~0.75 s reverb tail
whose decay was *measured by the path tracer*, not hand-tuned.

## Architecture — the two clocks

The load-bearing decision: the **simulation clock** and the **audio clock**
never share state except through `ParamBlock`.

```
┌────────────────────────────┐   ParamBlock    ┌───────────────────────────────┐
│ SIMULATION  (20 Hz thread) │  ───────────▶   │ AUDIO  (per-sample callback)  │
│                            │                 │                               │
│ omg-core                   │  taps: delay,   │ omg-dsp                       │
│  · image sources (ISM ≤3)  │  direction,     │  · fractional delays (Doppler)│
│  · stochastic path tracer  │  band gains     │  · per-tap 3-band shelf EQ    │
│    → echogram → RT60/level │                 │  · FOA ambisonic bus          │
│  · EMA temporal accumulation│ reverb: rt60[3]│  · 8-line FDN (traced RT60)   │
│    (kills MC flutter)      │  + level        │  · stereo decode (→ MagLS)    │
└────────────────────────────┘                 └───────────────────────────────┘
```

- Every parameter crossing the boundary goes through a one-pole smoother —
  simulation updates can never click, and moving delays *are* the Doppler.
- Tap index identity is stable across updates (deterministic ISM lattice
  enumeration), which is what makes per-tap smoothing valid.
- Native: thread + mutex mailbox. Web: Worker + SharedArrayBuffer, worklet
  reads with atomics. `ParamBlock` is the serialization contract.
- The late field is an FDN driven by traced per-band RT60, not a convolved
  impulse response — parameters morph artifact-free in real time.

## Crates

| crate | role | wasm? |
|---|---|---|
| `omg-core` | scene, materials, ISM, path tracer, `ParamBlock`. Zero deps. | yes |
| `omg-dsp` | delay lines, shelves, FDN, ambisonic bus, renderer. Alloc-free hot path. | yes (inside AudioWorklet) |
| `omg-app` | native demo binary: cpal live output + hound offline render | native only |

Planned:

| crate | role |
|---|---|
| `omg-gpu` | the tracer as wgsl compute via `wgpu` — same shaders on Metal (native) and WebGPU (browser) |
| `omg-web` | wasm bindings, Worker + AudioWorklet plumbing, SAB ring buffer, demo page |

## Roadmap

1. **M1 (this)** — shoebox ISM + stochastic RT60 → FDN, FOA panning, native. ✅
2. **M2: real binaural** ✅ — order-2 ambisonics (ACN/SN3D), all sources on
   one shared bus, single decode stage: 12 virtual speakers (icosahedron) ×
   measured MIT KEMAR HRIRs (128-tap, resampled to 48 kHz by
   `tools/make_hrir.py` into `assets/hrir_ico12.bin`; falls back to cardioid
   stereo if the asset is missing). Data: MIT Media Lab KEMAR HRTF
   measurements (Gardner & Martin), free with attribution. Still open:
   MagLS decoder, SOFA loading, head-tracking rotation of the bus.
3. **M3: web build** ✅ — `omg-web` (cdylib, no wasm-bindgen: plain C-ABI
   over linear memory): WorldSim in a Web Worker at 20 Hz, renderer inside
   the AudioWorklet, flat `ParamBlock`s over postMessage (~4 KB @ 20 Hz — no
   SharedArrayBuffer or COOP/COEP needed). Head yaw (device orientation /
   drag) is a fast path straight to the DSP: SH-bus z-rotation at the decode
   stage + per-block nearest-HRIR re-selection for point taps.

   ```sh
   sh tools/build_web.sh          # cargo build wasm + stage into web/
   python3 tools/serve.py         # http://localhost:8000/web/
   node tools/web_smoke.mjs       # headless pipeline test
   # Android (Chrome): plug in via USB, then
   adb reverse tcp:8000 tcp:8000  # phone's localhost:8000 → this machine
   # open http://localhost:8000/web/ on the phone — localhost is a secure
   # context, so AudioWorklet + orientation sensors work without HTTPS.
   ```
   Android is the mobile target (deviceorientationabsolute, no permission
   dance); iOS untested/out of scope for now.
4. **M4: arbitrary geometry** — triangle meshes + BVH; ISM needs path
   validation rays; tracer bounces off triangles. This is where `omg-gpu`
   (wgpu compute) replaces the CPU tracer.
5. **M5: diffraction** — UTD over an edge graph for the dominant occlusion
   paths (the single biggest realism item; see research notes). Until then
   occlusion is binary and will sound wrong behind obstacles.
6. **M6: multiple sources** — renderer already sums on one FOA bus;
   per-source early taps + one shared FDN.

## Known simplifications (deliberate, skeleton-stage)

- 3 frequency bands (low <250 Hz, mid, high >2.5 kHz); production wants 4–9.
- Stereo decode is virtual cardioids, not HRTF — headphones sound like
  wide stereo, not yet "outside your head". M2 fixes this.
- FDN low-band decay follows mid (no low shelf in the loop yet).
- Late-field level calibration is heuristic (energy after 80 ms, clamped).
- Reverb params are traced per-listener/source pair but the FDN is global —
  correct for one room, revisit for coupled spaces.
- No diffraction, no transmission, shoebox-only geometry.

## License & attribution

Code: MIT. Acoustic data and demo media:
- MIT KEMAR HRTF measurements (Gardner & Martin, MIT Media Lab) — free with attribution.
- Kimiko Ishizaka, *Open Goldberg Variations* — CC0.
- LibriVox *Alice's Adventures in Wonderland* — public domain.
- Street ambience: "Karlova 0001", Wikimedia Commons — public domain.
- Club track, projectile FX: synthesized by this repo's tools (CC0).

`assets/*.wav` are gitignored; regenerate via `tools/` (see build scripts) or
use the shipped `.ogg` versions (what the web demo loads).
