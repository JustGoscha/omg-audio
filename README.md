# omg-audio

**Live demo: <https://justgoscha.github.io/omg-audio/web/>** — headphones on.
Desktop: WASD + mouse, Space throws a whistling, exploding ball. Android
Chrome: on-screen joystick, turn your body to look around (device
orientation drives the binaural rendering).

A real-time path-traced sound propagation engine in Rust, compiled to both
native (reference build) and wasm (Web Worker + AudioWorklet). One scene —
a small city square with a house playing music, a great hall with a
narrator, a club with four synchronized speakers behind an entrance
vestibule, a two-storey old house, a colonnade and a kiosk — and the sound
field is *computed*, not authored:

- **Image-source early reflections** (order 3) + **stochastic path tracer**
  (4096 rays, 3 frequency bands) measuring per-room RT60 and late level.
- **Portals**: sources in other rooms render as virtual sources at
  doorways (BFS over the door graph), per-band muffled per door crossed,
  equal-power blended near thresholds. Coupled rooms carry their *own*
  reverb through the doorway as a directional wet emitter — you hear the
  hall being a hall from the corridor.
- **Wall transmission**: straight rays cross wall segments with mass-law
  attenuation (amplitude ∝ reference/thickness) and a fitted one-pole
  lowpass — European brick and concrete, not drywall. The club reads as
  *oomp-oomp* from the street and opens up as you walk in.
- **Aperture radiation**: doors and windows re-radiate what's inside —
  including order-2 reflections from the source room — so a window is
  audible off-axis, not just on the sight line.
- **Diffraction**: blocked outdoor paths bend around building corners
  (single and double) and over roof lines, priced by Kurze–Anderson
  knife-edge losses — the Fresnel number of each path's detour decides,
  per band, how much survives. Bass wraps around the club; over-the-roof
  arrivals actually come from above. Plus **facade reflections** outdoors.
- **Binaural output**: order-2 ambisonics decoded through 20 virtual
  speakers (dodecahedron) × measured MIT KEMAR HRIRs, plus *point
  rendering* — the strongest N paths per source get their own nearest-HRIR
  convolution from a 710-direction grid, N adapting to measured CPU load.
- **Ear adaptation (AGC)**: protects against ultra-loud content (club PA,
  explosions) with fast clamp and ~30 s recovery; it never boosts quiet.
- **Doppler by construction**: tap delays glide on motion; path identity
  changes crossfade. Room transitions cannot click or chirp.
- Dynamic sources (thrown projectiles: whistle in flight, bounce, explode),
  a world-anchored night-city ambience bed, HUD meters + spectrogram.

## Hear it

Web (what the live demo runs):

```sh
sh tools/build_web.sh          # cargo build wasm (+simd128) + stage into web/
python3 tools/serve.py         # http://localhost:8000/web/
node tools/web_smoke.mjs       # headless pipeline test
node tools/bench_web.mjs       # realtime-factor benchmark (point budgets)
```

Native:

```sh
cargo run --release              # live playback, Ctrl+C to quit
cargo run --release -- --render demo.wav --secs 12   # offline render + level report
cargo run --release -- --input assets/aria48.wav --render out.wav

# scripted walkthrough with fixed sources + 2D schematic video
cargo run --release -- --walkthrough --render walk.wav --json walk.json \
    [--music m.wav --voice v.wav --club c.wav]
python3 tools/render_viz.py walk.json frames 30
ffmpeg -framerate 30 -i frames/%05d.png -i walk.wav \
    -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest walkthrough.mp4
```

Env knobs (native): `OMG_POINT_TAPS=n` point-render budget,
`OMG_MUTE_TAPS` / `OMG_MUTE_OWN` / `OMG_MUTE_REMOTE` isolate signal paths.

Godot 4 (GDExtension, tested against 4.7):

```sh
sh tools/build_godot.sh        # cargo build the extension + stage into godot/bin/
/Applications/Godot.app/Contents/MacOS/Godot --headless --path godot -s test.gd   # smoke test
/Applications/Godot.app/Contents/MacOS/Godot --path godot                          # playable demo
```

`OmgEngine` (RefCounted) is the whole integration surface: `setup(rate)`,
`set_source_samples(i, PackedFloat32Array)`, `set_listener(x, y, yaw)`,
`set_head_yaw(yaw)`, `set_dynamic(slot, x, y, z, active)`, and
`render(frames) -> PackedVector2Array` which you push into an
`AudioStreamGenerator`. The simulation runs on its own 20 Hz thread inside
the extension; `render()` never blocks on it — the same two-clock
architecture as every other frontend. `godot/demo.gd` is the whole demo:
first-person walk (click to capture, WASD + mouse, Esc to release) with
synthesized music and club sources.

## Architecture — the two clocks

The load-bearing decision: the **simulation clock** and the **audio clock**
never share state except through `ParamBlock`.

```
┌─────────────────────────────────┐   ParamBlock   ┌──────────────────────────────────┐
│ SIMULATION  (20 Hz)             │  ───────────▶  │ AUDIO  (per-sample)              │
│ Web Worker / native thread     │                │ AudioWorklet / cpal callback     │
│                                 │  taps: key,    │                                  │
│ omg-core + omg-scene            │  delay, dir,   │ omg-dsp                          │
│  · image sources (ISM ≤3)      │  band gains    │  · fractional delays (Doppler)   │
│  · stochastic tracer → RT60    │                │  · per-tap shelf EQ + wall LP    │
│  · portals, transmission,      │  reverb:       │  · point HRIR convolution        │
│    apertures, diffraction      │  rt60[3],      │  · order-2 ambisonic bus         │
│  · EMA temporal accumulation   │  level[3],     │  · 8-line FDN (traced RT60) ×2   │
│    (kills MC flutter)          │  remote wet    │  · 20-speaker KEMAR decode + AGC │
└─────────────────────────────────┘                └──────────────────────────────────┘
```

- Every parameter crossing the boundary goes through a one-pole smoother —
  simulation updates can never click, and moving delays *are* the Doppler.
- Taps carry stable path-identity keys: same key ⇒ glide (motion), new key
  ⇒ crossfade from silence. Delay is never slid between two different
  paths — that would be a pitch chirp.
- Web transport is flat `ParamBlock`s over `postMessage` (~4 KB @ 20 Hz) —
  no SharedArrayBuffer, no COOP/COEP headers, so it hosts on GitHub Pages.
- Head yaw (mouse / device orientation) is a fast path straight into the
  DSP: SH-bus z-rotation at the decode stage + per-block nearest-HRIR
  re-selection for point taps.
- The late field is an FDN driven by traced per-band RT60, not a convolved
  impulse response — parameters morph artifact-free in real time.

## Rendering: bus + points

Two tiers, chosen per tap by measured loudness:

1. **Point rendering** — the strongest N taps per source (direct sound,
   prominent early reflections) each get their own convolution with the
   nearest of 710 measured KEMAR HRIRs: full localization sharpness where
   localization actually happens. Cost is linear in N.
2. **Ambisonic bus** — everything else (dense reflections, reverb tails,
   the ambience bed) sums onto one shared order-2 bus, decoded once through
   20 virtual speakers × KEMAR HRIRs. Fixed cost regardless of tap count,
   and the physically right tool for *diffuse* content, which has no single
   arrival direction to sharpen. Spatial resolution is bounded by the
   ambisonic order (~±30° blur at order 2), not the speaker count — 20
   speakers already saturates order 2, which is why the path to sharpness
   is tier 1, not more speakers.

N is not hardcoded: the AudioWorklet measures its own render load and walks
the budget between 8 and 32 with hysteresis — a throttled CPU (Low Power
Mode, phones) settles low, a desktop climbs to the cap. The HUD shows the
current value (`hrtf ×N`). Measured on an M1 at the worst-case scene
position (club/entrance doorway blend, ~446 live taps, SIMD kernel):

| point budget / source | realtime factor |
|---|---|
| 0 (bus only) | 3.0× |
| 8 | 2.5× |
| 24 | 2.1× |
| 32 | 1.9× |
| 64 | 1.3× |

Rendering *every* tap this way (up to 160/source × 6 sources) would be
~10× over budget — that, and only that, is why the bus tier exists.
The convolution kernel is written to vectorize (NEON / wasm simd128):
pre-reversed HRIRs turn the ring convolution into contiguous dot products
with explicit 4-lane accumulation.

## Crates

| crate | role | wasm? |
|---|---|---|
| `omg-core` | vec/rng, materials, shoebox raycast, ISM, path tracer, `ParamBlock`. Zero deps. | yes |
| `omg-dsp` | delays, smoothers, EQ, FDN, ambisonics, HRTF (bus + point), renderer, output stage. Alloc-free hot path. | yes |
| `omg-scene` | the world: rooms, doors, windows, materials/thickness, portal graph, transmission/aperture/diffraction routing, `WorldSim` | yes |
| `omg-web` | wasm C-ABI exports (`sim_*` for the Worker, `eng_*` for the Worklet), no wasm-bindgen | is the wasm |
| `omg-app` | native binary: cpal live output, offline render, walkthrough scripting, JSON export for the video tool | native |
| `omg-godot` | GDExtension (godot-rust): the engine as a Godot 4 class, KEMAR HRIRs baked in | native |

## Deliberate approximations (and what full fidelity would need)

Everything here is a *measured* trade against the real-time budget, not a
guess — `tools/bench_web.mjs` is the receipt:

- **3 frequency bands** (<250 Hz, 250–2500, >2500). Wall/air filtering is
  fitted continuously from the bands (one-pole lowpass through the band
  gains), so nothing sounds stepped; production engines use 4–9 bands.
- **Rectangular rooms + wall segments** (height-aware), not arbitrary
  meshes. Arbitrary geometry needs triangle BVH + path-validation rays —
  planned as a wgpu compute port of the tracer.
- **Diffraction is knife-edge Kurze–Anderson** (Fresnel-number insertion
  loss per edge, "rubber band" multi-edge construction) over corner and
  roof edges — the dominant behavior, but not full UTD wedge coefficients
  (interior wedge angle, reflection-boundary terms), and outdoors only;
  indoor doorframe diffraction is approximated by the aperture radiators.
- **Order-2 bus for the diffuse tier** — see the rendering section; the
  sharp tier bypasses it entirely.
- **Nearest-HRIR selection** (with 10 ms crossfades) rather than
  interpolation; the 710-point grid keeps neighbor error small.
- One FDN per source room pair (own + coupled), not a full acoustic
  radiance transfer.

## License & attribution

Code: Apache-2.0. Acoustic data and demo media:

- MIT KEMAR HRTF measurements (Gardner & Martin, MIT Media Lab) — free with attribution.
- Kimiko Ishizaka, *Open Goldberg Variations* — CC0.
- LibriVox *Alice's Adventures in Wonderland* — public domain.
- Night-city ambience: ["Ambient night city, far away party, walla, crowd,
  insects, air"](https://freesound.org/people/ValentinPetiteau/sounds/649075/)
  by Valentin Petiteau — CC0.
- Club track, projectile FX: synthesized by this repo's tools (CC0).

`assets/*.wav` are gitignored; regenerate via `tools/` or use the shipped
`.ogg` versions (what the web demo loads).
