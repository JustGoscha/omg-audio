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
  doorways (BFS over the door graph), each crossing priced by knife-edge
  diffraction at the jamb the path actually bends around — free on the
  sight line, bass-only around the corner; equal-power blended near
  thresholds. Coupled rooms carry their *own*
  reverb through the doorway as a directional wet emitter — you hear the
  hall being a hall from the corridor.
- **Wall transmission**: straight rays cross wall segments with mass-law
  attenuation (amplitude ∝ reference/thickness) and a fitted one-pole
  lowpass — European brick and concrete, not drywall. The club reads as
  *oomp-oomp* from the street and opens up as you walk in.
- **Aperture radiation**: doors and windows re-radiate what's inside —
  including order-2 reflections from the source room — so a window is
  audible off-axis, not just on the sight line.
- **Diffraction as an occlusion floor**: losing sight of a source (or of
  the window/door it radiates from) never cuts it — for every exit
  emitter the simulation finds the best bent path around the blocking
  geometry (single and double corners, over the roof) and occlusion
  floors at its Kurze–Anderson knife-edge loss. The Fresnel number of
  each path's detour decides, per band, how much survives: bass wraps
  around buildings, treble shadows. Dry sound, early reflections and
  coupled reverb all hand off to the bend together — a regression test
  literally walks through a building's shadow and asserts no level jump.
  Plus **facade reflections** outdoors.
- **Binaural output**: order-2 ambisonics decoded through 20 virtual
  speakers (dodecahedron) × measured MIT KEMAR HRIRs, plus *point
  rendering* — the strongest N paths per source get their own nearest-HRIR
  convolution from a 710-direction grid, N adapting to measured CPU load.
- **Ear adaptation**: an AGC protects against ultra-loud content (club
  PA, explosions) with fast clamp and ~30 s recovery — it never boosts
  quiet. On top of it, **hearing fatigue** (temporary threshold shift):
  seconds of demand ~10 dB over the target build up a muffle — a lowpass
  sweeping 18 kHz → 1.4 kHz with exposure depth — that releases over
  ~25 s, the dulled-ears feeling after stepping away from the speakers.
  The HUD shows `muffled N%` while it holds.
- **Doppler by construction**: tap delays glide on motion; path identity
  changes crossfade. Room transitions cannot click or chirp.
- **The ambient dome**: ambience and rain are not listener-glued beds —
  they are the *outdoor field*, an audio skybox sampled by rays. Every
  tick a deterministic fan of 512 rays traces from the listener against a
  real mesh of the scene (walls with true aperture holes, slabs, roofs,
  transmissive glass panes, swinging door leaves); rays that escape to
  the sky deliver the dome's sound from their departure direction,
  attenuated by whatever they bounced off or passed through. Standing
  outside, the whole dome arrives; indoors, the openings localize the
  world outside; a room with no direct opening receives it through the
  rooms that do — after threading two doorways, emergently. What rays
  cannot carry — the through-shell seep — comes from a Sabine power
  balance over the room-coupling graph (shared walls, slabs, roofs,
  exterior surfaces, live apertures, all derived from the scene). No
  per-room constants anywhere.
- **Doors are moving geometry**: a leaf's openness (0…1, the animated
  swing) prices every filter as the area-weighted energy mix of the open
  slit and the wood panel — opening a door *sweeps* the sound, it never
  switches it. `tools/env_probe.mjs` renders a walk through a doorway and
  a full door swing and asserts the level trajectory has no step.
- **Rain lands on things**: drops are modal impacts (glass/metal/stone
  mode tables) anchored to the surfaces that collect them — ticks on the
  actual window panes of your room, knocks and drumming from the roof
  only where the sky is really overhead (a storey above you silences it),
  splash pings on the ground outdoors — while the fused downpour noise
  pours in through the aperture inlets and seeps through the shell,
  band-shaped by the same pricing. Structure-borne impacts arrive duller
  and quieter than open-air splashes (a lowpassed impactor at a fraction
  of the level: the slab transmits a knock, not the impact).
- **Import loudness normalization**: every clip entering the engine is
  gated-RMS normalized to one reference — how a sound was recorded stops
  mattering; its mixer type (a real SPL scale, needle-drop → jet-engine)
  decides the energy it emits.
- Dynamic sources (thrown projectiles: whistle in flight, bounce, explode),
  a night-nature ambience, openable doors, simulated rain, a mixer panel
  with per-channel meters, HUD meters + spectrogram. The world: a square
  with a club, an old two-storey house, a great hall with a piano room, a
  stone cathedral far to the north (a flute inside it, its nave's reverb
  audible through the portal), and an underground bunker with a little
  radio — earth is modeled, so the bunker is sealed except its stair
  shaft.
- **Field-debug panel** (the `debug` button): what is actually playing
  right now, and why — per-source level, live/point tap slots, own- and
  remote-reverb sends, distance, ambience inlet occupancy, plus ~45 s
  scrolling graphs of per-source levels, total tap count, sim tick time
  and render load. Slot leaks, level pile-ups and perf spikes are
  measurable on sight instead of reconstructed from ear-memory.

## Demos

[![walkthrough preview](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-preview.gif)](https://github.com/JustGoscha/omg-audio/releases/tag/walkthroughs)

Every engine version re-renders the **same scripted 98 s walk** (living
room → corridor → great hall → outside → entrance → club) as a top-down
schematic video with binaural audio — the whole history is on the
[walkthroughs release](https://github.com/JustGoscha/omg-audio/releases/tag/walkthroughs),
headphones on:

| version | what it demonstrates |
| --- | --- |
| [v12 – huygens-doorways](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v12-huygens-doorways.mp4) | apertures re-radiate as Huygens sources; doorway wet pays Lambert obliquity — a real on-axis peak walking past an open door |
| [v11 – mesh-dome-cathedral](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v11-mesh-dome-cathedral.mp4) | ambient dome, mesh-emergent knife-edge occlusion, cathedral + bunker |
| [v10 – club-transmission](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v10-club-transmission.mp4) | mass-law wall transmission — *oomp-oomp* from the street |
| [v8 – point-hrtf](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v8-point-hrtf.mp4) | per-path nearest-HRIR convolution for the strongest paths |
| [v7 – hrtf-binaural](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v7-hrtf-binaural.mp4) | measured KEMAR HRIR binaural decode |
| [v6](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v6-audible-coupled-reverb.mp4) / [v5 – coupled reverb](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-v5-coupled-reverb.mp4) | coupled rooms carry their own reverb through doorways |
| [two-sources](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-two-sources.mp4) / [voice](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-voice.mp4) / [music](https://github.com/JustGoscha/omg-audio/releases/download/walkthroughs/walkthrough-music.mp4) | the earliest renders |

New versions join with
`gh release upload walkthroughs walkthrough-vN-<name>.mp4` after the
render recipe below.

## Hear it

Web (what the live demo runs):

```sh
sh tools/build_web.sh          # cargo build wasm (+simd128) + stage into web/
python3 tools/serve.py         # http://localhost:8000/web/
node tools/web_smoke.mjs       # headless pipeline test
node tools/env_probe.mjs       # ambience/rain continuity through doors + swings
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
`set_head(yaw, pitch, roll)`, `set_dynamic(slot, x, y, z, active)`, and
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
- Head orientation (mouse look, device orientation, camera face tracking)
  is a fast path straight into the DSP: a full yaw/pitch/roll SH rotation
  (exact block-diagonal 3×3 + 5×5 on the order-2 bus) at the decode stage
  + per-block nearest-HRIR re-selection for point taps.
- The 🎥 button in the demo enables **camera face tracking** (MediaPipe
  FaceLandmarker, fetched on demand): your real head movements — small
  turns, tilts, even leaning — drive the engine's head while the view
  stays put. Turn your head and hear the world hold still. `recenter`
  re-zeros on your current sitting pose.
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
| `omg-core` | vec/rng, materials, shoebox + triangle-mesh geometry (binned-SAH BVH, budgeted diffraction-edge extraction), ISM, path tracer, `ParamBlock`. Zero deps. | yes |
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
- **The demo scene authors rectangles**, but the core is no longer
  limited to them: `omg_core::mesh` is an arbitrary triangle mesh with a
  binned-SAH BVH — the same stochastic tracer runs over it (unit-tested
  to match the analytic shoebox on a box), segments query per-surface
  transmission crossings, and diffraction edges are auto-extracted with
  an explicit importance budget (dense meshes degrade by dropping short/
  shallow creases first, never by breaking). Measured on M1
  (`mesh_bench`, 2 520-tri city block): BVH build 2 ms, 2 M rays/s,
  full trace 79 ms at the 4096-ray desktop tier / 19 ms at the 1024-ray
  mobile tier — variance, not bias, is what degrades on cheaper tiers
  (EMA accumulation smooths it). On top of it, `omg_core::paths` finds
  propagation paths on a raw mesh with ZERO authoring — no rooms, no
  doors, no portals: wall thickness is emergent (entry/exit crossings
  paired for the mass law), and door jambs, building corners and roof
  lines are all just auto-extracted edges searched for single- and
  double-bend paths (166 µs per source-listener query at the default
  budget). The demo's OCCLUSION now runs on this: the old hand-built
  corner list, corner-visibility matrix and multi-roof rubber band are
  deleted — occlusion floors come from AutoPaths over the same world
  mesh the ambient dome traces (bound-ranked single bends, branch-and-
  bound double bends, an over-the-silhouette hull path for any number
  of buildings, bent energy averaged over each aperture's real extent,
  and per-cell caching with a fixed refresh budget). In-room acoustics
  remain per-room shoeboxes (which these rooms are); the directional
  late field and the wgpu compute port are the remaining milestone.
- **Diffraction is knife-edge Kurze–Anderson** (Fresnel-number insertion
  loss per edge, "rubber band" multi-edge construction) over corner, roof
  and door-jamb edges — the dominant behavior, but not full UTD wedge
  coefficients (interior wedge angle, reflection-boundary terms). And as
  an occlusion *floor* it preserves level and spectrum continuity, not
  arrival direction: deep in a shadow, bent energy still arrives from its
  emitter's direction rather than visibly from the corner or roof line
  (near the boundary, where localization is sharpest, the two coincide).
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
- Night ambience: ["nature_ambient_night_5-min"](https://freesound.org/people/edmondsr/sounds/711207/)
  by edmondsr — CC0.
- Cathedral flute: ["Natural Native Flute with Lush Reverb"](https://freesound.org/people/CVLTIV8R/sounds/816095/)
  by CVLTIV8R, performed by RJ Roush — CC0.
- Bunker radio: ["An old radio playing in a Sri Lankan sewing factory"](https://freesound.org/people/florianreichelt/sounds/423446/)
  by florianreichelt — CC0.
- Car motors: ["car_idle_loop"](https://freesound.org/people/AndrewAlexander/sounds/369053/)
  by AndrewAlexander, ["Bus Motor/Engine Sound Loop"](https://freesound.org/people/qubodup/sounds/54909/)
  by qubodup, ["Diesel Generator 03"](https://freesound.org/people/KVV_Audio/sounds/748274/)
  by KVV_Audio, ["Motor Loop"](https://freesound.org/people/soundjoao/sounds/325808/)
  by soundjoao — all CC0, loopified via `tools/make_motors.py`.
- Club track, projectile FX: synthesized by this repo's tools (CC0).
- Rain splat bank: 22 hits sliced from CC0 freesound recordings
  (sources listed in `tools/make_drops.py`).

`assets/*.wav` are gitignored; regenerate via `tools/` or use the shipped
`.ogg` versions (what the web demo loads).
