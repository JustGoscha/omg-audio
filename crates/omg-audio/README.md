# omg-audio

Geometric game audio. Give it your level's triangles and materials, your
sound sources and the listener: it traces how sound actually gets from
each source to your ears — through doorways, around corners, muffled
through walls, ringing in the room you are in and the one next door —
and renders it binaurally.

## Quick start

```rust
use omg_audio::{clip, Config, Geometry, Hrir, Material, Mixer, Play, Simulation, SourceDesc};

// 1. Your level: triangles + materials (meters, +z up)
let mut level = Geometry::new();
let brick = level.material(Material::BRICK);
let floor = level.material(Material::WOOD);
let grass = level.material(Material::GRASS);
level.add_mesh(&wall_positions, &wall_indices, brick); // from your level data
level.add_box([0.0, 0.0, -0.2], [8.0, 6.0, 0.0], floor);
level.ground_plane(-0.2, grass); // optional: open ground to the horizon

// 2. The simulation (any thread, 10–30 Hz)
let mut sim = Simulation::new(Config::default());
sim.set_geometry(level);
let radio = sim.add_source(SourceDesc::at([6.0, 4.0, 1.2]));
let door = sim.add_object([3.5, -0.1, 0.0], [4.5, 0.0, 2.2], Material::DOOR); // move/disable to open

// 3. The mixer (audio thread)
let mut mixer = Mixer::new(48_000.0, Hrir::load_dir("assets"));
mixer.play(radio, &clip(&radio_samples), Play::LOOP); // mono f32 at 48 kHz

// 4. Every game tick: move things, trace, hand the frame over
sim.set_listener(camera_pos, yaw, pitch, roll);
sim.set_source(radio, radio_pos, true);
mixer.apply(&sim.tick());

// 5. In your audio callback
mixer.render(&mut left, &mut right);
```

Prefer not to manage threads? `threaded::Engine::new(48_000.0, EngineConfig::default())`
returns an `Engine` (game thread: geometry, sources, `play`) and an
`AudioOut` (audio callback: `render`); a simulation thread and lock-free
handoffs sit in between. `cargo run --release -p omg-audio --example box_room`
renders a walk into a room to a WAV.

## The pieces

- **Geometry** — static triangles with a material each (`add_mesh`,
  `add_mesh_per_tri`, `add_box`, `ground_plane`). A simplified collision
  mesh is ideal; acoustics needs walls, floors and roofs, not door
  handles. Wall thickness comes from the geometry (front + back face);
  for single-plane walls set it per material: `set_thickness(id, 0.24)`.
- **Materials** — presets `STONE BRICK CONCRETE WOOD GLASS HEAVY_GLASS
  DOOR HEAVY_DOOR CARPET CURTAIN DRYWALL METAL GRASS EARTH`, or your own:
  `Material::simple(absorption, scattering)`, `Material::new(absorption[3],
  scattering, transmission[3])`. Bands: < 250 Hz, 250–2500 Hz, > 2500 Hz.
- **Objects** — boxes that move or change: doors, windows, crates,
  vehicles. Openness is geometry: shrink, move or disable the box.
- **Simulation** — `tick()` → `Frame`. `Quality::{Low, Medium, High,
  Ultra, Custom}`; `Compute::Gpu` (feature `gpu`) walks the rays on the
  GPU through wgpu with 8× the rays, falling back to the CPU without an
  adapter. With many sources, at most `max_traced_sources` (64) are traced
  per tick — loud, near and moving ones first.
- **Mixer** — real-time, allocation-free `render`. Mix-wide budgets
  (`set_budget(taps, points, voices)`) bound the cost whatever the source
  count: the loudest paths render, the loudest voices play, silent and
  muted sources cost nothing. `play_near` whispers straight into one ear.
- **Frame** — the only link between the two: `write_flat` / `read_flat`
  turn it into a flat `f32` array for a worker → AudioWorklet hop.

Coordinates: meters, right-handed, **+z up**; `yaw` turns left from +x.
Engines with y up (three.js, Godot) convert points with `from_y_up`.

## Performance (Apple M1, one core, 48 kHz, `examples/bench_sources.rs`)

25-building town (2.4k triangles + ground), every source playing at once,
listener walking:

| sources | sim tick, CPU (High) | sim tick, GPU async | mixer, default budget |
|--------:|---------------------:|--------------------:|----------------------:|
| 64      | ~16 ms               | ~3 ms               | ~1.3× real time       |
| 256     | ~19 ms               | ~6 ms               | ~1.3× real time       |

The mixer's cost does not grow with the source count; the budget sets it.

## On the web (planned JS API)

The split above is the web architecture: `Simulation` in a Worker,
`Mixer` in an AudioWorklet, `Frame`s as flat `Float32Array`s between
them. A thin npm package will wrap it, roughly:

```js
const omg = await OmgAudio.create({ audioContext, workerUrl });
omg.world.addMesh(positions, indices, 'stone');      // Float32Array, Uint32Array
const radio = omg.addSource({ position: [6, 1.2, -4] });
radio.play(audioBuffer, { loop: true });
omg.setListener(camera);                              // a three.js camera, y up
// three.js helper: walk a scene, read userData.acoustic = 'stone' | 'glass' | …
OmgAudio.three.addScene(omg.world, scene);
```
