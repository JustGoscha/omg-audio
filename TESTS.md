# TESTS — every effect, proven on its own

Each audio effect has an **isolated** test: a minimal synthetic setup
(a shoebox, two mesh boxes, a hand-built `ParamBlock`, a synthetic HRIR
set) that exercises that one effect through its public API and asserts
the physical claim quantitatively. Isolated tests never use the demo
world (`WorldSim`, `walkthrough::rooms()/doors()/sources()`, demo
coordinates). The demo scene is where the effects come together; its
tests (`crates/omg-scene/tests/*.rs` without the `fx_` prefix, the
`world.rs` unit tests, `tools/web_smoke.mjs`) check the combination.

Convention: one file per effect, `crates/<crate>/tests/fx_<effect>.rs`,
opening with a `//!` block that states the effect, the claim it
guarantees, and which scene tests exercise it combined. Shared helpers
live in each crate's `tests/common/mod.rs`. A test that exposes a real
bug stays in the suite asserting the CORRECT behaviour, marked
`#[ignore = "BUG: …"]`, until the bug is fixed.

## Running

```sh
export PATH="$HOME/.cargo/bin:$PATH"   # rustup, not Homebrew's rustc
cargo test --workspace --exclude omg-gpu --exclude omg-godot --release
cargo test -p omg-dsp --release --test fx_doppler       # one effect
cargo test --workspace --exclude omg-gpu --exclude omg-godot --release -- --ignored   # known bugs
```

omg-godot needs rustc ≥ 1.94. omg-gpu tests skip themselves without a
wgpu adapter.

## Effect → isolated test

### Propagation (omg-core)

| Effect | Test file |
|---|---|
| Distance spreading 1/r + direct delay | `crates/omg-core/tests/fx_distance_spreading.rs` |
| Air absorption | `crates/omg-core/tests/fx_air_absorption.rs` |
| Image-source early reflections | `crates/omg-core/src/lib.rs` (unit) |
| Late-field RT60 accuracy | `crates/omg-core/tests/fx_late_rt60.rs`, `src/lib.rs` (unit) |
| Late-field direction / anisotropy | `crates/omg-core/src/tracer.rs` (unit) |
| Mesh BVH tracing | `crates/omg-core/src/mesh.rs` (unit) |
| Traced early reflections (PT) | `crates/omg-core/tests/pt_golden.rs`, `pt_mesh_gate.rs` (scene crate) |
| Knife-edge diffraction kernel | `crates/omg-core/src/diffraction.rs` (unit) |
| Bent paths around buildings | `crates/omg-core/src/paths.rs` (unit) |
| Wall transmission (late field, mass law) | `crates/omg-core/tests/fx_wall_transmission.rs` |
| Wall transmission (early paths) | `crates/omg-core/tests/fx_early_wall_transmission.rs` |
| Furniture occlusion | `crates/omg-core/tests/pt_occluder.rs` |
| Furniture seep (soft bodies) | `crates/omg-core/tests/fx_furniture_seep.rs` |
| Furniture absorption (late) | `crates/omg-core/tests/fx_late_furniture.rs` |
| Door leaf in the late field | `crates/omg-core/tests/fx_door_leaf_late.rs` |
| Sim → renderer transport | `crates/omg-core/tests/fx_param_flat_transport.rs` |

### Scene simulation (omg-scene)

| Effect | Test file |
|---|---|
| Ground reflection | `crates/omg-scene/tests/fx_ground_reflection.rs` |
| Facade slap-back | `crates/omg-scene/tests/fx_facade_slapback.rs` |
| Speaker rig (club) | `crates/omg-scene/tests/fx_speaker_rig.rs` |
| Aperture re-radiation + Lambert | `crates/omg-scene/tests/fx_aperture_reradiation.rs` |
| Aperture hop, door openness | `crates/omg-scene/src/walkthrough.rs` (unit) |
| Moving-source wet compensation | `crates/omg-scene/tests/fx_moving_wet.rs` |
| Coupled / remote reverb (sim) | `crates/omg-scene/tests/fx_coupled_reverb.rs` |
| Glass panes, heavy doors | `crates/omg-scene/tests/fx_glass_heavy_doors.rs` |
| Ambient dome | `crates/omg-scene/tests/fx_ambient_dome.rs` |
| Shaped ambience (ground-area / volume emitters: surround at ear height, from below and fading when flying, toward the area from afar, roof and doorway pricing) | `crates/omg-scene/tests/fx_ambient_shapes.rs`, `src/ambient.rs` (unit); demo crickets: `tests/ambient_crickets.rs` |
| Outdoor-field power balance | `crates/omg-scene/tests/fx_outdoor_field.rs` |
| Door-graph routing | `crates/omg-scene/tests/fx_door_routing.rs` |
| Room blending at thresholds | `crates/omg-scene/tests/fx_room_blend.rs` |
| Quality tiers | `crates/omg-scene/tests/fx_quality_tiers.rs` |

### Engine v3 tracer (omg-core)

| Effect | Test file |
|---|---|
| v3 tracer: direct, walls, NEE, late field, diffraction; crossing a wall drops the room left behind | `crates/omg-core/tests/fx_v3_tracer.rs` |
| GPU walk results from before a wall crossing are discarded | `omg-core v3::fold::tests::a_walk_from_before_crossing_a_wall_is_discarded` |

### DSP (omg-dsp)

| Effect | Test file |
|---|---|
| Doppler (delay glide), also at 10 km and for fast flyovers | `crates/omg-dsp/tests/fx_doppler.rs` |
| Play log: recorded sources read at emission time, no distance limit | `crates/omg-dsp/src/playlog.rs` (unit tests), `omg-web tests::a_bell_10_km_away_arrives_29_s_late_reverb_included` |
| Path-identity crossfade | `crates/omg-dsp/tests/fx_path_crossfade.rs` |
| Per-tap band EQ, through-wall lowpass | `crates/omg-dsp/tests/fx_tap_eq.rs` |
| Own-room FDN reverb | `crates/omg-dsp/tests/fx_fdn_rt60.rs` |
| Late level at the FDN output: leaving a room takes its tail along, a silent source still rings out | `crates/omg-dsp/tests/fx_late_level.rs` |
| Remote (coupled) reverb render | `crates/omg-dsp/tests/fx_remote_reverb.rs` |
| FDN sleep when idle | `crates/omg-dsp/tests/fx_fdn_sleep.rs` |
| Tap ceiling, point budget | `crates/omg-dsp/tests/fx_tap_budget.rs` |
| HRTF bus decode | `crates/omg-dsp/tests/fx_hrtf_bus_decode.rs` |
| Point HRIR convolution | `crates/omg-dsp/tests/fx_point_conv.rs` |
| Cardioid stereo fallback | `crates/omg-dsp/tests/fx_stereo_fallback.rs` |
| Soft limiter | `crates/omg-dsp/tests/fx_soft_limiter.rs` |
| Head-pose smoothing | `crates/omg-dsp/tests/fx_head_smoothing.rs` |
| Delay line, smoother | `crates/omg-dsp/tests/fx_delay_smooth.rs` |
| Ambisonic encode + head rotation | `crates/omg-dsp/src/ambi.rs` (unit) |
| Ear adaptation (AGC), fatigue, tinnitus | `crates/omg-dsp/src/output.rs` (unit) |
| Near-field whisper ear stage | `crates/omg-dsp/src/near.rs` (unit) |
| Rain | `crates/omg-dsp/tests/fx_rain.rs`, `src/rain.rs` (unit) |
| Ambience | `crates/omg-dsp/src/ambience.rs` (unit) |
| Environment routing slots | `crates/omg-dsp/src/env.rs` (unit) |
| Band splitter | `crates/omg-dsp/src/bands.rs` (unit) |
| Loudness normalisation | `crates/omg-dsp/src/level.rs` (unit) |

### Web engine (omg-web, `src/lib.rs` tests)

| Effect | Test |
|---|---|
| FX one-shot voices | `fx_voices_mix_into_their_source_stop_end_and_cap` |
| Mixer faders, mute skip | `mixer_faders_ramp_clamp_and_mute_skips` |
| Master gain | `master_glides_and_clamps` |
| Final hard clamp | `final_hard_clamp_after_master` |
| Level meters | `meters_report_peak_and_rms_of_a_known_signal` |
| Whisper post-AGC, voice cap | `whisper_is_post_agc`, `whisper_voice_cap_is_four` |
| Dynamic slots reach the sim (footsteps) | `every_dynamic_slot_reaches_the_sim` |

## Not isolated yet

- **WorldSim-only:** environment assembly, bunker grade separation, LOD
  for quiet sources, bend floor and cache, dynamic slots, mute, the
  straight-through tap, remote-wet send assembly, furniture tables.
  Unblocking them needs a `WorldSim::from_scene(rooms, doors, sources,
  facades, furniture)` constructor, with `new()` delegating to it.
- **JS-only (web/main.js):** car spawn/pitch/fade envelope, ball
  physics and thump triggers, footstep cadence and surfaces, wildlife
  scheduling, jump/crouch, face tracking, door animation, the worklet
  load governor. These need the logic moved into a module that Node can
  import without the DOM.
- **Per-material rain drops and FDN sleep state** have no public hook;
  the rain and sleep tests work around it.

## Known bugs (ignored tests)

| Test | Bug |
|---|---|
| `fx_param_flat_transport::large_keys_stay_distinct_and_exact` | Tap keys travel as f32; keys ≥ 2^24 (bent paths, mesh PT) collide |
| `fx_late_rt60::live_room_rt60_is_not_truncated` | Late tracer stops rays after 64 bounces, cutting live-room tails |
| `fx_air_absorption::late_field_high_band_decays_faster_from_air` | Late tracer applies no air absorption |
| `fx_glass_heavy_doors::shut_heavy_door_dry_path_pays_the_steel_panel` | A shut heavy door's dry path is priced as wood |
| `fx_room_blend::crossfade_is_continuous_through_the_door_plane` | Room blend weights jump at the door plane |
| `fx_point_conv::retarget_during_a_crossfade_stays_continuous` | HRIR reselection mid-crossfade clicks |
| `fx_fdn_rt60::mid_t60_independent_of_hf_damping` | FDN high damping shortens the mid T60 |
| `fx_fdn_rt60::own_late_field_is_diffuse_for_short_rooms` | Own-room reverb leans left/up in short rooms |
| `fx_tap_eq::deep_band_cuts_reach_their_band_gain` | First-order shelves can't reach deep band cuts |
