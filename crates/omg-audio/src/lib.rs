//! # omg-audio
//!
//! Geometric game audio. Give it your level's triangles with materials,
//! your sound sources and the listener; it traces how sound actually gets
//! from each source to the listener — through doorways, around corners,
//! muffled through walls, reverberating in the room you stand in and the
//! one next door — and renders it binaurally.
//!
//! Three pieces, split by clock:
//!
//! - [`Geometry`]: the static level — triangles + [`Material`]s
//!   (presets like `Material::BRICK`, or your own), optional ground plane.
//! - [`Simulation`] (non-real-time): geometry, moving boxes (doors,
//!   windows, props), sources, listener. `tick()` 10–30 times a second
//!   → a [`Frame`].
//! - [`Mixer`] (real-time): plays clips on sources, `apply(&frame)`,
//!   `render(l, r)` in your audio callback.
//!
//! They only talk through `Frame`s, which serialize to flat f32 arrays
//! (`Frame::write_flat` / `read_flat`): the simulation can live in a
//! worker and the mixer in an AudioWorklet. On native targets,
//! [`threaded::Engine`] wires both up with a simulation thread and
//! lock-free handoffs.
//!
//! Coordinates: meters, right-handed, +z up ([`from_y_up`] converts).
//! See README.md for a quick start.

pub mod env;
pub mod frame;
pub mod geometry;
pub mod governor;
pub mod mixer;
pub mod sim;
// threads: native, and wasm under Emscripten (pthreads as Web Workers);
// not bare wasm32-unknown-unknown
#[cfg(all(feature = "threaded", any(not(target_arch = "wasm32"), target_os = "emscripten")))]
pub mod threaded;

pub use frame::{Frame, Listener, SkyCell, SourceId};
pub use geometry::{from_y_up, to_y_up, Geometry, MaterialId};
pub use mixer::{clip, normalize, Clip, Hrir, Meter, Mixer, Play, VoiceId, HRIR_GRID, HRIR_SPEAKERS};
pub use omg_core::material::Material;
pub use omg_core::params::{ParamBlock, Tap};
pub use omg_core::v3::walker::V3Walker;
pub use omg_core::v3::{tap_kind, Budget, DenseResponse};
pub use sim::{Compute, Config, DebugPath, ObjectId, PathKind, Quality, Simulation, SourceDesc};

/// Speed of sound the engine uses (m/s).
pub const SPEED_OF_SOUND: f32 = omg_core::SPEED_OF_SOUND;
