//! omg-godot: GDExtension exposing the omg-audio spatial engine to Godot 4.
//!
//! `OmgEngine` is a RefCounted you drive from GDScript: set source sample
//! loops, move the listener, pull rendered binaural stereo and push it into
//! an `AudioStreamGenerator`. The demo project in `godot/` shows the whole
//! loop in ~40 lines of GDScript. Simulation runs on its own thread at
//! 20 Hz; `render()` never blocks on it.

use godot::classes::RefCounted;
use godot::prelude::*;

mod engine;
use engine::SpatialEngine;

struct OmgAudioExtension;

#[gdextension]
unsafe impl ExtensionLibrary for OmgAudioExtension {}

#[derive(GodotClass)]
#[class(base=RefCounted, init)]
struct OmgEngine {
    engine: Option<SpatialEngine>,
    scratch: Vec<(f32, f32)>,
    base: Base<RefCounted>,
}

#[godot_api]
impl OmgEngine {
    /// Create the engine. Call once, with the mix rate you will also give
    /// your AudioStreamGenerator (48000 recommended).
    #[func]
    fn setup(&mut self, sample_rate: f32) {
        self.engine = Some(SpatialEngine::new(sample_rate));
    }

    /// Number of source slots (3 static scene sources + 3 dynamic).
    #[func]
    fn source_count(&self) -> i64 {
        engine::NSRC as i64
    }

    /// Set the looping mono sample buffer for source `i` (engine rate).
    #[func]
    fn set_source_samples(&mut self, i: i64, samples: PackedFloat32Array) {
        if let Some(e) = &mut self.engine {
            e.set_source_samples(i as usize, samples.to_vec());
        }
    }

    /// Listener position (world meters, scene plan) and walk yaw.
    #[func]
    fn set_listener(&mut self, x: f32, y: f32, yaw: f32) {
        if let Some(e) = &self.engine {
            e.set_listener(x, y, yaw);
        }
    }

    /// Fast head/camera orientation — applied at the DSP immediately.
    /// Yaw positive turns left, pitch positive looks up, roll positive
    /// tilts right (right ear down).
    #[func]
    fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        if let Some(e) = &mut self.engine {
            e.set_head(yaw, pitch, roll);
        }
    }

    /// Move a dynamic source (slots 0..3): position + height + active.
    #[func]
    fn set_dynamic(&mut self, slot: i64, x: f32, y: f32, z: f32, active: bool) {
        if let Some(e) = &self.engine {
            e.set_dynamic(slot as usize, x, y, z, active);
        }
    }

    /// Per-source point-HRTF budget (see the repo README).
    #[func]
    fn set_point_budget(&mut self, n: i64) {
        if let Some(e) = &mut self.engine {
            e.set_point_budget(n.max(0) as usize);
        }
    }

    /// Current ear-adaptation gain (for HUD meters).
    #[func]
    fn agc_gain(&self) -> f32 {
        self.engine.as_ref().map_or(1.0, |e| e.agc_gain())
    }

    /// Room index the listener is in (per the demo scene's room list).
    #[func]
    fn listener_room(&self) -> i64 {
        self.engine.as_ref().map_or(0, |e| e.room as i64)
    }

    /// Render `frames` of binaural stereo — push straight into an
    /// `AudioStreamGeneratorPlayback` with `push_buffer()`.
    #[func]
    fn render(&mut self, frames: i64) -> PackedVector2Array {
        let Some(e) = &mut self.engine else {
            return PackedVector2Array::new();
        };
        e.render(frames.max(0) as usize, &mut self.scratch);
        self.scratch
            .iter()
            .map(|(l, r)| Vector2::new(*l, *r))
            .collect()
    }
}
