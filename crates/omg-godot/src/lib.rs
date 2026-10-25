//! omg-godot: the omg-audio spatial engine as Godot 4 nodes (GDExtension).
//!
//! Drop an `OmgWorld` into a level, tag geometry with `OmgGeometry`
//! (or `omg_material` metadata), add `OmgSource`s and an `OmgListener`;
//! the world collects the level's triangles, traces them with engine v3
//! on its own thread and renders binaural audio on another. See
//! `nodes.rs` for the node API and `backend/` for the engine side (a
//! stand-in for the planned `omg-audio` engine crate).

use godot::prelude::*;

pub mod backend;
mod nodes;

struct OmgAudioExtension;

#[gdextension]
unsafe impl ExtensionLibrary for OmgAudioExtension {}
