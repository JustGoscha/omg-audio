//! Shared helpers for the isolated `fx_*` effect tests: synthetic scene
//! builders (rooms, doors, sources) that never touch the demo world, and
//! small tap/assert utilities. Each `fx_*.rs` binary includes this via
//! `mod common;` and uses a subset.
#![allow(dead_code)]

use omg_core::material::Material;
use omg_core::params::Tap;
use omg_core::NBANDS;
use omg_scene::walkthrough::{Door, Roof, RoomDef, SourceDef};

/// An enclosed box room at ground level (floor_z 0), uniform walls.
pub fn room(min: (f32, f32), max: (f32, f32), height: f32, mat: Material) -> RoomDef {
    RoomDef {
        name: "room",
        min,
        max,
        height,
        barrier_height: height + 0.3,
        floor_z: 0.0,
        walls: [mat; 6],
        wall_thickness: 0.2,
        outdoor: false,
        roof: Roof::Flat,
    }
}

/// A room stacked on a storey below (`floor_z` above ground).
pub fn storey(min: (f32, f32), max: (f32, f32), height: f32, floor_z: f32, mat: Material) -> RoomDef {
    RoomDef { floor_z, barrier_height: floor_z + height + 0.3, ..room(min, max, height, mat) }
}

/// The open air around a scene. `ground` becomes walls[4], which the
/// dome mesh uses as the ground plane's material.
pub fn outdoor(min: (f32, f32), max: (f32, f32), ground: Material) -> RoomDef {
    RoomDef {
        name: "outside",
        min,
        max,
        height: 30.0,
        barrier_height: 30.0,
        floor_z: 0.0,
        walls: [ground; 6],
        wall_thickness: 0.15,
        outdoor: true,
        roof: Roof::Flat,
    }
}

/// A fully open, light (wood) doorway 1.1 m × 2 m centered at z = 1.
pub fn door(rooms: (usize, usize), pos: (f32, f32), axis: usize) -> Door {
    Door {
        rooms,
        pos,
        axis,
        half: 0.55,
        height: 2.0,
        zc: 1.0,
        glass: false,
        openness: 1.0,
        heavy: false,
    }
}

/// A window pane (not a routing edge, a radiator with glass loss).
pub fn pane(rooms: (usize, usize), pos: (f32, f32), axis: usize, heavy: bool) -> Door {
    Door { glass: true, heavy, half: 1.0, height: 1.4, zc: 1.5, ..door(rooms, pos, axis) }
}

pub fn source(room: usize, pos: (f32, f32)) -> SourceDef {
    SourceDef { name: "src", pos, room, gain: 1.0, z: 1.5, emitters: &[] }
}

pub fn tap(taps: &[Tap], key: u32) -> Option<&Tap> {
    taps.iter().find(|t| t.key == key)
}

/// Earliest tap (the direct path in a box).
pub fn direct(taps: &[Tap]) -> &Tap {
    taps.iter().min_by(|a, b| a.delay_s.total_cmp(&b.delay_s)).expect("at least one tap")
}

pub fn power(taps: &[Tap], band: usize) -> f32 {
    taps.iter().map(|t| t.gains[band] * t.gains[band]).sum()
}

pub fn rel_close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-12)
}

#[track_caller]
pub fn assert_rel(a: f32, b: f32, tol: f32, what: &str) {
    assert!(rel_close(a, b, tol), "{what}: {a} vs {b} (rel tol {tol})");
}

#[track_caller]
pub fn assert_bands_rel(a: [f32; NBANDS], b: [f32; NBANDS], tol: f32, what: &str) {
    for k in 0..NBANDS {
        assert!(rel_close(a[k], b[k], tol), "{what} band {k}: {a:?} vs {b:?}");
    }
}
