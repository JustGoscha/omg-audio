//! Shared helpers for the isolated `fx_*` effect tests: synthetic
//! geometry builders and small numeric utilities. Nothing here knows
//! about the demo scene — every test builds its own minimal world.
#![allow(dead_code)]

use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::tracer::Echogram;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

pub fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Amplitude → dB.
pub fn db(a: f32) -> f32 {
    20.0 * a.max(1e-12).log10()
}

/// Energy → dB.
pub fn db_e(e: f32) -> f32 {
    10.0 * e.max(1e-24).log10()
}

/// A material with the same absorption in every band, no transmission:
/// isolates whatever frequency-dependent effect a test is probing.
pub const fn flat_opaque(absorption: f32, scattering: f32) -> Material {
    Material { absorption: [absorption; NBANDS], scattering, transmission: [0.0; NBANDS] }
}

/// Like `flat_opaque` but with a per-band transmission amplitude.
pub const fn flat_transmissive(absorption: f32, scattering: f32, transmission: [f32; NBANDS]) -> Material {
    Material { absorption: [absorption; NBANDS], scattering, transmission }
}

/// Face selector for `shell`: x·min, x·max, y·min, y·max, z·min, z·max.
pub const X_MIN: usize = 0;
pub const X_MAX: usize = 1;
pub const Y_MIN: usize = 2;
pub const Y_MAX: usize = 3;
pub const Z_MIN: usize = 4;
pub const Z_MAX: usize = 5;

/// Push the six faces of an axis-aligned room [min, max] as quads, one
/// surface per face, each with its own material id. `None` skips the
/// face (a shared wall already pushed by the neighbour room).
pub fn shell(mb: &mut MeshBuilder, min: Vec3, max: Vec3, faces: [Option<u16>; 6]) {
    let (a, b) = (min, max);
    let quads: [[Vec3; 4]; 6] = [
        [v(a.x, a.y, a.z), v(a.x, b.y, a.z), v(a.x, b.y, b.z), v(a.x, a.y, b.z)],
        [v(b.x, a.y, a.z), v(b.x, a.y, b.z), v(b.x, b.y, b.z), v(b.x, b.y, a.z)],
        [v(a.x, a.y, a.z), v(a.x, a.y, b.z), v(b.x, a.y, b.z), v(b.x, a.y, a.z)],
        [v(a.x, b.y, a.z), v(b.x, b.y, a.z), v(b.x, b.y, b.z), v(a.x, b.y, b.z)],
        [v(a.x, a.y, a.z), v(b.x, a.y, a.z), v(b.x, b.y, a.z), v(a.x, b.y, a.z)],
        [v(a.x, a.y, b.z), v(a.x, b.y, b.z), v(b.x, b.y, b.z), v(b.x, a.y, b.z)],
    ];
    for (q, m) in quads.iter().zip(faces) {
        if let Some(m) = m {
            mb.begin_surface();
            mb.quad(q[0], q[1], q[2], q[3], m);
        }
    }
}

/// Two rooms side by side along x sharing ONE wall at `x = split`:
/// A = [0, split] × [0, w] × [0, h], B = [split, 2·split] × … . The outer
/// shell uses `outer`, the shared wall uses `party`; if `door` is Some
/// ((y0, y1), z1) the party wall has a doorway hole y∈[y0,y1], z∈[0,z1].
pub fn two_rooms(
    split: f32,
    w: f32,
    h: f32,
    outer: Material,
    party: Material,
    door: Option<((f32, f32), f32)>,
) -> Mesh {
    let mut mb = MeshBuilder::new();
    let mo = mb.material(outer);
    let mp = mb.material(party);
    let o = Some(mo);
    // room A without its x·max face, room B without its x·min face
    shell(&mut mb, v(0.0, 0.0, 0.0), v(split, w, h), [o, None, o, o, o, o]);
    shell(&mut mb, v(split, 0.0, 0.0), v(2.0 * split, w, h), [None, o, o, o, o, o]);
    let x = split;
    mb.begin_surface();
    match door {
        None => mb.quad(v(x, 0.0, 0.0), v(x, w, 0.0), v(x, w, h), v(x, 0.0, h), mp),
        Some(((y0, y1), z1)) => {
            mb.quad(v(x, 0.0, 0.0), v(x, y0, 0.0), v(x, y0, h), v(x, 0.0, h), mp);
            mb.quad(v(x, y1, 0.0), v(x, w, 0.0), v(x, w, h), v(x, y1, h), mp);
            mb.quad(v(x, y0, z1), v(x, y1, z1), v(x, y1, h), v(x, y0, h), mp);
        }
    }
    mb.build()
}

/// Total echogram energy per band (all bins).
pub fn band_energy(e: &Echogram) -> [f32; NBANDS] {
    let mut s = [0.0f32; NBANDS];
    for bin in &e.bins {
        for b in 0..NBANDS {
            s[b] += bin[b];
        }
    }
    s
}

/// Sabine RT60 (s) for volume V (m³) and equivalent absorption area A (m²).
pub fn sabine(volume: f32, area_abs: f32) -> f32 {
    0.161 * volume / area_abs
}
