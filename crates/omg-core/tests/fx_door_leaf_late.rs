//! Effect: a door leaf reshaping the LATE-field coupling between two
//! rooms — `scene::WithPanels` overlay boxes over a mesh with a doorway
//! hole, traced by `tracer::trace` (C6d: "closing a door reshapes the
//! late field with zero portal code").
//!
//! Physical claim guaranteed: with an opaque party wall, the neighbour
//! room's late energy arrives ONLY through the doorway: an open door
//! couples the rooms; an opaque closed leaf cuts the coupling to exactly
//! zero; a real (partly transmissive) leaf lets a fraction through that
//! is bass-heavy relative to the open door; the source room gets
//! slightly LIVELIER when the door closes (its energy no longer drains
//! next door).
//!
//! Combined in the demo: `omg_scene::world::WorldSim` door leaves
//! (`dome::door_panels` → `late_panels`) — walkthrough door open/close.

mod common;

use common::{band_energy, db_e, flat_opaque, flat_transmissive, two_rooms, v};
use omg_core::material::Material;
use omg_core::mesh::Mesh;
use omg_core::rng::Rng;
use omg_core::scene::WithPanels;
use omg_core::tracer::{trace, Echogram};
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

const SRC: Vec3 = Vec3::new(2.0, 1.5, 1.5);
const LIS_A: Vec3 = Vec3::new(4.0, 3.5, 1.6);
const LIS_B: Vec3 = Vec3::new(9.5, 3.5, 1.6);
const HOLE: ((f32, f32), f32) = ((2.0, 3.0), 2.1);

fn world() -> Mesh {
    let wall = flat_opaque(0.1, 0.3);
    two_rooms(6.0, 5.0, 3.0, wall, wall, Some(HOLE))
}

fn leaf(m: Material) -> (Vec3, Vec3, Material) {
    (v(5.98, 2.0, 0.0), v(6.02, 3.0, 2.1), m)
}

fn energy(mesh: &Mesh, panels: &[(Vec3, Vec3, Material)], lis: Vec3, seed: u64) -> [f32; NBANDS] {
    let geo = WithPanels { base: mesh, panels };
    let mut rng = Rng::new(seed);
    let mut e = Echogram::new();
    trace(&geo, SRC, lis, 20_000, [1.0; 3], &mut rng, &mut e);
    band_energy(&e)
}

#[test]
fn open_door_couples_closed_opaque_leaf_seals() {
    let m = world();
    let open = energy(&m, &[], LIS_B, 1);
    assert!(open.iter().all(|&x| x > 1e-4), "open door: no energy next door {open:?}");
    let sealed = energy(&m, &[leaf(flat_opaque(0.1, 0.3))], LIS_B, 1);
    assert_eq!(sealed, [0.0; NBANDS], "opaque leaf leaked {sealed:?}");
}

#[test]
fn real_leaf_leaks_a_bass_heavy_fraction() {
    let m = world();
    let open = energy(&m, &[], LIS_B, 2);
    let t = [0.3, 0.1, 0.03];
    let closed = energy(&m, &[leaf(flat_transmissive(0.1, 0.3, t))], LIS_B, 3);
    let rel: [f32; NBANDS] = core::array::from_fn(|b| db_e(closed[b] / open[b]));
    assert!(rel.iter().all(|&r| r < -6.0), "closed leaf not quieter than open: {rel:?} dB");
    assert!(rel[0] > rel[1] && rel[1] > rel[2], "leak not bass-heavy: {rel:?} dB");
    assert!(rel[0] - rel[2] > 12.0, "low-high tilt only {:.1} dB", rel[0] - rel[2]);
}

#[test]
fn closing_the_door_keeps_energy_in_the_source_room() {
    let m = world();
    let open = energy(&m, &[], LIS_A, 4);
    let closed = energy(&m, &[leaf(flat_opaque(0.1, 0.3))], LIS_A, 4);
    for b in 0..NBANDS {
        let d = db_e(closed[b] / open[b]);
        assert!(d > 0.3, "band {b}: closing the door changed room A by only {d:.2} dB");
    }
}
