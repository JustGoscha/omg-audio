//! Effect: early (specular-path) transmission through MESH walls and
//! door leaves — `pt_mesh::mesh_record` on a synthetic two-room mesh,
//! with `extras` boxes as door leaves.
//!
//! Physical claim guaranteed: the direct path from one room into the
//! next pays exactly the party wall's per-band transmission
//! (gain = air(d)/d × T, bass-heavy, straight delay d/c); threading a
//! door HOLE is free (gain = air(d)/d); a door LEAF placed in the hole
//! (transient `Aabb` extra) turns "free" back into exactly its own
//! transmission; and a fully opaque wall leaves no direct record.
//! No portals, no room graph — geometry alone decides.
//!
//! Combined in the demo: omg-scene tests/pt_mesh_gate.rs (doorway &
//! masonry on the real world mesh) and `WorldSim`'s door states.

mod common;

use common::{flat_opaque, flat_transmissive, two_rooms, v};
use omg_core::material::air_attenuation;
use omg_core::mesh::Mesh;
use omg_core::pt::Aabb;
use omg_core::pt_mesh::{mesh_record, MeshRecord, SurfaceTable};
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};

const T_WALL: [f32; NBANDS] = [0.1, 0.02, 0.004];
const DOOR: ((f32, f32), f32) = ((2.0, 3.0), 2.1);
/// Line through the door hole (y = 2.5, z = 1.2 < 2.1).
const SRC_DOOR: Vec3 = Vec3::new(2.0, 2.5, 1.2);
const LIS_DOOR: Vec3 = Vec3::new(10.0, 2.5, 1.2);
/// Line through solid wall (y = 4.0, outside the hole).
const SRC_WALL: Vec3 = Vec3::new(2.0, 4.0, 1.2);
const LIS_WALL: Vec3 = Vec3::new(10.0, 4.0, 1.2);

fn mesh(t: [f32; NBANDS]) -> Mesh {
    two_rooms(6.0, 5.0, 3.0, flat_opaque(0.1, 0.1), flat_transmissive(0.05, 0.1, t), Some(DOOR))
}

fn direct(mesh: &Mesh, src: Vec3, lis: Vec3, extras: &[Aabb]) -> Option<MeshRecord> {
    let table = SurfaceTable::build(mesh);
    let mut buf = Vec::new();
    mesh_record(mesh, &table, &[], 0, src, lis, extras, &mut buf)
}

#[test]
fn direct_through_party_wall_pays_exact_transmission() {
    let m = mesh(T_WALL);
    let r = direct(&m, SRC_WALL, LIS_WALL, &[]).expect("direct through wall must exist");
    let d = 8.0;
    let air = air_attenuation(d);
    for b in 0..NBANDS {
        let want = air[b] / d * T_WALL[b];
        assert!((r.gains[b] - want).abs() < 1e-4 * want, "band {b}: {} want {want}", r.gains[b]);
    }
    assert!(r.gains[0] > r.gains[1] && r.gains[1] > r.gains[2], "not bass-heavy: {:?}", r.gains);
    assert!((r.delay_s - d / SPEED_OF_SOUND).abs() < 1e-6);
}

#[test]
fn door_hole_is_free() {
    let m = mesh(T_WALL);
    let r = direct(&m, SRC_DOOR, LIS_DOOR, &[]).expect("direct through the hole must exist");
    let d = 8.0;
    let air = air_attenuation(d);
    for b in 0..NBANDS {
        let want = air[b] / d;
        assert!((r.gains[b] - want).abs() < 1e-4 * want, "band {b}: {} want free {want}", r.gains[b]);
    }
}

#[test]
fn door_leaf_restores_its_own_transmission() {
    let m = mesh(T_WALL);
    let leaf_t = [0.3, 0.08, 0.01];
    let leaf = Aabb { min: v(5.98, 2.0, 0.0), max: v(6.02, 3.0, 2.1), transmission: leaf_t };
    let r = direct(&m, SRC_DOOR, LIS_DOOR, &[leaf]).expect("leaf is not opaque");
    let air = air_attenuation(8.0);
    for b in 0..NBANDS {
        let want = air[b] / 8.0 * leaf_t[b];
        assert!((r.gains[b] - want).abs() < 1e-4 * want, "band {b}: {} want {want}", r.gains[b]);
    }
    // a leaf elsewhere (not on the path) changes nothing
    let away = Aabb { min: v(1.0, 0.1, 0.0), max: v(1.2, 0.5, 2.0), transmission: [0.0; NBANDS] };
    let r2 = direct(&m, SRC_DOOR, LIS_DOOR, &[away]).unwrap();
    assert!((r2.gains[1] - air[1] / 8.0).abs() < 1e-6);
}

#[test]
fn opaque_wall_has_no_direct_record() {
    let m = mesh([0.0; NBANDS]);
    assert!(direct(&m, SRC_WALL, LIS_WALL, &[]).is_none(), "opaque wall let a direct record through");
    // while the hole in the same wall still passes
    assert!(direct(&m, SRC_DOOR, LIS_DOOR, &[]).is_some());
}
