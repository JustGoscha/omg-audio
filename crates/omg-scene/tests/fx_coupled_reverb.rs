//! Effect: COUPLED / REMOTE REVERB, sim side — `Sim::reverb_only` (the
//! source room's wet field at its exit door) and `Sim::reverb_world`
//! over arbitrary `AcousticGeometry` + panel overlays.
//!
//! Physical claims guaranteed:
//! (1) `reverb_only` measures exactly the late field `update` would for
//!     the same room and poses (same tracer, same budget), without taps,
//!     and it orders rooms physically (live ≫ dead RT60).
//! (2) `reverb_world` on a two-room geometry (A ↔ doorway ↔ B, source in
//!     A, listener in B) MEASURES the coupled wet field: present through
//!     the open doorway, arriving anisotropically FROM the doorway, and
//!     closing the opening with a panel overlay (`WithPanels`, a steel
//!     leaf) drops it markedly — no portal code involved.
//!
//! Combined in the demo: `WorldSim` remote wet (`sim_remotes[si]
//! .reverb_only` at the chain's exit door → `ParamBlock::remote`) and the
//! world late field (`world_late[si].reverb_world` with door-leaf
//! panels) — `tests/late_world_gate.rs`, `tests/club_boom.rs`.
//!
//! Global switch used: quality override 1 (gate age = 1, re-trace every
//! call) — set to the same value by every test in this binary.

mod common;

use common::*;
use omg_core::material::Material;
use omg_core::scene::{Shoebox, WithPanels};
use omg_core::vec3::Vec3;
use omg_scene::dome::build_world_mesh;
use omg_scene::quality;
use omg_scene::sim::Sim;
use omg_scene::walkthrough::HEAVY_PANEL_TRANSMISSION;

#[test]
fn reverb_only_is_the_rooms_late_field() {
    quality::set_override(1, 1);
    let room = Shoebox::new(Vec3::new(9.0, 7.0, 3.5), [Material::CONCRETE; 6]);
    let (src, door) = (Vec3::new(2.0, 3.0, 1.5), Vec3::new(8.7, 3.5, 1.5));
    let wet = Sim::new().reverb_only(&room, src, door);
    let full = Sim::new().update(&room, src, door, 0.0).reverb;
    assert_eq!(wet.rt60, full.rt60, "same measurement as the full update");
    assert_eq!(wet.level, full.level);

    let dead = Shoebox::new(Vec3::new(9.0, 7.0, 3.5), [Material::ACOUSTIC_TILE; 6]);
    let dry = Sim::new().reverb_only(&dead, src, door);
    assert!(wet.rt60[1] > 2.0 * dry.rt60[1], "concrete {:?} vs tile {:?}", wet.rt60, dry.rt60);
    assert!(wet.level[1] > dry.level[1], "a live room sends more wet");
}

#[test]
fn world_late_field_couples_through_a_doorway() {
    quality::set_override(1, 1);
    // live A (0..6) | dead B (6..12), shared wall x = 6, doorway at
    // y = 2.5: B's own tail dies fast, so its late field is A's, through
    // the opening
    let rooms = vec![
        room((0.0, 0.0), (6.0, 5.0), 3.0, Material::CONCRETE),
        room((6.0, 0.0), (12.0, 5.0), 3.0, Material::ACOUSTIC_TILE),
        outdoor((-30.0, -30.0), (40.0, 40.0), Material::CONCRETE),
    ];
    let doors = vec![door((0, 1), (6.0, 2.5), 0)];
    let (mesh, _) = build_world_mesh(&rooms, &doors);
    let src = Vec3::new(2.0, 2.5, 1.5);
    let lis = Vec3::new(8.5, 2.5, 1.6);
    let leaf = Material { absorption: [0.1; 3], scattering: 0.1, transmission: HEAVY_PANEL_TRANSMISSION };
    let closed = [(Vec3::new(5.96, 1.95, 0.0), Vec3::new(6.04, 3.05, 2.0), leaf)];

    let measure = |panels: &[(Vec3, Vec3, Material)]| {
        let world = WithPanels { base: &mesh, panels };
        let mut sim = Sim::new();
        let mut out = None;
        for _ in 0..24 {
            let mut budget = 1;
            out = Some(sim.reverb_world(&world, panels, src, lis, &mut budget));
            assert_eq!(budget, 0, "gate age 1: every call traces");
        }
        out.unwrap()
    };
    let (open, dir, aniso) = measure(&[]);
    let (shut, _, _) = measure(&closed);
    eprintln!("open {:?} dir {dir:?} aniso {aniso:.2} | closed {:?}", open.level, shut.level);
    assert!(open.level[1] > 1e-3, "A's wet reaches B through the doorway: {:?}", open.level);
    assert!(dir[0] < -0.7, "the tail arrives from the doorway side (−x): {dir:?}");
    assert!(aniso > 0.4, "coupled tail is directional: {aniso}");
    assert!(shut.level[1] < 0.25 * open.level[1], "a shut steel leaf drains B's wet: {:?} vs {:?}", shut.level, open.level);
    assert!(shut.level[2] < 0.3 * open.level[2], "and the highs most");
}
