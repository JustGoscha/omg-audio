//! Effect: OUTDOOR-FIELD POWER BALANCE — `environment::AcousticsGraph::new
//! (rooms, doors)`, `outdoor_field(doors)`, `roof_sky`.
//!
//! Physical claims guaranteed (steady-state Sabine balance over the
//! room coupling graph, outdoors pinned at unit energy): every enclosed
//! room gets a partial field in (0, 1), bass-favoured (mass-law
//! conduits); a room behind an OPEN door hears more than behind a shut
//! one, and swinging the leaf moves the field continuously and
//! monotonically; a room two doorways deep receives less than the room
//! adjacent to the outside — yet is supplied through it; and the roof
//! exposure (rain) is per storey: a room under another storey sees no
//! sky, the top storey sees all of it, a buried room none.
//!
//! Combined in the demo: `WorldSim` ambient bed (`acoustics.outdoor_field
//! (&self.doors)` every tick, rain on `roof_sky`) — `environment.rs`
//! unit tests and `world.rs::ambient_field_is_continuous_through_a_doorway`.

mod common;

use common::*;
use omg_core::material::Material;
use omg_core::NBANDS;
use omg_scene::environment::AcousticsGraph;
use omg_scene::walkthrough::{Door, RoomDef};

const OUT: usize = 2;

/// Front room A (door to the street at y = 0) and back room B behind it
/// (door A↔B at y = 5), B has no outside opening.
fn scene() -> (Vec<RoomDef>, Vec<Door>) {
    let rooms = vec![
        room((0.0, 0.0), (6.0, 5.0), 2.8, Material::BRICK),
        room((0.0, 5.0), (6.0, 10.0), 2.8, Material::BRICK),
        outdoor((-50.0, -50.0), (50.0, 50.0), Material::GRASS),
    ];
    let doors = vec![door((0, OUT), (3.0, 0.0), 1), door((0, 1), (3.0, 5.0), 1)];
    (rooms, doors)
}

#[test]
fn field_is_partial_bass_heavy_and_deeper_is_quieter() {
    let (rooms, doors) = scene();
    let g = AcousticsGraph::new(&rooms, &doors);
    let f = g.outdoor_field(&doors);
    assert_eq!(f[OUT], [1.0; NBANDS], "outdoors is the unit reference");
    for ri in [0usize, 1] {
        for b in 0..NBANDS {
            assert!(f[ri][b] > 1e-6 && f[ri][b] < 1.0, "room {ri} band {b}: {:?}", f[ri]);
        }
        assert!(f[ri][0] >= f[ri][2], "room {ri}: indoors favours lows: {:?}", f[ri]);
    }
    eprintln!("front {:?} back {:?}", f[0], f[1]);
    assert!(f[1][1] < 0.7 * f[0][1], "two doorways deep < adjacent: {:?} vs {:?}", f[1], f[0]);
    // the back room's supply really comes through the front room
    let mut sealed = doors.clone();
    sealed[1].openness = 0.0;
    let fs = g.outdoor_field(&sealed);
    assert!(fs[1][1] < 0.5 * f[1][1], "shutting A↔B starves B: {:?} vs {:?}", fs[1], f[1]);
}

#[test]
fn open_door_beats_closed_continuously() {
    let (rooms, mut doors) = scene();
    let g = AcousticsGraph::new(&rooms, &doors);
    let open = g.outdoor_field(&doors)[0];
    doors[0].openness = 0.0;
    let shut = g.outdoor_field(&doors)[0];
    // the slit admits energy ∝ its area: every 5% of swing moves the
    // field by a bounded share of the open→shut range (no snap anywhere)
    let mut prev = open;
    for step in 1..=20 {
        doors[0].openness = 1.0 - step as f32 / 20.0;
        let cur = g.outdoor_field(&doors)[0];
        for b in 0..NBANDS {
            assert!(cur[b] <= prev[b] + 1e-9, "closing never raises band {b}");
            let share = (prev[b] - cur[b]) / (open[b] - shut[b]);
            assert!(share < 0.12, "5% swing moved band {b} by {share:.3} of its range: {prev:?} -> {cur:?}");
        }
        prev = cur;
    }
    assert!(prev[1] < 0.3 * open[1], "shut street door: clearly quieter: {prev:?} vs {open:?}");
    // a steel leaf seals harder than wood
    doors[0].heavy = true;
    let steel = g.outdoor_field(&doors)[0];
    assert!(steel[1] < prev[1], "steel {steel:?} vs wood {prev:?}");
}

#[test]
fn roof_exposure_by_storey() {
    let rooms = vec![
        room((0.0, 0.0), (6.0, 5.0), 2.8, Material::BRICK),           // ground floor
        storey((0.0, 0.0), (6.0, 5.0), 2.6, 3.0, Material::BRICK),    // upper storey
        room((10.0, 0.0), (14.0, 5.0), 3.0, Material::BRICK),         // bungalow
        storey((20.0, 0.0), (26.0, 5.0), 2.2, -3.0, Material::CONCRETE), // buried
        outdoor((-50.0, -50.0), (50.0, 50.0), Material::GRASS),
    ];
    let doors = vec![door((0, 4), (3.0, 0.0), 1)];
    let g = AcousticsGraph::new(&rooms, &doors);
    assert!(g.roof_sky[0] < 0.05, "ground floor is roofed by the storey: {}", g.roof_sky[0]);
    assert!(g.roof_sky[1] > 0.95, "top storey sees the sky: {}", g.roof_sky[1]);
    assert!(g.roof_sky[2] > 0.95, "a bungalow sees the sky: {}", g.roof_sky[2]);
    assert!(g.roof_sky[3] < 0.05, "a buried ceiling is under earth: {}", g.roof_sky[3]);
    // the buried room is sealed by earth (no exterior wall, no roof)
    let f = g.outdoor_field(&doors);
    assert!(f[3][1] < 1e-6, "no door, no sky, below grade: {:?}", f[3]);
    assert!(f[2][1] > 1e-6, "an above-ground shell still leaks: {:?}", f[2]);
}
