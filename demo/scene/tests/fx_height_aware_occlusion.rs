//! Effect: HEIGHT-AWARE STRAIGHT-RAY OCCLUSION —
//! `walkthrough::straight_path_transmission_z`.
//!
//! Claims guaranteed:
//! - a ray that crosses a building's wall plane ABOVE its roof line
//!   (`barrier_height`) pays nothing for it: a bell hung above a solid
//!   tower shaft rings out over it;
//! - the same ray at ear height pays exactly what the 2D rule
//!   (`straight_path_transmission`) charges — the plan-view rule is the
//!   height-aware rule for paths that stay low;
//! - a ray passing UNDER an upper storey's floor does not pay that
//!   storey's walls;
//! - `open_end` leaves the aperture the segment ends on to the route that
//!   continues through it, and every other wall on the way is still paid.
//!
//! Combined in the demo: `crates/omg-scene/tests/bells_open_vs_indoor.rs`
//! (the bells in the open belfry stage above the 24 m shaft) and every
//! WorldSim straight-ray tap and outdoor occlusion.

mod common;

use common::{door, outdoor, room, storey};
use omg_core::material::Material;
use omg_scene::walkthrough::{straight_path_transmission, straight_path_transmission_z};

const BRICK: Material = Material::BRICK;

#[test]
fn over_the_roof_line_is_free() {
    // a 24 m solid shaft, roof line 24.5 m; source above it, listener far
    let rooms = vec![
        room((-3.0, -3.0), (3.0, 3.0), 24.0, BRICK),
        outdoor((-500.0, -500.0), (500.0, 500.0), Material::GRASS),
    ];
    let mut rooms = rooms;
    rooms[0].barrier_height = 24.5;
    let doors = [];
    let above = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 25.5), (0.0, 300.0, 1.6), false);
    assert!(above.iter().all(|&t| t > 0.999), "a ray clearing the roof line pays nothing: {above:?}");
    // the same bell hung INSIDE the shaft pays its wall
    let inside = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 22.5), (0.0, 300.0, 1.6), false);
    let wall = BRICK.transmission_at(rooms[0].wall_thickness);
    for b in 0..3 {
        assert!((inside[b] - wall[b]).abs() < 1e-6, "band {b}: {inside:?} vs wall {wall:?}");
    }
}

#[test]
fn low_rays_match_the_plan_view_rule() {
    let rooms = vec![
        room((10.0, -5.0), (20.0, 5.0), 3.0, BRICK),
        outdoor((-100.0, -100.0), (100.0, 100.0), Material::GRASS),
    ];
    let doors = [];
    let flat = straight_path_transmission(&rooms, &doors, (0.0, 0.0), (40.0, 0.0));
    let low = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 1.5), (40.0, 0.0, 1.6), false);
    for b in 0..3 {
        assert!((flat[b] - low[b]).abs() < 1e-7, "band {b}: plan {flat:?} vs 3D {low:?}");
    }
    assert!(low[1] < 0.1, "two brick walls must cost something: {low:?}");
}

#[test]
fn under_an_upper_storey_skips_its_walls() {
    // an upper storey on stilts (floor at 4 m): a ground-level ray under
    // it crosses no wall of it
    let rooms = vec![
        storey((10.0, -5.0), (20.0, 5.0), 3.0, 4.0, BRICK),
        outdoor((-100.0, -100.0), (100.0, 100.0), Material::GRASS),
    ];
    let doors = [];
    let under = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 1.6), (40.0, 0.0, 1.6), false);
    assert!(under.iter().all(|&t| t > 0.999), "under the storey: {under:?}");
    let through = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 5.5), (40.0, 0.0, 5.5), false);
    assert!(through[1] < 0.1, "through the storey's walls: {through:?}");
}

#[test]
fn open_end_leaves_the_final_aperture_to_the_route() {
    // a closed wooden door on the far wall of a room; a segment from
    // outside the NEAR wall to that door point
    let mut d = door((0, 1), (20.0, 0.0), 0);
    d.openness = 0.0;
    let rooms = vec![
        room((10.0, -5.0), (20.0, 5.0), 3.0, BRICK),
        outdoor((-100.0, -100.0), (100.0, 100.0), Material::GRASS),
    ];
    let doors = [d];
    let wall = BRICK.transmission_at(rooms[0].wall_thickness);
    let open = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 1.0), (20.0, 0.0, 1.0), true);
    let shut = straight_path_transmission_z(&rooms, &doors, (0.0, 0.0, 1.0), (20.0, 0.0, 1.0), false);
    let panel = doors[0].fill_amplitude();
    for b in 0..3 {
        // the near wall is always paid; only the end door is left out
        assert!((open[b] - wall[b]).abs() < 1e-6, "band {b}: open_end {open:?} vs near wall {wall:?}");
        assert!((shut[b] - wall[b] * panel[b]).abs() < 1e-6, "band {b}: closed end {shut:?}");
    }
}
