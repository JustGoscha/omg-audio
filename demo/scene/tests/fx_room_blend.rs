//! Effect: ROOM BLENDING AT THRESHOLDS — `walkthrough::blended_state_for
//! (rooms, doors, x, y, eye_z, yaw)` (BLEND_RADIUS = 1.5 m).
//!
//! Physical claims guaranteed: within BLEND_RADIUS of a walkable doorway
//! both connected rooms are simulated and crossfaded with EQUAL-POWER
//! weights (w_primary² + w_other² = 1 everywhere), the far room fading
//! in smoothly from 0 at the radius; outside the radius, near a pane, or
//! near a door that does not touch the listener's room, there is no
//! blend. Documented (`BlendedState`): continuous through the crossing —
//! both weights ≈ 0.707 at the door plane whichever side `room_of`
//! reports.
//!
//! Combined in the demo: every interactive `WorldSim::tick_at` /
//! `tick_at_z` (and the scripted walk via `blended_state_at`) —
//! `world.rs::ambient_field_is_continuous_through_a_doorway`,
//! `tests/world_traced_gate.rs::doorway_walk_in_is_continuous`.

mod common;

use common::*;
use omg_core::material::Material;
use omg_scene::walkthrough::{blended_state_for, Door, RoomDef, BLEND_RADIUS};

/// South room 0 (y 0..4) and north room 1 (y 4..8) sharing the y = 4
/// wall, doorway at x = 3; an unrelated room 2 far east with its own
/// door to room 3.
fn scene() -> (Vec<RoomDef>, Vec<Door>) {
    let rooms = vec![
        room((0.0, 0.0), (6.0, 4.0), 3.0, Material::BRICK),
        room((0.0, 4.0), (6.0, 8.0), 3.0, Material::BRICK),
        room((20.0, 0.0), (26.0, 4.0), 3.0, Material::BRICK),
        room((26.0, 0.0), (32.0, 4.0), 3.0, Material::BRICK),
    ];
    let doors = vec![door((0, 1), (3.0, 4.0), 1), door((2, 3), (26.0, 2.0), 0)];
    (rooms, doors)
}

/// (weight of room 0, weight of room 1) at a pose.
fn weights(rooms: &[RoomDef], doors: &[Door], x: f32, y: f32) -> (f32, f32) {
    let bs = blended_state_for(rooms, doors, x, y, 1.6, 0.0);
    let mut w = [0.0f32; 4];
    w[bs.primary.room] += bs.primary_weight;
    if let Some((o, wo)) = &bs.other {
        w[o.room] += wo;
    }
    (w[0], w[1])
}

#[test]
fn equal_power_crossfade_within_the_radius() {
    let (rooms, doors) = scene();
    let mut prev_far = -1.0f32;
    // approach the door from deep inside room 0 along x = 3
    for i in 0..=30 {
        let y = 0.5 + i as f32 * 0.1; // up to 3.5 (wall margin keeps ≥ 0.3 off the plane)
        let bs = blended_state_for(&rooms, &doors, 3.0, y, 1.6, 0.0);
        assert_eq!(bs.primary.room, 0);
        let dist = 4.0 - y.min(3.7);
        match &bs.other {
            None => {
                assert!(dist >= BLEND_RADIUS - 1e-4, "inside the radius must blend (dist {dist})");
                assert_eq!(bs.primary_weight, 1.0);
            }
            Some((o, wo)) => {
                assert_eq!(o.room, 1, "the room across THIS door");
                assert!(dist < BLEND_RADIUS + 1e-4);
                assert_rel(bs.primary_weight.powi(2) + wo.powi(2), 1.0, 1e-5, "equal power");
                assert!(*wo <= std::f32::consts::FRAC_1_SQRT_2 + 1e-6, "primary never loses the majority");
                assert!(*wo >= prev_far - 1e-6, "far room fades IN as the door nears");
                prev_far = *wo;
            }
        }
    }
    assert!(prev_far > 0.3, "close to the door the far room is clearly in: {prev_far}");
    // at the radius edge the far room starts from (near) zero
    let (_, w1) = weights(&rooms, &doors, 3.0, 4.0 - BLEND_RADIUS + 0.01);
    assert!(w1 < 0.02, "fade-in starts at zero: {w1}");
}

#[test]
fn no_blend_away_from_doors_at_panes_or_foreign_doors() {
    let (rooms, mut doors) = scene();
    // centre of room 0: far from everything
    assert!(blended_state_for(&rooms, &doors, 1.0, 1.0, 1.6, 0.0).other.is_none());
    // the scene's door, not the demo's: a pose in room 1 right at the
    // demo's Living↔Corridor door (4, 6) — which also links rooms 0 and
    // 1 — but 2 m from ours
    let bs = blended_state_for(&rooms, &doors, 4.0, 5.9, 1.6, 0.0);
    assert!(bs.other.is_none(), "blending must use the doors passed in");
    // a pane is not a threshold
    doors[0] = pane((0, 1), (3.0, 4.0), 1, false);
    assert!(blended_state_for(&rooms, &doors, 3.0, 3.6, 1.6, 0.0).other.is_none());
    // the nearest door connects two OTHER rooms (placed right beside
    // room 0's east wall): it is no threshold of room 0
    let (rooms2, mut doors2) = scene();
    doors2[1].pos = (6.0, 2.0);
    let bs = blended_state_for(&rooms2, &doors2, 5.7, 2.0, 1.6, 0.0);
    assert_eq!(bs.primary.room, 0);
    assert!(bs.other.is_none(), "a door not touching the listener's room never blends");
}

/// `BlendedState` doc: "Continuous through the crossing: at the door
/// plane both weights are ≈ 0.707 regardless of which side `room_of`
/// currently reports." Walk straight through the doorway in 1 cm steps:
/// each room's weight must change smoothly.
#[test]
#[ignore = "BUG: blend weights jump ~0.22 at the door plane — distance is measured from the wall-margin-CLAMPED listener (≥ 0.3 m), not the raw pose"]
fn crossfade_is_continuous_through_the_door_plane() {
    let (rooms, doors) = scene();
    let mut prev: Option<(f32, f32)> = None;
    let mut worst = 0.0f32;
    for i in 0..=300 {
        let y = 2.5 + i as f32 * 0.01;
        let (w0, w1) = weights(&rooms, &doors, 3.0, y);
        assert_rel(w0 * w0 + w1 * w1, 1.0, 1e-5, "equal power across the crossing");
        if let Some((p0, p1)) = prev {
            worst = worst.max((w0 - p0).abs()).max((w1 - p1).abs());
        }
        prev = Some((w0, w1));
    }
    let (a0, a1) = weights(&rooms, &doors, 3.0, 3.999);
    let (b0, b1) = weights(&rooms, &doors, 3.0, 4.001);
    eprintln!("plane: south side ({a0:.3}, {a1:.3})  north side ({b0:.3}, {b1:.3})  worst step {worst:.3}");
    assert!(worst < 0.03, "no weight may jump between 1 cm steps: worst {worst}");
    for w in [a0, a1, b0, b1] {
        assert!((w - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.05, "≈ 0.707 at the plane: {w}");
    }
}
