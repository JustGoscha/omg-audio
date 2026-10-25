//! Effect: DOOR-GRAPH ROUTING — `walkthrough::route_source`,
//! `door_chain`, `price_chain` on a synthetic room chain.
//!
//! Physical claims guaranteed: a source in another room is heard as a
//! virtual source ON the exit doorway into the listener's room (on its
//! plane, inside its opening); the chain is the BFS-shortest sequence of
//! walkable doors (glass panes are never routing edges); the pre-door
//! path length is source → door → … → exit door, so the Huygens level
//! (0.6/extra · muffle, see fx_aperture_reradiation) drops with every
//! hop; sight lines through aligned openings cost nothing, bends around
//! a jamb cost knife-edge loss (highs most); a shut door on the chain
//! charges the wet path its panel. Room indices are the scene's own —
//! nothing is sized by the demo world.
//!
//! Combined in the demo: `WorldSim` ISM pipeline for every cross-room
//! source (`route_source(def, state.room, …, &self.doors)`) and the
//! remote-wet radiators — `world.rs` tests `closing_a_door_muffles_the_portal`,
//! `shadow_boundaries_are_continuous`.

mod common;

use common::*;
use omg_core::NBANDS;
use omg_scene::walkthrough::{door_chain, price_chain, route_source, Door, Routed, DOOR_PANEL_TRANSMISSION};

// Room indices deliberately beyond the demo's 13 rooms.
const A: usize = 20;
const B: usize = 21;
const C: usize = 22;
const D: usize = 30;
const E: usize = 31;

/// A(0..6) | B(6..12) | C(12..18) in a row along x (doors at y = 2 in
/// the shared x-walls), plus a detour A → D → E → C and a glass pane
/// straight from A to C (not walkable).
fn doors() -> Vec<Door> {
    vec![
        door((A, B), (6.0, 2.0), 0),   // 0
        door((B, C), (12.0, 2.0), 0),  // 1
        door((A, D), (3.0, 4.0), 1),   // 2: detour, one hop longer
        door((D, E), (9.0, 6.0), 0),   // 3
        door((E, C), (15.0, 4.0), 1),  // 4
        pane((A, C), (6.0, 3.5), 0, false), // 5: glass, never routes
    ]
}

fn huygens(r: &Routed) -> f32 {
    0.6 / r.extra_dist.max(0.6) * r.muffle[1]
}

#[test]
fn bfs_finds_the_shortest_walkable_chain() {
    let ds = doors();
    assert_eq!(door_chain(A, C, &ds), vec![0, 1], "two hops via B, not three via D/E, never the pane");
    assert_eq!(door_chain(C, A, &ds), vec![1, 0], "reverse direction, reverse chain");
    assert_eq!(door_chain(A, E, &ds), vec![2, 3]);
    assert_eq!(door_chain(B, B, &ds), Vec::<usize>::new(), "same room: no doors");
    // route_source walks the same chain
    let r = route_source(&source(A, (2.0, 2.0)), C, (15.0, 2.0), &ds);
    assert_eq!(r.route.len(), 4, "source, two door vertices, listener");
    assert!((r.route[1].0 - 6.0).abs() < 1e-5 && (r.route[2].0 - 12.0).abs() < 1e-5);
}

#[test]
fn virtual_source_sits_at_the_exit_doorway() {
    let ds = doors();
    // off-axis listener in C: the chain must bend at the exit door
    let lis = (16.0, 3.6);
    let r = route_source(&source(A, (2.0, 2.0)), C, lis, &ds);
    let exit = &ds[1];
    assert!((r.virt_world.0 - exit.pos.0).abs() < 1e-5, "on the exit door plane: {:?}", r.virt_world);
    assert!((r.virt_world.1 - exit.pos.1).abs() <= exit.half, "inside the opening: {:?}", r.virt_world);
    assert_eq!(r.exit_axis, Some(exit.axis));
    assert!(!r.glass);
    // extra_dist = source → door 0 vertex → door 1 vertex
    let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    assert_rel(r.extra_dist, dist(r.route[0], r.route[1]) + dist(r.route[1], r.route[2]), 1e-5, "pre-door path");
    // same-room: the source itself, unrouted
    let same = route_source(&source(C, (14.0, 1.0)), C, lis, &ds);
    assert_eq!(same.virt_world, (14.0, 1.0));
    assert_eq!((same.extra_dist, same.exit_axis), (0.0, None));
    assert_eq!(same.muffle, [1.0; NBANDS]);
}

#[test]
fn level_drops_per_hop_and_bends_cost_highs() {
    let ds = doors();
    let src = source(A, (2.0, 2.0));
    // aligned openings: free sight line, only distance
    let in_b = route_source(&src, B, (9.0, 2.0), &ds);
    let in_c = route_source(&src, C, (15.0, 2.0), &ds);
    assert_eq!(in_b.muffle, [1.0; NBANDS], "on-axis hop is free");
    assert_eq!(in_c.muffle, [1.0; NBANDS], "aligned chain is free");
    assert_rel(in_b.extra_dist, 4.0, 1e-5, "A → door AB");
    assert_rel(in_c.extra_dist, 10.0, 1e-5, "A → AB → BC");
    assert!(huygens(&in_c) < huygens(&in_b), "one more hop, quieter");
    // off-axis in C: the exit jamb bends the path — knife-edge, highs first
    let bent = route_source(&src, C, (13.0, 3.9), &ds);
    assert!(bent.muffle[1] < 0.9, "bend must cost: {:?}", bent.muffle);
    assert!(bent.muffle[2] < bent.muffle[0], "and favour lows: {:?}", bent.muffle);
    assert!(huygens(&bent) < huygens(&in_c));
}

#[test]
fn shut_door_on_the_chain_charges_the_wet_path() {
    let mut ds = doors();
    ds[0].openness = 0.0;
    let chain = door_chain(A, C, &ds);
    assert_eq!(chain, vec![0, 1], "topology ignores leaf state");
    let r = price_chain((2.0, 2.0), &chain, (15.0, 2.0), &ds);
    assert_bands_rel(r.wet_trans, DOOR_PANEL_TRANSMISSION, 1e-6, "shut wooden leaf on the wet path");
    for b in 0..NBANDS {
        assert!(r.muffle[b] <= DOOR_PANEL_TRANSMISSION[b] + 1e-6, "dry path pays the panel too");
    }
}
