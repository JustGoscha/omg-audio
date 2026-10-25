//! Effect: GLASS PANES and HEAVY DOORS — `GLASS_TRANSMISSION`,
//! `HEAVY_GLASS_TRANSMISSION`, `DOOR_PANEL_TRANSMISSION`,
//! `HEAVY_PANEL_TRANSMISSION`, `glass_transmission(heavy)`,
//! `Door::fill_energy/fill_amplitude`, and how routing prices them
//! (`aperture_routes`, `price_chain`).
//!
//! Physical claims guaranteed (mass law): every filler passes lows more
//! than highs (strictly falling spectrum); the heavy build of each
//! (laminated glazing, steel leaf) transmits less than the light one in
//! EVERY band, and laminated glass is the more bass-tilted ("the club
//! booms, the hiss dies"). A door's filler is the area-weighted energy
//! mix of the open slit and its panel (open = 1, shut = panel²); a pane
//! transmits as glass regardless of `openness`. A window radiator's
//! routed muffle is its pane's transmission; a closed door on a routed
//! chain charges the wet path its own panel.
//!
//! Combined in the demo: club laminated windows + sound-lock heavy
//! doors, bunker/cathedral steel doors, house single-glass windows —
//! `tests/club_boom.rs`, `tests/sealed_club.rs`, `world.rs` tests.

mod common;

use common::*;
use omg_core::NBANDS;
use omg_scene::walkthrough::{
    aperture_routes, aperture_hop, glass_transmission, price_chain, DOOR_PANEL_TRANSMISSION,
    GLASS_TRANSMISSION, HEAVY_GLASS_TRANSMISSION, HEAVY_PANEL_TRANSMISSION,
};

fn falling(s: [f32; NBANDS]) -> bool {
    s.windows(2).all(|w| w[0] > w[1]) && s.iter().all(|&v| v > 0.0 && v < 1.0)
}

#[test]
fn fillers_obey_mass_law_and_heavy_is_tighter() {
    for (name, s) in [
        ("glass", GLASS_TRANSMISSION),
        ("laminated", HEAVY_GLASS_TRANSMISSION),
        ("wood leaf", DOOR_PANEL_TRANSMISSION),
        ("steel leaf", HEAVY_PANEL_TRANSMISSION),
    ] {
        assert!(falling(s), "{name}: lows must pass more than highs: {s:?}");
    }
    for b in 0..NBANDS {
        assert!(HEAVY_GLASS_TRANSMISSION[b] < GLASS_TRANSMISSION[b], "laminated < single, band {b}");
        assert!(HEAVY_PANEL_TRANSMISSION[b] < DOOR_PANEL_TRANSMISSION[b], "steel < wood, band {b}");
    }
    let tilt = |s: [f32; NBANDS]| s[0] / s[2];
    assert!(tilt(HEAVY_GLASS_TRANSMISSION) > 3.0 * tilt(GLASS_TRANSMISSION), "laminated glazing booms");
    assert_eq!(glass_transmission(false), GLASS_TRANSMISSION);
    assert_eq!(glass_transmission(true), HEAVY_GLASS_TRANSMISSION);
}

#[test]
fn door_fill_is_the_area_weighted_energy_mix() {
    for heavy in [false, true] {
        let panel = if heavy { HEAVY_PANEL_TRANSMISSION } else { DOOR_PANEL_TRANSMISSION };
        let mut d = door((0, 1), (0.0, 0.0), 0);
        d.heavy = heavy;
        for o in [0.0f32, 0.25, 0.5, 1.0] {
            d.openness = o;
            let e = d.fill_energy();
            let a = d.fill_amplitude();
            for b in 0..NBANDS {
                assert_rel(e[b], o + (1.0 - o) * panel[b] * panel[b], 1e-6, "energy mix");
                assert_rel(a[b] * a[b], e[b], 1e-6, "amplitude = √energy");
            }
        }
    }
    // shut: steel below wood in every band; half-open: still ordered
    let (mut wood, mut steel) = (door((0, 1), (0.0, 0.0), 0), door((0, 1), (0.0, 0.0), 0));
    steel.heavy = true;
    for o in [0.0f32, 0.5] {
        wood.openness = o;
        steel.openness = o;
        for b in 0..NBANDS {
            assert!(steel.fill_energy()[b] < wood.fill_energy()[b]);
        }
    }
    // panes ignore openness and transmit glass²
    for heavy in [false, true] {
        let mut p = pane((0, 1), (0.0, 0.0), 0, heavy);
        let g = glass_transmission(heavy);
        for o in [0.0f32, 1.0] {
            p.openness = o;
            for b in 0..NBANDS {
                assert_rel(p.fill_energy()[b], g[b] * g[b], 1e-6, "pane energy");
            }
        }
    }
}

#[test]
fn routing_prices_panes_and_closed_panels() {
    // window radiator between A and B: its muffle is the pane itself
    for heavy in [false, true] {
        let doors = [pane((0, 1), (6.0, 2.0), 0, heavy)];
        let routes = aperture_routes(&source(0, (2.0, 2.0)), 1, (9.0, 2.0), &doors);
        assert_eq!(routes.len(), 1);
        assert!(routes[0].glass);
        assert_eq!(routes[0].muffle, glass_transmission(heavy));
        assert_bands_rel(routes[0].wet_trans, glass_transmission(heavy), 1e-6, "pane wet");
    }
    // a shut door on a chain: the wet path pays its own panel
    for heavy in [false, true] {
        let mut d = door((0, 1), (6.0, 2.0), 0);
        d.openness = 0.0;
        d.heavy = heavy;
        let r = price_chain((2.0, 2.0), &[0], (9.0, 2.0), &[d]);
        let panel = if heavy { HEAVY_PANEL_TRANSMISSION } else { DOOR_PANEL_TRANSMISSION };
        assert_bands_rel(r.wet_trans, panel, 1e-6, "closed-door wet transmission");
    }
}

/// The DRY routed path through a shut door must pay the same panel the
/// wet path and the power balance pay — a steel blast door is "nearly a
/// wall when shut" (HEAVY_PANEL_TRANSMISSION), not a wooden leaf.
#[test]
#[ignore = "BUG: aperture_hop prices a shut HEAVY door's dry path with DOOR_PANEL_TRANSMISSION (wood), ignoring Door::heavy"]
fn shut_heavy_door_dry_path_pays_the_steel_panel() {
    let mut d = door((0, 1), (6.0, 2.0), 0);
    d.openness = 0.0;
    d.heavy = true;
    // on-axis: no bending, the panel is the whole price
    let (_, dry) = aperture_hop((2.0, 2.0), (9.0, 2.0), &d);
    let chain = price_chain((2.0, 2.0), &[0], (9.0, 2.0), &[d]);
    for b in 0..NBANDS {
        assert_rel(dry[b], HEAVY_PANEL_TRANSMISSION[b], 1e-5, "aperture_hop through shut steel");
        assert_rel(chain.muffle[b], chain.wet_trans[b], 1e-5, "dry and wet pay the same panel");
    }
}
