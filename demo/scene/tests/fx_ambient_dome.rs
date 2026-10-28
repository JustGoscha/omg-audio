//! Effect: AMBIENT DOME — `dome::DomeProbe::new(rooms, doors)` +
//! `sample(eye, &door_panels(doors))`.
//!
//! Physical claims guaranteed, on a minimal scene (open ground + one
//! box room with one doorway): standing in the open, the whole dome
//! arrives — the upper hemisphere directly, the lower one off the ground
//! at the ground's energy reflectance, i.e. ½ + ½(1 − α_ground); indoors
//! only a small fraction arrives, localized TOWARD the doorway; swinging
//! the leaf shut drains the inflow monotonically down to the shell's
//! bass-only mass-law seep; a glass pane in the opening passes the dome
//! with the pane's spectrum (laminated darker than single glazing).
//!
//! Combined in the demo: `WorldSim`'s ambient bed (city hum / rain
//! directions) — `dome.rs` unit tests on the demo world and
//! `world.rs::ambient_field_is_continuous_through_a_doorway`.

mod common;

use common::*;
use omg_core::material::Material;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;
use omg_scene::dome::{door_panels, DomeBin, DomeProbe, DOME_BINS};
use omg_scene::walkthrough::{Door, RoomDef};

/// Box room 8 × 6 × 3 (brick) at the origin, doorway in its south wall
/// (y = 0) at x = 4, inside a 200 m concrete-ground field.
fn scene(ground: Material) -> (Vec<RoomDef>, Vec<Door>) {
    let rooms = vec![
        room((0.0, 0.0), (8.0, 6.0), 3.0, Material::BRICK),
        outdoor((-100.0, -100.0), (100.0, 100.0), ground),
    ];
    (rooms, vec![door((0, 1), (4.0, 0.0), 1)])
}

fn sample(rooms: &[RoomDef], doors: &[Door], eye: Vec3) -> [DomeBin; DOME_BINS] {
    DomeProbe::new(rooms, doors).sample(eye, &door_panels(doors))
}

fn total(bins: &[DomeBin; DOME_BINS]) -> [f32; NBANDS] {
    core::array::from_fn(|b| bins.iter().map(|k| k.energy[b]).sum())
}

const INDOOR: Vec3 = Vec3 { x: 4.0, y: 4.0, z: 1.6 };

#[test]
fn open_air_hears_the_whole_dome() {
    for ground in [Material::CONCRETE, Material::GRASS] {
        let (rooms, doors) = scene(ground);
        // far from the building: nothing but sky and ground
        let t = total(&sample(&rooms, &doors, Vec3::new(-60.0, -60.0, 1.6)));
        for b in 0..NBANDS {
            let expect = 0.5 + 0.5 * (1.0 - ground.absorption[b]);
            assert_rel(t[b], expect, 0.02, "dome = sky + ground-reflected sky");
        }
    }
}

#[test]
fn indoors_hears_a_sliver_localized_at_the_doorway() {
    let (rooms, doors) = scene(Material::CONCRETE);
    let open = total(&sample(&rooms, &doors, Vec3::new(-60.0, -60.0, 1.6)))[1];
    let bins = sample(&rooms, &doors, INDOOR);
    let inside = total(&bins)[1];
    eprintln!("open {open:.4} inside {inside:.4}");
    assert!(inside < 0.2 * open, "indoors is a fraction of the dome: {inside} vs {open}");
    assert!(inside > 1e-4, "but the open doorway lets it in");
    // the dominant sector faces the doorway: from (4, 4) toward (4, 0) = −y
    let best = bins.iter().max_by(|a, b| a.energy[1].total_cmp(&b.energy[1])).unwrap();
    assert!(best.dir[1] < -0.6, "inflow must come from the door (−y): {:?}", best.dir);
    // energy-weighted mean direction also points at it
    let (mut sx, mut sy) = (0.0f32, 0.0f32);
    for k in &bins {
        sx += k.dir[0] * k.energy[1];
        sy += k.dir[1] * k.energy[1];
    }
    assert!(sy < -2.0 * sx.abs(), "net inflow toward −y: ({sx}, {sy})");
}

#[test]
fn closing_the_leaf_drains_the_inflow_to_bass_seep() {
    let (rooms, mut doors) = scene(Material::CONCRETE);
    let mut prev = [f32::MAX; NBANDS];
    let mut open = [0.0f32; NBANDS];
    for step in 0..=8 {
        doors[0].openness = 1.0 - step as f32 / 8.0;
        let t = total(&sample(&rooms, &doors, INDOOR));
        if step == 0 {
            open = t;
        }
        for b in 0..NBANDS {
            assert!(t[b] <= prev[b] + 1e-7, "closing never raises inflow (step {step}, band {b})");
        }
        prev = t;
    }
    eprintln!("open {open:?} shut {prev:?}");
    assert!(prev[1] < 0.2 * open[1], "a shut leaf leaves only panel/shell seep");
    assert!(prev[0] > 8.0 * prev[2], "sealed seep is bass-only (mass law): {prev:?}");
    // a steel leaf seals tighter than wood
    doors[0].heavy = true;
    let steel = total(&sample(&rooms, &doors, INDOOR));
    assert!(steel[1] <= prev[1], "steel {steel:?} vs wood {prev:?}");
}

#[test]
fn glass_passes_the_dome_with_its_spectrum() {
    let (rooms, mut doors) = scene(Material::CONCRETE);
    let mut tilt = [0.0f32; 2];
    for (i, heavy) in [false, true].into_iter().enumerate() {
        doors[0] = pane((0, 1), (4.0, 0.0), 1, heavy);
        let t = total(&sample(&rooms, &doors, INDOOR));
        eprintln!("heavy {heavy}: {t:?}");
        assert!(t[0] > 1e-4, "the room hears through its pane");
        assert!(t[0] > 2.0 * t[2], "glass favours lows: {t:?}");
        tilt[i] = t[0] / t[2];
        // the bass/treble tilt is the pane's energy tilt (g_lo/g_hi)²,
        // give or take the shell's own mass-law seep (with real values — a
        // laminated pane nearly as tight as the wall — the concrete's bass
        // seep tilts it up to ~2×)
        let g = omg_scene::walkthrough::glass_transmission(heavy);
        let pane_tilt = (g[0] / g[2]).powi(2);
        assert!((0.7..2.0).contains(&(tilt[i] / pane_tilt)), "tilt {} vs pane {pane_tilt}", tilt[i]);
    }
    assert!(tilt[1] > 2.0 * tilt[0], "laminated glazing is darker: {tilt:?}");
}
