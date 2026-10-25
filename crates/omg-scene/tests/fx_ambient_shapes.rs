//! Effect: SHAPED AMBIENCE — `ambient::field(emitters, cal, eye, vis)`:
//! an ambience bed whose emitters have a shape in the world (a ground
//! area of crickets, a volume) instead of a dome around the ear.
//!
//! Physical claims guaranteed, on synthetic shapes (a 400 m × 400 m
//! cricket meadow, a 40 m forest box, one roofed box room):
//!  - at ear height inside the meadow the field SURROUNDS: every azimuth
//!    sector carries a similar share, the energy vector is short, and
//!    what bias there is points down (the emitters are in the grass);
//!  - rising above it, the field comes from BELOW, narrows (energy vector
//!    grows toward 1) and fades (the area's subtended solid angle
//!    shrinks; spreading + air absorption), highs fastest (air);
//!  - walking away horizontally it collapses toward the meadow and
//!    fades; 10 km out it is a point at the meadow's bearing, all but
//!    gone;
//!  - a volume surrounds from all sides (zenith included) when you are
//!    in it and points at itself from outside;
//!  - the direct field is priced by geometry: hovering just over a roof,
//!    the meadow below is muffled by the slab (mass law, highs first);
//!    indoors its strongest direct share arrives through the doorway.
//!
//! Combined in the demo: `WorldSim`'s ambience bed (crickets over the
//! outdoor grass, `walkthrough::ambient_emitters`) — see
//! `tests/ambient_crickets.rs` (flying, 10 km, indoors, dome A/B).

mod common;

use common::*;
use omg_core::material::Material;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;
use omg_scene::ambient::{self, AmbientEmitter, AmbientShape, ShapedField};
use omg_scene::dome::{door_panels, DomeProbe, DOME_BINS};

fn meadow() -> Vec<AmbientEmitter> {
    vec![AmbientEmitter {
        shape: AmbientShape::GroundArea { min: (-200.0, -200.0), max: (200.0, 200.0), z: 0.1, holes: vec![] },
        level: 1.0,
    }]
}

/// Calibrated like a scene: the free field at ear height in the middle
/// of the meadow is 1 per band.
fn cal(em: &[AmbientEmitter]) -> [f32; NBANDS] {
    ambient::calibrate(em, Vec3::new(0.0, 0.0, 1.6), [1.0; NBANDS])
}

fn free(em: &[AmbientEmitter], eye: Vec3) -> ShapedField {
    let mut buf = Vec::new();
    ambient::field(em, cal(em), eye, |_, _| [1.0; NBANDS], &mut buf)
}

fn total(f: &ShapedField, b: usize) -> f32 {
    f.direct_total()[b]
}

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-30).log10()
}

#[test]
fn ear_height_in_the_meadow_surrounds_with_a_downward_bias() {
    let em = meadow();
    let f = free(&em, Vec3::new(0.0, 0.0, 1.6));
    assert_rel(total(&f, 1), 1.0, 1e-3, "calibrated reference");
    let (dir, r) = f.mean_dir();
    // surround: a short energy vector, every azimuth sector alive
    assert!(r < 0.35, "near-diffuse: energy vector {r}");
    assert!(dir.z < -0.9, "what bias there is points down (grass): {dir:?}");
    let sectors: Vec<f32> = (0..8).map(|k| f.direct[k][1]).collect();
    let mean = sectors.iter().sum::<f32>() / 8.0;
    for (k, s) in sectors.iter().enumerate() {
        assert!(*s > 0.7 * mean && *s < 1.3 * mean, "sector {k} carries its share: {sectors:?}");
    }
    assert!(f.direct[8][1] < 1e-6, "nothing from the zenith: {}", f.direct[8][1]);
    // and near-horizontal: the mean elevation of the arrivals is shallow
    // (equal energy per octave of distance — far grass counts)
    for k in 0..8 {
        let d = f.dir_sum[k] * (1.0 / f.direct[k][1]);
        let el = d.z.asin().to_degrees();
        assert!(el < -3.0 && el > -35.0, "sector {k} elevation {el}°");
    }
}

#[test]
fn rising_above_the_meadow_it_comes_from_below_narrows_and_fades() {
    let em = meadow();
    let mut prev: Option<(f32, f32, f32)> = None; // (mid, r, hi/lo)
    for h in [1.6f32, 3.0, 30.0, 300.0] {
        let f = free(&em, Vec3::new(0.0, 0.0, h));
        let (dir, r) = f.mean_dir();
        let mid = total(&f, 1);
        let tilt = total(&f, 2) / total(&f, 0);
        println!("h {h:>5} m: mid {:+.1} dB, energy vector {r:.2}, dir z {:+.2}, hi/lo {:+.1} dB", db(mid), dir.z, db(tilt));
        assert!(dir.z < -0.99, "h {h}: from straight below: {dir:?}");
        if let Some((pm, pr, pt)) = prev {
            assert!(mid < pm, "h {h}: quieter than lower: {mid} vs {pm}");
            assert!(r > pr, "h {h}: narrower than lower: {r} vs {pr}");
            assert!(tilt <= pt * 1.001, "h {h}: highs fade faster (air): {tilt} vs {pt}");
        }
        prev = Some((mid, r, tilt));
        match h as u32 {
            30 => assert!(db(mid) < -3.0 && db(mid) > -12.0, "30 m: {} dB", db(mid)),
            300 => {
                assert!(db(mid) < -12.0 && db(mid) > -30.0, "300 m: {} dB", db(mid));
                assert!(r > 0.75, "300 m: a lobe below, {r}");
            }
            _ => {}
        }
    }
}

#[test]
fn walking_away_it_collapses_toward_the_meadow_and_fades() {
    let em = meadow();
    let inside = total(&free(&em, Vec3::new(0.0, 0.0, 1.6)), 1);
    let mut prev = inside;
    for x in [400.0f32, 1000.0, 10_000.0] {
        let f = free(&em, Vec3::new(x, 0.0, 1.6));
        let (dir, r) = f.mean_dir();
        let mid = total(&f, 1);
        println!("{x:>6} m east: mid {:+.1} dB, energy vector {r:.2}, dir {dir:?}", db(mid / inside));
        assert!(dir.x < -0.95, "{x} m: toward the meadow (west): {dir:?}");
        assert!(r > 0.8, "{x} m: a compact direction: {r}");
        assert!(mid < prev, "{x} m: fades with distance");
        prev = mid;
    }
    assert!(db(prev / inside) < -60.0, "10 km out it is all but gone: {} dB", db(prev / inside));
}

#[test]
fn a_volume_surrounds_from_inside_and_points_at_itself_from_outside() {
    let em = vec![AmbientEmitter {
        shape: AmbientShape::Box { min: Vec3::new(-20.0, -20.0, 0.0), max: Vec3::new(20.0, 20.0, 15.0) },
        level: 1.0,
    }];
    let c = ambient::calibrate(&em, Vec3::new(0.0, 0.0, 7.5), [1.0; NBANDS]);
    let mut buf = Vec::new();
    let mut at = |p: Vec3| ambient::field(&em, c, p, |_, _| [1.0; NBANDS], &mut buf);
    let f = at(Vec3::new(0.0, 0.0, 7.5));
    let (_, r) = f.mean_dir();
    assert!(r < 0.1, "inside the volume: surround, energy vector {r}");
    assert!(f.direct[8][1] > 0.02 * total(&f, 1), "the canopy above sounds too");
    let g = at(Vec3::new(0.0, -300.0, 1.6));
    let (dir, r) = g.mean_dir();
    assert!(dir.y > 0.95 && r > 0.9, "from outside it points at itself: {dir:?} {r}");
    assert!(total(&g, 1) < 0.05, "and is far quieter outside");
}

/// Box room 8 × 6 × 3 (brick, concrete slab roof) at the origin with a
/// doorway in its south wall (y = 0) at x = 4, in a grass field; the
/// meadow of crickets covers the field around it (not under the house).
fn house(door_open: bool) -> (DomeProbe, Vec<omg_scene::dome::Panel>, Vec<AmbientEmitter>) {
    let mut r = room((0.0, 0.0), (8.0, 6.0), 3.0, Material::BRICK);
    r.walls[5] = Material::CONCRETE;
    let rooms = vec![r, outdoor((-100.0, -100.0), (100.0, 100.0), Material::GRASS)];
    let mut d = door((0, 1), (4.0, 0.0), 1);
    d.openness = if door_open { 1.0 } else { 0.0 };
    let doors = vec![d];
    let em = vec![AmbientEmitter {
        shape: AmbientShape::GroundArea {
            min: (-100.0, -100.0),
            max: (100.0, 100.0),
            z: 0.1,
            holes: vec![((0.0, 0.0), (8.0, 6.0))],
        },
        level: 1.0,
    }];
    (DomeProbe::new(&rooms, &doors), door_panels(&doors), em)
}

fn priced(p: &DomeProbe, leaves: &[omg_scene::dome::Panel], em: &[AmbientEmitter], eye: Vec3) -> ShapedField {
    let mut hits = Vec::new();
    let mut buf = Vec::new();
    let c = ambient::calibrate(em, Vec3::new(-40.0, -40.0, 1.6), [1.0; NBANDS]);
    ambient::field(em, c, eye, |a, b| p.transmission(a, b, leaves, &mut hits), &mut buf)
}

#[test]
fn hovering_over_a_roof_the_meadow_below_is_muffled_by_the_slab() {
    let (p, leaves, em) = house(true);
    let over_roof = priced(&p, &leaves, &em, Vec3::new(4.0, 3.0, 4.0));
    let over_grass = priced(&p, &leaves, &em, Vec3::new(-30.0, 3.0, 4.0));
    // the steep cone straight down is the slab's: compare the visible
    // fraction of the free field
    let vis = |f: &ShapedField, b: usize| f.direct_total()[b] / f.free[b];
    println!(
        "visible over roof {:.3}/{:.3}/{:.3}, over grass {:.3}/{:.3}/{:.3}",
        vis(&over_roof, 0),
        vis(&over_roof, 1),
        vis(&over_roof, 2),
        vis(&over_grass, 0),
        vis(&over_grass, 1),
        vis(&over_grass, 2)
    );
    assert!(vis(&over_grass, 1) > 0.97, "open grass: nothing in the way");
    assert!(vis(&over_roof, 1) < 0.9 * vis(&over_grass, 1), "the roof shadows the meadow below");
    assert!(vis(&over_roof, 2) < vis(&over_roof, 0), "and what it passes is darker (mass law)");
}

#[test]
fn indoors_the_direct_share_arrives_through_the_doorway() {
    let (p, leaves, em) = house(true);
    let f = priced(&p, &leaves, &em, Vec3::new(4.0, 4.0, 1.6));
    let open_air = priced(&p, &leaves, &em, Vec3::new(-40.0, -40.0, 1.6));
    let best = (0..DOME_BINS).max_by(|&a, &b| f.direct[a][2].total_cmp(&f.direct[b][2])).unwrap();
    let d = f.dir_sum[best] * (1.0 / f.dir_sum[best].length());
    assert!(d.y < -0.6, "strongest treble arrives from the doorway (south): {d:?}");
    assert!(
        f.direct_total()[2] < 0.2 * open_air.direct_total()[2],
        "indoors only a sliver arrives direct"
    );
    // shut the door: the treble that remains is the shell's mass law
    let (ps, ls, em) = house(false);
    let shut = priced(&ps, &ls, &em, Vec3::new(4.0, 4.0, 1.6));
    assert!(shut.direct_total()[2] < 0.1 * f.direct_total()[2], "a shut leaf blocks the doorway's treble");
}
