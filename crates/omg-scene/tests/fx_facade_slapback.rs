//! Effect: FACADE SLAP-BACK — `Sim::update_outdoor` with `Facade`s, tap
//! keys 10 + facade index.
//!
//! Physical claim guaranteed: a building wall (axis-aligned vertical
//! plane with a finite lateral extent and per-band reflection) adds ONE
//! first-order specular reflection exactly where the image-source method
//! says it exists: source and listener on the same side, and the
//! reflection point (listener → mirrored source ∩ plane) inside the
//! wall's extent. The tap's delay is |S' − L|/c, it arrives from the
//! wall's direction, its gain is air(d)·refl/d. No facade, opposite
//! sides, or a reflection point off the end of the wall ⇒ no tap.
//!
//! Combined in the demo: `WorldSim`'s outdoor ISM path passes the scene's
//! facades (built from the demo buildings in `world.rs`) to every
//! outdoor source — heard on the square between the Hall, Club and Old
//! House (`world.rs` unit tests, `doorway_contrast_walking_past`).

mod common;

use common::*;
use omg_core::material::{air_attenuation, Material};
use omg_core::params::Tap;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};
use omg_scene::sim::{Facade, Sim};

const REFL: [f32; NBANDS] = [0.9, 0.7, 0.5];

fn wall_x(plane: f32, lo: f32, hi: f32) -> Facade {
    Facade { axis: 0, plane, lo, hi, refl: REFL }
}

fn taps(facades: &[Facade], src: Vec3, lis: Vec3) -> Vec<Tap> {
    Sim::new().update_outdoor(&Material::GRASS, facades, src, lis, 0.0, 0.0, [1.0; NBANDS]).taps
}

#[test]
fn facade_adds_one_exact_image_reflection() {
    let src = Vec3::new(5.0, 0.0, 1.5);
    let lis = Vec3::new(10.0, 2.0, 1.6);
    let t = taps(&[wall_x(20.0, -10.0, 10.0)], src, lis);
    assert_eq!(t.len(), 3, "direct + ground + one slap-back");
    let s = tap(&t, 10).expect("slap-back tap, key 10");

    let img = Vec3::new(2.0 * 20.0 - src.x, src.y, src.z);
    let d = (img - lis).length();
    assert_rel(s.delay_s, d / SPEED_OF_SOUND, 1e-6, "slap-back delay = |S' − L|/c");
    // equivalently source → wall → listener, and later than the direct path
    let direct = tap(&t, 0).unwrap();
    assert!(s.delay_s > direct.delay_s + 0.05, "a 20 m-away wall slaps back ≥ 50 ms late");
    // from the wall (+x), along the image direction
    let dir = (img - lis).normalize();
    for k in 0..3 {
        assert!((s.dir[k] - [dir.x, dir.y, dir.z][k]).abs() < 1e-5, "direction {:?} vs {dir:?}", s.dir);
    }
    assert!(s.dir[0] > 0.95);
    let air = air_attenuation(d);
    for b in 0..NBANDS {
        assert_rel(s.gains[b], air[b] / d * REFL[b], 1e-5, "slap-back gain");
    }
}

#[test]
fn facade_along_y_mirrors_y() {
    let src = Vec3::new(0.0, 3.0, 1.5);
    let lis = Vec3::new(4.0, 6.0, 1.6);
    let f = Facade { axis: 1, plane: 12.0, lo: -20.0, hi: 20.0, refl: REFL };
    let t = taps(&[f], src, lis);
    let s = tap(&t, 10).expect("slap-back off the y-wall");
    let img = Vec3::new(src.x, 24.0 - src.y, src.z);
    assert_rel(s.delay_s, (img - lis).length() / SPEED_OF_SOUND, 1e-6, "y-wall delay");
    assert!(s.dir[1] > 0.9, "arrives from +y: {:?}", s.dir);
}

#[test]
fn no_facade_or_invalid_geometry_means_no_slap_back() {
    let src = Vec3::new(5.0, 0.0, 1.5);
    let lis = Vec3::new(10.0, 2.0, 1.6);
    // absent
    let t = taps(&[], src, lis);
    assert!(t.iter().all(|t| t.key < 10), "no facade, no slap-back");
    // listener on the far side of the wall
    let t = taps(&[wall_x(20.0, -10.0, 10.0)], src, Vec3::new(25.0, 2.0, 1.6));
    assert!(tap(&t, 10).is_none(), "opposite sides: the wall cannot mirror");
    // reflection point at y = 1.2 (t = 0.4 along L→S'): a wall spanning
    // only y ∈ [3, 10] misses it, one spanning [1, 1.5] catches it
    let t = taps(&[wall_x(20.0, 3.0, 10.0)], src, lis);
    assert!(tap(&t, 10).is_none(), "reflection point beyond the wall's end");
    let t = taps(&[wall_x(20.0, 1.0, 1.5)], src, lis);
    assert!(tap(&t, 10).is_some(), "a narrow wall right at the reflection point reflects");
}

#[test]
fn several_facades_key_by_index() {
    let src = Vec3::new(5.0, 5.0, 1.5);
    let lis = Vec3::new(8.0, 6.0, 1.6);
    let walls = [
        wall_x(20.0, -50.0, 50.0),
        Facade { axis: 1, plane: 30.0, lo: -50.0, hi: 50.0, refl: REFL },
        wall_x(-30.0, 3.0, 4.0), // misses: reflection point near y ≈ 5.x
    ];
    let t = taps(&walls, src, lis);
    assert!(tap(&t, 10).is_some() && tap(&t, 11).is_some());
    assert!(tap(&t, 12).is_none());
    assert_eq!(t.len(), 4);
}
