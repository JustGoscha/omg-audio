//! Effect: OUTDOOR GROUND REFLECTION — `Sim::update_outdoor`, tap key 1.
//!
//! Physical claim guaranteed: in open air the listener hears, besides the
//! direct path, one image source mirrored in the ground plane z = 0. Its
//! extra delay is exactly the image-source path difference
//! (|S' − L| − |S − L|)/c (→ 2·h_s·h_l/(c·d) at range), it arrives from
//! BELOW, and its per-band gain is the direct law (air · 1/d) times the
//! ground material's amplitude reflection √(1 − α) — so a hard ground
//! reflects more than grass, highs suffer most. The pre-room path of a
//! routed source (extra_dist, muffle) shifts both taps alike. No reverb.
//!
//! Combined in the demo: every outdoor source state in `WorldSim`'s ISM
//! pipeline (`ism_source_block` → `update_outdoor(&Material::GRASS, …)`),
//! e.g. the street/field listeners of `world.rs` unit tests and
//! `tests/wildlife.rs` (traced mode builds its own ground in the mesh).

mod common;

use common::*;
use omg_core::material::{air_attenuation, Material};
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};
use omg_scene::sim::Sim;

fn outdoor(ground: &Material, src: Vec3, lis: Vec3, yaw: f32, extra: f32, muffle: [f32; NBANDS]) -> Vec<omg_core::params::Tap> {
    let pb = Sim::new().update_outdoor(ground, &[], src, lis, yaw, extra, muffle);
    assert_eq!(pb.reverb.level, [0.0; NBANDS], "open air has no enclosing reverb");
    pb.taps
}

#[test]
fn ground_image_has_exact_delay_direction_and_gain() {
    let src = Vec3::new(0.0, 0.0, 1.5);
    let lis = Vec3::new(10.0, 0.0, 1.6);
    let taps = outdoor(&Material::GRASS, src, lis, 0.0, 0.0, [1.0; NBANDS]);
    assert_eq!(taps.len(), 2, "direct + ground, nothing else without facades");
    let d = tap(&taps, 0).expect("direct tap");
    let g = tap(&taps, 1).expect("ground tap");

    let d0 = (src - lis).length();
    let d1 = (Vec3::new(src.x, src.y, -src.z) - lis).length();
    assert_rel(d.delay_s, d0 / SPEED_OF_SOUND, 1e-6, "direct delay");
    assert_rel(g.delay_s, d1 / SPEED_OF_SOUND, 1e-6, "ground delay");
    assert_rel(g.delay_s - d.delay_s, (d1 - d0) / SPEED_OF_SOUND, 1e-4, "image extra delay");

    // arrives from below, along the image direction
    assert!(g.dir[2] < -0.25, "ground tap must arrive from below: {:?}", g.dir);
    assert_rel(g.dir[2], -(src.z + lis.z) / d1, 1e-5, "ground elevation");
    assert!(d.dir[2].abs() < 0.02, "direct tap is level: {:?}", d.dir);

    // gain = air(d1)/d1 · √(1 − α) per band; direct = air(d0)/d0
    let refl = Material::GRASS.reflection_amplitude();
    let (a0, a1) = (air_attenuation(d0), air_attenuation(d1));
    for b in 0..NBANDS {
        assert_rel(d.gains[b], a0[b] / d0, 1e-5, "direct gain");
        assert_rel(g.gains[b], a1[b] / d1 * refl[b], 1e-5, "ground gain");
        assert!(g.gains[b] < d.gains[b], "a reflection is never louder than the direct path");
    }
    // grass absorbs highs: the ground tap is darker than the direct one
    assert!(g.gains[2] / d.gains[2] < g.gains[0] / d.gains[0]);
}

#[test]
fn ground_gain_scales_with_material() {
    let src = Vec3::new(0.0, 0.0, 1.5);
    let lis = Vec3::new(10.0, 3.0, 1.6);
    let grass = outdoor(&Material::GRASS, src, lis, 0.0, 0.0, [1.0; NBANDS]);
    let conc = outdoor(&Material::CONCRETE, src, lis, 0.0, 0.0, [1.0; NBANDS]);
    let (rg, rc) = (Material::GRASS.reflection_amplitude(), Material::CONCRETE.reflection_amplitude());
    for b in 0..NBANDS {
        // the direct path does not care what the ground is
        assert_eq!(tap(&grass, 0).unwrap().gains[b], tap(&conc, 0).unwrap().gains[b]);
        let ratio = tap(&conc, 1).unwrap().gains[b] / tap(&grass, 1).unwrap().gains[b];
        assert_rel(ratio, rc[b] / rg[b], 1e-5, "ground gain ratio = reflection ratio");
        assert!(ratio > 1.0, "hard ground reflects more than grass (band {b})");
    }
    // the delay is geometry only
    assert_eq!(tap(&grass, 1).unwrap().delay_s, tap(&conc, 1).unwrap().delay_s);
}

#[test]
fn ground_delay_follows_image_geometry() {
    // far field: Δ ≈ 2·h_s·h_l / d
    let lis = Vec3::new(100.0, 0.0, 1.6);
    let taps = outdoor(&Material::GRASS, Vec3::new(0.0, 0.0, 1.5), lis, 0.0, 0.0, [1.0; NBANDS]);
    let dt = tap(&taps, 1).unwrap().delay_s - tap(&taps, 0).unwrap().delay_s;
    let approx = 2.0 * 1.5 * 1.6 / 100.0 / SPEED_OF_SOUND;
    assert_rel(dt, approx, 0.03, "far-field path difference");

    // the higher the source, the later the bounce; at grade it merges
    let mut prev = -1.0f32;
    for h in [0.01f32, 0.5, 1.5, 3.0, 6.0] {
        let t = outdoor(&Material::GRASS, Vec3::new(0.0, 0.0, h), Vec3::new(10.0, 0.0, 1.6), 0.0, 0.0, [1.0; NBANDS]);
        let dt = tap(&t, 1).unwrap().delay_s - tap(&t, 0).unwrap().delay_s;
        assert!(dt > prev, "extra delay must grow with source height: {dt} after {prev}");
        prev = dt;
    }
    let t = outdoor(&Material::GRASS, Vec3::new(0.0, 0.0, 0.01), Vec3::new(10.0, 0.0, 0.01), 0.0, 0.0, [1.0; NBANDS]);
    let dt = tap(&t, 1).unwrap().delay_s - tap(&t, 0).unwrap().delay_s;
    assert!(dt * 48_000.0 < 0.01, "both at grade: the bounce coincides with the direct path ({dt} s)");
}

#[test]
fn routed_path_and_yaw_apply_to_both_taps() {
    let src = Vec3::new(0.0, 0.0, 1.5);
    let lis = Vec3::new(10.0, 0.0, 1.6);
    let base = outdoor(&Material::GRASS, src, lis, 0.0, 0.0, [1.0; NBANDS]);
    let muffle = [0.8, 0.5, 0.2];
    let extra = 7.0;
    let routed = outdoor(&Material::GRASS, src, lis, 0.0, extra, muffle);
    for key in [0u32, 1] {
        let (b0, r0) = (tap(&base, key).unwrap(), tap(&routed, key).unwrap());
        assert_rel(r0.delay_s - b0.delay_s, extra / SPEED_OF_SOUND, 1e-4, "pre-path delay");
        let d = b0.delay_s * SPEED_OF_SOUND;
        let (ab, ar) = (air_attenuation(d), air_attenuation(d + extra));
        for b in 0..NBANDS {
            // total path law: muffle · air(d + extra) / (d + extra) · refl
            let expect = b0.gains[b] * muffle[b] * (ar[b] / ab[b]) * d / (d + extra);
            assert_rel(r0.gains[b], expect, 1e-4, "routed gain");
        }
    }
    // yaw: facing +y (π/2), a world direction (x, y) reads as (y, −x)
    // in the listener frame (+x forward, +y left)
    let lis2 = Vec3::new(10.0, 3.0, 1.6);
    let base = outdoor(&Material::GRASS, src, lis2, 0.0, 0.0, [1.0; NBANDS]);
    let turned = outdoor(&Material::GRASS, src, lis2, core::f32::consts::FRAC_PI_2, 0.0, [1.0; NBANDS]);
    for key in [0u32, 1] {
        let (b0, t0) = (tap(&base, key).unwrap(), tap(&turned, key).unwrap());
        assert!(b0.dir[1].abs() > 0.2, "probe must have a lateral component");
        assert!((t0.dir[0] - b0.dir[1]).abs() < 1e-5 && (t0.dir[1] + b0.dir[0]).abs() < 1e-5,
            "yaw rotation: {:?} -> {:?}", b0.dir, t0.dir);
        assert_eq!(t0.dir[2], b0.dir[2], "yaw leaves elevation alone");
        assert_eq!(t0.gains, b0.gains);
    }
}
