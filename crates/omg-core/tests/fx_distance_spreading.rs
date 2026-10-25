//! Effect: geometric (spherical) spreading of the DIRECT path + its
//! propagation delay, in isolation.
//!
//! Physical claim guaranteed: a point source's direct-path amplitude
//! falls as 1/r (−6.02 dB per doubling of distance, before air
//! absorption), it arrives r/c seconds late, from the source's direction,
//! and the near-field clamp (0.3 m) keeps the gain finite as r → 0.
//! Both early-field solvers — `ism::image_source_taps` and the PT
//! `pt::record_for` — must obey the same law exactly.
//!
//! Combined in the demo: every source's direct tap in
//! `omg_scene::world::WorldSim` (walkthrough approach/retreat tests)
//! rides on this law together with air, occlusion and transmission.

mod common;

use common::{db, v};
use omg_core::ism::image_source_taps;
use omg_core::material::{air_attenuation, Material};
use omg_core::params::Tap;
use omg_core::pt::record_for;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};

/// A room so large its walls are irrelevant; order 0 = direct only.
fn big_room() -> Shoebox {
    Shoebox::new(v(100.0, 100.0, 100.0), [Material::CONCRETE; 6])
}

const LIS: Vec3 = Vec3::new(50.0, 50.0, 50.0);

fn direct_tap(src: Vec3) -> Tap {
    let mut taps = Vec::new();
    image_source_taps(&big_room(), src, LIS, 0, &mut taps);
    assert_eq!(taps.len(), 1, "order 0 must yield exactly the direct path");
    taps[0]
}

/// Source at distance d from the listener along a fixed oblique axis.
fn src_at(d: f32) -> Vec3 {
    let axis = v(0.6, -0.48, 0.64).normalize();
    LIS + axis * d
}

#[test]
fn direct_gain_is_exactly_air_over_r() {
    for d in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        let t = direct_tap(src_at(d));
        let air = air_attenuation(d);
        for b in 0..NBANDS {
            let want = air[b] / d;
            let rel = (t.gains[b] - want).abs() / want;
            assert!(rel < 1e-4, "d={d} band {b}: gain {} want {want}", t.gains[b]);
        }
    }
}

#[test]
fn minus_six_db_per_doubling() {
    let ds = [1.0f32, 2.0, 4.0, 8.0];
    let taps: Vec<Tap> = ds.iter().map(|&d| direct_tap(src_at(d))).collect();
    for i in 1..ds.len() {
        for b in 0..NBANDS {
            // pure spreading: divide out air absorption first
            let a0 = taps[i - 1].gains[b] / air_attenuation(ds[i - 1])[b];
            let a1 = taps[i].gains[b] / air_attenuation(ds[i])[b];
            let step = db(a1) - db(a0);
            assert!(
                (step - (-6.0206)).abs() < 0.01,
                "band {b} {}→{} m: spreading step {step:.3} dB, want −6.02",
                ds[i - 1],
                ds[i]
            );
        }
        // and raw (with air) the low band is still −6.02 ± 0.02 dB at
        // these room-scale distances — air is negligible below 250 Hz
        let raw = db(taps[i].gains[0]) - db(taps[i - 1].gains[0]);
        assert!((raw + 6.0206).abs() < 0.02, "raw low-band step {raw:.3} dB");
    }
}

#[test]
fn delay_is_distance_over_c_and_direction_points_at_source() {
    for d in [1.0f32, 2.0, 4.0, 8.0, 40.0] {
        let src = src_at(d);
        let t = direct_tap(src);
        let want = d / SPEED_OF_SOUND;
        assert!((t.delay_s - want).abs() < 2e-6, "d={d}: delay {} want {want}", t.delay_s);
        let to_src = (src - LIS).normalize();
        let dot = t.dir[0] * to_src.x + t.dir[1] * to_src.y + t.dir[2] * to_src.z;
        assert!(dot > 0.99999, "d={d}: dir {:?} not toward source", t.dir);
    }
    // axis-aligned 49 m: 142.857 ms
    let t = direct_tap(LIS + v(0.0, 0.0, 49.0));
    assert!((t.delay_s - 0.142857).abs() < 2e-6, "49 m delay {}", t.delay_s);
}

#[test]
fn near_field_clamp_keeps_gain_finite() {
    // inside 0.3 m the spreading is clamped: no 1/r singularity
    let clamp = direct_tap(src_at(0.3)).gains;
    for d in [0.0f32, 0.01, 0.1, 0.29] {
        let g = direct_tap(src_at(d)).gains;
        for b in 0..NBANDS {
            assert!(g[b].is_finite(), "d={d}: non-finite gain");
            assert!((g[b] - clamp[b]).abs() < 1e-4 * clamp[b], "d={d} band {b}: {} vs clamp {}", g[b], clamp[b]);
        }
    }
    assert!((clamp[0] - 1.0 / 0.3).abs() < 1e-3, "clamp gain {} want 1/0.3", clamp[0]);
    // just outside the clamp the 1/r law resumes
    let g1 = direct_tap(src_at(0.6)).gains[0];
    assert!((db(g1) - db(clamp[0]) + 6.0206).abs() < 0.02);
}

#[test]
fn pt_direct_record_obeys_the_same_law() {
    let room = big_room();
    for d in [1.0f32, 2.0, 4.0, 8.0] {
        let src = src_at(d);
        let ism = direct_tap(src);
        let pt = record_for(&room, &[], 0, src, LIS).expect("direct path must exist");
        assert!((pt.delay_s - ism.delay_s).abs() < 1e-6, "d={d}: PT delay {} ISM {}", pt.delay_s, ism.delay_s);
        for b in 0..NBANDS {
            let rel = (pt.gains[b] - ism.gains[b]).abs() / ism.gains[b];
            assert!(rel < 1e-4, "d={d} band {b}: PT {} ISM {}", pt.gains[b], ism.gains[b]);
        }
    }
}
