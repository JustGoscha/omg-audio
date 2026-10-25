//! Effect: frequency-dependent air absorption, in isolation
//! (`material::AIR_ABSORPTION_PER_M`, `material::air_attenuation`).
//!
//! Physical claim guaranteed: amplitude decays as exp(−a_b·d) per band;
//! highs attenuate far more than lows; the loss is monotonic in distance,
//! exactly unity at 0 m, multiplicative over path segments
//! (air(a+b) = air(a)·air(b)), and its per-km magnitudes are in the
//! ISO 9613-1 ballpark (20 °C / 50 % RH) for the band centres
//! 125 Hz / 790 Hz / 5.6 kHz. The early-field solvers apply exactly this
//! factor on top of 1/r spreading.
//!
//! Combined in the demo: long outdoor paths in `omg_scene::world::WorldSim`
//! (street / courtyard sources) — air turns far sources dull.
//!
//! Also documents a gap: the stochastic late-field tracer applies NO air
//! absorption (see the ignored test at the bottom).

mod common;

use common::{flat_opaque, v};
use omg_core::ism::image_source_taps;
use omg_core::material::{air_attenuation, Material, AIR_ABSORPTION_PER_M};
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::NBANDS;

#[test]
fn unity_at_zero_distance() {
    let a = air_attenuation(0.0);
    for b in 0..NBANDS {
        assert_eq!(a[b], 1.0, "band {b} at 0 m");
    }
}

#[test]
fn highs_attenuate_more_than_lows() {
    for d in [1.0f32, 10.0, 100.0, 300.0, 1000.0] {
        let a = air_attenuation(d);
        assert!(a[0] > a[1] && a[1] > a[2], "d={d}: {a:?} not low > mid > high");
    }
    // and the coefficient table itself is ordered
    assert!(AIR_ABSORPTION_PER_M[0] < AIR_ABSORPTION_PER_M[1]);
    assert!(AIR_ABSORPTION_PER_M[1] < AIR_ABSORPTION_PER_M[2]);
}

#[test]
fn monotonic_in_distance() {
    let mut prev = air_attenuation(0.0);
    for i in 1..=200 {
        let d = i as f32 * 5.0; // 5 m .. 1 km
        let a = air_attenuation(d);
        for b in 0..NBANDS {
            assert!(a[b] < prev[b], "band {b}: not strictly decreasing at {d} m");
            assert!(a[b] > 0.0 && a[b] <= 1.0);
        }
        prev = a;
    }
}

#[test]
fn magnitude_at_300m_matches_formula() {
    let a = air_attenuation(300.0);
    // closed form, written out independently of the constant table:
    // exp(-1e-4·300), exp(-5e-4·300), exp(-3e-3·300)
    let want = [0.970_446f32, 0.860_708, 0.406_570];
    for b in 0..NBANDS {
        assert!((a[b] - want[b]).abs() < 1e-4, "band {b}: {} want {}", a[b], want[b]);
        let formula = (-AIR_ABSORPTION_PER_M[b] * 300.0).exp();
        assert!((a[b] - formula).abs() < 1e-6);
    }
}

#[test]
fn per_km_loss_is_in_the_iso_9613_ballpark() {
    // dB per km = −20·log10(air(1000 m))
    let a = air_attenuation(1000.0);
    let db_km: [f32; NBANDS] = core::array::from_fn(|b| -20.0 * a[b].log10());
    // ISO 9613-1, 20 °C / 50 % RH: ≈0.4 dB/km @125 Hz, ≈3–5 dB/km
    // @0.8–1 kHz, ≈25–60 dB/km @4–8 kHz. Generous brackets: this is a
    // three-band fit, the claim is "right order of magnitude per band".
    assert!(db_km[0] > 0.1 && db_km[0] < 2.0, "low {:.2} dB/km", db_km[0]);
    assert!(db_km[1] > 1.5 && db_km[1] < 10.0, "mid {:.2} dB/km", db_km[1]);
    assert!(db_km[2] > 15.0 && db_km[2] < 70.0, "high {:.2} dB/km", db_km[2]);
}

#[test]
fn multiplicative_over_segments() {
    for (x, y) in [(3.0f32, 7.0f32), (50.0, 250.0), (0.5, 999.5)] {
        let (ax, ay, axy) = (air_attenuation(x), air_attenuation(y), air_attenuation(x + y));
        for b in 0..NBANDS {
            assert!((ax[b] * ay[b] - axy[b]).abs() < 1e-5 * axy[b].max(1e-3), "band {b}: {x}+{y}");
        }
    }
}

#[test]
fn far_direct_tap_carries_the_air_tilt() {
    // the ISM direct tap at 300 m: after dividing out 1/r, the spectral
    // tilt (high/low) is exactly the air ratio — a far source is dull
    let room = Shoebox::new(v(1000.0, 1000.0, 1000.0), [Material::CONCRETE; 6]);
    let lis = v(500.0, 500.0, 500.0);
    let mut taps = Vec::new();
    for d in [10.0f32, 300.0] {
        image_source_taps(&room, v(500.0 + d, 500.0, 500.0), lis, 0, &mut taps);
        let g = taps[0].gains;
        let air = air_attenuation(d);
        let tilt = g[2] / g[0];
        assert!((tilt - air[2] / air[0]).abs() < 1e-4, "d={d}: tilt {tilt}");
    }
    image_source_taps(&room, v(800.0, 500.0, 500.0), lis, 0, &mut taps);
    let far_tilt_db = 20.0 * (taps[0].gains[2] / taps[0].gains[0]).log10();
    assert!(
        far_tilt_db < -7.0 && far_tilt_db > -9.0,
        "300 m high-vs-low tilt {far_tilt_db:.2} dB, want ≈ −7.6 dB"
    );
}

/// The late field travels HUNDREDS of metres (a 2.5 s tail ≈ 850 m of
/// path): air absorption is what makes large live halls' tails darker
/// (Sabine's 4mV term). With frequency-FLAT wall absorption the high-band
/// RT60 must therefore come out clearly shorter than the low band:
/// expected ≈ 0.73× here (air alone limits RT60_high to ≈ 6.7 s).
#[test]
#[ignore = "BUG: tracer::trace applies no air absorption along ray paths — late-field RT60 is frequency-flat in a flat-absorption hall"]
fn late_field_high_band_decays_faster_from_air() {
    let room = Shoebox::new(v(20.0, 15.0, 10.0), [flat_opaque(0.15, 0.3); 6]);
    let mut rng = Rng::new(42);
    let mut e = Echogram::new();
    trace(&room, v(5.0, 5.0, 3.0), v(14.0, 9.0, 4.0), 20_000, [1.0; 3], &mut rng, &mut e);
    let rt = estimate_reverb(&e).rt60;
    let ratio = rt[2] / rt[0];
    assert!(ratio < 0.85, "rt60 {rt:?}: high/low = {ratio:.3}, air should darken the tail");
}
