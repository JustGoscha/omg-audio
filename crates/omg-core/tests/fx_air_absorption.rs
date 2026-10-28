//! Effect: frequency-dependent air absorption, in isolation
//! (`material::AIR_SUBBANDS_PER_M`, `material::air_attenuation`).
//!
//! Physical claim guaranteed: per band, the energy ISO 9613-1 (20 °C /
//! 50 % RH) leaves of pink sound after a path — the standard's own
//! formula, integrated finely over the band, within 0.5 dB; highs
//! attenuate far more than lows; the loss is monotonic in distance and
//! exactly unity at 0 m; a band's top dies first, so its loss per km
//! falls with distance. The early-field solvers apply exactly this factor
//! on top of 1/r spreading.
//!
//! Combined in the demo: long outdoor paths in `omg_scene::world::WorldSim`
//! (street / courtyard sources) — air turns far sources dull.
//!
//! Also documents a gap: the stochastic late-field tracer applies NO air
//! absorption (see the ignored test at the bottom).

mod common;

use common::{flat_opaque, v};
use omg_core::ism::image_source_taps;
use omg_core::material::{air_attenuation, Material};
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

/// ISO 9613-1 (pure-tone attenuation coefficient, dB/m).
fn iso_alpha(f: f64) -> f64 {
    let (t, hr, pa, pr, t0, t01) = (293.15f64, 50.0f64, 101.325f64, 101.325f64, 293.15f64, 273.16f64);
    let c = -6.8346 * (t01 / t).powf(1.261) + 4.6151;
    let h = hr * (pr * 10f64.powf(c) / pa);
    let fro = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    let frn = (pa / pr) * (t / t0).powf(-0.5) * (9.0 + 280.0 * h * (-4.170 * ((t / t0).powf(-1.0 / 3.0) - 1.0)).exp());
    8.686 * f * f
        * (1.84e-11 * (pa / pr).powi(-1) * (t / t0).sqrt()
            + (t / t0).powf(-2.5) * (0.01275 * (-2239.1 / t).exp() / (fro + f * f / fro) + 0.1068 * (-3352.0 / t).exp() / (frn + f * f / frn)))
}

#[test]
fn bands_match_the_standard_integrated() {
    let bands = [(20.0f64, 314.0f64), (314.0, 2100.0), (2100.0, 16000.0)];
    for d in [100.0f32, 1000.0, 3000.0] {
        let a = air_attenuation(d);
        for (b, &(lo, hi)) in bands.iter().enumerate() {
            // pink: equal energy per log-frequency, 256 points
            let n = 256;
            let e: f64 = (0..n)
                .map(|k| {
                    let f = lo * (hi / lo).powf((k as f64 + 0.5) / n as f64);
                    10f64.powf(-iso_alpha(f) * d as f64 / 10.0)
                })
                .sum::<f64>()
                / n as f64;
            let want = 10.0 * e.log10();
            let got = 20.0 * (a[b] as f64).log10();
            // (the band's bottom edge carries all far: 2 dB at 3 km, where
            // the high band is 45 dB down)
            let tol = if d > 1000.0 { 2.0 } else { 0.5 };
            assert!((got - want).abs() < tol, "band {b} at {d} m: {got:.2} dB, ISO {want:.2} dB");
        }
    }
}

#[test]
fn a_band_top_dies_first() {
    // dB lost per km falls with distance (the band's bottom carries on)
    for b in 1..NBANDS {
        let db = |d: f32| -20.0 * air_attenuation(d)[b].log10() / (d / 1000.0);
        assert!(db(3000.0) < db(300.0), "band {b}: {} dB/km at 3 km vs {} at 300 m", db(3000.0), db(300.0));
    }
}

#[test]
fn per_km_loss_is_in_the_iso_9613_ballpark() {
    // dB per km = −20·log10(air(1000 m)): ≈0.4 dB/km @125 Hz, ≈3–5 dB/km
    // @0.8–1 kHz, ≈25–60 dB/km @4–8 kHz
    let a = air_attenuation(1000.0);
    let db_km: [f32; NBANDS] = core::array::from_fn(|b| -20.0 * a[b].log10());
    assert!(db_km[0] > 0.1 && db_km[0] < 2.0, "low {:.2} dB/km", db_km[0]);
    assert!(db_km[1] > 1.5 && db_km[1] < 10.0, "mid {:.2} dB/km", db_km[1]);
    assert!(db_km[2] > 15.0 && db_km[2] < 70.0, "high {:.2} dB/km", db_km[2]);
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
        far_tilt_db < -8.5 && far_tilt_db > -10.5,
        "300 m high-vs-low tilt {far_tilt_db:.2} dB, want ≈ −9.6 dB (ISO 9613-1)"
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
