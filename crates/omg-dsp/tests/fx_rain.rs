//! Effect: rain drop rendering (rain.rs `Rain`): the recorded-splat
//! bank (`set_bank`) and the per-surface modal drop synthesis, driven
//! through the public API only (Environment → set_environment,
//! set_intensity, process).
//!
//! Claims:
//!  - with a bank loaded, recorded slices are actually played: the
//!    output difference between a silent bank and a tonal bank (same RNG
//!    trajectory, so everything else cancels exactly) is that tone,
//!    pitch-varied ±15 %;
//!  - slices are dulled at load (one-pole ~3 kHz): bright bank content is
//!    played much quieter than low content of the same amplitude;
//!  - bank hits on a pane land AT the pane's direction;
//!  - materials by location: outdoor ground "plats" are short thumps
//!    (nearly all energy in the first 20 ms of each event); drops landing
//!    on window panes (glass) are brighter than drops on a roof overhead
//!    (stone/metal knocks through the structure + drumming).
//!
//! Isolation: synthetic `Environment`s (no scene), deterministic (the
//! engine's RNG has a fixed seed). Not duplicated: ramping, dryness,
//! crest/darkness and roof-gating have unit tests in src/rain.rs; the
//! src materials test re-implements the modal recurrence on private
//! tables — here the real renderer is driven.
//! Combined in the demo: omg-scene src/environment.rs
//! `roof_exposure_respects_storeys` / dome tests produce the
//! Environment this consumes.

mod common;
use common::*;
use omg_dsp::ambi::NCH;
use omg_dsp::env::{EnvRoute, EnvWindow, Environment};
use omg_dsp::rain::{Rain, BANK_SLOT_S};

const SETTLE: usize = 48_000 * 12; // intensity smoother τ = 6 s

fn outdoor() -> Environment {
    let dirs = [[0.0f32, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]];
    Environment {
        routes: dirs
            .iter()
            .enumerate()
            .map(|(k, d)| EnvRoute { id: 100 + k as u32, dir: *d, gains: [1.0; 3], dist: 35.0 })
            .collect(),
        ..Default::default()
    }
}

fn panes(n: usize) -> Environment {
    Environment {
        enclosure: 1.0,
        roof_gain: 0.0,
        windows: vec![EnvWindow { dir: [1.0, 0.0, 0.0], gain: 1.0 }; n],
        ..Default::default()
    }
}

fn roof() -> Environment {
    Environment { enclosure: 1.0, roof_gain: 1.0, ..Default::default() }
}

/// 8 slices of a tone (whole-slice, amplitude `amp`).
fn tone_bank(freq: f32, amp: f32) -> Vec<f32> {
    let slot = (BANK_SLOT_S * SR).round() as usize;
    (0..8 * slot).map(|k| amp * sine_at(freq, k % slot)).collect()
}

/// Settle, then return `n` bus frames.
fn render(env: &Environment, intensity: f32, bank: Option<Vec<f32>>, n: usize) -> Vec<[f32; NCH]> {
    let mut r = Rain::new(SR);
    if let Some(b) = bank {
        r.set_bank(b);
    }
    r.set_environment(env);
    r.set_intensity(intensity);
    for _ in 0..SETTLE {
        r.process(&mut [0.0f32; NCH]);
    }
    (0..n)
        .map(|_| {
            let mut b = [0.0f32; NCH];
            r.process(&mut b);
            b
        })
        .collect()
}

fn w(frames: &[[f32; NCH]]) -> Vec<f32> {
    frames.iter().map(|f| f[0]).collect()
}

/// Contribution of the bank to the output: tonal bank minus silent bank.
fn bank_part(env: &Environment, freq: f32) -> Vec<[f32; NCH]> {
    let a = render(env, 1.0, Some(tone_bank(freq, 0.0)), 48_000);
    let b = render(env, 1.0, Some(tone_bank(freq, 0.5)), 48_000);
    a.iter().zip(&b).map(|(x, y)| core::array::from_fn(|k| y[k] - x[k])).collect()
}

#[test]
fn bank_slices_are_played() {
    let env = outdoor();
    let diff = w(&bank_part(&env, 300.0));
    let total = w(&render(&env, 1.0, Some(tone_bank(300.0, 0.5)), 48_000));
    assert!(rms(&diff) > 0.05 * rms(&total), "bank hits audible: {} vs {}", rms(&diff), rms(&total));
    // the bank's contribution IS the tone (rate 0.85–1.15 ⇒ 255–345 Hz)
    let inband = energy(&Biquad::bandpass(300.0, 1.0).run(&diff));
    assert!(inband > 0.6 * energy(&diff), "bank part off-tone: {:.2}", inband / energy(&diff));
    // and loading a bank changes the drop mix at all (vs pure synthesis)
    let none = w(&render(&env, 1.0, None, 48_000));
    let silent = w(&render(&env, 1.0, Some(tone_bank(300.0, 0.0)), 48_000));
    assert!(none.iter().zip(&silent).any(|(a, b)| a != b));
}

#[test]
fn bank_is_dulled_at_load() {
    let env = outdoor();
    let low = energy(&w(&bank_part(&env, 300.0)));
    let high = energy(&w(&bank_part(&env, 9000.0)));
    // one-pole 3 kHz: |H(9 kHz)| ≈ 0.3 → energy ≈ 0.1×
    assert!(high < 0.25 * low, "bright bank content must be dulled: {high:.3e} vs {low:.3e}");
}

#[test]
fn bank_hits_land_on_the_pane() {
    let diff = bank_part(&panes(1), 300.0);
    let ew: f64 = diff.iter().map(|f| (f[0] as f64).powi(2)).sum();
    let ex: f64 = diff.iter().map(|f| f[0] as f64 * f[3] as f64).sum();
    assert!(ew > 0.0, "indoor window bank hits must exist");
    assert!(ex / ew > 0.7, "bank hits should come from the pane (+x): {:.2}", ex / ew);
}

/// Outdoors below the hiss knee only ground plats sound: every event
/// is a short thump. The output is bit-exact zero between events, so
/// each non-zero run is one event (or an occasional overlap of two).
/// Typical (median) event: ≥ 98 % of its energy in the first 20 ms, and
/// rung out (to the drop kill floor) within 100 ms.
#[test]
fn ground_plats_are_short_thumps() {
    let y = w(&render(&outdoor(), 0.2, None, 48_000 * 6));
    let mut runs: Vec<(usize, f64)> = Vec::new(); // (length, head energy fraction)
    let mut k = 0;
    while k < y.len() {
        if y[k] == 0.0 {
            k += 1;
            continue;
        }
        let start = k;
        while k < y.len() && y[k] != 0.0 {
            k += 1;
        }
        let run = &y[start..k];
        runs.push((run.len(), energy(&run[..run.len().min(960)]) / energy(run)));
    }
    assert!(runs.len() >= 10, "need drizzle events: {}", runs.len());
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let head = median(runs.iter().map(|r| r.1).collect());
    let len_ms = median(runs.iter().map(|r| r.0 as f64).collect()) / 48.0;
    assert!(head > 0.98, "plats must be dead within 20 ms: median head fraction {head:.3}");
    assert!(len_ms < 100.0, "plat lifetime {len_ms:.0} ms");
}

/// Panes tick brighter than the roof (structure-filtered stone + metal
/// knocks over the low drumming bed). Brightness = first-difference
/// energy / energy (energy-weighted, so quiet ringing tails don't count).
#[test]
fn panes_are_brighter_than_the_roof() {
    let bright = |x: Vec<f32>| {
        let d: Vec<f32> = x.windows(2).map(|p| p[1] - p[0]).collect();
        energy(&d) / energy(&x)
    };
    let glass = bright(w(&render(&panes(4), 0.45, None, 48_000 * 2)));
    let stone = bright(w(&render(&roof(), 0.45, None, 48_000 * 2)));
    assert!(glass > 2.0 * stone, "panes {glass:.4} vs roof {stone:.4}");
}
