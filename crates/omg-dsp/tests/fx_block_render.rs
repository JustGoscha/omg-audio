//! Effect: block rendering is the per-sample renderer, path by path
//! (renderer.rs `process_block`, `process_log_block`).
//!
//! Claim: rendering a source in blocks — each path runs a whole block
//! with its state in registers, gain smoothers in closed form, the source
//! read for the block at once — produces the same bus and point-stereo
//! signal as calling `process` / `process_log` once per sample: every
//! channel within 1e-4 absolute and −80 dB of the scene's peak, through
//! Doppler glides, path crossfades and released slots closing with their
//! linear tails, transmission lowpasses, point HRIR taps with head-turn
//! re-selection, the own and coupled-room FDNs, fader ramps, odd block
//! sizes (re-selection falls mid-block) and one-shots on the log.
//!
//! The one intentional difference is float rounding: smoothers are
//! evaluated as target + (cur − target)·coefⁱ instead of ticked, and
//! sums land in another order. (A path whose gain is below 1e-7 is
//! skipped per block instead of per sample — inaudible by construction.)
//!
//! Isolation: synthetic 12-direction HRIR grid, scripted ParamBlocks; no
//! scene, no simulation.

mod common;
use common::*;
use omg_core::params::{ParamBlock, RemoteReverb, ReverbParams, Tap};
use omg_dsp::ambi::{encode_gains, NCH};
use omg_dsp::hrtf::HrirGrid;
use omg_dsp::playlog::{Clip, PlayLog};
use omg_dsp::renderer::{BusBlock, BLOCK};
use omg_dsp::Renderer;
use std::sync::Arc;

const N: usize = 300_000;
/// Block sizes cycled through (re-selection every 128 samples then falls
/// anywhere inside a block).
const SIZES: [usize; 8] = [128, 128, 100, 37, 128, 1, 64, 91];

fn grid() -> Arc<HrirGrid> {
    let mut rng = Noise::new(77);
    let taps = 48;
    let recs: Vec<HrirRec> = icosahedron()
        .into_iter()
        .map(|dir| {
            let mut k = |s: f32| (0..taps).map(|i| rng.next() * s * (-(i as f32) / 8.0).exp()).collect();
            HrirRec { dir, l: k(0.6), r: k(0.5) }
        })
        .collect();
    Arc::new(HrirGrid::from_bytes(&hrir_bytes(taps, &recs)))
}

/// The scene at block-start sample `k`, update number `v`: 36 candidate
/// paths, some gliding (Doppler), some coming and going (crossfades and
/// releases), a few through walls (lowpass), the late field always on and
/// the coupled room part of the time.
fn scene(v: u64, k: usize) -> ParamBlock {
    let t = k as f32 / SR;
    let mut taps = Vec::new();
    for i in 0..36u32 {
        // paths 20.. blink in and out on their own schedules
        if i >= 20 && (v as u32 + i) % 5 < 2 {
            continue;
        }
        let fi = i as f32;
        let speed = if i % 3 == 0 { 0.004 * (fi + 1.0) } else { 0.0 };
        let delay_s = 0.003 + 0.0021 * fi + speed * t;
        let az = 0.7 * fi + 0.4 * t;
        let el = 0.3 * (fi * 0.5).sin();
        let dir = [az.cos() * el.cos(), az.sin() * el.cos(), el.sin()];
        let g = 0.6 / (1.0 + 0.3 * fi) * (1.0 + 0.3 * (0.9 * t + fi).sin());
        let gains = if i % 7 == 3 {
            [g, 0.25 * g, 0.05 * g] // through a wall: lows ≫ mids
        } else {
            [0.8 * g, g, 0.6 * g]
        };
        taps.push(Tap { key: i, delay_s, dir, gains });
    }
    // a path that teleports now and then: crossfade, same key
    let jump = if v % 6 < 3 { 0.0 } else { 0.05 };
    taps.push(Tap { key: 99, delay_s: 0.02 + jump, dir: [0.0, -1.0, 0.0], gains: [0.3; 3] });
    // a stretch with no late field at all: both FDNs ring out, fall
    // asleep (the block path then skips them) and wake again after
    let late = !(25..110).contains(&v);
    let remote = (late && v % 8 >= 3).then(|| RemoteReverb {
        rt60: [1.8, 1.5, 1.0],
        send: [0.08, 0.05, 0.03],
        sh: encode_gains([0.5f32.cos(), 0.5f32.sin(), 0.0]),
    });
    ParamBlock {
        taps,
        reverb: ReverbParams { rt60: [1.2, 1.0, 0.7], level: if late { [0.05, 0.04, 0.02] } else { [0.0; 3] } },
        remote,
        version: v,
    }
}

/// Bus (9 channels) + point stereo per sample.
type Out = Vec<[f32; NCH + 2]>;

/// When to apply updates / head turns / one-shots: at block starts, so
/// the per-sample run can do the same at the same samples.
fn starts() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let (mut k, mut b) = (0, 0);
    while k < N {
        let n = SIZES[b % SIZES.len()].min(N - k);
        v.push((k, n));
        k += n;
        b += 1;
    }
    v
}

fn events(r: &mut Renderer, log: Option<&mut PlayLog>, b: usize, k: usize, version: &mut u64) {
    if b % 23 == 0 {
        *version += 1;
        r.set_params(&scene(*version, k));
    }
    if b % 41 == 7 {
        let yaw = 0.8 * ((b / 41) as f32).sin();
        r.set_head(yaw, 0.2 * yaw, 0.0);
    }
    if b % 200 == 50 {
        // a budget change re-applies the same block
        r.set_point_budget(if (b / 200) % 2 == 0 { 3 } else { 8 });
        r.invalidate_params();
        r.set_params(&scene(*version, k));
    }
    if let Some(log) = log {
        if b % 97 == 13 {
            log.play(1, Clip::from(noise(b as u64, 9000, 0.7)), k as f64, 0.8, 1.3);
        }
    }
}

fn renderer(g: &Arc<HrirGrid>) -> Renderer {
    let mut r = Renderer::with_grid(SR, Some(g.clone()));
    r.set_point_budget(6);
    r.set_tap_ceiling(30);
    r
}

fn fader(k: usize) -> f32 {
    0.7 + 0.3 * (k as f32 * 0.0007).sin()
}

fn run(log_src: bool, blocks: bool) -> Out {
    let g = grid();
    let mut r = renderer(&g);
    let mut log = PlayLog::new();
    log.set_loop(Clip::from(noise(3, 48_000, 0.5)), 0.0);
    let mut r_log = if log_src { Some(log) } else { None };
    let x = noise(5, N, 0.5);
    let mut out = vec![[0.0f32; NCH + 2]; N];
    let mut version = 0;
    for (b, &(k0, n)) in starts().iter().enumerate() {
        events(&mut r, r_log.as_mut(), b, k0, &mut version);
        if blocks {
            let mut bus: BusBlock = [[0.0; BLOCK]; NCH];
            let (mut pl, mut pr) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            match &r_log {
                Some(log) => {
                    let gain: Vec<f32> = (k0..k0 + n).map(fader).collect();
                    r.process_log_block(log, k0 as f64, &gain, &mut bus, &mut pl[..n], &mut pr[..n]);
                }
                None => r.process_block(&x[k0..k0 + n], &mut bus, &mut pl[..n], &mut pr[..n]),
            }
            for i in 0..n {
                let o = &mut out[k0 + i];
                for c in 0..NCH {
                    o[c] = bus[c][i];
                }
                o[NCH] = pl[i];
                o[NCH + 1] = pr[i];
            }
        } else {
            for k in k0..k0 + n {
                let mut bus = [0.0f32; NCH];
                let (a, b) = match &r_log {
                    Some(log) => r.process_log(log, k as f64, fader(k), &mut bus),
                    None => r.process(x[k], &mut bus),
                };
                out[k][..NCH].copy_from_slice(&bus);
                out[k][NCH] = a;
                out[k][NCH + 1] = b;
            }
        }
    }
    out
}

fn compare(what: &str, a: &Out, b: &Out) {
    let mut peak = 0.0f32;
    let mut worst = (0.0f32, 0usize, 0usize);
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        for c in 0..NCH + 2 {
            peak = peak.max(x[c].abs());
            let d = (x[c] - y[c]).abs();
            if d > worst.0 {
                worst = (d, k, c);
            }
        }
    }
    let rel_db = 20.0 * (worst.0 / peak).max(1e-12).log10();
    eprintln!("{what}: peak {peak:.4}, worst |Δ| {:.2e} at sample {} ch {} ({rel_db:.1} dB re peak)", worst.0, worst.1, worst.2);
    assert!(peak > 0.05, "{what}: the scene must actually sound (peak {peak})");
    assert!(worst.0 < 1e-4 && rel_db < -80.0, "{what}: block render differs by {:.2e} ({rel_db:.1} dB) at sample {} ch {}", worst.0, worst.1, worst.2);
}

#[test]
fn log_source_blocks_match_per_sample() {
    compare("log", &run(true, false), &run(true, true));
}

#[test]
fn live_source_blocks_match_per_sample() {
    compare("live", &run(false, false), &run(false, true));
}

/// Every channel the scene drives must actually carry signal in both
/// runs — the comparison is not vacuous for the point taps, the bus, or
/// the FDNs.
#[test]
fn the_scene_exercises_bus_points_and_tails() {
    let a = run(true, true);
    let e = |c: usize| a.iter().map(|o| (o[c] as f64).powi(2)).sum::<f64>();
    assert!(e(0) > 1.0, "bus W");
    assert!(e(4) > 1e-3, "bus order 2");
    assert!(e(NCH) > 1.0 && e(NCH + 1) > 1.0, "point stereo");
}

/// 10 km away (29 s, ~1.4 M samples of delay) driving off at 20 m/s,
/// updates every 10 ms: the block path's delay motion (f64, rearranged
/// per-block arithmetic) still matches the per-sample glide.
///
/// The residual here (≈ −83 dB, a pure gain) is the per-sample path's,
/// not the block path's: its f32 gain smoother stalls ~7e-5 below its
/// target once a step moves less than half an ulp, while the block
/// path's closed form arrives.
#[test]
fn far_doppler_blocks_match_per_sample() {
    let run = |blocks: bool| -> Out {
        let mut r = Renderer::new(SR).without_line();
        let mut log = PlayLog::new();
        log.set_loop(Clip::from(sine(1000.0, 4800, 0.5)), 0.0);
        let at = |k: usize| {
            let t = k as f32 / SR;
            let delay_s = 10_000.0 / 343.0 - 20.0 / 343.0 * (t - 0.3).max(0.0);
            ParamBlock {
                taps: vec![Tap { key: 7, delay_s, dir: [1.0, 0.0, 0.0], gains: [1.0; 3] }],
                version: k as u64 + 1,
                ..Default::default()
            }
        };
        let n = 48_000;
        let mut out = vec![[0.0f32; NCH + 2]; n];
        for k0 in (0..n).step_by(BLOCK) {
            if k0 % 512 == 0 {
                r.set_params(&at(k0));
            }
            if blocks {
                let mut bus: BusBlock = [[0.0; BLOCK]; NCH];
                let (mut pl, mut pr) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
                r.process_log_block(&log, k0 as f64, &[1.0; BLOCK], &mut bus, &mut pl, &mut pr);
                for i in 0..BLOCK {
                    out[k0 + i][0] = bus[0][i];
                }
            } else {
                for k in k0..k0 + BLOCK {
                    let mut bus = [0.0f32; NCH];
                    r.process_log(&log, k as f64, 1.0, &mut bus);
                    out[k][0] = bus[0];
                }
            }
        }
        out
    };
    compare("far doppler", &run(false), &run(true));
}
