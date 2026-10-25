//! Effect: point-rendered direct paths — nearest-HRIR convolution with a
//! ~10 ms crossfade on direction change (hrtf.rs `HrirGrid`, `PointConv`).
//!
//! Claims:
//!  - `HrirGrid::nearest` picks the closest measured direction;
//!  - while the direction holds, output is exactly the source convolved
//!    with that direction's HRIR pair;
//!  - a direction change crossfades linearly over 10 ms (480 samples at
//!    48 kHz): no click, then exactly the new HRIR;
//!  - re-selecting the current index does not restart a fade;
//!  - through the renderer, a point tap's binaural output carries the
//!    HRIR's ILD and leaves the SH bus untouched.
//!
//! Isolation: synthetic 3-direction grid (left / front / right) with
//! impulse-pair HRIRs built in-test; reference outputs by direct
//! convolution. No assets, no scene.
//! Combined in the demo: every point-rendered direct path in
//! omg-web/omg-app; head motion drives `set_dir` via Renderer (see
//! fx_head_smoothing.rs).

mod common;
use common::*;
use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::NCH;
use omg_dsp::hrtf::{HrirGrid, PointConv};
use omg_dsp::Renderer;
use std::sync::Arc;

const TAPS: usize = 32;
const LEFT: [f32; 3] = [0.0, 1.0, 0.0];
const FRONT: [f32; 3] = [1.0, 0.0, 0.0];
const RIGHT: [f32; 3] = [0.0, -1.0, 0.0];

/// (dir, hrir_l, hrir_r) for the synthetic grid: near ear direct, far ear
/// 20 samples late at 0.3; front: both ears 10 samples, 0.65.
fn recs() -> Vec<HrirRec> {
    vec![
        HrirRec { dir: LEFT, l: impulse(TAPS, 0, 1.0), r: impulse(TAPS, 20, 0.3) },
        HrirRec { dir: FRONT, l: impulse(TAPS, 10, 0.65), r: impulse(TAPS, 10, 0.65) },
        HrirRec { dir: RIGHT, l: impulse(TAPS, 20, 0.3), r: impulse(TAPS, 0, 1.0) },
    ]
}

fn grid() -> HrirGrid {
    HrirGrid::from_bytes(&hrir_bytes(TAPS, &recs()))
}

#[test]
fn nearest_picks_the_closest_direction() {
    let g = grid();
    assert_eq!(g.len(), 3);
    assert_eq!(g.taps, TAPS);
    assert_eq!(g.nearest(normalize([0.2, 1.0, 0.1])), 0);
    assert_eq!(g.nearest(normalize([1.0, 0.3, -0.2])), 1);
    assert_eq!(g.nearest(normalize([-0.5, -1.0, 0.0])), 2);
}

/// Hold left, switch to right at n = 4800: the output follows the exact
/// linear-crossfade model y = f·prev + (1−f)·cur, f: 1 → 0 over 480
/// samples; step size never exceeds the steady signal's own.
#[test]
fn direction_change_crossfades_over_10ms() {
    let g = grid();
    let x = sine(1000.0, 9600, 1.0);
    let rl = recs();
    let ref_l = (convolve(&x, &rl[0].l), convolve(&x, &rl[0].r));
    let ref_r = (convolve(&x, &rl[2].l), convolve(&x, &rl[2].r));

    let mut pc = PointConv::new(TAPS, SR);
    pc.reset(g.nearest(LEFT));
    let mut out = Vec::new();
    for (n, s) in x.iter().enumerate() {
        if n == 4800 {
            pc.set_dir(&g, RIGHT);
        }
        if n > 4800 && n < 4900 {
            pc.set_dir(&g, RIGHT); // same index again: must not restart
        }
        out.push(pc.process(&g, *s));
    }
    for n in 0..9600 {
        let f = if n < 4800 { 1.0 } else { (1.0 - (n - 4800) as f32 / 480.0).max(0.0) };
        let want_l = f * ref_l.0[n] + (1.0 - f) * ref_r.0[n];
        let want_r = f * ref_l.1[n] + (1.0 - f) * ref_r.1[n];
        assert!(
            (out[n].0 - want_l).abs() < 1e-4 && (out[n].1 - want_r).abs() < 1e-4,
            "n {n}: {:?} vs ({want_l}, {want_r})",
            out[n]
        );
    }
    let l: Vec<f32> = out.iter().map(|o| o.0).collect();
    let steady = max_step(&l[4000..4800]);
    assert!(max_step(&l[4790..5400]) <= 1.05 * steady, "crossfade click");
}

/// Renderer re-selects point HRIRs every 128 samples (2.7 ms) — faster
/// than the 10 ms fade. A second change arriving mid-fade (A→B, then
/// B→C 128 samples later, e.g. a head turn sweeping a dense grid)
/// restarts the fade from PURE `prev` = B, discarding the A-weighted part
/// of the blend in one sample: y jumps by f·|A−B| (f ≈ 0.73 here).
/// Correct behavior: continuous output (no step beyond the signal's own).
#[test]
#[ignore = "BUG: PointConv::set_dir mid-crossfade snaps to pure prev — output steps by f·|A−B| (≈0.7 of a full-scale HRIR difference) when re-selected within 10 ms"]
fn retarget_during_a_crossfade_stays_continuous() {
    let g = grid();
    let x = sine(1000.0, 9600, 1.0);
    let mut pc = PointConv::new(TAPS, SR);
    pc.reset(g.nearest(LEFT));
    let mut l = Vec::new();
    for (n, s) in x.iter().enumerate() {
        if n == 4800 {
            pc.set_dir(&g, FRONT);
        }
        if n == 4928 {
            pc.set_dir(&g, RIGHT);
        }
        l.push(pc.process(&g, *s).0);
    }
    let steady = max_step(&l[4000..4800]);
    let worst = max_step(&l[4790..5600]);
    assert!(worst <= 1.1 * steady, "mid-fade retarget step {worst:.3} vs steady slope {steady:.3}");
}

/// Through the renderer with a grid: the strongest tap is point-rendered
/// — its stereo carries the left HRIR's ILD and nothing reaches the bus.
#[test]
fn renderer_point_tap_uses_the_hrir() {
    let mut r = Renderer::with_grid(SR, Some(Arc::new(grid())));
    r.set_params(&ParamBlock {
        taps: vec![Tap { key: 1, delay_s: 0.002, dir: LEFT, gains: [0.5; 3] }],
        version: 1,
        ..Default::default()
    });
    let x = noise(8, 24_000, 0.5);
    let (mut l, mut rr, mut bus_e) = (Vec::new(), Vec::new(), 0.0f64);
    for s in x {
        let mut bus = [0.0f32; NCH];
        let (a, b) = r.process(s, &mut bus);
        l.push(a);
        rr.push(b);
        bus_e += bus.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    }
    assert_eq!(r.debug_stats().1, 1);
    assert_eq!(bus_e, 0.0, "point tap must not also feed the bus");
    let ild = db_pow(energy(&l[12_000..]) / energy(&rr[12_000..]));
    assert!((ild - db(1.0 / 0.3) as f64).abs() < 0.2, "ILD {ild:.2} dB, HRIR says {:.2}", db(1.0 / 0.3));
    assert_eq!(xcorr_peak_lag(&l[12_000..], &rr[12_000..], 30), 20, "ITD from the HRIR");
}
