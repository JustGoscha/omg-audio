//! Effect: render-cost governors — the per-source tap ceiling
//! (`set_tap_ceiling`) and the point-render budget (`set_point_budget`).
//!
//! Claims:
//!  - ceiling N keeps exactly the N STRONGEST incoming taps (by mid
//!    gain); the weak ones are dropped, never the strong ones;
//!  - lowering the ceiling mid-play releases the evicted taps with the
//!    normal slot fade (click-free), never a cut;
//!  - with an HRIR grid, the point budget N point-renders exactly the N
//!    strongest taps (they leave the SH bus and come out as binaural
//!    stereo); without a grid nothing is point-rendered;
//!  - changing the budget re-slots taps with a crossfade whose sum is
//!    level-continuous.
//!
//! Isolation: `Renderer::new` / `Renderer::with_grid` on a 2-direction
//! synthetic `HrirGrid` (unit-impulse kernels, built in-test), DC input
//! so every tap's contribution is exactly its gain; `debug_stats()` for
//! slot occupancy. No scene.
//! Combined in the demo: the web worklet's load governor drives both
//! knobs live (omg-web); no scene test pins it.

mod common;
use common::*;
use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::NCH;
use omg_dsp::hrtf::HrirGrid;
use omg_dsp::Renderer;
use std::sync::Arc;

/// Six front taps, mid gains 1, ½, ¼, … (distinct subset sums), shuffled
/// so input order says nothing about strength.
fn block(version: u64) -> ParamBlock {
    let order = [3u32, 0, 5, 1, 4, 2];
    ParamBlock {
        taps: order
            .iter()
            .map(|&k| {
                let g = 0.5f32.powi(k as i32);
                Tap { key: k, delay_s: 0.001 * (k + 1) as f32, dir: [1.0, 0.0, 0.0], gains: [g; 3] }
            })
            .collect(),
        version,
        ..Default::default()
    }
}

fn sum_top(n: usize) -> f32 {
    (0..n).map(|k| 0.5f32.powi(k as i32)).sum()
}

/// DC through the renderer for `n` samples; returns (bus W, pl) per sample.
fn run(r: &mut Renderer, n: usize) -> Vec<(f32, f32)> {
    (0..n)
        .map(|_| {
            let mut bus = [0.0f32; NCH];
            let (pl, _) = r.process(1.0, &mut bus);
            (bus[0], pl)
        })
        .collect()
}

/// Grid with unit-gain-sum kernels × 1.25, so a point tap's DC output
/// (× POINT_GAIN 0.8) equals its bus contribution exactly.
fn grid() -> Arc<HrirGrid> {
    let rec = |dir| HrirRec { dir, l: impulse(8, 0, 1.25), r: impulse(8, 0, 1.25) };
    Arc::new(HrirGrid::from_bytes(&hrir_bytes(8, &[rec([0.0, 1.0, 0.0]), rec([0.0, -1.0, 0.0])])))
}

#[test]
fn ceiling_keeps_the_strongest_taps() {
    for n in [1usize, 3, 6] {
        let mut r = Renderer::new(SR);
        r.set_tap_ceiling(n);
        r.set_params(&block(1));
        let y = run(&mut r, 24_000);
        let (live, points, gsum, _, _) = r.debug_stats();
        assert_eq!((live, points), (n as u32, 0), "ceiling {n}");
        assert!((gsum - sum_top(n)).abs() < 1e-3, "ceiling {n}: kept gain {gsum}, strongest {}", sum_top(n));
        assert!((y.last().unwrap().0 - sum_top(n)).abs() < 1e-3, "ceiling {n}: output {:?}", y.last());
    }
    // 0 is clamped to 1 (a source never goes fully dark by ceiling)
    let mut r = Renderer::new(SR);
    r.set_tap_ceiling(0);
    r.set_params(&block(1));
    run(&mut r, 24_000);
    assert_eq!(r.debug_stats().0, 1);
}

#[test]
fn lowering_the_ceiling_fades_evicted_taps() {
    let mut r = Renderer::new(SR);
    r.set_params(&block(1));
    let before = run(&mut r, 24_000);
    r.set_tap_ceiling(2);
    r.set_params(&block(2));
    let after = run(&mut r, 48_000);
    assert_eq!(r.debug_stats().0, 2, "settled to the ceiling");
    // continuity: the first sample after the change is where we were
    let (a, b) = (before.last().unwrap().0, after[0].0);
    assert!((a - b).abs() < 0.01 * a, "eviction must not step: {a} → {b}");
    let step = after.windows(2).map(|w| (w[1].0 - w[0].0).abs()).fold(0.0, f32::max);
    assert!(step < 1e-3, "eviction must glide: max step {step}");
    assert!((after.last().unwrap().0 - sum_top(2)).abs() < 1e-3);
}

#[test]
fn point_budget_point_renders_the_strongest() {
    for n in [0usize, 2, 6] {
        let mut r = Renderer::with_grid(SR, Some(grid()));
        r.set_point_budget(n);
        r.set_params(&block(1));
        let y = run(&mut r, 24_000);
        let (w, pl) = *y.last().unwrap();
        assert_eq!(r.debug_stats().1, n as u32, "budget {n}");
        assert!((pl - sum_top(n)).abs() < 1e-3, "budget {n}: point stereo {pl}, strongest {}", sum_top(n));
        assert!((w - (sum_top(6) - sum_top(n))).abs() < 1e-3, "budget {n}: bus keeps the rest: {w}");
    }
    // no grid ⇒ nothing is point-rendered, whatever the budget
    let mut r = Renderer::new(SR);
    r.set_point_budget(6);
    r.set_params(&block(1));
    let y = run(&mut r, 4800);
    assert_eq!(r.debug_stats().1, 0);
    assert!(y.iter().all(|s| s.1 == 0.0));
}

/// Budget 2 → 1: tap #1 moves from point to bus via a crossfade (an extra
/// slot is live meanwhile) and point + bus stays level-continuous.
#[test]
fn budget_change_reslots_with_a_level_continuous_crossfade() {
    let mut r = Renderer::with_grid(SR, Some(grid()));
    r.set_point_budget(2);
    r.set_params(&block(1));
    run(&mut r, 24_000);
    r.set_point_budget(1);
    r.set_params(&block(2));
    assert_eq!(r.debug_stats().0, 7, "old point slot fades while the bus slot fades in");
    let y = run(&mut r, 48_000);
    let total = sum_top(6);
    let worst = y.iter().map(|(w, pl)| (w + pl - total).abs()).fold(0.0, f32::max);
    assert!(worst < 1e-3, "re-slot must be level-continuous: worst dev {worst}");
    let (live, points, ..) = r.debug_stats();
    assert_eq!((live, points), (6, 1));
}
