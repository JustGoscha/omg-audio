//! Per-path render cost: one renderer, P bus paths (no HRIR grid, late
//! field off), a looped recording read from a PlayLog — per-sample
//! `process_log` vs `process_log_block`. Prints ns and cycles (at an
//! assumed 3.2 GHz) per path per sample, best of five runs each.
//!
//!     cargo run --release -p omg-dsp --example bench_paths [P] [points]
//!
//! With `points`, every path is point-rendered through the HRIR grid in
//! assets/ (nearest-HRIR convolution) instead of the bus.

use omg_core::params::{ParamBlock, Tap};
use omg_dsp::ambi::NCH;
use omg_dsp::playlog::{Clip, PlayLog};
use omg_dsp::renderer::{BusBlock, BLOCK};
use omg_dsp::Renderer;
use std::hint::black_box;
use std::time::Instant;

const SR: f32 = 48_000.0;

fn scene(v: u64, paths: usize, t: f32) -> ParamBlock {
    let taps = (0..paths as u32)
        .map(|i| {
            let fi = i as f32;
            let az = 0.7 * fi;
            Tap {
                key: i,
                // every path moving a little: Doppler glides throughout
                delay_s: 0.003 + 0.0011 * fi + 0.002 * (t + fi).sin(),
                dir: [az.cos(), az.sin(), 0.1],
                gains: [0.4, 0.5 / (1.0 + 0.05 * fi), 0.3],
            }
        })
        .collect();
    ParamBlock { taps, version: v, ..Default::default() }
}

fn main() {
    let paths: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(96);
    let grid = (std::env::args().nth(2).as_deref() == Some("points"))
        .then(|| std::sync::Arc::new(omg_dsp::hrtf::HrirGrid::load("assets/hrir_grid.bin").expect("assets/hrir_grid.bin")));
    let noise: Vec<f32> =
        (0..48_000u32).map(|i| ((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0 - 1.0) * 0.2).collect();
    let mut log = PlayLog::new();
    log.set_loop(Clip::from(noise), 0.0);
    let secs = 2.0;
    let n = (secs * SR) as usize;
    let gain = [1.0f32; BLOCK];
    let mut best = [f64::MAX; 2];
    for run in 0..10 {
        let blocks = run % 2 == 1;
        let mut r = Renderer::with_grid(SR, grid.clone()).without_line();
        r.set_tap_ceiling(paths);
        r.set_point_budget(if grid.is_some() { paths } else { 0 });
        let mut v = 1;
        r.set_params(&scene(v, paths, 0.0));
        let mut acc = 0.0f32;
        let t0 = Instant::now();
        let mut k = 0;
        while k < n {
            if k % 2400 == 0 {
                v += 1;
                r.set_params(&scene(v, paths, k as f32 / SR));
            }
            if blocks {
                let mut bus: BusBlock = [[0.0; BLOCK]; NCH];
                let (mut pl, mut pr) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
                r.process_log_block(&log, k as f64, &gain, &mut bus, &mut pl, &mut pr);
                acc += bus[0][BLOCK - 1] + pl[BLOCK - 1];
            } else {
                for j in 0..BLOCK {
                    let mut bus = [0.0f32; NCH];
                    let (a, _) = r.process_log(&log, (k + j) as f64, 1.0, &mut bus);
                    acc += bus[0] + a;
                }
            }
            k += BLOCK;
        }
        black_box(acc);
        let b = &mut best[blocks as usize];
        *b = b.min(t0.elapsed().as_secs_f64());
    }
    for blocks in [false, true] {
        let dt = best[blocks as usize];
        let ns = dt * 1e9 / (n as f64 * paths as f64);
        println!(
            "{:>12}: {paths} paths, {:.1}x real time, {ns:.2} ns = {:.0} cycles @3.2GHz per path per sample",
            if blocks { "block" } else { "per-sample" },
            secs as f64 / dt,
            ns * 3.2
        );
    }
}
