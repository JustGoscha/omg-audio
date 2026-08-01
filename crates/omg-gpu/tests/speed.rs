//! Rough dispatch-cost probe (ignored by default): GPU trace wall time
//! at the engine budget and at 8×, vs the CPU tracer on this machine.
//! Run: cargo test -p omg-gpu --release --test speed -- --ignored --nocapture

use omg_core::rng::Rng;
use omg_core::tracer::{trace, Echogram};

#[path = "../../omg-core/tests/trace_golden.rs"]
mod golden;

#[test]
#[ignore]
fn speed() {
    let Some(gpu) = omg_gpu::GpuTracer::new() else {
        eprintln!("SKIP: no adapter");
        return;
    };
    let cfgs = golden::golden_configs();
    let cfg = &cfgs[0];
    let mut echo = Echogram::new();
    for rays in [4096u32, 32768] {
        gpu.trace(&cfg.room, cfg.src, cfg.lis, rays, [1.0; 3], 7, &mut echo); // warm
        let t0 = std::time::Instant::now();
        let n = 20;
        for k in 0..n {
            gpu.trace(&cfg.room, cfg.src, cfg.lis, rays, [1.0; 3], k, &mut echo);
        }
        println!(
            "GPU {rays:>6} rays: {:.2} ms/trace (sync incl. readback)",
            t0.elapsed().as_secs_f64() * 1000.0 / n as f64
        );
    }
    let mut rng = Rng::new(7);
    let t0 = std::time::Instant::now();
    let n = 20;
    for _ in 0..n {
        trace(&cfg.room, cfg.src, cfg.lis, 4096, [1.0; 3], &mut rng, &mut echo);
    }
    println!(
        "CPU   4096 rays: {:.2} ms/trace",
        t0.elapsed().as_secs_f64() * 1000.0 / n as f64
    );
}
