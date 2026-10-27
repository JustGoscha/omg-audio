//! GPU walk time in the demo town (GPU timestamps: the walk's own time),
//! listener in the living room, every walkthrough source playing.
//!
//!     cargo run --release -p omg-web --features gpu --example gpu_town
//!
//! RAYS=early,late pins the budget (default: medium's), NODIFF=1 turns
//! diffraction off, SYNC=1 walks every ray whole (no late pool).
use omg_audio::{Compute, Config, Quality, Simulation};
use std::time::{Duration, Instant};

fn main() {
    let sync = std::env::var_os("SYNC").is_some();
    let mut b = Quality::Medium.budget(true);
    if let Ok(spec) = std::env::var("RAYS") {
        let v: Vec<u32> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        b.rays = v[0];
        b.late_rays = v[1];
    }
    b.diffraction = std::env::var_os("NODIFF").is_none();
    if let Ok(n) = std::env::var("NEE") {
        b.nee = n.parse().unwrap();
    }
    let compute = if sync { Compute::GpuSync } else { Compute::Gpu };
    let mut sim = Simulation::new(Config { compute, quality: Quality::Custom(b), ..Config::default() });
    let t0 = Instant::now();
    let town = omg_web::town::Town::new(&mut sim);
    eprintln!("town: {} sources, built in {:.0} ms", town.sources.iter().map(Vec::len).sum::<usize>(), t0.elapsed().as_secs_f64() * 1e3);
    if !sim.gpu_active() {
        println!("no GPU");
        return;
    }
    sim.set_listener([3.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    let (mut sum, mut n, mut worst, mut chunk) = (0.0f32, 0, 0.0f32, 0.0f32);
    let mut tick_ms = 0.0f64;
    for k in 0..80 {
        let t = Instant::now();
        sim.tick();
        let dt = t.elapsed();
        if k >= 20 {
            let g = sim.gpu_walk_ms();
            sum += g;
            worst = worst.max(g);
            chunk = chunk.max(sim.gpu_walk_max_ms());
            n += 1;
            tick_ms += dt.as_secs_f64() * 1e3;
        }
        if !sync {
            std::thread::sleep(Duration::from_millis(50).saturating_sub(dt));
        }
    }
    println!(
        "{} early (nee {}) {} late{}{}: GPU walk {:.1} ms (worst {:.1}), longest dispatch {:.1} ms, CPU tick {:.1} ms",
        b.rays,
        b.nee,
        b.late_rays,
        if b.diffraction { "" } else { ", no diffraction" },
        if sync { ", whole walk" } else { ", pooled" },
        sum / n as f32,
        worst,
        chunk,
        tick_ms / n as f64
    );
}
