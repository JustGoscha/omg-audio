//! The simulation tick's cost in the town (every source of the
//! walkthrough active), web settings and native ones, v4 against v5.
//!
//!     cargo run --release -p omg-web --example tick_ms
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;
use std::time::Instant;

fn run(name: &str, cfg: Config) {
    let mut sim = Simulation::new(cfg);
    let _town = Town::new(&mut sim);
    let path = [(3.0f32, 3.0f32), (8.0, 20.0), (18.0, 30.0), (27.0, 32.0), (8.0, 60.0)];
    let (mut ms, mut n) = (0.0f64, 0);
    for k in 0..200 {
        let a = path[(k / 40) % path.len()];
        sim.set_listener([a.0 + 0.05 * (k % 40) as f32, a.1, 1.6], 0.0, 0.0, 0.0);
        let t = Instant::now();
        sim.tick();
        if k % 40 >= 10 {
            ms += t.elapsed().as_secs_f64() * 1e3;
            n += 1;
        }
    }
    println!("{name:28} {:.1} ms/tick", ms / n as f64);
}

fn main() {
    let web = Config { quality: Quality::Medium, compute: Compute::Cpu, kernel_thread: false, max_kernels: 8, max_kernel_s: 2.5, kernel_step_samples: 65536, ..Config::default() };
    let native = Config { quality: Quality::High, compute: Compute::Cpu, ..Config::default() };
    run("web (low) v5", Config { quality: Quality::Low, v5: true, ..web });
    run("web (medium) v4", web);
    run("web (medium) v5", Config { v5: true, ..web });
    run("native (high, CPU) v4", native);
    run("native (high, CPU) v5", Config { v5: true, ..native });
}
