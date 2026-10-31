//! What a moving door costs: the town with every source sounding, the
//! ears by the club's door while it swings shut and open again (1 s each
//! way) — a full tick every 50 ms and exact frames at 120 Hz between —
//! per engine: the tick and the frame between ticks (ms), averaged over
//! the swings and over a still door.
//!
//!     cargo run --release -p omg-web --example door_ms -- v6.1 v6.2
//!     COMPUTE=gpu cargo run --release -p omg-web --features gpu --example door_ms
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;
use std::time::Instant;

/// The club's door (`assess`'s CLUB_DOOR).
const CLUB_DOOR: usize = 4;

fn run(mode: &str) {
    // COMPUTE=gpu: the walk on the GPU, as the Godot host runs it (build
    // with --features gpu)
    let gpu = std::env::var("COMPUTE").as_deref() == Ok("gpu");
    let cfg = Config {
        quality: Quality::High,
        compute: if gpu { Compute::Gpu } else { Compute::Cpu },
        v4: false,
        v5: false,
        v6: mode.starts_with("v6"),
        v61: mode == "v6.1" || mode == "v6.2",
        v62: mode == "v6.2",
        kernel_thread: false,
        ..Config::default()
    };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    sim.set_listener([25.0, 30.6, 1.6], 0.0, 0.0, 0.0);
    let (mut tick, mut exact) = ([0.0f64; 2], [0.0f64; 2]);
    let (mut nt, mut nx) = ([0usize; 2], [0usize; 2]);
    let frame = 1.0 / 120.0;
    let mut t = 0.0f64;
    let mut next_tick = 0.0f64;
    while t < 12.0 {
        // still for 2 s, then a swing shut and open (1 s each), …
        let ph = t % 4.0;
        let open = if ph < 2.0 { 1.0 } else if ph < 3.0 { 1.0 - (ph - 2.0) } else { ph - 3.0 };
        town.set_door(&mut sim, CLUB_DOOR, open as f32);
        let moving = (2.0..4.0).contains(&ph) as usize;
        sim.set_time(t);
        let c = Instant::now();
        if t >= next_tick {
            sim.tick();
            next_tick += 0.05;
            if t > 1.0 {
                tick[moving] += c.elapsed().as_secs_f64() * 1e3;
                nt[moving] += 1;
            }
        } else if sim.tick_exact().is_some() && t > 1.0 {
            exact[moving] += c.elapsed().as_secs_f64() * 1e3;
            nx[moving] += 1;
        }
        t += frame;
    }
    println!(
        "{mode:5} still: tick {:5.2} ms, frame {:5.3} ms | door moving: tick {:5.2} ms, frame {:5.3} ms",
        tick[0] / nt[0].max(1) as f64,
        exact[0] / nx[0].max(1) as f64,
        tick[1] / nt[1].max(1) as f64,
        exact[1] / nx[1].max(1) as f64
    );
}

fn main() {
    let modes: Vec<String> = std::env::args().skip(1).collect();
    for m in if modes.is_empty() { vec!["v6.1".into(), "v6.2".into()] } else { modes } {
        run(&m);
    }
}
