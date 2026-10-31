//! The club's dense (scattered) response tick by tick, the listener
//! standing still: its energy per band past the first 0.25 s (late) and
//! before, summed over the club's speakers, every 10 ticks. Nothing
//! moves, so the estimate should only settle — any jump is one sample's.
//!
//!   SPOT=square|north|entrance  NEE=n  TICKS=600  SEED=n
//!   cargo run --release -p omg-web --example dense_track
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn main() {
    let mut b = Quality::High.budget(false);
    if let Some(n) = env("NEE").and_then(|n| n.parse().ok()) {
        b.nee = n;
        b.late_nee = n;
    }
    if let Some(n) = env("LATE").and_then(|n| n.parse().ok()) {
        b.late_rays = n;
    }
    if let Some(n) = env("RAYS").and_then(|n| n.parse().ok()) {
        b.rays = n;
    }
    let mut cfg = Config { quality: Quality::Custom(b), compute: Compute::Cpu, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() };
    if let Some(s) = env("SEED").and_then(|s| s.parse().ok()) {
        cfg.seed = s;
    }
    let mut sim = Simulation::new(cfg);
    let dbg = env("DBG").is_some();
    sim.set_debug_paths(dbg);
    let mut town = Town::new(&mut sim);
    let src = env("SRC").and_then(|s| s.parse().ok()).unwrap_or(2usize);
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != src);
    }
    let pos = match env("SPOT").as_deref() {
        Some("north") => [27.0, 48.0, 1.6],
        Some("entrance") => [25.0, 30.6, 1.6],
        Some("inside") => [31.0, 36.0, 1.6],
        _ => [15.0, 32.0, 1.6],
    };
    let ticks: usize = env("TICKS").and_then(|n| n.parse().ok()).unwrap_or(600);
    for k in 0..ticks {
        sim.set_listener_at(pos, 0.0, 0.0, 0.0, k as f64 * 0.05);
        sim.set_time(k as f64 * 0.05);
        sim.tick();
        if dbg {
            let ps = sim.debug_paths();
            if let Some(p) = ps.iter().filter(|p| p.kind == omg_audio::PathKind::Sampled).max_by(|a, b| a.energy.total_cmp(&b.energy)) {
                if p.energy > 1e-3 {
                    let pts: Vec<String> = p.points.iter().map(|q| format!("({:.1},{:.1},{:.1})", q[0], q[1], q[2])).collect();
                    println!("   tick {} strongest sample {:.1} dB: {}", k + 1, 10.0 * p.energy.log10(), pts.join(" "));
                }
            }
        }
        if k % 10 != 9 {
            continue;
        }
        let (mut early, mut late) = ([0.0f64; 3], [0.0f64; 3]);
        for &id in &town.sources[src] {
            let Some(d) = sim.dense_response(id) else { continue };
            for (i, bin) in d.bins.iter().enumerate() {
                let dst = if i as f32 * d.bin_s < 0.25 { &mut early } else { &mut late };
                for b in 0..3 {
                    dst[b] += bin[b] as f64;
                }
            }
        }
        let db = |e: [f64; 3]| e.map(|x| 10.0 * x.max(1e-30).log10());
        let (e, l) = (db(early), db(late));
        println!("tick {:4} t {:5.2}  early {:6.1} {:6.1} {:6.1}  late {:6.1} {:6.1} {:6.1}", k + 1, (k + 1) as f32 * 0.05, e[0], e[1], e[2], l[0], l[1], l[2]);
    }
}
