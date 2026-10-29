//! The simulation tick's cost moving against standing still: the town with
//! every walkthrough source active (as the Godot demo plays it), the
//! listener flying at SPEED m/s (8: Godot's fly) back and forth along x
//! inside the cathedral, then still at the same place; per tick, its
//! wall-clock. A tick over the 50 ms period leaves the simulation thread
//! no time for the exact frames between ticks: paths then move in steps.
//!
//!   MODE=v3|v5|v6  SPEED=8  QUALITY=0..3 | LEVEL=0..10  SECS=10  THREADS=n
//!   cargo run --release -p omg-web --example tick_move
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;
use std::time::Instant;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn main() {
    let mode = env("MODE").unwrap_or_else(|| "v6".into());
    let quality = match (env("LEVEL").and_then(|v| v.parse::<usize>().ok()), env("QUALITY").as_deref()) {
        (Some(l), _) => Quality::Level(l),
        (_, Some("0")) => Quality::Low,
        (_, Some("1")) => Quality::Medium,
        (_, Some("3")) => Quality::Ultra,
        _ => Quality::High,
    };
    // DIFF=0: no diffraction (edge bends, two-edge paths); RAYS, BOUNCES,
    // LATE, NEE: single levers over the tier
    let quality = {
        let mut b = quality.budget(false);
        let num = |k: &str| env(k).and_then(|v| v.parse::<u32>().ok());
        if env("DIFF").as_deref() == Some("0") {
            b.diffraction = false;
        }
        if let Some(n) = num("RAYS") {
            b.rays = n;
        }
        if let Some(n) = num("BOUNCES") {
            b.bounces = n;
        }
        if let Some(n) = num("LATE") {
            b.late_rays = n;
        }
        if let Some(n) = num("NEE") {
            b.nee = n as usize;
        }
        Quality::Custom(b)
    };
    let mut cfg = Config { quality, compute: Compute::Cpu, v4: mode == "v4" || mode == "v5", v5: mode == "v5", v6: mode == "v6", ..Config::default() };
    // THREADS=n: the walk's threads (default: the machine's default)
    if let Some(n) = env("THREADS").and_then(|v| v.parse().ok()) {
        cfg.walk_threads = n;
    }
    eprintln!("walk threads: {}", cfg.walk_threads);
    let mut sim = Simulation::new(cfg);
    let _town = Town::new(&mut sim);
    let speed: f32 = env("SPEED").and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let secs: f32 = env("SECS").and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let (x0, x1, y) = (2.0f32, 14.0f32, 60.0f32);
    // ROUTE=fly: from the field north through the cathedral's south wall
    // (y 52) to the flute and on (Godot's recorded scenario), every tick
    // over 1.5× the median printed by part
    let fly = env("ROUTE").as_deref() == Some("fly");
    let pos = move |t: f32| -> [f32; 3] {
        if fly {
            return [4.0, (40.0 + t * speed).min(62.0), 2.0];
        }
        // back and forth between x0 and x1
        let span = x1 - x0;
        let d = (t * speed) % (2.0 * span);
        [if d < span { x0 + d } else { x1 - (d - span) }, y, 2.0]
    };
    let mut run = |moving: bool| -> (Vec<f64>, [f64; 8]) {
        let mut ms = Vec::new();
        let mut parts = [0.0f64; 8];
        let n = (secs / 0.05) as usize;
        for k in 0..n {
            let t = k as f32 * 0.05;
            let p = if moving { pos(t) } else { [8.0, y, 2.0] };
            sim.set_listener_at(p, 0.0, 0.0, 0.0, t as f64 + if moving { 0.0 } else { 1000.0 });
            let t0 = Instant::now();
            sim.tick();
            let total = t0.elapsed().as_secs_f64() * 1e3;
            ms.push(total);
            if fly && moving {
                let c = sim.tick_costs();
                println!(
                    "  t {t:5.2} y {:5.1}: {total:5.1} ms (direct {:.1} edges {:.1} 2-edge {:.1} early {:.1} late {:.1} chains {:.1} fold {:.1})",
                    p[1], c.direct, c.edges - c.edges2, c.edges2, c.early, c.late, c.chains, c.fold
                );
            }
            if k >= 10 {
                let c = sim.tick_costs();
                let v = [c.direct, c.edges - c.edges2, c.edges2, c.early, c.late, c.chains, c.fold];
                for (p, x) in parts.iter_mut().zip(v) {
                    *p += x as f64;
                }
                parts[7] += total - v.iter().map(|&x| x as f64).sum::<f64>();
            }
        }
        (ms, parts)
    };
    let report = |name: &str, v: &[f64]| {
        let mut s = v[10..].to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let over = s.iter().filter(|&&x| x > 50.0).count();
        println!("{mode} {name:6}: tick median {:5.1} ms, p90 {:5.1}, max {:5.1}; over the 50 ms period {over} of {}", s[s.len() / 2], s[s.len() * 9 / 10], s[s.len() - 1], s.len());
    };
    let names = ["direct", "edges", "2-edge", "early", "late", "chains", "fold", "other"];
    let part_line = |parts: &[f64; 8], n: usize| {
        let v: Vec<String> = names.iter().zip(parts).map(|(k, p)| format!("{k} {:.1}", p / n as f64)).collect();
        println!("        ms per tick: {}", v.join(" · "));
    };
    let (m, pm) = run(true);
    report("moving", &m);
    part_line(&pm, m.len() - 10);
    let (s, ps) = run(false);
    report("still", &s);
    part_line(&ps, s.len() - 10);
}
