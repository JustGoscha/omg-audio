//! Portals against the walk alone (`Config::portals`): one source alone
//! (the club, its inner door open), the ears standing at each point, a
//! few seeds — per tick the source's field (all of it but the direct
//! sound and the exact paths: the estimated early field and the reverb,
//! low and mid band) — where it settles (the last half's mean: both ways
//! must agree, the estimate unbiased either way) and how it gets there:
//! the ticks it jumps 6 dB or more over the tick before (the bursts), its
//! spread across seeds.
//!
//!     cargo run --release -p omg-web --example portal_ab
//!     AT=x,y;x,y (24.2,-7.4;26,7;-14,20;8,46)  TICKS=n (240)  SEEDS=n (4)
//!     SOURCE=name (club)  OPEN=i,j (4)  GEOMETRY=<file>  LEVEL=n (else High)
//!     K=k: the walk's sphere radius per metre of path (the rung's)
//!     RAW=1: what the walk deposits a tick (unaveraged) instead of the field
use omg_audio::{Config, Frame, Quality, Simulation, SourceId};
use omg_web::town::Town;

fn field(f: &Frame, ids: &[SourceId]) -> [f64; 2] {
    let mut e = [0.0f64; 2];
    for &id in ids {
        let Some(pb) = f.block(id) else { continue };
        for (b, eb) in e.iter_mut().enumerate() {
            *eb += (pb.early.level[b] as f64).powi(2) + (pb.reverb.level[b] as f64).powi(2);
            *eb += pb.remote.map_or(0.0, |r| (r.send[b] as f64).powi(2));
        }
    }
    e
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn main() {
    let env = |k: &str| std::env::var(k).ok();
    let pts: Vec<[f32; 2]> = env("AT")
        .unwrap_or_else(|| "24.2,-7.4;26,7;-14,20;8,46".into())
        .split(';')
        .filter_map(|p| {
            let v: Vec<f32> = p.split(',').filter_map(|x| x.parse().ok()).collect();
            (v.len() == 2).then(|| [v[0], v[1]])
        })
        .collect();
    let ticks: usize = env("TICKS").and_then(|v| v.parse().ok()).unwrap_or(240);
    let seeds: u64 = env("SEEDS").and_then(|v| v.parse().ok()).unwrap_or(4);
    let name = env("SOURCE").unwrap_or_else(|| "club".into());
    let open = env("OPEN").unwrap_or_else(|| "4".into());
    let quality = env("LEVEL").and_then(|v| v.parse().ok()).map_or(Quality::High, Quality::Level);
    let quality = match env("K").and_then(|v| v.parse::<f32>().ok()) {
        Some(k) => Quality::Custom(omg_core::v3::Budget { k, ..quality.budget(false) }),
        None => quality,
    };
    let geometry = env("GEOMETRY");
    // (settled low, settled mid, jumps, ms a tick)
    let run = |p: [f32; 2], seed: u64, portals: bool| -> (f64, f64, usize, f64) {
        let cfg = Config { quality, v4: false, v5: false, v6: true, v61: true, v62: true, portals, seed, kernel_thread: false, ..Config::default() };
        let mut sim = Simulation::new(cfg);
        let mut town = Town::new(&mut sim);
        if let Some(g) = geometry.as_ref() {
            sim.set_geometry(omg_audio::Geometry::load(g).expect("GEOMETRY: a dumped geometry"));
        }
        let src = town.defs.iter().position(|d| d.name == name).expect("the source");
        for i in 0..town.defs.len() {
            town.set_muted(&mut sim, i, i != src);
        }
        for i in open.split(',').filter_map(|x| x.parse::<usize>().ok()) {
            town.set_door(&mut sim, i, 1.0);
        }
        let ids = town.sources[src].clone();
        sim.set_listener([p[0], p[1], 1.6], 0.0, 0.0, 0.0);
        let (mut acc, mut n, mut jumps, mut prev) = ([0.0f64; 2], 0usize, 0usize, f64::NAN);
        let clock = std::time::Instant::now();
        for k in 0..ticks {
            let f = sim.tick();
            let e = if env("RAW").as_deref() == Some("1") {
                // what the walk deposited this tick (low, mid), not averaged
                let w = ids.iter().filter_map(|&id| sim.walked_energy(id)).fold([0.0f64; 2], |a, e| [a[0] + e[0] as f64, a[1] + e[1] as f64]);
                w
            } else {
                field(&f, &ids)
            };
            let lv = db(e[0] + e[1]);
            if k > 0 && lv - prev >= 6.0 {
                jumps += 1;
            }
            prev = lv;
            if k >= ticks / 2 {
                acc[0] += e[0];
                acc[1] += e[1];
                n += 1;
            }
        }
        let ms = clock.elapsed().as_secs_f64() * 1e3 / ticks as f64;
        if env("COSTS").as_deref() == Some("1") {
            let c = sim.tick_costs();
            eprintln!("      last tick: early {:.1} late {:.1} portals {:.1} chains {:.1} fold {:.1} ms", c.early, c.late, c.portals, c.chains, c.fold);
        }
        (db(acc[0] / n as f64), db(acc[1] / n as f64), jumps, ms)
    };
    println!("portal_ab: {name} alone (doors {open} open), {ticks} ticks standing, {seeds} seeds — field (estimated early + reverb) settled low / mid dB, ticks jumping ≥ 6 dB, ms a tick");
    for p in &pts {
        println!("  at ({}, {}):", p[0], p[1]);
        for portals in [false, true] {
            let r: Vec<(f64, f64, usize, f64)> = (1..=seeds).map(|s| run(*p, s, portals)).collect();
            let lows: Vec<String> = r.iter().map(|x| format!("{:6.1}", x.0)).collect();
            let mids: Vec<String> = r.iter().map(|x| format!("{:6.1}", x.1)).collect();
            let jumps: usize = r.iter().map(|x| x.2).sum();
            let ms = r.iter().map(|x| x.3).sum::<f64>() / r.len() as f64;
            let mean = |v: &dyn Fn(&(f64, f64, usize, f64)) -> f64| db(r.iter().map(|x| 10f64.powf(v(x) / 10.0)).sum::<f64>() / r.len() as f64);
            println!(
                "    {:8} low {} (mean {:6.1})  mid {} (mean {:6.1})  jumps {jumps:3}  {ms:5.1} ms",
                if portals { "portals" } else { "walk" },
                lows.join(""),
                mean(&|x| x.0),
                mids.join(""),
                mean(&|x| x.1)
            );
        }
    }
}
