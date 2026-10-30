//! Down into the bunker and back up: the radio's level (mid band, dB:
//! its taps, early field, reverb and remote send together) as the ears
//! walk from the street through the blast door, down the stair, through
//! the hatch to the radio, wait, and walk back out to where they started
//! — against the level a listener standing there all along settles at.
//! Leaving a room, its sound should go as fast as its paths do.
//!
//!     cargo run --release -p omg-web --example cellar
//!     MODE=v6.2 (default) | v6.1 | v6 | v5   SEED=n
//!     GEOMETRY=<file>: a host's dumped level (the Godot demo's)
//!     COMPUTE=gpu: the walk on the GPU (build with --features gpu)
//!     PACE=1: in real time, as a host runs (the GPU walk lands while the
//!     frames go on; unpaced it never lands in time)
use omg_audio::{Compute, Config, Frame, Quality, Simulation, SourceId};
use omg_web::town::Town;

fn level(f: &Frame, ids: &[SourceId]) -> f64 {
    let mut e = 0.0f64;
    for &id in ids {
        let Some(pb) = f.block(id) else { continue };
        e += pb.taps.iter().map(|t| (t.gains[1] as f64).powi(2)).sum::<f64>();
        e += (pb.early.level[1] as f64).powi(2) + (pb.reverb.level[1] as f64).powi(2);
        e += pb.remote.map_or(0.0, |r| (r.send[1] as f64).powi(2));
    }
    10.0 * e.max(1e-30).log10()
}

fn config() -> Config {
    let mode = std::env::var("MODE").unwrap_or_else(|_| "v6.2".into());
    let seed = std::env::var("SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    Config {
        quality: Quality::High,
        v4: mode == "v5",
        v5: mode == "v5",
        v6: mode.starts_with("v6"),
        v61: mode == "v6.1" || mode == "v6.2",
        v62: mode == "v6.2",
        seed,
        kernel_thread: false,
        compute: if std::env::var("COMPUTE").as_deref() == Ok("gpu") { Compute::Gpu } else { Compute::Cpu },
        ..Config::default()
    }
}

fn town(sim: &mut Simulation) -> Town {
    let town = Town::new(sim);
    if let Ok(path) = std::env::var("GEOMETRY") {
        sim.set_geometry(omg_audio::Geometry::load(&path).expect("GEOMETRY: a dumped geometry"));
    }
    town
}

fn main() {
    // (time s, position): the street, the blockhouse, the stair's foot,
    // through the hatch, the radio's room — and back
    let way: [(f64, [f32; 3]); 10] = [
        (0.0, [26.0, 7.0, 1.6]),
        (2.0, [26.0, 7.0, 1.6]),
        (4.2, [32.7, 7.0, 1.6]),
        (5.4, [33.6, 7.0, -1.4]),
        (6.4, [36.5, 9.0, -1.4]),
        (9.4, [36.5, 9.0, -1.4]),
        (10.4, [33.6, 7.0, -1.4]),
        (11.6, [32.7, 7.0, 1.6]),
        (13.8, [26.0, 7.0, 1.6]),
        (30.0, [26.0, 7.0, 1.6]),
    ];
    let at = |t: f64| -> [f32; 3] {
        for w in way.windows(2) {
            let ((t0, a), (t1, b)) = (w[0], w[1]);
            if t <= t1 {
                let u = (((t - t0) / (t1 - t0)) as f32).clamp(0.0, 1.0);
                return core::array::from_fn(|k| a[k] + (b[k] - a[k]) * u);
            }
        }
        way[way.len() - 1].1
    };
    let radio = |town: &Town| -> Vec<SourceId> {
        let i = town.defs.iter().position(|d| d.name == "radio").expect("the radio");
        town.sources[i].clone()
    };
    // the reference: standing at the end all along
    let reference = {
        let mut sim = Simulation::new(config());
        let town = town(&mut sim);
        let ids = radio(&town);
        sim.set_listener(way[0].1, 0.0, 0.0, 0.0);
        let mut acc = Vec::new();
        for k in 0..200 {
            if std::env::var("PACE").as_deref() == Ok("1") {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            let f = sim.tick();
            if k >= 100 {
                acc.push(level(&f, &ids));
            }
        }
        10.0 * (acc.iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>() / acc.len() as f64).log10()
    };
    let mut sim = Simulation::new(config());
    let town = town(&mut sim);
    let ids = radio(&town);
    println!("walk: {}", if sim.gpu_active() { "GPU" } else { "CPU" });
    println!("cellar: the radio, mid band dB; standing in the street all along it settles at {reference:+.1} dB");
    let mut row: Vec<String> = Vec::new();
    let pace = std::env::var("PACE").as_deref() == Ok("1");
    let start = std::time::Instant::now();
    for k in 0..(30.0 * 120.0) as usize {
        let t = k as f64 / 120.0;
        if pace {
            let ahead = t - start.elapsed().as_secs_f64();
            if ahead > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(ahead));
            }
        }
        sim.set_time(t);
        sim.set_listener_at(at(t), 0.0, 0.0, 0.0, t);
        let f = if k % 6 == 0 { Some(sim.tick()) } else { sim.tick_exact() };
        if k % 60 == 0 {
            if let Some(f) = f.as_ref() {
                let p = at(t);
                let v: Vec<u64> = ids.iter().filter_map(|&id| f.block(id).map(|b| b.version)).collect();
                let n = f.block(ids[0]).map_or(0, |b| b.taps.len());
                let parts = f.block(ids[0]).map_or(String::new(), |pb| {
                    let db = |e: f64| 10.0 * e.max(1e-30).log10();
                    let taps: f64 = pb.taps.iter().filter(|t| t.key != omg_core::v3::KEY_DIRECT).map(|t| (t.gains[1] as f64).powi(2)).sum();
                    let direct: f64 = pb.taps.iter().filter(|t| t.key == omg_core::v3::KEY_DIRECT).map(|t| (t.gains[1] as f64).powi(2)).sum();
                    format!("direct {:+.0} taps {:+.0} early {:+.0} reverb {:+.0} (rt {:.1} s) remote {:+.0}", db(direct), db(taps), db((pb.early.level[1] as f64).powi(2)), db((pb.reverb.level[1] as f64).powi(2)), pb.reverb.rt60[1], db(pb.remote.map_or(0.0, |r| (r.send[1] as f64).powi(2))))
                });
                let others: Vec<String> = town.defs.iter().zip(&town.sources).filter(|(d, _)| ["piano", "flute", "club"].iter().any(|n| d.name.starts_with(n))).map(|(d, s)| format!("{} {:+.1}", d.name, level(f, s))).collect();
                row.push(format!("{t:5.1} s ({:5.1}, {:4.1}, {:4.1})  {:+6.1} dB   version {v:?}, {n} taps | {} | {parts}", p[0], p[1], p[2], level(f, &ids), others.join(", ")));
            }
        }
    }
    for r in row {
        println!("   {r}");
    }
}
