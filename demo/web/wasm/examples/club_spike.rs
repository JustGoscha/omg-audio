//! A pocket of loudness caught at its tick: the club alone, the ears at one
//! point, per tick the club's traced field (mid band, dB, the direct sound
//! left out), and at a tick it jumps over 6 dB the strongest sampled paths
//! of that tick — the ray that brought it, vertex by vertex
//! (`examples/club_map` finds the points).
//!
//!     cargo run --release -p omg-web --example club_spike
//!     AT=x,y,z (38,40,1.6)  SEED=n  TICKS=n (60)  GEOMETRY=<file>
//!     SOURCE=name (club): another of the town's sources alone  OPEN=i,j (doors)
//!     PORTALS=1: the field through openings from the source's side
use omg_audio::{Config, PathKind, Quality, Simulation};
use omg_web::town::Town;

fn main() {
    let env = |k: &str| std::env::var(k).ok();
    let at: Vec<f32> = env("AT").map_or(vec![38.0, 40.0, 1.6], |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let seed: u64 = env("SEED").and_then(|v| v.parse().ok()).unwrap_or(1);
    // PORTALS=1: what reaches the ears through openings from the source's side
    let portals = env("PORTALS").as_deref() == Some("1");
    let cfg = Config { quality: Quality::High, v4: false, v5: false, v6: true, v61: true, v62: true, portals, seed, kernel_thread: false, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    if let Some(path) = env("GEOMETRY") {
        sim.set_geometry(omg_audio::Geometry::load(&path).expect("geometry"));
    }
    let name = env("SOURCE").unwrap_or_else(|| "club".into());
    let club = town.defs.iter().position(|d| d.name == name).expect("the source");
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != club);
    }
    // OPEN=i,j: open those doors (the club's inner door is 4)
    if let Some(open) = env("OPEN") {
        for i in open.split(',').filter_map(|x| x.parse::<usize>().ok()) {
            town.set_door(&mut sim, i, 1.0);
        }
    }
    let ids = town.sources[club].clone();
    sim.set_listener([at[0], at[1], at[2]], 0.0, 0.0, 0.0);
    sim.set_debug_paths(true);
    let ticks: usize = env("TICKS").and_then(|v| v.parse().ok()).unwrap_or(60);
    let mut prev = 0.0f64;
    for k in 0..ticks {
        let f = sim.tick();
        let mut e = 0.0f64;
        for &id in &ids {
            if let Some(pb) = f.block(id) {
                e += pb.taps.iter().filter(|t| t.key != omg_core::v3::KEY_DIRECT).map(|t| (t.gains[1] as f64).powi(2)).sum::<f64>();
                e += (pb.early.level[1] as f64).powi(2) + (pb.reverb.level[1] as f64).powi(2) + pb.remote.map_or(0.0, |r| (r.send[1] as f64).powi(2));
            }
        }
        let jump = 10.0 * (e / prev.max(1e-30)).log10();
        println!("tick {k:3}: {:+6.1} dB ({jump:+5.1})", 10.0 * e.max(1e-30).log10());
        if k > 0 && jump > 6.0 {
            let paths = sim.debug_paths();
            for p in paths.iter().filter(|p| ids.contains(&p.source) && p.kind == PathKind::Sampled).take(5) {
                let pts: Vec<String> = p.points.iter().map(|q| format!("({:.2}, {:.2}, {:.2})", q[0], q[1], q[2])).collect();
                println!("     sampled path {:+.1} dB: {}", 10.0 * (p.energy as f64).max(1e-30).log10(), pts.join(" → "));
            }
        }
        prev = e;
    }
}
