//! What the engine plays of a source's mirror echoes: the town, a source
//! standing at SRC, the ears at LIS, TICKS ticks (60); the exactly solved
//! mirror chains (`Simulation::debug_paths`), each with its reflections,
//! delay after the direct sound and level under it (mid band) — to set
//! beside `echo_ref` / `echo_ref2` / `echo_ref3`'s exhaustive lists.
//!
//!     LIS=-13.1,0,1.6 SRC=-17.1,0,0.7 cargo run --release -p omg-web --example echo_check
//!     MODE=v6.2 (default) | v6.1 | v6 | v5   MIN=refl (1: every order)
//!     TAPS=1: the taps the renderer plays as well (the early bins too)
use omg_audio::{Config, PathKind, Quality, Simulation, SourceDesc};
use omg_web::town::Town;

fn env3(k: &str, d: [f32; 3]) -> [f32; 3] {
    let v: Vec<f32> = std::env::var(k).ok().map_or(Vec::new(), |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    if v.len() == 3 {
        [v[0], v[1], v[2]]
    } else {
        d
    }
}

fn main() {
    let lis = env3("LIS", [-13.1, 0.0, 1.6]);
    let sp = env3("SRC", [-17.1, 0.0, 0.7]);
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let min: usize = std::env::var("MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mode = std::env::var("MODE").unwrap_or_else(|_| "v6.2".into());
    let cfg = Config {
        quality: Quality::High,
        v4: mode == "v5",
        v5: mode == "v5",
        v6: mode.starts_with("v6"),
        v61: mode == "v6.1" || mode == "v6.2",
        v62: mode == "v6.2",
        kernel_thread: false,
        ..Config::default()
    };
    let mut sim = Simulation::new(cfg);
    let _town = Town::new(&mut sim);
    let src = sim.add_source(SourceDesc::at(sp));
    sim.set_listener(lis, 0.0, 0.0, 0.0);
    sim.set_debug_paths(true);
    let mut taps = Vec::new();
    for _ in 0..ticks {
        let f = sim.tick();
        if let Some(pb) = f.block(src) {
            taps = pb.taps.iter().map(|t| (t.key, t.delay_s - pb.onset_s, t.gains[1])).collect();
        }
    }
    let paths: Vec<_> = sim.debug_paths().into_iter().filter(|p| p.source == src).collect();
    let len = |pts: &[[f32; 3]]| -> f32 {
        pts.windows(2).map(|w| ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2) + (w[1][2] - w[0][2]).powi(2)).sqrt()).sum()
    };
    let direct = paths.iter().find(|p| p.kind == PathKind::Direct).map(|p| (p.energy, len(&p.points)));
    let Some((de, dl)) = direct else {
        println!("no direct path");
        return;
    };
    println!("echo_check ({mode}): source {sp:?}, ears {lis:?}, direct {dl:.1} m; mirror chains solved (≥ {min} reflections):");
    let mut rows: Vec<(f32, f32, usize, String)> = paths
        .iter()
        .filter(|p| p.kind == PathKind::Mirror && p.points.len() >= min + 2)
        .map(|p| {
            let pts: Vec<String> = p.points[1..p.points.len() - 1].iter().map(|q| format!("({:.1}, {:.1}, {:.1})", q[0], q[1], q[2])).collect();
            ((len(&p.points) - dl) / omg_core::SPEED_OF_SOUND * 1e3, 10.0 * (p.energy / de).log10(), p.points.len() - 2, pts.join(" → "))
        })
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (dt, db, n, pts) in rows.iter().take(24) {
        println!("   +{dt:6.1} ms  {db:+6.1} dB  {n} refl  {pts}");
    }
    // TAPS=1: the taps the renderer plays (mid band), against the direct
    // tap's level: the solved chains and the strongest traced bins
    if std::env::var_os("TAPS").is_some() {
        let d = taps.iter().find(|t| t.0 == omg_core::v3::KEY_DIRECT).map_or(1.0, |t| t.2);
        taps.sort_by(|a, b| a.1.total_cmp(&b.1));
        println!("   taps played (mid band, dB from the direct tap):");
        for (key, dt, g) in &taps {
            println!("      +{:6.1} ms  {:+6.1} dB  key {key:#x}", dt * 1e3, 20.0 * (g / d).log10());
        }
    }
}
