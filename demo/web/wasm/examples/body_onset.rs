//! A clap from a source that walks with the ears (the Godot demo's
//! hands, 0.4 m under them): its response the moment it is switched on
//! against one traced all along — the hands kept on cost every tick what
//! a moving source costs (its mirror chains named and solved each tick),
//! so a host would switch them on only to clap. Per tick after the
//! switch, the hands' level (mid band, dB) by part, both ways.
//!
//!     cargo run --release -p omg-web --example body_onset
//!     AT=x,y (the seminar: 5.5,17)  GEOMETRY=<file>  LEVEL=n (2)
use omg_audio::{Config, Frame, Quality, Simulation, SourceDesc, SourceId};
use omg_web::town::Town;

fn parts(f: &Frame, id: SourceId) -> [f64; 4] {
    let Some(pb) = f.block(id) else { return [0.0; 4] };
    let taps: f64 = pb.taps.iter().map(|t| (t.gains[1] as f64).powi(2)).sum();
    [taps, (pb.early.level[1] as f64).powi(2), (pb.reverb.level[1] as f64).powi(2), pb.reverb.rt60[1] as f64]
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn main() {
    let env = |k: &str| std::env::var(k).ok();
    let at: Vec<f32> = env("AT").map_or(vec![5.5, 17.0], |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let level: usize = env("LEVEL").and_then(|v| v.parse().ok()).unwrap_or(2);
    let run = |always: bool| -> Vec<[f64; 4]> {
        let cfg = Config { quality: Quality::Level(level), v6: true, v61: true, v62: true, kernel_thread: false, ..Config::default() };
        let mut sim = Simulation::new(cfg);
        let mut town = Town::new(&mut sim);
        if let Some(path) = env("GEOMETRY") {
            sim.set_geometry(omg_audio::Geometry::load(&path).expect("GEOMETRY: a dumped geometry"));
        }
        for i in 0..town.defs.len() {
            town.set_muted(&mut sim, i, true);
        }
        let ears = [at[0], at[1], 1.6];
        let hands = sim.add_source(SourceDesc { active: always, ..SourceDesc::at([ears[0], ears[1], 1.2]) });
        let mut out = Vec::new();
        for k in 0..120 {
            let t = k as f64 * 0.05;
            sim.set_time(t);
            // the ears stand, as for a clap; the hands switched on at 3 s
            sim.set_listener_at(ears, 0.0, 0.0, 0.0, t);
            sim.set_source_at(hands, [ears[0], ears[1], 1.2], always || k >= 60, t);
            let f = sim.tick();
            if k >= 60 {
                out.push(parts(&f, hands));
            }
        }
        out
    };
    let (on, switched) = (run(true), run(false));
    println!("body_onset at ({}, {}): the hands' level, mid band dB (taps, early, reverb, rt60) — on all along | switched on at the clap", at[0], at[1]);
    for (k, (a, b)) in on.iter().zip(&switched).enumerate().take(24) {
        println!(
            "  tick {k:2}: {:6.1} {:6.1} {:6.1} {:4.2} s | {:6.1} {:6.1} {:6.1} {:4.2} s",
            db(a[0]), db(a[1]), db(a[2]), a[3], db(b[0]), db(b[1]), db(b[2]), b[3]
        );
    }
}
