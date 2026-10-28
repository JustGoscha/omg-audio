//! How loud the club arrives around the town: the web's town (the Godot
//! level's layout), only the club playing, its paths' and fields' energy
//! per band at the ears — the transfer from the club's speakers, in dB re
//! the dance floor — at the places of a building-acoustics estimate, for
//! the club's doors (D3 lobby outer, D4 club inner) open/shut and the
//! church portal (D6) and the seminar door (D2) shut/open.
//!
//!     cargo run --release -p omg-web --example club_levels [-- v3]
//!
//! ONLY=P0,P10 picks places; SETTLE / AVG ticks (60 / 40).
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;

const PLACES: [(&str, f32, f32); 13] = [
    ("P0 dance floor", 27.0, 32.0),
    ("P1 lobby", 21.0, 31.0),
    ("P2 street at the lobby", 18.0, 31.0),
    ("P3 2 m E of the window", 34.0, 32.0),
    ("P4 20 m W", 0.0, 31.0),
    ("P5 40 m W", -20.0, 31.0),
    ("P6 20 m N", 27.0, 58.0),
    ("P7 50 m N", 27.0, 88.0),
    ("P8 100 m N", 27.0, 138.0),
    ("P9 outside the portal", 8.0, 50.0),
    ("P10 church", 8.0, 64.0),
    ("P11 outside the seminar door", 7.0, 26.0),
    ("P12 seminar room", 9.0, 17.0),
];

fn main() {
    let v4 = !std::env::args().any(|a| a == "v3");
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (settle, avg) = (env("SETTLE", 60), env("AVG", 40));
    let only: Option<Vec<String>> = std::env::var("ONLY").ok().map(|s| s.split(',').map(|x| x.to_string()).collect());
    let cfg = Config { quality: Quality::High, compute: Compute::Cpu, v4, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    let club = 2usize;
    for i in 0..town.sources.len() {
        if i != club {
            town.set_muted(&mut sim, i, true);
        }
    }
    println!("engine {} · club transfer per band (low / mid / high, dB re the dance floor; total in brackets: paths + reverb, dB re 1)", if v4 { "v4" } else { "v3" });
    // door states: (name, D3, D4, D6, D2)
    let states = [("default (D3 open, D4 shut)", 1.0, 0.0), ("club open (D3, D4 open)", 1.0, 1.0), ("club shut (D3, D4 shut)", 0.0, 0.0)];
    let mut floor: Option<[f32; 3]> = None;
    for &(sname, d3, d4) in &states {
        for (pname, d6, d2) in [("church portal, seminar door shut", 0.0f32, 0.0f32), ("church portal, seminar door open", 1.0, 1.0)] {
            println!("— {sname}; {pname}");
            for (i, d) in [(3usize, d3), (4, d4), (6, d6), (2, d2)] {
                town.set_door(&mut sim, i, d);
            }
            for &(name, x, y) in &PLACES {
                if only.as_ref().is_some_and(|o| !o.iter().any(|p| name.starts_with(p.as_str()))) {
                    continue;
                }
                sim.set_listener([x, y, 1.6], 0.0, 0.0, 0.0);
                for _ in 0..settle {
                    sim.tick();
                }
                let (mut e, mut exact) = ([0.0f64; 3], [0.0f64; 3]);
                for _ in 0..avg {
                    let f = sim.tick();
                    for (id, b) in &f.blocks {
                        if !town.sources[club].contains(id) {
                            continue;
                        }
                        for k in 0..3 {
                            let taps: f64 = b.taps.iter().map(|t| (t.gains[k] as f64).powi(2)).sum();
                            let late = (b.reverb.level[k] as f64).powi(2) + b.remote.as_ref().map_or(0.0, |r| (r.send[k] as f64).powi(2));
                            e[k] += taps + late;
                            exact[k] += taps;
                        }
                    }
                }
                let db = |x: f64| 10.0 * (x / avg as f64).max(1e-20).log10() as f32;
                let g: [f32; 3] = core::array::from_fn(|k| db(e[k]));
                if name.starts_with("P0") && floor.is_none() {
                    floor = Some(g);
                }
                let f0 = floor.unwrap_or(g);
                let paths: [f32; 3] = core::array::from_fn(|k| db(exact[k]) - g[k]);
                println!(
                    "  {name:30} {:+6.1} / {:+6.1} / {:+6.1} dB   ({:+.1} / {:+.1} / {:+.1})   paths share {:+.0} / {:+.0} / {:+.0} dB",
                    g[0] - f0[0], g[1] - f0[1], g[2] - f0[2], g[0], g[1], g[2], paths[0], paths[1], paths[2]
                );
            }
        }
    }
}
