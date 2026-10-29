//! A source's traced level tick by tick, the listener standing still:
//! what of it is exact (direct, bends, mirror chains) and what estimated
//! (the dense response), per tick — the estimate's own noise, before any
//! mixer; and the reverberation time it plays (v3/v6/v6.1: the parametric
//! reverb's; v4/v5: the dense response's decay). MODE=v3|v4|v5|v6|v6.1,
//! SRC=logical source, AT=x,y,z.
//!
//!     cargo run --release -p omg-web --example probe
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;

fn main() {
    let mode = std::env::var("MODE").unwrap_or_else(|_| "v5".into());
    let src: usize = std::env::var("SRC").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let at: Vec<f32> = std::env::var("AT").unwrap_or_else(|_| "15,32,1.6".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let ticks: usize = std::env::var("TICKS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let cfg = Config { quality: Quality::High, compute: Compute::Cpu, v4: mode == "v4" || mode == "v5", v5: mode == "v5", v6: mode.starts_with("v6"), v61: mode == "v6.1" || mode == "v6.2", v62: mode == "v6.2", kernel_thread: false, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    // DOOR=i@tick: door i shuts at that tick (doors 3 and i open before)
    let door: Option<(usize, usize)> = std::env::var("DOOR").ok().map(|d| {
        let (i, t) = d.split_once('@').unwrap();
        (i.parse().unwrap(), t.parse().unwrap())
    });
    if let Some((i, _)) = door {
        town.set_door(&mut sim, 3, 1.0);
        town.set_door(&mut sim, i, 1.0);
    }
    let ids = town.sources[src].clone();
    if std::env::var("ONLY").is_ok() {
        for i in 0..town.defs.len() {
            town.set_muted(&mut sim, i, i != src);
        }
    }
    sim.set_listener([at[0], at[1], at[2]], 0.0, 0.0, 0.0);
    let debug = std::env::var("PATHS").is_ok();
    sim.set_debug_paths(debug);
    println!(" tick   direct  bends  chains(n)   dense    reverb(lvl²)   total  rt60 (mid)  [dB, mid band, all of the source's emitters]");
    let db = |x: f32| 10.0 * x.max(1e-30).log10();
    for t in 0..ticks {
        if let Some((i, at)) = door {
            if t == at {
                town.set_door(&mut sim, i, 0.0);
            }
        }
        let f = sim.tick();
        let (mut d, mut b, mut c, mut nc, mut dense, mut rev) = (0.0f32, 0.0f32, 0.0f32, 0usize, 0.0f32, 0.0f32);
        let mut rt60 = 0.0f32;
        let mut early_d = 0.0f32;
        for id in &ids {
            let Some(pb) = f.block(*id) else { continue };
            for tap in &pb.taps {
                let e = tap.gains[1] * tap.gains[1];
                match tap.key {
                    1 => d += e,
                    20_000..=8_999_999 => b += e,
                    k if k >= 9_000_000 => {
                        c += e;
                        nc += 1;
                    }
                    _ => c += e,
                }
            }
            rev += pb.reverb.level[1] * pb.reverb.level[1] + pb.remote.as_ref().map_or(0.0, |r| r.send[1] * r.send[1]);
            rt60 = rt60.max(pb.reverb.rt60[1]);
            if let Some(dr) = sim.dense_response(*id) {
                dense += dr.bins.iter().map(|x| x[1]).sum::<f32>();
                early_d += dr.bins.iter().take((0.08 / dr.bin_s) as usize).map(|x| x[1]).sum::<f32>();
            }
        }
        let parametric = mode == "v3" || mode.starts_with("v6");
        let total = d + b + c + if parametric { rev } else { dense };
        if t % 20 == 19 || std::env::var("EVERY").is_ok() {
            println!("{t:5}  {:6.1} {:6.1} {:6.1}({nc:3}) {:7.1}   {:7.1}      {:7.1}   {rt60:5.2} s   early dense {:7.1}", db(d), db(b), db(c), db(dense), db(rev), db(total), db(early_d));
        }
    }
    if debug {
        for p in sim.debug_paths().iter().filter(|p| ids.contains(&p.source) && p.kind != omg_audio::PathKind::Sampled) {
            let pts: Vec<String> = p.points.iter().map(|q| format!("({:.1},{:.1},{:.1})", q[0], q[1], q[2])).collect();
            println!("{:?} {:.1} dB: {}", p.kind, 10.0 * p.energy.max(1e-30).log10(), pts.join(" "));
        }
    }
}
