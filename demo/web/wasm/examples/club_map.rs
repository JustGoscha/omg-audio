//! The club heard around the town, mapped: only the club playing, the ears
//! standing at every point of a grid (1.6 m up) — a fresh simulation at
//! each, settled (walked from point to point, each inherits the last's
//! averages) — its level
//! (mid band, dB, its speakers' fields summed) by part — direct, taps,
//! early field, reverb, remote — and the points that stand out of their
//! neighbours ("pockets": louder than every neighbour's median by `POCKET`
//! dB), with what makes them.
//!
//!     cargo run --release -p omg-web --example club_map
//!     AREA=x0,x1,y0,y1 (14,42,18,46)  STEP=m (1)  TICKS=n (40)  POCKET=dB (6)
//!     MODE=v6.2 | v6.1 | v6 | v5   GEOMETRY=<file>  SEED=n
use omg_audio::{Config, Frame, Quality, Simulation, SourceId};
use omg_web::town::Town;

fn parts(f: &Frame, ids: &[SourceId]) -> [f64; 5] {
    let mut p = [0.0f64; 5];
    for &id in ids {
        let Some(pb) = f.block(id) else { continue };
        for t in &pb.taps {
            let e = (t.gains[1] as f64).powi(2);
            if t.key == omg_core::v3::KEY_DIRECT {
                p[0] += e;
            } else {
                p[1] += e;
            }
        }
        p[2] += (pb.early.level[1] as f64).powi(2);
        p[3] += (pb.reverb.level[1] as f64).powi(2);
        p[4] += pb.remote.map_or(0.0, |r| (r.send[1] as f64).powi(2));
    }
    p
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn main() {
    let env = |k: &str| std::env::var(k).ok();
    let area: Vec<f32> = env("AREA").map_or(vec![14.0, 42.0, 18.0, 46.0], |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let step: f32 = env("STEP").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let ticks: usize = env("TICKS").and_then(|v| v.parse().ok()).unwrap_or(40);
    let pocket: f64 = env("POCKET").and_then(|v| v.parse().ok()).unwrap_or(6.0);
    let mode = env("MODE").unwrap_or_else(|| "v6.2".into());
    let cfg = Config {
        quality: Quality::High,
        v4: mode == "v5",
        v5: mode == "v5",
        v6: mode.starts_with("v6"),
        v61: mode == "v6.1" || mode == "v6.2",
        v62: mode == "v6.2",
        seed: env("SEED").and_then(|v| v.parse().ok()).unwrap_or(1),
        kernel_thread: false,
        ..Config::default()
    };
    let geometry = env("GEOMETRY");
    // a simulation with the club alone
    let fresh = || {
        let mut sim = Simulation::new(cfg);
        let mut town = Town::new(&mut sim);
        if let Some(path) = geometry.as_ref() {
            sim.set_geometry(omg_audio::Geometry::load(path).expect("GEOMETRY: a dumped geometry"));
        }
        let club = town.defs.iter().position(|d| d.name == "club").expect("the club");
        for i in 0..town.defs.len() {
            town.set_muted(&mut sim, i, i != club);
        }
        let ids = town.sources[club].clone();
        (sim, ids)
    };
    let (nx, ny) = (((area[1] - area[0]) / step) as usize + 1, ((area[3] - area[2]) / step) as usize + 1);
    // per point: the level by part, the second half of the ticks there
    let mut grid: Vec<Vec<[f64; 5]>> = vec![vec![[0.0; 5]; nx]; ny];
    for (j, row) in grid.iter_mut().enumerate() {
        for (i, cell) in row.iter_mut().enumerate() {
            let (x, y) = (area[0] + i as f32 * step, area[2] + j as f32 * step);
            let (mut sim, ids) = fresh();
            sim.set_listener([x, y, 1.6], 0.0, 0.0, 0.0);
            for k in 0..ticks {
                let f = sim.tick();
                if k >= ticks / 2 {
                    let p = parts(&f, &ids);
                    for (c, v) in cell.iter_mut().zip(p) {
                        *c += v / (ticks - ticks / 2) as f64;
                    }
                }
            }
        }
    }
    let total = |p: &[f64; 5]| db(p.iter().sum());
    println!("club_map ({mode}): the club alone, mid band dB, x {}…{} y {}…{} every {step} m (rows north to south)", area[0], area[1], area[2], area[3]);
    print!("        ");
    for i in (0..nx).step_by(2) {
        print!("{:>6}", area[0] + i as f32 * step);
    }
    println!();
    for j in (0..ny).rev() {
        print!("y {:5.1} ", area[2] + j as f32 * step);
        for i in (0..nx).step_by(2) {
            print!("{:>6.0}", total(&grid[j][i]));
        }
        println!();
    }
    // pockets: louder than the median of the 8 neighbours by `pocket` dB
    println!("pockets (louder than their neighbours' median by {pocket} dB):");
    let mut n = 0;
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let mut nb: Vec<f64> = Vec::new();
            for dj in [-1i32, 0, 1] {
                for di in [-1i32, 0, 1] {
                    if di != 0 || dj != 0 {
                        nb.push(total(&grid[(j as i32 + dj) as usize][(i as i32 + di) as usize]));
                    }
                }
            }
            nb.sort_by(|a, b| a.total_cmp(b));
            let med = 0.5 * (nb[3] + nb[4]);
            let here = total(&grid[j][i]);
            if here - med > pocket {
                n += 1;
                let p = &grid[j][i];
                println!(
                    "   ({:.1}, {:.1}): {:+.1} dB, {:+.1} over its neighbours — direct {:+.0} taps {:+.0} early {:+.0} reverb {:+.0} remote {:+.0}",
                    area[0] + i as f32 * step,
                    area[2] + j as f32 * step,
                    here,
                    here - med,
                    db(p[0]),
                    db(p[1]),
                    db(p[2]),
                    db(p[3]),
                    db(p[4])
                );
            }
        }
    }
    println!("{n} pockets");
}
