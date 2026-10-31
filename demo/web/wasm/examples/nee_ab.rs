//! How close every source's scattered response comes to a reference after
//! a short estimate (TICKS, from a standing start), the whole town traced:
//! per source and listener spot, early (< 0.25 s) and late energy per band
//! against a long, many-ray run (REF=1 writes it). The tracer's estimators
//! compared on equal footing.
//!
//!   REF=1 TICKS=100 RAYS=8192 LATE=2048 NEE=0 … > ref.txt
//!   NEE=3 TICKS=40 SEEDS=4 … -- ref.txt
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

const SPOTS: [[f32; 3]; 5] = [[15.0, 32.0, 1.6], [27.0, 48.0, 1.6], [10.0, 19.0, 1.6], [20.0, 60.0, 1.6], [31.0, 36.0, 1.6]];

/// Per spot, per engine source: [early low, mid, high, late low, mid, high].
fn run(seed: u64, ticks: usize) -> Vec<Vec<[f64; 6]>> {
    let mut b = Quality::High.budget(false);
    let num = |k: &str| env(k).and_then(|n| n.parse::<usize>().ok());
    if let Some(n) = num("NEE") {
        b.nee = n;
    }
    if let Some(n) = num("LATE_NEE") {
        b.late_nee = n;
    }
    if let Some(n) = num("LATE") {
        b.late_rays = n as u32;
    }
    if let Some(n) = num("RAYS") {
        b.rays = n as u32;
    }
    let mut out = Vec::new();
    for pos in SPOTS {
        let cfg = Config { quality: Quality::Custom(b), compute: Compute::Cpu, kernel_thread: false, kernel_step_samples: 1 << 24, seed, ..Config::default() };
        let mut sim = Simulation::new(cfg);
        let town = Town::new(&mut sim);
        for k in 0..ticks {
            sim.set_listener_at(pos, 0.0, 0.0, 0.0, k as f64 * 0.05);
            sim.tick();
        }
        let mut row = Vec::new();
        for ids in &town.sources {
            for &id in ids {
                let mut e = [0.0f64; 6];
                if let Some(d) = sim.dense_response(id) {
                    for (i, bin) in d.bins.iter().enumerate() {
                        let o = if i as f32 * d.bin_s < 0.25 { 0 } else { 3 };
                        for b in 0..3 {
                            e[o + b] += bin[b] as f64;
                        }
                    }
                }
                row.push(e);
            }
        }
        out.push(row);
    }
    out
}

fn main() {
    let ticks = env("TICKS").and_then(|n| n.parse().ok()).unwrap_or(40);
    if env("REF").is_some() {
        for row in run(99, ticks) {
            let v: Vec<String> = row.iter().map(|e| e.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(",")).collect();
            println!("{}", v.join(";"));
        }
        return;
    }
    let path = std::env::args().nth(1).expect("reference file");
    let text = std::fs::read_to_string(path).unwrap();
    let reference: Vec<Vec<[f64; 6]>> = text
        .lines()
        .map(|l| l.split(';').map(|e| {
            let v: Vec<f64> = e.split(',').map(|x| x.parse().unwrap()).collect();
            core::array::from_fn(|i| v[i])
        }).collect())
        .collect();
    let seeds: u64 = env("SEEDS").and_then(|n| n.parse().ok()).unwrap_or(4);
    // error in dB per (part, band) — capped at 20 dB (a source no sample
    // reached) — over sources whose reference is within 60 dB of the
    // loudest at that spot (what can be heard)
    let range = 10f64.powf(-env("RANGE_DB").and_then(|n| n.parse::<f64>().ok()).unwrap_or(60.0) / 10.0);
    let mut errs: [Vec<f64>; 6] = Default::default();
    for seed in 1..=seeds {
        let got = run(seed, ticks);
        for (s, row) in got.iter().enumerate() {
            for k in 0..6 {
                let top = reference[s].iter().map(|e| e[k]).fold(0.0, f64::max);
                for (i, e) in row.iter().enumerate() {
                    let r = reference[s][i][k];
                    if r <= top * range || r <= 0.0 {
                        continue;
                    }
                    errs[k].push((10.0 * (e[k].max(1e-30) / r).log10()).abs().min(20.0));
                }
            }
        }
    }
    let names = ["early low", "early mid", "early high", "late low", "late mid", "late high"];
    for k in 0..6 {
        let v = &mut errs[k];
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let over = |x: f64| 100.0 * v.iter().filter(|&&e| e > x).count() as f64 / n;
        println!("{:10}  mean {:5.2} dB  median {:5.2} dB  >3 dB {:4.1} %  >6 dB {:4.1} %  ({} cases)", names[k], mean, v[v.len() / 2], over(3.0), over(6.0), v.len());
    }
}
