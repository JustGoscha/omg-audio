//! A recorded tour (`demo/godot/demo/record.gd`, scenario `tour`) read
//! back step by step: per step of the walk (`.tour.csv`) its level and its
//! three bands (under 250 Hz, 250 Hz – 2.5 kHz, over 2.5 kHz; dB of the
//! output), and what stands out — sustained jumps (a level 10 dB over the
//! median of the 2 s before, held 150 ms or more) that no scripted event
//! explains (a clap, the bells, an owl, a car, a door, the curtains, the
//! 0.6 s after one), with where the ears were.
//!
//!     cargo run --release -p omg-web --example tour_scan -- <base>
//!     (<base>.wav, <base>.tour.csv, <base>.frames.csv)
use std::fs;

fn wav(path: &str) -> (u32, Vec<f32>, Vec<f32>) {
    let b = fs::read(path).expect("wav");
    let sr = u32::from_le_bytes(b[24..28].try_into().unwrap());
    let mut i = 12;
    let mut data = &b[0..0];
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let n = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            data = &b[i + 8..(i + 8 + n).min(b.len())];
            break;
        }
        i += 8 + n;
    }
    let (mut l, mut r) = (Vec::new(), Vec::new());
    for f in data.chunks_exact(8) {
        l.push(f32::from_le_bytes(f[0..4].try_into().unwrap()));
        r.push(f32::from_le_bytes(f[4..8].try_into().unwrap()));
    }
    (sr, l, r)
}

#[derive(Clone, Copy)]
struct Biq {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Biq {
    fn new(fc: f64, sr: f64, high: bool) -> Self {
        let w = std::f64::consts::TAU * fc / sr;
        let (c, al) = (w.cos(), w.sin() / std::f64::consts::SQRT_2);
        let a0 = 1.0 + al;
        let b = if high { [(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0] } else { [(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0] };
        Self { b: [b[0] / a0, b[1] / a0, b[2] / a0], a: [-2.0 * c / a0, (1.0 - al) / a0], z: [0.0; 2] }
    }
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn main() {
    let base = std::env::args().nth(1).expect("base path");
    let (sr, l, r) = wav(&format!("{base}.wav"));
    let sr_f = sr as f64;
    // the steps: (start s, kind, phase)
    let steps: Vec<(f64, String, String)> = fs::read_to_string(format!("{base}.tour.csv"))
        .expect("tour.csv")
        .lines()
        .skip(1)
        .filter_map(|line| {
            let c: Vec<&str> = line.split(',').collect();
            Some((c.first()?.parse().ok()?, c.get(2)?.to_string(), c.get(3)?.to_string()))
        })
        .collect();
    let frames: Vec<(f64, f64, f64)> = fs::read_to_string(format!("{base}.frames.csv"))
        .unwrap_or_default()
        .lines()
        .skip(1)
        .filter_map(|line| {
            let c: Vec<&str> = line.split(',').collect();
            Some((c.first()?.parse::<f64>().ok()? / 1000.0, c.get(1)?.parse().ok()?, c.get(2)?.parse().ok()?))
        })
        .collect();
    let at = |t: f64| -> (f64, f64) {
        let k = frames.partition_point(|f| f.0 < t).min(frames.len().saturating_sub(1));
        frames.get(k).map_or((0.0, 0.0), |f| (f.1, f.2))
    };
    // 50 ms windows: total and the three bands (mean square of L and R)
    let w = (0.05 * sr_f) as usize;
    let (mut lo_l, mut lo_r) = (Biq::new(250.0, sr_f, false), Biq::new(250.0, sr_f, false));
    let (mut mh_l, mut mh_r, mut ml_l, mut ml_r) = (Biq::new(250.0, sr_f, true), Biq::new(250.0, sr_f, true), Biq::new(2500.0, sr_f, false), Biq::new(2500.0, sr_f, false));
    let (mut hi_l, mut hi_r) = (Biq::new(2500.0, sr_f, true), Biq::new(2500.0, sr_f, true));
    let mut win: Vec<[f64; 4]> = Vec::new();
    for c in 0..l.len() / w {
        let mut e = [0.0f64; 4];
        for i in c * w..(c + 1) * w {
            let (x, y) = (l[i] as f64, r[i] as f64);
            e[0] += 0.5 * (x * x + y * y);
            let (a, b) = (lo_l.tick(x), lo_r.tick(y));
            e[1] += 0.5 * (a * a + b * b);
            let (a, b) = (ml_l.tick(mh_l.tick(x)), ml_r.tick(mh_r.tick(y)));
            e[2] += 0.5 * (a * a + b * b);
            let (a, b) = (hi_l.tick(x), hi_r.tick(y));
            e[3] += 0.5 * (a * a + b * b);
        }
        win.push(e.map(|v| v / w as f64));
    }
    let t_of = |k: usize| (k as f64 + 0.5) * 0.05;
    println!("{base}: {:.1} s, {} steps", l.len() as f64 / sr_f, steps.len());
    println!("  step  start  phase         kind      mean dB   max dB   low / mid / high dB");
    for (i, (t0, kind, phase)) in steps.iter().enumerate() {
        let t1 = steps.get(i + 1).map_or(l.len() as f64 / sr_f, |s| s.0);
        let ks: Vec<usize> = (0..win.len()).filter(|&k| t_of(k) >= *t0 && t_of(k) < t1).collect();
        if ks.is_empty() {
            continue;
        }
        let mean = |b: usize| db(ks.iter().map(|&k| win[k][b]).sum::<f64>() / ks.len() as f64);
        let max = ks.iter().map(|&k| db(win[k][0])).fold(f64::MIN, f64::max);
        println!("  {i:4} {t0:6.1}  {phase:13} {kind:8} {:8.1} {max:8.1}   {:6.1} {:6.1} {:6.1}", mean(0), mean(1), mean(2), mean(3));
    }
    // what stands out: 10 dB over the median of the 2 s before, held 150 ms,
    // not in the 0.6 s after a scripted event
    let events: Vec<f64> = steps.iter().filter(|s| !matches!(s.1.as_str(), "go" | "wait")).map(|s| s.0).collect();
    let mut found = 0;
    let mut k = 40;
    println!("  standing out (10 dB over the 2 s before, 150 ms or more, no event):");
    while k < win.len() {
        let mut prev: Vec<f64> = (k - 40..k).map(|j| db(win[j][0])).collect();
        prev.sort_by(|a, b| a.total_cmp(b));
        let med = prev[20];
        let run = (k..win.len()).take_while(|&j| db(win[j][0]) > med + 10.0).count();
        let t = t_of(k);
        let near_event = events.iter().any(|&e| t >= e - 0.05 && t <= e + 0.6);
        if run >= 3 && !near_event {
            let (x, y) = at(t);
            let phase = steps.iter().rev().find(|s| s.0 <= t).map_or("", |s| s.2.as_str());
            let peak = (k..k + run).map(|j| db(win[j][0])).fold(f64::MIN, f64::max);
            println!("    {t:7.2} s at ({x:.1}, {y:.1}) [{phase}]: {:.0} ms up to {peak:.1} dB, {:+.1} over the 2 s before", run as f64 * 50.0, peak - med);
            found += 1;
            k += run;
        } else {
            k += 1;
        }
    }
    println!("  {found} moments");
    // dropouts: 15 dB under the median of the 2 s before, 150 ms or more
    // (the ears inside something solid — the route through a kiosk — or
    // the engine falling silent)
    let mut drops = 0;
    let mut k = 40;
    println!("  dropouts (15 dB under the 2 s before, 150 ms or more, no event):");
    while k < win.len() {
        let mut prev: Vec<f64> = (k - 40..k).map(|j| db(win[j][0])).collect();
        prev.sort_by(|a, b| a.total_cmp(b));
        let med = prev[20];
        let run = (k..win.len()).take_while(|&j| db(win[j][0]) < med - 15.0).count();
        let t = t_of(k);
        let near_event = events.iter().any(|&e| t >= e - 0.05 && t <= e + 0.6);
        if run >= 3 && !near_event {
            let (x, y) = at(t);
            let phase = steps.iter().rev().find(|s| s.0 <= t).map_or("", |s| s.2.as_str());
            let low = (k..k + run).map(|j| db(win[j][0])).fold(f64::MAX, f64::min);
            println!("    {t:7.2} s at ({x:.1}, {y:.1}) [{phase}]: {:.0} ms down to {low:.1} dB, {:+.1} under the 2 s before", run as f64 * 50.0, low - med);
            drops += 1;
            k += run;
        } else {
            k += 1;
        }
    }
    println!("  {drops} dropouts");
}
