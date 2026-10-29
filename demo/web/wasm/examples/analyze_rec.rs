//! Reads a recording of what Godot played (`demo/godot/demo/record.gd`:
//! the WAV, its audio callbacks `.wav.csv`, the game frames
//! `.frames.csv`) and reports, by where the ears were (outside, crossing
//! the wall, inside — the fly scenario's y):
//!
//!  - artifacts: with `--tone` the flute played a pure 500 Hz alone, so
//!    the output above 3 kHz (4th-order high-pass) is what the engine
//!    made — pops, clicks, crackle — per 5 ms window, under the whole;
//!  - jumps: the 10 ms level off its own course (median of nine) by
//!    more than 2 / 4 dB;
//!  - the audio thread: callbacks with frames rendered late (the output
//!    thread behind), the slowest take, irregular callback spacing;
//!  - the engine: tick time, load, budget changes, late frames.
//!
//!     cargo run --release -p omg-web --example analyze_rec -- <base>
//!     (<base>.wav, <base>.wav.csv, <base>.frames.csv)
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

fn csv(path: &str) -> Vec<Vec<f64>> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .skip(1)
        .map(|l| l.split(',').map(|x| x.parse().unwrap_or(0.0)).collect())
        .collect()
}

#[derive(Clone, Copy)]
struct Biq {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Biq {
    fn hp(fc: f64, sr: f64) -> Self {
        let w = std::f64::consts::TAU * fc / sr;
        let (c, al) = (w.cos(), w.sin() / std::f64::consts::SQRT_2);
        let a0 = 1.0 + al;
        Self { b: [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0], a: [-2.0 * c / a0, (1.0 - al) / a0], z: [0.0; 2] }
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
    let sr = sr as f64;
    let calls = csv(&format!("{base}.wav.csv"));
    let frames = csv(&format!("{base}.frames.csv"));
    // the ears' y at a time (s), from the game frames
    let y_at = |t: f64| -> f64 {
        let ms = t * 1000.0;
        let k = frames.partition_point(|f| f[0] < ms).min(frames.len().saturating_sub(1));
        frames.get(k).map_or(0.0, |f| f[2])
    };
    let wall = 52.0;
    let phase = |y: f64| if y < wall - 1.0 { 0 } else if y < wall + 1.5 { 1 } else { 2 };
    let names = ["outside", "crossing", "inside"];
    println!("{base}: {:.1} s at {} Hz", l.len() as f64 / sr, sr);

    // artifacts: above 3 kHz, 5 ms windows
    let (mut h1, mut h2, mut h3, mut h4) = (Biq::hp(3000.0, sr), Biq::hp(3000.0, sr), Biq::hp(3000.0, sr), Biq::hp(3000.0, sr));
    let w = (0.005 * sr) as usize;
    let mut art: [Vec<(f64, f64)>; 3] = Default::default();
    let (mut eh, mut et) = (0.0f64, 0.0f64);
    for i in 0..l.len() {
        let (a, b) = (l[i] as f64, r[i] as f64);
        let ha = {
            let y = h1.tick(a);
            h2.tick(y)
        };
        let hb = {
            let y = h3.tick(b);
            h4.tick(y)
        };
        eh += ha * ha + hb * hb;
        et += a * a + b * b;
        if (i + 1) % w == 0 {
            let t = i as f64 / sr;
            if t > 0.05 {
                art[phase(y_at(t))].push((t, db(eh / et.max(1e-30))));
            }
            eh = 0.0;
            et = 0.0;
        }
    }
    println!("  artifacts (above 3 kHz, under the whole; a pure tone's output has none):");
    for (p, v) in art.iter().enumerate() {
        if v.is_empty() {
            continue;
        }
        let mut s: Vec<f64> = v.iter().map(|x| x.1).collect();
        s.sort_by(|a, b| a.total_cmp(b));
        let over = |d: f64| v.iter().filter(|x| x.1 > d).count();
        println!("    {:8} {:4} windows: median {:6.1} dB, worst {:6.1} dB, over −40 dB {:3}, over −30 dB {:3}", names[p], v.len(), s[s.len() / 2], s[s.len() - 1], over(-40.0), over(-30.0));
    }
    let mut worst: Vec<(f64, f64)> = art.iter().flatten().copied().filter(|x| x.1 > -30.0).collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    if !worst.is_empty() {
        let w: Vec<String> = worst.iter().take(8).map(|(t, d)| format!("{t:.3}s {d:.0}dB (y {:.1})", y_at(*t))).collect();
        println!("    loudest: {}", w.join(", "));
    }

    // jumps: 10 ms level against its median of nine
    let w = (0.010 * sr) as usize;
    let lv: Vec<(f64, f64)> = l
        .chunks(w)
        .zip(r.chunks(w))
        .enumerate()
        .map(|(k, (a, b))| ((k * w) as f64 / sr, db(a.iter().zip(b).map(|(x, y)| (x * x + y * y) as f64).sum::<f64>() / w as f64)))
        .collect();
    let mut jumps = [[0usize; 3]; 3];
    for k in 4..lv.len().saturating_sub(4) {
        let mut m: Vec<f64> = lv[k - 4..=k + 4].iter().map(|x| x.1).collect();
        m.sort_by(|a, b| a.total_cmp(b));
        let d = (lv[k].1 - m[4]).abs();
        let p = phase(y_at(lv[k].0));
        jumps[p][0] += 1;
        jumps[p][1] += (d > 2.0) as usize;
        jumps[p][2] += (d > 4.0) as usize;
    }
    println!("  level jumps (10 ms off its 90 ms course):");
    for p in 0..3 {
        if jumps[p][0] > 0 {
            println!("    {:8} {:4} windows: >2 dB {:3}, >4 dB {:3}", names[p], jumps[p][0], jumps[p][1], jumps[p][2]);
        }
    }

    // the audio thread
    let late: f64 = calls.iter().map(|c| c[3]).sum();
    let late_calls = calls.iter().filter(|c| c[3] > 0.0).count();
    let take_max = calls.iter().map(|c| c[2]).fold(0.0, f64::max);
    let mut gaps: Vec<f64> = calls.windows(2).map(|c| c[1][0] - c[0][0]).collect();
    gaps.sort_by(|a, b| a.total_cmp(b));
    let (gmed, gmax) = (gaps.get(gaps.len() / 2).copied().unwrap_or(0.0), gaps.last().copied().unwrap_or(0.0));
    println!(
        "  audio thread: {} callbacks of {} frames; {} with frames rendered late ({} frames); slowest take {:.0} µs; spacing median {:.1} ms, max {:.1} ms",
        calls.len(),
        calls.first().map_or(0.0, |c| c[1]),
        late_calls,
        late,
        take_max,
        gmed,
        gmax
    );

    // the engine
    if !frames.is_empty() {
        let col = |i: usize| frames.iter().map(|f| f[i]).collect::<Vec<f64>>();
        let (tick, load, taps) = (col(3), col(4), col(5));
        let tmax = tick.iter().cloned().fold(0.0, f64::max);
        let lmax = load.iter().cloned().fold(0.0, f64::max);
        let changes = taps.windows(2).filter(|w| w[0] != w[1]).count();
        println!(
            "  engine: tick {:.1}–{:.1} ms, load up to {:.2}, budget changes {} (taps {:.0} → {:.0}), late frames {:.0}",
            tick.iter().cloned().fold(f64::MAX, f64::min),
            tmax,
            lmax,
            changes,
            taps.first().copied().unwrap_or(0.0),
            taps.last().copied().unwrap_or(0.0),
            frames.last().map_or(0.0, |f| f[8])
        );
    }
}
