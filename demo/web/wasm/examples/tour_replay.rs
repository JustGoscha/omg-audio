//! A stretch of a recorded tour (`tools/record_tour.sh`) walked again
//! headless: the ears follow the take's frames (`<base>.frames.csv`) from
//! `T0` to `T1` with one source playing, and per 50 ms its level by band
//! (low, mid, high; dB, all its parts) and the low band by part — to tell
//! whether something heard in the take is the engine's or the host's.
//!
//!     cargo run --release -p omg-web --example tour_replay -- <base>
//!     T0=s T1=s (the stretch)  SOURCE=name (club)  OPEN=i,j (doors open)
//!     Z=m (ears' height: 1.6)  MODE=v6.2 | v6.1 | v6  SEED=n
//!     LEVEL=n: the quality ladder's rung (0 … 10; the take's `level`
//!     column says where the host's governor was), else High
//!     GEOMETRY=<file>: the host's dumped level   COMPUTE=gpu   PACE=1
//!     MIX=1: rendered through the mixer too (no HRIR; the source plays
//!     bass-heavy noise, or CLIP=<raw f32 mono 48 kHz>): the output's
//!     level under 250 Hz and in all (dBFS) per 50 ms, after the ear (and
//!     the ear: the level it is adapted to, dB SPL, its gain, its reflex)
use omg_audio::{clip, Compute, Config, Frame, Hrir, Mixer, Play, Quality, Simulation, SourceId};
use omg_web::town::Town;
use std::fs;

/// Per band: direct, taps, early, reverb, remote (energy).
fn parts(f: &Frame, ids: &[SourceId]) -> [[f64; 5]; 3] {
    let mut p = [[0.0f64; 5]; 3];
    for &id in ids {
        let Some(pb) = f.block(id) else { continue };
        for (b, pb_b) in p.iter_mut().enumerate() {
            for t in &pb.taps {
                let e = (t.gains[b] as f64).powi(2);
                pb_b[if t.key == omg_core::v3::KEY_DIRECT { 0 } else { 1 }] += e;
            }
            pb_b[2] += (pb.early.level[b] as f64).powi(2);
            pb_b[3] += (pb.reverb.level[b] as f64).powi(2);
            pb_b[4] += pb.remote.map_or(0.0, |r| (r.send[b] as f64).powi(2));
        }
    }
    p
}

/// Two-pole low-pass (RBJ, Q 0.707).
struct Lp {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Lp {
    fn new(fc: f64, sr: f64) -> Self {
        let w = std::f64::consts::TAU * fc / sr;
        let (c, al) = (w.cos(), w.sin() / std::f64::consts::SQRT_2);
        let a0 = 1.0 + al;
        Self { b: [(1.0 - c) / 2.0 / a0, (1.0 - c) / a0, (1.0 - c) / 2.0 / a0], a: [-2.0 * c / a0, (1.0 - al) / a0], z: [0.0; 2] }
    }
    fn tick(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// Low-passed noise (xorshift, two one-poles).
fn noise(n: usize, seed: u32, lp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    let (mut a, mut b) = (0.0f32, 0.0f32);
    let c = 1.0 - (-std::f32::consts::TAU * lp / 48_000.0).exp();
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let w = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
            a += c * (w - a);
            b += c * (a - b);
            b
        })
        .collect()
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn main() {
    let env = |k: &str| std::env::var(k).ok();
    let base = std::env::args().nth(1).expect("base path");
    let t0: f64 = env("T0").and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let t1: f64 = env("T1").and_then(|v| v.parse().ok()).unwrap_or(t0 + 10.0);
    let z: f32 = env("Z").and_then(|v| v.parse().ok()).unwrap_or(1.6);
    let name = env("SOURCE").unwrap_or_else(|| "club".into());
    let mode = env("MODE").unwrap_or_else(|| "v6.2".into());
    let pace = env("PACE").as_deref() == Some("1");
    let cfg = Config {
        quality: env("LEVEL").and_then(|v| v.parse().ok()).map_or(Quality::High, Quality::Level),
        v6: mode.starts_with("v6"),
        v61: mode == "v6.1" || mode == "v6.2",
        v62: mode == "v6.2",
        seed: env("SEED").and_then(|v| v.parse().ok()).unwrap_or(1),
        kernel_thread: false,
        compute: if env("COMPUTE").as_deref() == Some("gpu") { Compute::Gpu } else { Compute::Cpu },
        ..Config::default()
    };
    // the take's frames: (s, x, y)
    let frames: Vec<(f64, f32, f32)> = fs::read_to_string(format!("{base}.frames.csv"))
        .expect("frames.csv")
        .lines()
        .skip(1)
        .filter_map(|line| {
            let c: Vec<&str> = line.split(',').collect();
            Some((c.first()?.parse::<f64>().ok()? / 1000.0, c.get(1)?.parse().ok()?, c.get(2)?.parse().ok()?))
        })
        .collect();
    let at = |t: f64| -> [f32; 3] {
        let k = frames.partition_point(|f| f.0 < t).clamp(1, frames.len() - 1);
        let (a, b) = (frames[k - 1], frames[k]);
        let u = (((t - a.0) / (b.0 - a.0).max(1e-6)) as f32).clamp(0.0, 1.0);
        [a.1 + (b.1 - a.1) * u, a.2 + (b.2 - a.2) * u, z]
    };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    if let Some(path) = env("GEOMETRY") {
        sim.set_geometry(omg_audio::Geometry::load(&path).expect("GEOMETRY: a dumped geometry"));
    }
    let src = town.defs.iter().position(|d| d.name == name).unwrap_or_else(|| panic!("no source {name}"));
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != src);
    }
    for i in env("OPEN").unwrap_or_default().split(',').filter_map(|x| x.parse::<usize>().ok()) {
        town.set_door(&mut sim, i, 1.0);
    }
    let ids = town.sources[src].clone();
    // MIX=1: the mixer as a host runs it
    let mut mix = (env("MIX").as_deref() == Some("1")).then(|| {
        let mut mix = Mixer::new(48_000.0, Hrir::none());
        let mut x = match env("CLIP") {
            Some(path) => fs::read(path).expect("CLIP").chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            None => {
                let mut x = noise(4 * 48_000, 3, 180.0);
                for (c, m) in x.iter_mut().zip(noise(4 * 48_000, 5, 2_000.0)) {
                    *c += 0.1 * m;
                }
                x
            }
        };
        omg_audio::normalize(&mut x);
        for &id in &ids {
            mix.play(id, &clip(&x), Play::LOOP);
        }
        mix
    });
    let (mut lp_l, mut lp_r) = (Lp::new(250.0, 48_000.0), Lp::new(250.0, 48_000.0));
    let mut out = [0.0f64; 2];
    let mut n_out = 0usize;
    println!("tour_replay ({mode}, walk on the {}): {name} alone, {t0}…{t1} s of {base}", if sim.gpu_active() { "GPU" } else { "CPU" });
    println!("     t s   (x, y)          low   mid  high dB | low: direct taps early reverb remote{}", if mix.is_some() { " | out: low, all dBFS" } else { "" });
    let start = std::time::Instant::now();
    let rate = 120.0;
    let mut acc = [[0.0f64; 5]; 3];
    let mut n = 0;
    for k in 0..((t1 - t0) * rate) as usize {
        let t = t0 + k as f64 / rate;
        if pace {
            let ahead = (t - t0) - start.elapsed().as_secs_f64();
            if ahead > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(ahead));
            }
        }
        sim.set_time(t);
        sim.set_listener_at(at(t), 0.0, 0.0, 0.0, t);
        let f = if k % 6 == 0 { Some(sim.tick()) } else { sim.tick_exact() };
        if let (Some(mix), Some(f)) = (mix.as_mut(), f.as_ref()) {
            mix.apply(f);
        }
        if let Some(mix) = mix.as_mut() {
            let (mut l, mut r) = ([0.0f32; 400], [0.0f32; 400]);
            mix.render(&mut l, &mut r);
            for (a, b) in l.iter().zip(&r) {
                let (x, y) = (lp_l.tick(*a as f64), lp_r.tick(*b as f64));
                out[0] += 0.5 * (x * x + y * y);
                out[1] += 0.5 * ((*a as f64).powi(2) + (*b as f64).powi(2));
            }
            n_out += 400;
        }
        if let Some(f) = f.as_ref() {
            let p = parts(f, &ids);
            for b in 0..3 {
                for c in 0..5 {
                    acc[b][c] += p[b][c];
                }
            }
            n += 1;
        }
        if (k + 1) % 6 == 0 && n > 0 {
            let a = acc.map(|b| b.map(|v| v / n as f64));
            let tot = |b: usize| db(a[b].iter().sum());
            let q = at(t);
            let o = match mix.as_ref() {
                Some(mix) if n_out > 0 => {
                    let (ear, reflex) = mix.ear();
                    format!(" | {:6.1} {:6.1}  ear {ear:5.1} gain {:+5.1} reflex {reflex:4.1}", db(out[0] / n_out as f64), db(out[1] / n_out as f64), 20.0 * mix.agc_gain().log10())
                }
                _ => String::new(),
            };
            println!(
                "  {t:6.2}  ({:5.1}, {:4.1})  {:5.0} {:5.0} {:5.0}  | {:5.0} {:5.0} {:5.0} {:5.0} {:5.0}{o}",
                q[0], q[1], tot(0), tot(1), tot(2), db(a[0][0]), db(a[0][1]), db(a[0][2]), db(a[0][3]), db(a[0][4])
            );
            out = [0.0; 2];
            n_out = 0;
            acc = [[0.0; 5]; 3];
            n = 0;
        }
    }
}
