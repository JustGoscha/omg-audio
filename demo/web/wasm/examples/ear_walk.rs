//! The ear on a walk: the web's town (the Godot level's layout) with the
//! club (bass-heavy noise) and the fountain (noise) playing; the listener
//! waits on the square, walks through the open sound-lock onto the dance
//! floor, stays, walks back out and on to the church's portal. Every
//! 0.5 s: where, the output's level (dBFS, after the ear), the level the
//! ear is adapted to (dB SPL), its gain and the reflex (dB).
//!
//!     cargo run --release -p omg-web --example ear_walk
use omg_audio::{clip, Compute, Config, Hrir, Mixer, Play, Quality, Simulation};
use omg_web::town::Town;

const SR: f32 = 48_000.0;
const BLOCK: usize = 480;

fn noise(n: usize, seed: u32, lp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    let (mut a, mut b) = (0.0f32, 0.0f32);
    let c = 1.0 - (-std::f32::consts::TAU * lp / SR).exp();
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

fn main() {
    let cfg = Config { quality: Quality::Medium, compute: Compute::Cpu, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    let mut mix = Mixer::new(SR, Hrir::none());
    // the club: 104 dB at 1 m over its emitters, a bass-heavy spectrum
    // (its mids ~20 dB under its bass); the fountain: 76 dB
    let mut club = noise(4 * 48_000, 3, 180.0);
    let mid = noise(4 * 48_000, 5, 2_000.0);
    for (c, m) in club.iter_mut().zip(&mid) {
        *c += 0.1 * m;
    }
    omg_audio::normalize(&mut club);
    let mut water = noise(4 * 48_000, 9, 6_000.0);
    omg_audio::normalize(&mut water);
    let (club_i, fountain_i) = (2usize, 14usize);
    for i in 0..town.sources.len() {
        if i != club_i && i != fountain_i {
            town.set_muted(&mut sim, i, true);
        }
    }
    for &id in &town.sources[club_i] {
        mix.play(id, &clip(&club), Play::LOOP);
    }
    for &id in &town.sources[fountain_i] {
        mix.play(id, &clip(&water), Play::LOOP);
    }
    town.set_door(&mut sim, 4, 1.0);
    // (x, y, seconds to get there, seconds to stay)
    let path = [((18.0f32, 30.0f32), 0.0f32, 10.0f32), ((21.0, 31.0), 2.5, 0.0), ((27.0, 32.0), 4.5, 12.0), ((21.0, 31.0), 4.5, 0.0), ((18.0, 30.0), 2.5, 0.0), ((8.0, 46.0), 12.0, 15.0)];
    let mut pos = path[0].0;
    let (mut t, mut blocks) = (0.0f32, 0usize);
    let (mut acc, mut n) = (0.0f64, 0usize);
    println!("   t     x     y   out dBFS   ear at dB SPL   gain dB   reflex dB");
    let step = |pos: (f32, f32), sim: &mut Simulation, mix: &mut Mixer, t: &mut f32, blocks: &mut usize, acc: &mut f64, n: &mut usize| {
        sim.set_listener([pos.0, pos.1, 1.6], 0.0, 0.0, 0.0);
        if *blocks % 5 == 0 {
            let f = sim.tick();
            mix.apply(&f);
        }
        *blocks += 1;
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        mix.render(&mut l, &mut r);
        *acc += l.iter().chain(r.iter()).map(|x| (x * x) as f64).sum::<f64>();
        *n += 2 * BLOCK;
        *t += BLOCK as f32 / SR;
        if *blocks % 50 == 0 {
            let (ear, reflex) = mix.ear();
            println!(
                "{:5.1} {:5.1} {:5.1}   {:7.1}        {:6.1}       {:+6.1}     {:5.1}",
                *t,
                pos.0,
                pos.1,
                10.0 * (*acc / *n as f64).max(1e-20).log10(),
                ear,
                20.0 * mix.agc_gain().log10(),
                reflex
            );
            *acc = 0.0;
            *n = 0;
        }
    };
    for &(to, go, stay) in &path {
        let from = pos;
        let k_go = (go * SR / BLOCK as f32) as usize;
        for k in 0..k_go {
            let a = (k + 1) as f32 / k_go as f32;
            pos = (from.0 + (to.0 - from.0) * a, from.1 + (to.1 - from.1) * a);
            step(pos, &mut sim, &mut mix, &mut t, &mut blocks, &mut acc, &mut n);
        }
        pos = to;
        for _ in 0..(stay * SR / BLOCK as f32) as usize {
            step(pos, &mut sim, &mut mix, &mut t, &mut blocks, &mut acc, &mut n);
        }
    }
}
