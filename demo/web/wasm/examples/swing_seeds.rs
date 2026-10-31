//! The club's inner door swinging shut (town_system's door test) over
//! seeds: its climb back and worst treble per seed. SEEDS=n (6)
use omg_audio::{clip, Compute, Config, Hrir, Mixer, Play, Quality, Simulation};
use omg_web::town::Town;

const SR: f32 = 48_000.0;
thread_local! { static CLICK_AT: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) }; static AFTER: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) }; }

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-30).log10()
}

fn only(town: &mut Town, sim: &mut Simulation, keep: usize) {
    for i in 0..town.defs.len() {
        town.set_muted(sim, i, i != keep);
    }
}

fn swing(seed: u64) -> (f32, f32) {
    let run = |swing: bool| -> (Vec<f32>, Vec<f32>) {
        let mut sim = Simulation::new(Config { quality: Quality::Level(3), compute: Compute::Cpu, v6: true, seed, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() });
        let mut town = Town::new(&mut sim);
        let club = 2;
        only(&mut town, &mut sim, club);
        town.set_door(&mut sim, 3, 1.0);
        town.set_door(&mut sim, 4, if swing { 1.0 } else { 0.0 });
        let mut mix = Mixer::with_capacity(SR, Hrir::builtin(), 32);
        mix.set_ear(false);
        let cycle = (SR / 500.0) as usize;
        let tone = clip(&(0..cycle * 10).map(|k| 0.5 * (std::f32::consts::TAU * (k % cycle) as f32 / cycle as f32).sin()).collect::<Vec<f32>>());
        for &id in &town.sources[club] {
            mix.play(id, &tone, Play::LOOP);
        }
        let lis = [21.0f32, 31.0, 1.6];
        let block = 400usize;
        let (mut l, mut r) = (Vec::new(), Vec::new());
        let mut open = 1.0f32;
        for b in 0..(8.0 * SR) as usize / block {
            let t = (b * block) as f32 / SR;
            if swing && t >= 4.0 {
                open *= (-7.0 * block as f32 / SR).exp();
                if open < 0.001 {
                    open = 0.0;
                }
                town.set_door(&mut sim, 4, open);
            }
            sim.set_listener(lis, 0.0, 0.0, 0.0);
            if b % 6 == 0 {
                mix.apply(&sim.tick());
            } else if let Some(f) = sim.tick_exact() {
                mix.apply(&f);
            }
            let (mut bl, mut br) = (vec![0.0; block], vec![0.0; block]);
            mix.render(&mut bl, &mut br);
            l.extend(bl);
            r.extend(br);
        }
        (l, r)
    };
    // the level per 10 ms; the treble per 5 ms (a triple difference: the
    // tone 54 dB under what it passes at 4 kHz) under the tone
    let w = (0.01 * SR) as usize;
    let levels = |(l, r): &(Vec<f32>, Vec<f32>)| -> Vec<f32> { l.chunks(w).zip(r.chunks(w)).map(|(a, b)| db(a.iter().zip(b).map(|(x, y)| 0.5 * (x * x + y * y)).sum::<f32>() / w as f32)).collect() };
    let mean = |lev: &[f32], a: usize, b: usize| db(lev[a..b].iter().map(|d| 10f32.powf(d / 10.0)).sum::<f32>() / (b - a) as f32);
    let swung = run(true);
    let lev = levels(&swung);
    let shut = mean(&levels(&run(false)), 650, 800);
    let after = mean(&lev, 650, 800);
    let (mut lowest, mut climb) = (f32::MAX, 0.0f32);
    for k in 400..520 {
        lowest = lowest.min(lev[k]);
        climb += (lev[k] - lev[k - 1]).max(0.0);
    }
    let mut clicks = f32::MIN;
    let m: Vec<f32> = swung.0.iter().zip(&swung.1).map(|(a, b)| 0.5 * (a + b)).collect();
    let h = (0.005 * SR) as usize;
    for k in (4.0 * SR) as usize / h..(5.2 * SR) as usize / h {
        let (a, b) = (k * h, (k + 1) * h);
        let hf: f32 = (a..b).map(|i| (m[i] - 3.0 * m[i - 1] + 3.0 * m[i - 2] - m[i - 3]).powi(2)).sum();
        let all: f32 = (a..b).map(|i| m[i] * m[i]).sum();
        let c = db(hf / all.max(1e-30));
        if c > clicks {
            CLICK_AT.with(|t| t.set(a as f32 / SR));
        }
        clicks = clicks.max(c);
    }
    eprintln!("open {:.1} dB, shut all along {shut:.1} dB; the swing: lowest {:+.1} dB from it, climbing back {climb:.1} dB, treble at worst {clicks:.1} dB under the tone; 1.5–3 s after the swing {:+.1} dB", lev[350], lowest - shut, after - shut);
    AFTER.with(|t| t.set(after - shut));
    (climb, clicks)
}

fn main() {
    let n: u64 = std::env::var("SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let seeds: Vec<u64> = match std::env::var("SEED") {
        Ok(v) => vec![u64::from_str_radix(v.trim_start_matches("0x"), 16).unwrap_or(0x0e3_5eed)],
        Err(_) => std::iter::once(0x0e3_5eed).chain(1..=n).collect(),
    };
    for s in seeds {
        let (climb, clicks) = swing(s);
        println!("seed {s:#x}: climbing back {climb:.1} dB, treble at worst {clicks:.1} dB at {:.3} s, 1.5–3 s after {:+.1} dB", CLICK_AT.with(|t| t.get()), AFTER.with(|t| t.get()));
    }
}
