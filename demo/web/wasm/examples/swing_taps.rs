//! The club's inner door swinging shut (the town test's scenario: the
//! ears at (25, 31), door 4 closing from 4 s at 7/s): per tick from 3.9 s
//! the club's taps — key, delay, mid gain — that appear, go, or step in
//! delay or gain from the tick before, to find what clicks.
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;

fn main() {
    let mut sim = Simulation::new(Config { quality: Quality::Level(3), compute: Compute::Cpu, v6: true, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() });
    let mut town = Town::new(&mut sim);
    let club = 2;
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != club);
    }
    town.set_door(&mut sim, 3, 1.0);
    town.set_door(&mut sim, 4, 1.0);
    let ids = town.sources[club].clone();
    let sr = 48_000.0f32;
    let block = 400usize;
    let mut open = 1.0f32;
    let mut prev: std::collections::HashMap<(u32, u32), (f32, f32)> = std::collections::HashMap::new();
    for b in 0..(5.4 * sr) as usize / block {
        let t = (b * block) as f32 / sr;
        if t >= 4.0 {
            open *= (-7.0 * block as f32 / sr).exp();
            if open < 0.001 {
                open = 0.0;
            }
            town.set_door(&mut sim, 4, open);
        }
        sim.set_listener([25.0, 31.0, 1.6], 0.0, 0.0, 0.0);
        if b % 6 != 0 {
            continue;
        }
        let f = sim.tick();
        let mut now: std::collections::HashMap<(u32, u32), (f32, f32)> = std::collections::HashMap::new();
        for (k, &id) in ids.iter().enumerate() {
            let Some(pb) = f.block(id) else { continue };
            for tap in &pb.taps {
                now.insert((k as u32, tap.key), (tap.delay_s, 20.0 * tap.gains[1].abs().max(1e-9).log10()));
            }
        }
        if t >= 3.9 {
            let mut ev: Vec<String> = Vec::new();
            for (key, &(d, g)) in &now {
                match prev.get(key) {
                    None if g > -60.0 => ev.push(format!("+{}:{} {:.1}ms {:.0}dB", key.0, key.1, d * 1e3, g)),
                    Some(&(d0, g0)) if (d - d0).abs() > 0.0005 && g > -60.0 => ev.push(format!("~{}:{} delay {:.2}→{:.2}ms {:.0}dB", key.0, key.1, d0 * 1e3, d * 1e3, g)),
                    Some(&(_, g0)) if (g - g0).abs() > 6.0 && g.max(g0) > -60.0 => ev.push(format!("^{}:{} {:.0}→{:.0}dB", key.0, key.1, g0, g)),
                    _ => {}
                }
            }
            for (key, &(_, g0)) in &prev {
                if !now.contains_key(key) && g0 > -60.0 {
                    ev.push(format!("-{}:{} {:.0}dB", key.0, key.1, g0));
                }
            }
            ev.sort();
            if !ev.is_empty() {
                println!("{t:.2} s (open {open:.3}): {}", ev.join("  "));
            }
        }
        prev = now;
    }
}
