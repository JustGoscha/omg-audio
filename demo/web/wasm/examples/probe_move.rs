//! A moving source's paths tick by tick (the car slot driving past at
//! SPEED m/s, 300 m off): the direct tap and every tap within 3 ms of it.
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;

fn main() {
    let v: f32 = std::env::var("SPEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let cfg = Config { quality: Quality::High, compute: Compute::Cpu, kernel_thread: false, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != 11);
    }
    let lis = [-100.0f32, 100.0, 1.6];
    let id = town.sources[11][0];
    for k in 0..120 {
        let t = k as f64 * 0.05;
        sim.set_time(t);
        town.set_dynamic(&mut sim, 11 - omg_scene::walkthrough::DYN_BASE, Some(([-400.0, -150.0 + v * t as f32, 0.7], 1.0)));
        sim.set_listener_at(lis, 0.0, 0.0, 0.0, t);
        let f = sim.tick();
        let Some(pb) = f.block(id) else { continue };
        let d = pb.taps.iter().find(|t| t.key == 1).map(|t| (t.delay_s, t.gains[1]));
        if k % 4 == 0 {
            let (dd, dg) = d.unwrap_or((0.0, 0.0));
            let near: Vec<String> = pb
                .taps
                .iter()
                .filter(|t| (t.delay_s - dd).abs() < 0.003)
                .map(|t| format!("{}:{:.3}ms {:.1}dB", t.key, (t.delay_s - dd) * 1e3, 20.0 * t.gains[1].max(1e-9).log10()))
                .collect();
            println!("t {t:4.2}  direct {:.4}s {:.1}dB  taps {}  near: {}", dd, 20.0 * dg.max(1e-9).log10(), pb.taps.len(), near.join(", "));
        }
    }
}
