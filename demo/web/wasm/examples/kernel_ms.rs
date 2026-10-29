//! How long building one convolution kernel takes (the club's response
//! from the entrance: a few seconds of ringing room), at full length and
//! capped.
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;
use std::time::Instant;

fn main() {
    let cfg = Config { quality: Quality::High, compute: Compute::Cpu, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    let mut town = Town::new(&mut sim);
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != 2);
    }
    sim.set_listener([21.0, 30.2, 1.6], 0.0, 0.0, 0.0);
    for _ in 0..40 {
        sim.tick();
    }
    let id = town.sources[2][0];
    let d = sim.dense_response(id).expect("a response").clone();
    println!("response: {} bins ({:.2} s)", d.bins.len(), d.bins.len() as f32 * d.bin_s);
    for max_s in [10.0f32, 2.5, 0.5] {
        let t = Instant::now();
        let k = omg_dsp::dense::build_kernel_max(&d, 48_000.0, 1.0, 1, max_s);
        println!("kernel ≤ {max_s:4.1} s: {:.2} s long, built in {:.1} ms", k.len as f32 / 48_000.0, t.elapsed().as_secs_f64() * 1e3);
    }
}
