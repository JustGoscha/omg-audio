//! Headless: the web demo's town and engine settings, no browser. The
//! listener steps from the living room out into the square; how many
//! ticks until every convolved source's kernel reflects where they stand
//! (the reverb follows the listener), and what the kernel building costs.
use omg_audio::{Compute, Config, Quality, Simulation};
use omg_web::town::Town;
use std::time::Instant;

fn web_config() -> Config {
    Config {
        quality: Quality::Medium,
        compute: Compute::Cpu,
        kernel_thread: false,
        max_kernels: 8,
        max_kernel_s: 2.5,
        kernel_step_samples: 65536,
        kernel_every_ticks: 1,
        ..Config::default()
    }
}

#[test]
fn the_reverb_follows_the_listener_out_of_a_room() {
    let mut sim = Simulation::new(web_config());
    let _town = Town::new(&mut sim);
    sim.set_listener([3.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    for _ in 0..80 {
        sim.tick();
    }
    // step out: the square
    sim.set_listener([12.0, 29.0, 1.7], 0.0, 0.0, 0.0);
    let t0 = Instant::now();
    let mut stale_ticks = Vec::new();
    let mut convolving = Vec::new();
    let mut ms = 0.0f64;
    for tick in 0..120 {
        let a = Instant::now();
        let f = sim.tick();
        ms += a.elapsed().as_secs_f64() * 1e3;
        // a kernel is stale while it was built from a response far from
        // the source's current one
        let stale = f
            .kernels
            .iter()
            .filter(|(id, _)| sim.kernel_staleness(*id).is_some_and(|c| c > 0.5))
            .count();
        stale_ticks.push(stale);
        convolving.push(f.kernels.len());
        if tick % 10 == 0 {
            eprintln!("tick {tick}: {stale} of {} kernels stale", f.kernels.len());
        }
    }
    eprintln!("120 ticks: {:.1} ms/tick, {:.0} ms total", ms / 120.0, t0.elapsed().as_secs_f64() * 1e3);
    let last_stale = stale_ticks.iter().rposition(|&n| n > 0).map_or(0, |t| t + 1);
    eprintln!("kernels follow the move after {last_stale} ticks ({:.2} s)", last_stale as f32 * 0.05);
    assert!(last_stale <= 10, "stale room reverb for {last_stale} ticks after stepping out");
    let full = convolving.iter().position(|&n| n >= 8).map_or(usize::MAX, |t| t + 1);
    eprintln!("all eight convolving again after {full} ticks");
    assert!(full <= 30, "convolution back within 1.5 s: {full} ticks");
}
