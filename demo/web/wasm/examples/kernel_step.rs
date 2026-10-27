//! How long building a v4 kernel takes, a step at a time (the web builds
//! in the worker's tick, without threads).
use omg_core::v3::DenseResponse;
use omg_dsp::dense::KernelBuilder;
use std::time::Instant;

fn main() {
    let bin_s = 0.002;
    let bins: Vec<[f32; 6]> = (0..1200)
        .map(|i| {
            let e = (-(i as f32) * bin_s * 6.9 / 1.2).exp();
            [e, e, e * 0.6, 0.1 * e, 0.0, 0.0]
        })
        .collect();
    let resp = DenseResponse { bin_s, bins, ..Default::default() };
    for step in [16384usize, 65536] {
        let mut b = KernelBuilder::new(&resp, 48_000.0, 1.0, 7, 2.5);
        let mut n = 0;
        let t0 = Instant::now();
        let mut worst = 0.0f64;
        loop {
            let t = Instant::now();
            let done = b.step(step);
            worst = worst.max(t.elapsed().as_secs_f64() * 1e3);
            n += 1;
            if done {
                break;
            }
        }
        println!("step {step}: {n} steps, {:.1} ms total, worst step {:.2} ms", t0.elapsed().as_secs_f64() * 1e3, worst);
    }
}
