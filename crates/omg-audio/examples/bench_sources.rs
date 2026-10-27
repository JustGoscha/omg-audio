//! Scaling benchmark: 64 and 256 sources in a small town (25 buildings
//! with doorways, a ground plane). Prints the simulation's tick time (CPU
//! walk; GPU walk too with `--features gpu` and an adapter) and the
//! mixer's real-time factor (audio seconds rendered per wall second, all
//! sources playing loops at once).
//!
//!     cargo run --release -p omg-audio --example bench_sources [--features gpu]
//!
//! `OMG_BUDGET=taps,points,voices` overrides the mixer's budgets.

use omg_audio::{clip, Compute, Config, Frame, Geometry, Hrir, Material, Mixer, Play, Simulation, SourceDesc};
use std::time::Instant;

const SR: f32 = 48_000.0;

/// 5×5 buildings, 10×8×6 m, 25 cm brick walls, a doorway each, on grass.
fn town() -> Geometry {
    let mut g = Geometry::new();
    let brick = g.material(Material::BRICK);
    let roof = g.material(Material::CONCRETE);
    let floor = g.material(Material::WOOD);
    let grass = g.material(Material::GRASS);
    let t = 0.25;
    for i in 0..5 {
        for j in 0..5 {
            let (x0, y0) = (i as f32 * 16.0 - 40.0, j as f32 * 14.0 - 35.0);
            let (x1, y1, h) = (x0 + 10.0, y0 + 8.0, 6.0);
            g.add_box([x0, y0, 0.0], [x1, y1, 0.05], floor);
            g.add_box([x0, y0, h], [x1, y1, h + t], roof);
            g.add_box([x0, y0, 0.0], [x0 + t, y1, h], brick);
            g.add_box([x1 - t, y0, 0.0], [x1, y1, h], brick);
            g.add_box([x0, y1 - t, 0.0], [x1, y1, h], brick);
            // front wall with a 1.2 × 2.2 m doorway
            let dx = x0 + 4.0;
            g.add_box([x0, y0, 0.0], [dx, y0 + t, h], brick);
            g.add_box([dx + 1.2, y0, 0.0], [x1, y0 + t, h], brick);
            g.add_box([dx, y0, 2.2], [dx + 1.2, y0 + t, h], brick);
        }
    }
    g.ground_plane(0.0, grass);
    g
}

fn sources(sim: &mut Simulation, n: usize) {
    for k in 0..n {
        // golden-angle spiral over the town, every third one indoors-ish
        let a = k as f32 * 2.399_963;
        let r = 4.0 + 45.0 * ((k as f32 + 0.5) / n as f32).sqrt();
        sim.add_source(SourceDesc { position: [r * a.cos(), r * a.sin(), 1.5], gain: 1.0, active: true });
    }
}

fn bench_sim(n: usize, compute: Compute) -> Option<(f64, usize, Vec<Frame>)> {
    let mut sim = Simulation::new(Config { compute, v4: std::env::var_os("OMG_V3").is_none(), ..Config::default() });
    sim.set_geometry(town());
    if compute != Compute::Cpu && !sim.gpu_active() {
        return None;
    }
    sources(&mut sim, n);
    sim.set_listener([2.0, 3.0, 1.7], 0.0, 0.0, 0.0);
    let mut frames = Vec::new();
    for _ in 0..10 {
        frames.push(sim.tick()); // warm up: every source traced once
    }
    let ticks = 30;
    let t0 = Instant::now();
    for k in 0..ticks {
        // the listener walks: nothing is cached
        sim.set_listener([2.0 + k as f32 * 0.1, 3.0, 1.7], 0.0, 0.0, 0.0);
        frames.push(sim.tick());
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / ticks as f64;
    Some((ms, sim.traced_last_tick(), frames))
}

fn bench_mixer(n: usize, frames: &[Frame], hrir: &Hrir) -> (f64, usize) {
    let mut m = Mixer::with_capacity(SR, hrir.clone(), n);
    if let Ok(b) = std::env::var("OMG_BUDGET") {
        let v: Vec<usize> = b.split(',').map(|x| x.parse().unwrap()).collect();
        m.set_budget(v[0], v[1], v[2]);
    }
    m.apply(&frames[0]);
    let noise: Vec<f32> = (0..SR as u32).map(|i| ((i.wrapping_mul(2654435761) >> 8) as f32 / 8388608.0 - 1.0) * 0.2).collect();
    let bed = clip(&noise);
    for (id, _) in &frames[0].blocks {
        m.play(*id, &bed, Play::LOOP);
    }
    let (mut l, mut r) = (vec![0.0f32; 128], vec![0.0f32; 128]);
    let secs = 3.0;
    let blocks = (secs * SR / 128.0) as usize;
    let per_frame = (0.05 * SR / 128.0) as usize; // a frame every 50 ms
    let mut fi = 0;
    let mut apply_s = 0.0f64;
    let t0 = Instant::now();
    for b in 0..blocks {
        if b % per_frame == 0 {
            fi = (fi + 1) % frames.len();
            let ta = Instant::now();
            m.apply(&frames[fi]);
            apply_s += ta.elapsed().as_secs_f64();
        }
        m.render(&mut l, &mut r);
    }
    let (_, live, points) = m.occupancy();
    eprintln!("      (apply {:.0} ms of {:.0} ms; {live} live paths, {points} point-HRTF)", apply_s * 1e3, t0.elapsed().as_secs_f64() * 1e3);
    (secs as f64 / t0.elapsed().as_secs_f64(), m.rendering())
}

fn main() {
    let hrir = Hrir::builtin();
    println!(
        "omg-audio scaling bench — town of 25 buildings ({} tris + ground), HRIR {}",
        town().triangle_count(),
        if hrir.is_binaural() { "binaural (built in)" } else { "none (cardioid)" }
    );
    for n in [64usize, 256] {
        let (cpu_ms, traced, frames) = bench_sim(n, Compute::Cpu).unwrap();
        println!("{n:>4} sources  sim CPU  {cpu_ms:7.2} ms/tick  ({traced} traced per tick)");
        for (name, c) in [("GPU (async)", Compute::Gpu), ("GPU (sync) ", Compute::GpuSync)] {
            match bench_sim(n, c) {
                Some((ms, traced, _)) => println!("{n:>4} sources  sim {name} {ms:7.2} ms/tick  ({traced} traced per tick)"),
                None => println!("{n:>4} sources  sim {name}   n/a (build with --features gpu; needs an adapter)"),
            }
        }
        let (rtf, rendering) = bench_mixer(n, &frames, &hrir);
        println!("{n:>4} sources  mixer    {rtf:7.1}× real time  ({rendering} rendering under the default budget)");
        let (rtf, rendering) = bench_mixer(n, &frames, &Hrir::none());
        println!("{n:>4} sources  mixer    {rtf:7.1}× real time  (no HRIR, {rendering} rendering)");
    }
}
