//! GPU walk time per tick (synchronous: the tick waits for its walk) by
//! quality tier and engine, 19 sources in bench_sources' town; and the
//! GPU tier's early / late halves on their own.
//!
//!     cargo run --release -p omg-audio --features gpu --example gpu_walk_ms

use omg_audio::{Compute, Config, Geometry, Material, Quality, Simulation, SourceDesc};
use std::time::Instant;

fn town() -> Geometry {
    let mut g = Geometry::new();
    let brick = g.material(Material::BRICK);
    let roof = g.material(Material::CONCRETE);
    let grass = g.material(Material::GRASS);
    for i in 0..5 {
        for j in 0..5 {
            let (x0, y0) = (i as f32 * 16.0 - 40.0, j as f32 * 14.0 - 35.0);
            let (x1, y1, h, t) = (x0 + 10.0, y0 + 8.0, 6.0, 0.25);
            g.add_box([x0, y0, h], [x1, y1, h + t], roof);
            g.add_box([x0, y0, 0.0], [x0 + t, y1, h], brick);
            g.add_box([x1 - t, y0, 0.0], [x1, y1, h], brick);
            g.add_box([x0, y1 - t, 0.0], [x1, y1, h], brick);
            g.add_box([x0, y0, 0.0], [x0 + 4.0, y0 + t, h], brick);
            g.add_box([x0 + 5.2, y0, 0.0], [x1, y0 + t, h], brick);
        }
    }
    g.ground_plane(0.0, grass);
    g
}

/// (whole tick ms, the GPU walk's own ms: submit to read-back)
fn walk_ms(quality: Quality, v4: bool, n: usize) -> Option<(f64, f64)> {
    let mut sim = Simulation::new(Config { compute: Compute::GpuSync, quality, v4, ..Config::default() });
    sim.set_geometry(town());
    if !sim.gpu_active() {
        return None;
    }
    for k in 0..n {
        let a = k as f32 * 2.399_963;
        let r = 4.0 + 45.0 * ((k as f32 + 0.5) / n as f32).sqrt();
        sim.add_source(SourceDesc { position: [r * a.cos(), r * a.sin(), 1.5], gain: 1.0, active: true });
    }
    for k in 0..5 {
        sim.set_listener([2.0 + k as f32 * 0.1, 3.0, 1.7], 0.0, 0.0, 0.0);
        sim.tick();
    }
    let ticks = 10;
    let t0 = Instant::now();
    let mut gpu = 0.0f64;
    for k in 0..ticks {
        sim.set_listener([2.5 + k as f32 * 0.1, 3.0, 1.7], 0.0, 0.0, 0.0);
        sim.tick();
        gpu += sim.gpu_walk_ms() as f64;
    }
    Some((t0.elapsed().as_secs_f64() * 1e3 / ticks as f64, gpu / ticks as f64))
}

fn main() {
    if std::env::var("POOL").is_ok() {
        pool_compare();
        return;
    }
    let n = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
    if let Ok(spec) = std::env::var("RAYS") {
        // RAYS=early,late: one custom budget, v4
        let v: Vec<u32> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        let diffraction = std::env::var("NODIFF").is_err();
        let b = omg_core::v3::Budget { rays: v[0], late_rays: v[1], diffraction, ..Quality::Medium.budget(true) };
        let (m, g) = walk_ms(Quality::Custom(b), true, n).expect("gpu");
        println!("v4 {} early, {} late: tick {m:.1} ms (gpu {g:.1})", v[0], v[1]);
        return;
    }
    for v4 in [false, true] {
        for (name, q) in [("low", Quality::Low), ("medium", Quality::Medium), ("high", Quality::High)] {
            let full = q.budget(true);
            let early = Quality::Custom(omg_core::v3::Budget { late_rays: 0, ..full });
            let late = Quality::Custom(omg_core::v3::Budget { rays: 1, ..full });
            let f = |q| walk_ms(q, v4, n).map_or("-".into(), |(m, g)| format!("{m:.1} (gpu {g:.1})"));
            println!(
                "{} {name:6} ({} early × {} refl, {} late): tick {} ms · early only {} · late only {}",
                if v4 { "v4" } else { "v3" },
                full.rays,
                full.bounces,
                full.late_rays,
                f(q),
                f(early),
                f(late)
            );
        }
    }
}

/// POOL=1: the live (async, pooled) walk against the synchronous whole
/// walk — walk time per tick and the settled dense response of a source
/// indoors (total energy, RT60). WARM=ticks before measuring (40; the live
/// pool fills over seconds), EARLY= / LATE= ray counts over Medium's,
/// TRACE=1 the live walk's dense energy every 0.5 s from the start.
#[allow(dead_code)]
pub fn pool_compare() {
    use omg_audio::Compute;
    let run = |compute: Compute| -> Option<(f32, f32, [f32; 3])> {
        let mut q = Quality::Medium.budget(true);
        if let Ok(v) = std::env::var("EARLY") { q.rays = v.parse().unwrap(); }
        if let Ok(v) = std::env::var("LATE") { q.late_rays = v.parse().unwrap(); }
        let mut sim = Simulation::new(Config { compute, quality: Quality::Custom(q), ..Config::default() });
        sim.set_geometry(town());
        if compute != Compute::Cpu && !sim.gpu_active() {
            return None;
        }
        // listener and source inside one building (x0 -8, y0 -7: 10 × 8 × 6 m)
        let id = sim.add_source(SourceDesc { position: [-6.0, -5.0, 1.5], gain: 1.0, active: true });
        sim.set_listener([-1.0, -2.0, 1.7], 0.0, 0.0, 0.0);
        let (mut ms, mut n) = (0.0f32, 0);
        let mut tail = Vec::new();
        let warm = std::env::var("WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(40usize);
        for k in 0..warm + 80 {
            let t0 = Instant::now();
            sim.tick();
            if std::env::var("TRACE").is_ok() && k % 10 == 0 && compute == Compute::Gpu {
                let e = sim.dense_response(id).map_or(0.0, |d| d.bins.iter().map(|b| b[1]).sum::<f32>());
                eprintln!("  t {:.1} s: dense {:.2} dB", k as f32 * 0.05, 10.0 * e.max(1e-12).log10());
            }
            if k >= warm {
                ms += sim.gpu_walk_ms();
                n += 1;
                if let Some(d) = sim.dense_response(id) {
                    tail.push((d.bins.iter().map(|b| b[1]).sum::<f32>(), d.rt60()));
                }
            }
            let left = 0.05 - t0.elapsed().as_secs_f32();
            if left > 0.0 && compute == Compute::Gpu {
                std::thread::sleep(std::time::Duration::from_secs_f32(left));
            }
        }
        let e = tail.iter().map(|t| t.0).sum::<f32>() / tail.len().max(1) as f32;
        let rt = tail.last().map_or([0.0; 3], |t| t.1);
        Some((ms / n.max(1) as f32, e, rt))
    };
    for (name, c) in [("cpu", Compute::Cpu), ("gpu sync (whole walk)", Compute::GpuSync), ("gpu live (pooled)", Compute::Gpu)] {
        match run(c) {
            Some((ms, e, rt)) => println!("{name:24} walk {ms:5.1} ms · dense energy {:.2} dB · RT60 {:.2} / {:.2} / {:.2} s", 10.0 * e.log10(), rt[0], rt[1], rt[2]),
            None => println!("{name}: no GPU"),
        }
    }
}
