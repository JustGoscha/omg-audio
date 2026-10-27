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

fn walk_ms(quality: Quality, v4: bool, n: usize) -> Option<f64> {
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
    for k in 0..ticks {
        sim.set_listener([2.5 + k as f32 * 0.1, 3.0, 1.7], 0.0, 0.0, 0.0);
        sim.tick();
    }
    Some(t0.elapsed().as_secs_f64() * 1e3 / ticks as f64)
}

fn main() {
    let n = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
    for v4 in [false, true] {
        for (name, q) in [("low", Quality::Low), ("medium", Quality::Medium), ("high", Quality::High)] {
            let full = q.budget(true);
            let early = Quality::Custom(omg_core::v3::Budget { late_rays: 0, ..full });
            let late = Quality::Custom(omg_core::v3::Budget { rays: 1, ..full });
            let f = |q| walk_ms(q, v4, n).map_or("-".into(), |m| format!("{m:.1}"));
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
