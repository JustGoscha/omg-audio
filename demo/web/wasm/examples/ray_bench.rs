//! How fast the level's rays are cast: from the points a recorded tour
//! walked (`<base>.frames.csv`, the ears at 1.6 m), rays in random
//! directions, each followed over 8 diffuse bounces as the walk does —
//! the mesh's raycast timed alone, and a checksum of what it hit (the same
//! across BVH changes).
//!
//!     cargo run --release -p omg-web --example ray_bench -- <base>
//!     GEOMETRY=<file> (else the web town)  RAYS=n (100000)
use omg_audio::{Config, Simulation};
use omg_core::rng::Rng;
use omg_core::vec3::Vec3;
use omg_web::town::Town;
use std::fs;

fn main() {
    let base = std::env::args().nth(1).expect("a tour's base path");
    let mut sim = Simulation::new(Config::default());
    let _town = Town::new(&mut sim);
    if let Ok(path) = std::env::var("GEOMETRY") {
        sim.set_geometry(omg_audio::Geometry::load(&path).expect("GEOMETRY"));
    }
    let mesh = sim.mesh();
    let spots: Vec<Vec3> = fs::read_to_string(format!("{base}.frames.csv"))
        .expect("frames")
        .lines()
        .skip(1)
        .step_by(50)
        .filter_map(|l| {
            let v: Vec<&str> = l.split(',').collect();
            Some(Vec3::new(v.get(1)?.parse().ok()?, v.get(2)?.parse().ok()?, 1.6))
        })
        .collect();
    let rays: usize = std::env::var("RAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    // the rays first (their starts and directions drawn up front), then
    // the bounces timed: each hit's point and a fresh diffuse direction
    let mut rng = Rng::new(7);
    let starts: Vec<(Vec3, Vec3)> = (0..rays).map(|i| (spots[i % spots.len()], rng.unit_sphere())).collect();
    let dirs: Vec<Vec3> = (0..rays * 8).map(|_| rng.unit_sphere()).collect();
    for round in 0..3 {
        let clock = std::time::Instant::now();
        let (mut casts, mut hits, mut sum) = (0usize, 0usize, 0.0f64);
        for (i, &(o, d)) in starts.iter().enumerate() {
            let (mut o, mut d) = (o, d);
            for k in 0..8 {
                casts += 1;
                let Some((t, tri)) = mesh.raycast(o, d) else { break };
                hits += 1;
                sum += t as f64 + tri as f64 * 1e-6;
                let n = mesh.tri_normal(tri);
                let n = if n.dot(d) > 0.0 { n * -1.0 } else { n };
                o = o + d * t + n * 1e-3;
                let r = dirs[i * 8 + k];
                d = if r.dot(n) < 0.0 { r * -1.0 } else { r };
            }
        }
        let ns = clock.elapsed().as_secs_f64() * 1e9 / casts as f64;
        println!("round {round}: {casts} casts, {hits} hits, {ns:.0} ns a cast, checksum {sum:.3}");
    }
}
