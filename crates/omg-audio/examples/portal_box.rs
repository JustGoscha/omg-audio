//! Portals checked where the answer is known: a closed room, the source
//! and the ears in it, and an "opening" — an aperture with no leaf,
//! blocking nothing — across the room between them. What crosses it is
//! counted by the record (`Config::portals`) or by the walk; the room
//! sounds the same either way. Per seed, the field the ears settle at
//! (what the walk deposits a tick, its mean over the ticks, per band,
//! dB), the walk alone against with portals.
//!
//!     cargo run --release -p omg-audio --example portal_box
//!     SCATTER=s (0.5)  ABS=a (0.1)  TICKS=n (200)  SEEDS=n (3)  DIFF=0 (no edges)
//!     SRC=x,y,z (3,4,1.5)  LIS=x,y,z (7,4.5,1.6)  K=k (the sphere radius per
//!     metre of path the walk catches mirror paths in: the rung's)
use omg_audio::{Config, Geometry, Material, Quality, Simulation, SourceDesc};

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

fn main() {
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
    let scatter = env("SCATTER").unwrap_or(0.5);
    let abs = env("ABS").unwrap_or(0.1);
    let ticks = env("TICKS").unwrap_or(200.0) as usize;
    let seeds = env("SEEDS").unwrap_or(3.0) as u64;
    let mat = Material::new([abs; 3], scatter, [0.0; 3]);
    let pt = |k: &str, d: [f32; 3]| -> [f32; 3] {
        std::env::var(k).ok().map_or(d, |s| {
            let v: Vec<f32> = s.split(',').filter_map(|x| x.parse().ok()).collect();
            [v[0], v[1], v[2]]
        })
    };
    let (src, lis) = (pt("SRC", [3.0, 4.0, 1.5]), pt("LIS", [7.0, 4.5, 1.6]));
    let run = |seed: u64, portals: bool| -> [f64; 3] {
        let mut g = Geometry::new();
        let m = g.material(mat);
        let (lx, ly, lz, t) = (10.0f32, 8.0f32, 4.0f32, 0.3f32);
        g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
        g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
        g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
        g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
        g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
        g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
        let mut b = Quality::High.budget(false);
        b.diffraction = std::env::var("DIFF").as_deref() != Ok("0");
        if let Some(k) = std::env::var("K").ok().and_then(|v| v.parse::<f32>().ok()) {
            b.k = k;
        }
        let cfg = Config { quality: Quality::Custom(b), v4: false, v5: false, v6: true, v61: true, v62: true, portals, seed, kernel_thread: false, ..Config::default() };
        let mut sim = Simulation::new(cfg);
        sim.set_geometry(g);
        // the opening: across the room at x = 5, no leaf
        let o = sim.add_object([4.96, 0.0, 0.0], [5.04, ly, lz], Material::DOOR);
        sim.set_object_aperture(o, [4.96, 0.0, 0.0], [5.04, ly, lz]);
        sim.set_object_enabled(o, false);
        let id = sim.add_source(SourceDesc::at(src));
        sim.set_listener(lis, 0.0, 0.0, 0.0);
        let mut acc = [0.0f64; 3];
        let mut n = 0;
        for _ in 0..ticks {
            sim.tick();
            if let Some(e) = sim.walked_energy(id) {
                for b in 0..3 {
                    acc[b] += e[b] as f64;
                }
                n += 1;
            }
        }
        acc.map(|v| db(v / n as f64))
    };
    println!("portal_box: a 10 × 8 × 4 m room (absorption {abs}, scattering {scatter}), an opening across it between source and ears");
    for portals in [false, true] {
        let r: Vec<String> = (1..=seeds).map(|s| run(s, portals)).map(|v| format!("{:6.2}/{:6.2}/{:6.2}", v[0], v[1], v[2])).collect();
        println!("  {:8} walked low/mid/high dB per seed: {}", if portals { "portals" } else { "walk" }, r.join("  "));
    }
}
