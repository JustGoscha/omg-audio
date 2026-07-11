//! Mesh pipeline benchmark: BVH build, raycast throughput, and the full
//! stochastic trace over arbitrary geometry vs the analytic shoebox.
//! Numbers feed the simulation budgets (see README).

use omg_core::material::Material;
use omg_core::mesh::{budget_edges, extract_edges, MeshBuilder};
use omg_core::paths::{AutoPaths, PathBudget};
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::tracer::{trace, Echogram};
use omg_core::vec3::Vec3;
use std::time::Instant;

fn main() {
    // A demo-world-sized scene: 9 buildings' worth of boxes + clutter.
    let mut mb = MeshBuilder::new();
    let m = mb.material(Material::BRICK);
    let mut rng = Rng::new(9);
    for i in 0..9 {
        let x = (i % 3) as f32 * 15.0;
        let y = (i / 3) as f32 * 15.0;
        mb.abox(Vec3::new(x, y, 0.0), Vec3::new(x + 8.0, y + 6.0, 3.0), m);
    }
    for _ in 0..200 {
        // clutter: benches, bins, columns
        let p = Vec3::new(rng.next_f32() * 40.0, rng.next_f32() * 40.0, 0.0);
        let s = Vec3::new(
            0.3 + rng.next_f32(),
            0.3 + rng.next_f32(),
            0.5 + rng.next_f32() * 1.5,
        );
        mb.abox(p, p + s, m);
    }
    // enclosing shell so the trace has a closed volume
    mb.abox(Vec3::new(-5.0, -5.0, -0.1), Vec3::new(50.0, 50.0, 12.0), m);

    let t0 = Instant::now();
    let mesh = mb.build();
    println!(
        "mesh: {} tris, BVH build {:.2} ms",
        mesh.tri_count(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    // raycast throughput
    let t0 = Instant::now();
    let mut hits = 0u32;
    const N: u32 = 200_000;
    for _ in 0..N {
        let o = Vec3::new(
            rng.next_f32() * 40.0,
            rng.next_f32() * 40.0,
            rng.next_f32() * 3.0,
        );
        if mesh.raycast(o, rng.unit_sphere()).is_some() {
            hits += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "raycast: {:.1} M rays/s ({:.0}% hit)",
        N as f64 / dt / 1e6,
        100.0 * hits as f64 / N as f64
    );

    // full trace: mesh vs analytic shoebox, demo ray budget
    let src = Vec3::new(4.0, 3.0, 1.5);
    let lis = Vec3::new(20.0, 20.0, 1.6);
    let mut echo = Echogram::new();
    let t0 = Instant::now();
    trace(&mesh, src, lis, 4096, [1.0; 3], &mut rng, &mut echo);
    println!("trace mesh (4096 rays, desktop tier): {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
    let t0 = Instant::now();
    trace(&mesh, src, lis, 1024, [1.0; 3], &mut rng, &mut echo);
    println!("trace mesh (1024 rays, mobile tier): {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);

    let sbox = Shoebox::new(Vec3::new(8.0, 6.0, 3.0), [Material::BRICK; 6]);
    let t0 = Instant::now();
    trace(&sbox, src, Vec3::new(6.0, 2.0, 1.6), 4096, [1.0; 3], &mut rng, &mut echo);
    println!("trace shoebox (4096 rays): {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);

    // automatic path finding (replaces portal routing): per-source cost
    let mut ap = AutoPaths::new(&mesh, 64);
    let mut paths = Vec::new();
    let t0 = Instant::now();
    const Q: u32 = 1000;
    for i in 0..Q {
        let l = Vec3::new(20.0 + (i % 7) as f32, 20.0 + (i % 5) as f32, 1.6);
        ap.find(&mesh, src, l, PathBudget::default(), &mut paths);
    }
    println!(
        "auto paths (direct + bent, default budget): {:.0} µs/query, {} paths at last query",
        t0.elapsed().as_secs_f64() * 1e6 / Q as f64,
        paths.len()
    );

    // diffraction edges + budget
    let t0 = Instant::now();
    let edges = extract_edges(&mesh, 40.0);
    let n_all = edges.len();
    let top = budget_edges(edges, 64);
    println!(
        "edges: {} candidates -> budget 64 (extract {:.2} ms); top importance {:.1}, cutoff {:.1}",
        n_all,
        t0.elapsed().as_secs_f64() * 1e3,
        top.first().map_or(0.0, |e| e.importance),
        top.last().map_or(0.0, |e| e.importance),
    );
}
