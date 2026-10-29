//! A deterministic reference for first-order echoes: every triangle of
//! the town's mesh (and the ground) as a mirror between a source and the
//! listener — the listener's image across its plane, the reflection point
//! inside the triangle, both legs clear of the mesh — with its delay after
//! the direct sound and its level under it (mid band: the surface's
//! mirror share, 1 − absorption − transmission², × (1 − scattering), and
//! the spreading over the longer path; air left out). What the chain
//! finder should name, solved without searching.
//!
//!     cargo run --release -p omg-web --example echo_ref -- 12 30 50
//!
//! The arguments are the car's places along the road (y; default the
//! `assess cartone` pass at 1 s steps). LIS=x,y,z ROAD=x (the car at
//! (ROAD, y, 0.7)), FLOOR=dB (default −30: weaker ones are not listed).
use omg_core::mesh::Mesh;
use omg_core::vec3::Vec3;
use omg_scene::{dome, walkthrough};

fn env3(k: &str, d: [f32; 3]) -> Vec3 {
    let v: Vec<f32> = std::env::var(k).ok().map_or(Vec::new(), |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let v = if v.len() == 3 { [v[0], v[1], v[2]] } else { d };
    Vec3::new(v[0], v[1], v[2])
}

/// Is the segment a → b clear of the mesh (but for `skip`, the mirror)?
fn clear(mesh: &Mesh, a: Vec3, b: Vec3, skip: Option<u32>) -> bool {
    let d = b - a;
    let len = d.length();
    let dir = d * (1.0 / len);
    let mut o = a;
    let mut gone = 0.0;
    // step past the mirror's own triangle if the ray grazes it
    for _ in 0..4 {
        match mesh.raycast(o, dir) {
            Some((t, tri)) if gone + t < len - 1e-3 => {
                if Some(tri) != skip {
                    return false;
                }
                o = o + dir * (t + 1e-3);
                gone += t + 1e-3;
            }
            _ => return true,
        }
    }
    false
}

fn main() {
    let lis = env3("LIS", [-100.0, 100.0, 1.6]);
    let road: f32 = std::env::var("ROAD").ok().and_then(|v| v.parse().ok()).unwrap_or(-130.0);
    let floor: f32 = std::env::var("FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(-30.0);
    let ys: Vec<f32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let ys = if ys.is_empty() { (0..11).map(|k| -50.0 + 27.78 * k as f32).collect() } else { ys };
    let (mesh, _) = dome::build_world_mesh(&walkthrough::rooms(), &walkthrough::doors());
    println!("echo_ref: {} triangles; listener ({:.1}, {:.1}, {:.1}); the car on x = {road}", mesh.indices.len(), lis.x, lis.y, lis.z);
    for y in ys {
        let src = Vec3::new(road, y, 0.7);
        let d_dir = (src - lis).length();
        let open = clear(&mesh, lis, src, None);
        println!("car at y {y:6.1}: direct {d_dir:6.1} m{}", if open { "" } else { " (blocked)" });
        // (delay after the direct ms, level re direct dB, point, what)
        let mut found: Vec<(f32, f32, Vec3, String)> = Vec::new();
        for (i, t) in mesh.indices.iter().enumerate() {
            let (p0, p1, p2) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
            let n = (p1 - p0).cross(p2 - p0);
            let area2 = n.length();
            if area2 < 1e-9 {
                continue;
            }
            let n = n * (1.0 / area2);
            let (sl, ss) = ((lis - p0).dot(n), (src - p0).dot(n));
            // both on one side, off the plane
            if sl * ss <= 0.0 || sl.abs() < 1e-3 || ss.abs() < 1e-3 {
                continue;
            }
            let image = lis - n * (2.0 * sl);
            let q = src + (image - src) * (ss / (ss + sl));
            // inside the triangle (barycentric, edges included)
            let inside = [(p0, p1), (p1, p2), (p2, p0)].iter().all(|&(a, b)| (b - a).cross(q - a).dot(n) >= -1e-4);
            if !inside {
                continue;
            }
            let len = (src - image).length();
            let m = &mesh.materials[mesh.tri_material[i] as usize];
            let thr = (1.0 - m.absorption[1] - m.transmission[1] * m.transmission[1]).max(0.0) * (1.0 - m.scattering);
            let db = 10.0 * thr.max(1e-12).log10() + 20.0 * (d_dir / len).log10();
            if db < floor {
                continue;
            }
            let off = q + n * (1e-3 * sl.signum());
            if !clear(&mesh, off, src, Some(i as u32)) || !clear(&mesh, off, lis, Some(i as u32)) {
                continue;
            }
            let dt = (len - d_dir) / omg_core::SPEED_OF_SOUND * 1e3;
            // one reflection point a face (coplanar neighbours share an edge)
            if found.iter().any(|f| (f.2 - q).length() < 0.05) {
                continue;
            }
            found.push((dt, db, q, format!("tri {i} n ({:.2}, {:.2}, {:.2}) abs {:.2} scat {:.2}", n.x, n.y, n.z, m.absorption[1], m.scattering)));
        }
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (dt, db, q, what) in found {
            println!("   +{dt:6.1} ms  {db:+6.1} dB  at ({:6.1}, {:6.1}, {:4.1})  {what}", q.x, q.y, q.z);
        }
    }
}
