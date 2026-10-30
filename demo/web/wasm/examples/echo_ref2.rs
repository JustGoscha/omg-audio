//! A deterministic reference for second-order echoes: every pair of the
//! town mesh's triangles (and the ground) as two mirrors between a source
//! and the listener — the listener mirrored across the first, that image
//! across the second, both reflection points inside their triangles, all
//! three legs clear — with its delay after the direct sound and its level
//! under it (mid band: both mirrors' shares, the spreading over the
//! path). What the engine should play, solved without searching
//! (`echo_ref` does first order).
//!
//!     cargo run --release -p omg-web --example echo_ref2 -- -10 12
//!
//! The arguments: the car's places along the road (y). LIS=x,y,z ROAD=x,
//! FLOOR=dB (default −30).
use omg_core::mesh::Mesh;
use omg_core::vec3::Vec3;
use omg_scene::{dome, walkthrough};

fn env3(k: &str, d: [f32; 3]) -> Vec3 {
    let v: Vec<f32> = std::env::var(k).ok().map_or(Vec::new(), |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let v = if v.len() == 3 { [v[0], v[1], v[2]] } else { d };
    Vec3::new(v[0], v[1], v[2])
}

fn clear(mesh: &Mesh, a: Vec3, b: Vec3, skip: &[u32]) -> bool {
    let d = b - a;
    let len = d.length();
    let dir = d * (1.0 / len);
    let (mut o, mut gone) = (a, 0.0);
    for _ in 0..6 {
        match mesh.raycast(o, dir) {
            Some((t, tri)) if gone + t < len - 1e-3 => {
                if !skip.contains(&tri) {
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

struct Tri {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    n: Vec3,
    d: f32,
    share: f32,
}

fn inside(t: &Tri, q: Vec3) -> bool {
    [(t.p0, t.p1), (t.p1, t.p2), (t.p2, t.p0)].iter().all(|&(a, b)| (b - a).cross(q - a).dot(t.n) >= -1e-4)
}

fn main() {
    let lis = env3("LIS", [-100.0, 100.0, 1.6]);
    let road: f32 = std::env::var("ROAD").ok().and_then(|v| v.parse().ok()).unwrap_or(-130.0);
    let floor: f32 = std::env::var("FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(-30.0);
    let ys: Vec<f32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let ys = if ys.is_empty() { vec![-10.0, 12.0, 40.0] } else { ys };
    let (mesh, _) = dome::build_town_mesh(&walkthrough::rooms(), &walkthrough::doors());
    let tris: Vec<Tri> = mesh
        .indices
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let (p0, p1, p2) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
            let n = (p1 - p0).cross(p2 - p0);
            let a = n.length();
            if a < 1e-6 {
                return None;
            }
            let n = n * (1.0 / a);
            let m = &mesh.materials[mesh.tri_material[i] as usize];
            let share = (1.0 - m.absorption[1] - m.transmission[1] * m.transmission[1]).max(0.0) * (1.0 - m.scattering);
            Some(Tri { p0, p1, p2, n, d: n.dot(p0), share })
        })
        .collect();
    let ids: Vec<u32> = (0..mesh.indices.len() as u32)
        .filter(|&i| {
            let t = mesh.indices[i as usize];
            let (p0, p1, p2) = (mesh.positions[t[0] as usize], mesh.positions[t[1] as usize], mesh.positions[t[2] as usize]);
            (p1 - p0).cross(p2 - p0).length() >= 1e-6
        })
        .collect();
    println!("echo_ref2: {} triangles; listener ({:.1}, {:.1}, {:.1}); the car on x = {road}", tris.len(), lis.x, lis.y, lis.z);
    for y in ys {
        let src = Vec3::new(road, y, 0.7);
        let direct = (src - lis).length();
        let mut found: Vec<(f32, f32, Vec3, Vec3, u32, u32)> = Vec::new();
        for (i1, t1) in tris.iter().enumerate() {
            let s1 = t1.n.dot(lis) - t1.d;
            if s1.abs() < 1e-3 || t1.share < 0.05 {
                continue;
            }
            let im1 = lis - t1.n * (2.0 * s1);
            for (i2, t2) in tris.iter().enumerate() {
                if i2 == i1 || t2.share < 0.05 {
                    continue;
                }
                let a = t2.n.dot(im1) - t2.d;
                let b = t2.n.dot(src) - t2.d;
                if a * b <= 0.0 || a.abs() < 1e-3 || b.abs() < 1e-3 {
                    continue;
                }
                let im2 = im1 - t2.n * (2.0 * a);
                let len = (src - im2).length();
                let db = 10.0 * (t1.share * t2.share).log10() + 20.0 * (direct / len).log10();
                if db < floor {
                    continue;
                }
                // the reflection point on the second mirror: src → im2 crosses its plane
                let q2 = src + (im2 - src) * (b / (b - (t2.n.dot(im2) - t2.d)));
                if !inside(t2, q2) {
                    continue;
                }
                // then q2 → im1 crosses the first mirror's plane
                let c = t1.n.dot(q2) - t1.d;
                let e = t1.n.dot(im1) - t1.d;
                if c * e >= 0.0 {
                    continue;
                }
                let q1 = q2 + (im1 - q2) * (c / (c - e));
                if !inside(t1, q1) {
                    continue;
                }
                let skip = [ids[i1], ids[i2]];
                let (o1, o2) = (q1 + t1.n * (1e-3 * s1.signum()), q2 + t2.n * (1e-3 * b.signum()));
                if !clear(&mesh, lis, o1, &skip) || !clear(&mesh, o1, o2, &skip) || !clear(&mesh, o2, src, &skip) {
                    continue;
                }
                if found.iter().any(|f| (f.2 - q1).length() < 0.05 && (f.3 - q2).length() < 0.05) {
                    continue;
                }
                let dt = (len - direct) / omg_core::SPEED_OF_SOUND * 1e3;
                found.push((dt, db, q1, q2, ids[i1], ids[i2]));
            }
        }
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("car at y {y:6.1}: direct {direct:6.1} m, {} second-order echoes over {floor} dB", found.len());
        for (dt, db, q1, q2, a, b) in found.iter().take(12) {
            println!("   +{dt:6.1} ms  {db:+6.1} dB  tri {a} at ({:.1}, {:.1}, {:.1}) → tri {b} at ({:.1}, {:.1}, {:.1})", q1.x, q1.y, q1.z, q2.x, q2.y, q2.z);
        }
    }
}
