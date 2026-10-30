//! A deterministic reference for third-order echoes: every triple of the
//! town mesh's triangles as three mirrors between a source and the
//! listener — the listener mirrored across the first, that image across
//! the second, that across the third, every reflection point inside its
//! triangle, all four legs clear — with its delay after the direct sound
//! and its level under it (mid band: the three mirrors' shares, the
//! spreading over the path). `echo_ref` does first order, `echo_ref2`
//! second.
//!
//!     cargo run --release -p omg-web --example echo_ref3 -- -10 12
//!
//! The arguments: the source's places along x = ROAD (y). LIS=x,y,z
//! ROAD=x, Z=the source's height (0.7), FLOOR=dB (default −30). The
//! doors stand as the town has them (closed but for the glass ones'
//! panes): a leg through a leaf is no mirror chain.
use omg_core::mesh::Mesh;
use omg_core::vec3::Vec3;
use omg_scene::{dome, walkthrough};

fn env3(k: &str, d: [f32; 3]) -> Vec3 {
    let v: Vec<f32> = std::env::var(k).ok().map_or(Vec::new(), |s| s.split(',').filter_map(|x| x.parse().ok()).collect());
    let v = if v.len() == 3 { [v[0], v[1], v[2]] } else { d };
    Vec3::new(v[0], v[1], v[2])
}

/// Whether a → b passes through the box lo..hi (a closed door's leaf).
fn through(a: Vec3, b: Vec3, (lo, hi): ([f32; 3], [f32; 3])) -> bool {
    let (a, d) = ([a.x, a.y, a.z], [b.x - a.x, b.y - a.y, b.z - a.z]);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for k in 0..3 {
        if d[k].abs() < 1e-9 {
            if a[k] < lo[k] || a[k] > hi[k] {
                return false;
            }
            continue;
        }
        let (u, v) = ((lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]);
        t0 = t0.max(u.min(v));
        t1 = t1.min(u.max(v));
    }
    t0 < t1
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
    let zs: f32 = std::env::var("Z").ok().and_then(|v| v.parse().ok()).unwrap_or(0.7);
    let floor: f32 = std::env::var("FLOOR").ok().and_then(|v| v.parse().ok()).unwrap_or(-30.0);
    let ys: Vec<f32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let ys = if ys.is_empty() { vec![-10.0, 12.0, 40.0] } else { ys };
    let (mesh, _) = dome::build_town_mesh(&walkthrough::rooms(), &walkthrough::doors());
    // the doors' leaves as the town has them (the mesh leaves their holes
    // open): a leg through a closed one is a transmitted path, no mirror
    // chain
    let leaves: Vec<([f32; 3], [f32; 3])> = walkthrough::doors().iter().filter_map(|d| omg_web::town::aperture(d).map(|(lo, hi, _)| (lo, hi))).collect();
    let open = |a: Vec3, b: Vec3| !leaves.iter().any(|&l| through(a, b, l));
    let mut ids: Vec<u32> = Vec::new();
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
            ids.push(i as u32);
            let n = n * (1.0 / a);
            let m = &mesh.materials[mesh.tri_material[i] as usize];
            let share = (1.0 - m.absorption[1] - m.transmission[1] * m.transmission[1]).max(0.0) * (1.0 - m.scattering);
            Some(Tri { p0, p1, p2, n, d: n.dot(p0), share })
        })
        .collect();
    println!("echo_ref3: {} triangles; listener ({:.1}, {:.1}, {:.1}); the source on x = {road}, z = {zs}", tris.len(), lis.x, lis.y, lis.z);
    let cross = |t: &Tri, a: Vec3, b: Vec3| -> Option<Vec3> {
        // where a → b crosses t's plane, if it does
        let (sa, sb) = (t.n.dot(a) - t.d, t.n.dot(b) - t.d);
        if sa * sb >= 0.0 {
            return None;
        }
        Some(a + (b - a) * (sa / (sa - sb)))
    };
    for y in ys {
        let src = Vec3::new(road, y, zs);
        let direct = (src - lis).length();
        let mut found: Vec<(f32, f32, [Vec3; 3], [u32; 3])> = Vec::new();
        for (i1, t1) in tris.iter().enumerate() {
            let s1 = t1.n.dot(lis) - t1.d;
            if s1.abs() < 1e-3 || t1.share < 0.05 {
                continue;
            }
            let im1 = lis - t1.n * (2.0 * s1);
            for (i2, t2) in tris.iter().enumerate() {
                if i2 == i1 || t2.share < 0.05 || 10.0 * (t1.share * t2.share).log10() < floor {
                    continue;
                }
                let a2 = t2.n.dot(im1) - t2.d;
                if a2.abs() < 1e-3 {
                    continue;
                }
                let im2 = im1 - t2.n * (2.0 * a2);
                for (i3, t3) in tris.iter().enumerate() {
                    if i3 == i2 || t3.share < 0.05 {
                        continue;
                    }
                    let a3 = t3.n.dot(im2) - t3.d;
                    let b3 = t3.n.dot(src) - t3.d;
                    if a3 * b3 <= 0.0 || a3.abs() < 1e-3 || b3.abs() < 1e-3 {
                        continue;
                    }
                    let im3 = im2 - t3.n * (2.0 * a3);
                    let len = (src - im3).length();
                    let db = 10.0 * (t1.share * t2.share * t3.share).log10() + 20.0 * (direct / len).log10();
                    if db < floor {
                        continue;
                    }
                    let Some(q3) = cross(t3, src, im3) else { continue };
                    if !inside(t3, q3) {
                        continue;
                    }
                    let Some(q2) = cross(t2, q3, im2) else { continue };
                    if !inside(t2, q2) {
                        continue;
                    }
                    let Some(q1) = cross(t1, q2, im1) else { continue };
                    if !inside(t1, q1) {
                        continue;
                    }
                    let skip = [ids[i1], ids[i2], ids[i3]];
                    // each leg from a millimetre off its mirror to a
                    // millimetre off the next: no triangle may lie between
                    // (not even one of the chain's own — the first leg
                    // ran through the second mirror's wall to a face
                    // behind it)
                    let off = |t: &Tri, q: Vec3, toward: Vec3| q + t.n * (1e-3 * (t.n.dot(toward) - t.d).signum());
                    let (o1, o2, o3) = (off(t1, q1, lis), off(t2, q2, q1), off(t3, q3, src));
                    let (o2b, o3b) = (off(t2, q2, q3), off(t3, q3, q2));
                    let (o1b, o2c) = (off(t1, q1, q2), off(t2, q2, q1));
                    if !clear(&mesh, lis, o1, &[]) || !clear(&mesh, o1b, o2c, &[]) || !clear(&mesh, o2b, o3b, &[]) || !clear(&mesh, o3, src, &[]) {
                        continue;
                    }
                    if !open(lis, q1) || !open(q1, q2) || !open(q2, q3) || !open(q3, src) {
                        continue;
                    }
                    let _ = (o2, &skip);
                    if found.iter().any(|f| (f.2[0] - q1).length() < 0.05 && (f.2[1] - q2).length() < 0.05 && (f.2[2] - q3).length() < 0.05) {
                        continue;
                    }
                    let dt = (len - direct) / omg_core::SPEED_OF_SOUND * 1e3;
                    found.push((dt, db, [q1, q2, q3], skip));
                }
            }
        }
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("source at y {y:6.1}: direct {direct:6.1} m, {} third-order echoes over {floor} dB", found.len());
        for (dt, db, q, t) in found.iter().take(12) {
            if std::env::var_os("PLANES").is_some() {
                for &id in t {
                    let k = ids.iter().position(|&x| x == id).unwrap();
                    let tr = &tris[k];
                    println!("      tri {id}: normal ({:.2}, {:.2}, {:.2}), d {:.2}, corners ({:.1}, {:.1}, {:.1}) ({:.1}, {:.1}, {:.1}) ({:.1}, {:.1}, {:.1}), mirror share {:.2}", tr.n.x, tr.n.y, tr.n.z, tr.d, tr.p0.x, tr.p0.y, tr.p0.z, tr.p1.x, tr.p1.y, tr.p1.z, tr.p2.x, tr.p2.y, tr.p2.z, tr.share);
                }
            }
            println!(
                "   +{dt:6.1} ms  {db:+6.1} dB  tri {} ({:.1}, {:.1}, {:.1}) → tri {} ({:.1}, {:.1}, {:.1}) → tri {} ({:.1}, {:.1}, {:.1})",
                t[0], q[0].x, q[0].y, q[0].z, t[1], q[1].x, q[1].y, q[1].z, t[2], q[2].x, q[2].y, q[2].z
            );
        }
    }
}
