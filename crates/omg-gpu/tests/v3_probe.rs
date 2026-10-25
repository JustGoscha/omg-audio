//! Engine v3 GPU kernel, deterministic half: the geometric primitives
//! the walk is built from — World::transmission (mesh crossings deduped
//! and paired into walls of real thickness, overlay panels, earth) and
//! World::raycast (BVH + panels) — and the diffraction walk's nearest
//! exposed edge along a segment must agree with the CPU segment by
//! segment, not just on average. Thousands of random segments through
//! the demo world, a thick wall, a doorway leaf and a bunker below
//! grade. A handful of exact float ties (a segment grazing a shared
//! patch edge) may resolve differently; anything systematic fails.
//! Self-skips without an adapter.

use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::rng::Rng;
use omg_core::scene::{AcousticGeometry, WithPanels};
use omg_core::v3::fold::{edge_nearest, edge_set, WalkJob};
use omg_core::v3::{Budget, World};
use omg_core::vec3::Vec3;
use omg_gpu::GpuV3Tracer;

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn job(w: &World, es: &omg_core::v3::EdgeSet) -> WalkJob {
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    WalkJob::build(w, v(0.0, 0.0, 0.0), &[], Budget::default(), id, 1, false, &[], Some(es))
}

fn check(name: &str, mesh: &Mesh, w: &World, segs: &[(Vec3, Vec3)]) {
    let Some(mut g) = GpuV3Tracer::new(mesh) else {
        eprintln!("SKIP {name}: no wgpu adapter");
        return;
    };
    let mut es = edge_set(w);
    let job = job(w, &es);
    let got = g.probe(&job, segs);
    let mut hits = Vec::new();
    let geo = WithPanels { base: mesh, panels: w.panels };
    let (mut t_bad, mut r_bad, mut e_bad, mut opaque, mut partial, mut bends) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for (i, &(a, b)) in segs.iter().enumerate() {
        let want = w.transmission(a, b, &mut hits);
        let (tg, ht, hn, _, _, eg) = got[i];
        let close = (0..3).all(|k| (tg[k] - want[k]).abs() <= 1e-5 + 1e-3 * want[k].abs());
        if !close {
            t_bad += 1;
            if t_bad <= 5 {
                eprintln!("{name}: seg {i} {a:?}→{b:?} trans cpu {want:?} gpu {tg:?}");
            }
        }
        if want[1] < 1e-6 {
            opaque += 1;
        } else if want[1] < 0.999 {
            partial += 1;
        }
        let d = (b - a).normalize();
        let hc = geo.raycast_hit(a, d);
        let ok = match &hc {
            None => ht >= 3e38,
            Some(h) => {
                (ht - h.t).abs() < 1e-3 * h.t.max(1.0)
                    && (h.normal.dot(v(hn[0], hn[1], hn[2])) > 0.999 || (ht - h.t).abs() < 1e-4)
            }
        };
        if !ok {
            r_bad += 1;
            if r_bad <= 5 {
                eprintln!("{name}: ray {i} cpu {:?} gpu t {ht} n {hn:?}", hc.map(|h| (h.t, h.normal)));
            }
        }
        // the walk's bend query (no edge excluded)
        let ec = edge_nearest(&mut es, a, d, (b - a).length());
        bends += ec.is_some() as usize;
        let same = match (ec, eg) {
            (None, None) => true,
            (Some((tc, dc)), Some((tg, dg))) => (tc - tg).abs() < 1e-2 && (dc - dg).abs() < 1e-3,
            _ => false,
        };
        if !same {
            e_bad += 1;
            if e_bad <= 5 {
                eprintln!("{name}: edge {i} {a:?}→{b:?} cpu {ec:?} gpu {eg:?}");
            }
        }
    }
    eprintln!(
        "{name}: {} segments ({opaque} opaque, {partial} partial, {bends} pass an edge; {} usable edges): mismatches transmission {t_bad}, raycast {r_bad}, edge {e_bad}",
        segs.len(),
        job.n_edges()
    );
    let tol = (segs.len() / 500).max(1);
    assert!(t_bad <= tol, "{name}: {t_bad} transmission mismatches");
    assert!(r_bad <= tol, "{name}: {r_bad} raycast mismatches");
    assert!(e_bad <= tol, "{name}: {e_bad} nearest-edge mismatches");
}

fn random_segs(rng: &mut Rng, n: usize, lo: Vec3, hi: Vec3) -> Vec<(Vec3, Vec3)> {
    let mut p = || {
        v(
            lo.x + (hi.x - lo.x) * rng.next_f32(),
            lo.y + (hi.y - lo.y) * rng.next_f32(),
            lo.z + (hi.z - lo.z) * rng.next_f32(),
        )
    };
    (0..n).map(|_| (p(), p())).collect()
}

#[test]
fn v3_probe_demo_world_segments_match_cpu() {
    let rooms = omg_scene::walkthrough::rooms();
    let doors = omg_scene::walkthrough::doors();
    let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
    let w = World::new(&mesh, &[]);
    let mut rng = Rng::new(3);
    // street scale and room scale
    let mut segs = random_segs(&mut rng, 3000, v(-20.0, -20.0, 0.2), v(120.0, 120.0, 12.0));
    segs.extend(random_segs(&mut rng, 3000, v(10.0, 20.0, 0.2), v(40.0, 45.0, 3.0)));
    check("demo world", &mesh, &w, &segs);
}

#[test]
fn v3_probe_thick_wall_and_panels_match_cpu() {
    let mut mb = MeshBuilder::new();
    let floor = mb.material(Material { absorption: [0.1; 3], scattering: 0.9, transmission: [0.0; 3] });
    let wall = mb.material(Material { absorption: [0.1; 3], scattering: 0.3, transmission: [0.1, 0.03, 0.01] });
    mb.abox(v(-60.0, -60.0, -1.0), v(60.0, 60.0, 0.0), floor);
    mb.abox(v(4.85, -60.0, 0.0), v(5.15, 60.0, 30.0), wall);
    let mesh = mb.build();
    let leaf = Material { transmission: [0.05, 0.02, 0.01], ..Material::WOOD_PANEL };
    let panels = [(v(1.0, -2.0, 0.0), v(1.1, 2.0, 2.1), leaf)];
    let w = World::new(&mesh, &panels);
    let mut rng = Rng::new(5);
    let segs = random_segs(&mut rng, 4000, v(-10.0, -10.0, 0.001), v(15.0, 10.0, 6.0));
    check("thick wall + leaf", &mesh, &w, &segs);
}

#[test]
fn v3_probe_bunker_earth_matches_cpu() {
    let mut mb = MeshBuilder::new();
    let wall = mb.material(Material { absorption: [0.05; 3], scattering: 0.2, transmission: [0.04, 0.005, 0.0006] });
    mb.abox(v(0.0, 0.0, -3.2), v(6.0, 8.0, -3.0), wall);
    mb.abox(v(0.0, 0.0, -1.0), v(6.0, 8.0, -0.8), wall);
    let ground = mb.material(Material { absorption: [0.1; 3], scattering: 0.3, transmission: [0.1, 0.02, 0.005] });
    mb.quad(v(-50.0, -50.0, 0.0), v(50.0, -50.0, 0.0), v(50.0, 50.0, 0.0), v(-50.0, 50.0, 0.0), ground);
    let mesh = mb.build();
    let pockets = [(v(0.0, 0.0, -3.0), v(6.0, 8.0, -0.8)), (v(10.0, 0.0, -2.0), v(12.0, 3.0, -0.5))];
    let w = World { grade: Some(0.0), pockets: &pockets, ..World::new(&mesh, &[]) };
    let mut rng = Rng::new(9);
    let segs = random_segs(&mut rng, 4000, v(-5.0, -5.0, -4.0), v(15.0, 12.0, 4.0));
    check("bunker", &mesh, &w, &segs);
}
