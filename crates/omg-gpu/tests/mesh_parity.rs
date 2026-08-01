//! C6d kernel K2 acceptance. (1) Parity: the GPU world-mesh tracer must
//! agree with the CPU tracer over the SAME mesh (statistical — averaged
//! seeds, golden-style tolerances). (2) Panels: covering the doorway
//! hole with a panel overlay must collapse the outside listener's
//! received energy on both backends. (3) Speed probe (printed, asserted
//! loosely): the whole point is that one dense world trace fits the
//! per-tick slot. Self-skips without an adapter.

use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::rng::Rng;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::vec3::Vec3;

/// The tracer-test room: a shell with a doorway hole in the x=8 wall.
fn holed_room() -> Mesh {
    let mut mb = MeshBuilder::new();
    let m = mb.material(Material::CONCRETE);
    let v = Vec3::new;
    mb.quad(v(0.0, 0.0, 0.0), v(0.0, 6.0, 0.0), v(0.0, 6.0, 3.0), v(0.0, 0.0, 3.0), m);
    mb.quad(v(0.0, 0.0, 0.0), v(8.0, 0.0, 0.0), v(8.0, 0.0, 3.0), v(0.0, 0.0, 3.0), m);
    mb.quad(v(0.0, 6.0, 0.0), v(8.0, 6.0, 0.0), v(8.0, 6.0, 3.0), v(0.0, 6.0, 3.0), m);
    mb.quad(v(0.0, 0.0, 0.0), v(8.0, 0.0, 0.0), v(8.0, 6.0, 0.0), v(0.0, 6.0, 0.0), m);
    mb.quad(v(0.0, 0.0, 3.0), v(8.0, 0.0, 3.0), v(8.0, 6.0, 3.0), v(0.0, 6.0, 3.0), m);
    mb.quad(v(8.0, 0.0, 0.0), v(8.0, 2.5, 0.0), v(8.0, 2.5, 3.0), v(8.0, 0.0, 3.0), m);
    mb.quad(v(8.0, 3.5, 0.0), v(8.0, 6.0, 0.0), v(8.0, 6.0, 3.0), v(8.0, 3.5, 3.0), m);
    mb.quad(v(8.0, 2.5, 2.1), v(8.0, 3.5, 2.1), v(8.0, 3.5, 3.0), v(8.0, 2.5, 3.0), m);
    mb.build()
}

fn db(v: f32) -> f32 {
    20.0 * v.max(1e-9).log10()
}

#[test]
fn mesh_kernel_matches_cpu_tracer() {
    let mesh = holed_room();
    let Some(gpu) = omg_gpu::GpuMeshTracer::new(&mesh) else {
        eprintln!("SKIP mesh_kernel_matches_cpu_tracer: no wgpu adapter");
        return;
    };
    // inside listener (reverberant) and outside listener (through-hole)
    let src = Vec3::new(2.0, 3.0, 1.5);
    for (name, lis) in [("inside", Vec3::new(6.0, 2.0, 1.6)), ("outside", Vec3::new(11.0, 3.0, 1.6))]
    {
        let mut cpu = Echogram::new();
        let mut acc = Echogram::new();
        for k in 0..4u64 {
            let mut rng = Rng::new(400 + k);
            trace(&mesh, src, lis, 16_384, [1.0; 3], &mut rng, &mut acc);
            cpu.ema(&acc, 1.0 / (k + 1) as f32);
        }
        let mut gpu_e = Echogram::new();
        let mut acc2 = Echogram::new();
        for k in 0..4u32 {
            gpu.trace(src, lis, 16_384, [1.0; 3], 0xC6D2 ^ k.wrapping_mul(0x85EB_CA6B), &[], &mut acc2);
            gpu_e.ema(&acc2, 1.0 / (k + 1) as f32);
        }
        let (pc, pg) = (estimate_reverb(&cpu), estimate_reverb(&gpu_e));
        eprintln!(
            "{name}: rt60 cpu {:?} gpu {:?} · level cpu {:?} gpu {:?}",
            pc.rt60, pg.rt60, pc.level, pg.level
        );
        for b in 0..3 {
            let r = pg.rt60[b] / pc.rt60[b];
            assert!((0.8..1.25).contains(&r), "{name} band {b} rt60 ratio {r}");
            let dl = (db(pg.level[b]) - db(pc.level[b])).abs();
            assert!(dl < 2.0, "{name} band {b} level diff {dl:.2} dB");
        }
        // late direction agreement (outside: points at the hole)
        let (dc, ac) = cpu.late_direction(0.05);
        let (dg, ag) = gpu_e.late_direction(0.05);
        let dot = dc[0] * dg[0] + dc[1] * dg[1] + dc[2] * dg[2];
        assert!(dot > 0.9, "{name}: late directions disagree, dot {dot}");
        assert!((ac - ag).abs() < 0.15, "{name}: anisotropy {ac} vs {ag}");
    }
}

#[test]
fn panel_over_the_hole_seals_it() {
    let mesh = holed_room();
    let Some(gpu) = omg_gpu::GpuMeshTracer::new(&mesh) else {
        eprintln!("SKIP panel_over_the_hole_seals_it: no wgpu adapter");
        return;
    };
    let src = Vec3::new(2.0, 3.0, 1.5);
    let lis = Vec3::new(11.0, 3.0, 1.6);
    let sum = |e: &Echogram| -> f32 { e.bins.iter().map(|b| b[1]).sum() };

    let mut open = Echogram::new();
    gpu.trace(src, lis, 16_384, [1.0; 3], 7, &[], &mut open);
    let leaf = omg_gpu::layout::GpuPanel {
        pmin: [7.96, 2.5, 0.0],
        scattering: Material::WOOD_PANEL.scattering,
        pmax: [8.04, 3.5, 2.1],
        _p0: 0,
        absorption: Material::WOOD_PANEL.absorption,
        _p1: 0,
    };
    let mut closed = Echogram::new();
    gpu.trace(src, lis, 16_384, [1.0; 3], 7, &[leaf], &mut closed);
    eprintln!("through-hole energy: open {:.6} closed {:.6}", sum(&open), sum(&closed));
    assert!(
        sum(&closed) < 0.15 * sum(&open),
        "the leaf must seal the hole: {} vs {}",
        sum(&closed),
        sum(&open)
    );
}

#[test]
fn world_trace_speed_probe() {
    let rooms = omg_scene::walkthrough::rooms();
    let doors = omg_scene::walkthrough::doors();
    let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
    let Some(gpu) = omg_gpu::GpuMeshTracer::new(&mesh) else {
        eprintln!("SKIP world_trace_speed_probe: no wgpu adapter");
        return;
    };
    let src = Vec3::new(2.0, 3.0, 1.5);
    let lis = Vec3::new(4.4, 7.5, 1.6);
    let mut e = Echogram::new();
    gpu.trace(src, lis, 8192, [1.0; 3], 1, &[], &mut e); // warm
    let t0 = std::time::Instant::now();
    for k in 0..10u32 {
        gpu.trace(src, lis, 8192, [1.0; 3], k, &[], &mut e);
    }
    let ms = t0.elapsed().as_secs_f64() * 100.0;
    // CPU reference at the tier budget
    let mut rng = Rng::new(3);
    let t1 = std::time::Instant::now();
    for _ in 0..10 {
        trace(&mesh, src, lis, 512, [1.0; 3], &mut rng, &mut e);
    }
    let cpu_ms = t1.elapsed().as_secs_f64() * 100.0;
    eprintln!("world trace: GPU 8192 rays {ms:.2} ms/dispatch vs CPU 512 rays {cpu_ms:.2} ms");
    assert!(ms < 25.0, "one world dispatch must fit the tick comfortably: {ms:.2} ms");
}
