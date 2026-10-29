//! Engine v3 on the GPU (ENGINE_V3.md V3.3, kernel K5) against the CPU
//! walk it ports. Both run the same estimator from the same listener;
//! the GPU's RNG and float order differ, so parity is statistical: means
//! over independent ticks, level in dB, RT60 as a ratio — the same
//! discipline as mesh_parity.rs. Scenes, each isolating one mechanism:
//! - free field: direct only, every ray reaches the sky;
//! - mirror floor: specular chains into the source sphere (→ image source);
//! - Lambertian floor: next-event estimation (→ 1/(2h²));
//! - shoebox: both families, early taps and the late RT60/level;
//! - shadowed facade: NEE into a barrier's shadow;
//! - thick wall: NEE segments priced through paired faces (mass law);
//! - panel over a doorway: overlay boxes reflect and transmit;
//! - bunker: earth swallows escapes and segments, pockets don't;
//! - eight sources, two sampled per bounce: importance sampling unbiased;
//! - diffraction (V3.4): rays bending over a barrier and reflecting on,
//!   the knife-edge shadow, and the demo world at its real budget.
//! Self-skips without an adapter.

#[path = "../../omg-core/tests/common/mod.rs"]
mod common;

use common::{flat_opaque, flat_transmissive, v};
use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::v3::fold::Walk;
use omg_core::v3::{Budget, Source, SourceField, Tracer, World, KEY_DIRECT};
use omg_core::vec3::Vec3;
use omg_gpu::GpuV3Tracer;

fn gpu(mesh: &Mesh, name: &str) -> Option<GpuV3Tracer> {
    let g = GpuV3Tracer::new(mesh);
    if g.is_none() {
        eprintln!("SKIP {name}: no wgpu adapter");
    }
    g
}

fn cpu_tick(t: &mut Tracer, w: &World, lis: Vec3, srcs: &[Source], b: Budget) -> Vec<SourceField> {
    t.trace(w, lis, srcs, b)
}

fn gpu_tick(g: &mut GpuV3Tracer, t: &mut Tracer, w: &World, lis: Vec3, srcs: &[Source], b: Budget) -> Vec<SourceField> {
    t.trace_walked(w, lis, srcs, b, None, &mut |job| Walk::Fresh(g.walk(job).expect("walk")))
        .expect("job within caps")
}

/// One-shot budget: a single family, no smoothing.
fn shot(rays: u32, bounces: u32, nee: usize, late_rays: u32) -> Budget {
    Budget { rays, bounces, k: 0.06, ema: 1.0, nee, late_rays, late_ema: 1.0, late_nee: nee.min(1), diffraction: false, ..Budget::default() }
}

/// Median per-source `reflected` over `runs` independent ticks, both
/// ways. Medians, not means: the estimator is heavy-tailed (a rare ray
/// that transmits through a wall on the importance-floored branch and
/// then grazes a near source sphere carries 10× a typical tick), so a
/// mean over a handful of ticks measures which side drew the spike.
fn median_reflected(
    g: &mut GpuV3Tracer,
    w: &World,
    lis: Vec3,
    srcs: &[Source],
    b: Budget,
    runs: u64,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    quantile_reflected(g, w, lis, srcs, b, runs, 0.5)
}

/// The `q` quantile of per-tick `reflected` (a bimodal field — rare
/// events on top of a steady base — compares on its lower quartile).
fn quantile_reflected(
    g: &mut GpuV3Tracer,
    w: &World,
    lis: Vec3,
    srcs: &[Source],
    b: Budget,
    runs: u64,
    q: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut c = vec![[Vec::new(), Vec::new(), Vec::new()]; srcs.len()];
    let mut gg = c.clone();
    for k in 0..runs {
        let fc = cpu_tick(&mut Tracer::new(100 + k), w, lis, srcs, b);
        let fg = gpu_tick(g, &mut Tracer::new(900 + k), w, lis, srcs, b);
        for i in 0..srcs.len() {
            for band in 0..3 {
                c[i][band].push(fc[i].reflected[band]);
                gg[i][band].push(fg[i].reflected[band]);
            }
        }
    }
    let med = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f32 * q).round() as usize]
    };
    let fold = |mut x: Vec<[Vec<f32>; 3]>| -> Vec<[f32; 3]> {
        x.iter_mut().map(|s| [med(&mut s[0]), med(&mut s[1]), med(&mut s[2])]).collect()
    };
    (fold(c), fold(gg))
}

fn db_e(e: f32) -> f32 {
    10.0 * e.max(1e-30).log10()
}

fn one(id: u32, pos: Vec3) -> [Source; 1] {
    [Source { id, pos, weight: 1.0 }]
}

fn floor_slab(m: Material) -> Mesh {
    let mut mb = MeshBuilder::new();
    let id = mb.material(m);
    // ±150 m: the analytic answers converge long before (the Lambertian
    // tail beyond R is ~(h/R)²), and a ±2 km slab tessellates into a
    // prim buffer past the device's 256 MB binding limit
    mb.abox(v(-150.0, -150.0, -1.0), v(150.0, 150.0, 0.0), id);
    mb.build()
}

#[test]
fn v3_free_field_direct_only_and_full_sky() {
    let mesh = MeshBuilder::new().build();
    let Some(mut g) = gpu(&mesh, "v3_free_field") else { return };
    let w = World::new(&mesh, &[]);
    let (lis, src) = (v(0.0, 0.0, 1.6), v(10.0, 0.0, 1.6));
    let b = Budget { rays: 4096, late_rays: 256, diffraction: false, ..Budget::default() };
    let mut tg = Tracer::new(1);
    let f = gpu_tick(&mut g, &mut tg, &w, lis, &one(0, src), b).remove(0);
    assert_eq!(f.taps.len(), 1, "free field: direct only, got {:?}", f.taps);
    assert_eq!(f.taps[0].key, KEY_DIRECT);
    assert!(f.reflected.iter().all(|&e| e == 0.0), "{:?}", f.reflected);
    // every early ray escapes with no reflection: the sky holds 1.0,
    // weighted toward no direction in particular
    let mut tc = Tracer::new(2);
    cpu_tick(&mut tc, &w, lis, &one(0, src), b);
    let sum = |t: &Tracer| t.sky().iter().map(|(_, e)| e[1]).sum::<f32>();
    let (sg, sc) = (sum(&tg), sum(&tc));
    eprintln!("free-field sky: cpu {sc:.5} gpu {sg:.5}");
    assert!((sg - 1.0).abs() < 1e-3 && (sc - 1.0).abs() < 1e-3);
    let mean = |t: &Tracer| {
        t.sky().iter().fold([0.0f32; 3], |a, (d, e)| [a[0] + d[0] * e[1], a[1] + d[1] * e[1], a[2] + d[2] * e[1]])
    };
    let m = mean(&tg);
    assert!(m.iter().all(|x| x.abs() < 0.02), "isotropic sky: {m:?}");
}

#[test]
fn v3_mirror_floor_matches_cpu_and_the_image_source() {
    let mesh = floor_slab(flat_opaque(0.0, 0.0));
    let Some(mut g) = gpu(&mesh, "v3_mirror") else { return };
    let w = World::new(&mesh, &[]);
    let h = 2.0f32;
    let (lis, src) = (v(0.0, 0.0, h), v(6.0, 0.0, h));
    let l_img = (36.0f32 + 4.0 * h * h).sqrt();
    let air = omg_core::material::air_attenuation(l_img);
    let want = air[1] * air[1] / (l_img * l_img);
    let (c, gg) = median_reflected(&mut g, &w, lis, &one(0, src), shot(20_000, 1, 0, 0), 7);
    let (rc, rg) = (c[0][1] / want, gg[0][1] / want);
    eprintln!("mirror: cpu ×{rc:.3} gpu ×{rg:.3} of the image source");
    assert!((rg - 1.0).abs() < 0.15, "gpu mirror ×{rg:.3}");
    assert!((rg / rc - 1.0).abs() < 0.12, "gpu/cpu {:.3}", rg / rc);
    // arrives from below at the image delay; debug capture draws it
    let mut t = Tracer::new(5);
    t.debug = true;
    let f = gpu_tick(&mut g, &mut t, &w, lis, &one(0, src), shot(20_000, 1, 0, 0)).remove(0);
    let refl = f.taps.iter().filter(|t| t.key != KEY_DIRECT).max_by(|a, b| a.gains[1].total_cmp(&b.gains[1])).unwrap();
    assert!(refl.dir[2] < -0.2, "from below: {:?}", refl.dir);
    assert!((refl.delay_s - l_img / omg_core::SPEED_OF_SOUND).abs() < 0.002);
    let walked: Vec<_> = t.debug_paths.iter().filter(|p| p.2.len() == 3).collect();
    assert!(!walked.is_empty(), "debug capture: {} paths", t.debug_paths.len());
    let p = &walked[0].2;
    assert!((p[0] - lis).length() < 1e-4 && (p[2] - src).length() < 1e-4 && p[1].z.abs() < 1e-3, "{p:?}");
}

#[test]
fn v3_lambertian_floor_matches_cpu_and_analytic() {
    let mesh = floor_slab(flat_opaque(0.0, 1.0));
    let Some(mut g) = gpu(&mesh, "v3_lambertian") else { return };
    let w = World::new(&mesh, &[]);
    let h = 2.0f32;
    let p = v(0.0, 0.0, h);
    let (c, gg) = median_reflected(&mut g, &w, p, &one(0, p), shot(20_000, 1, 0, 0), 7);
    let want = 1.0 / (2.0 * h * h);
    let (rc, rg) = (c[0][0] / want, gg[0][0] / want);
    eprintln!("lambertian: cpu ×{rc:.4} gpu ×{rg:.4} of 1/(2h²)");
    assert!((rg - 1.0).abs() < 0.1, "gpu diffuse floor ×{rg:.3}");
    assert!((rg / rc - 1.0).abs() < 0.05, "gpu/cpu {:.4}", rg / rc);
}

fn shoebox(alpha: f32, scatter: f32) -> Mesh {
    let mut mb = MeshBuilder::new();
    let m = mb.material(flat_opaque(alpha, scatter));
    common::shell(&mut mb, v(0.0, 0.0, 0.0), v(8.0, 6.0, 3.0), [Some(m); 6]);
    mb.build()
}

#[test]
fn v3_shoebox_early_and_late_match_cpu() {
    let mesh = shoebox(0.25, 0.5);
    let Some(mut g) = gpu(&mesh, "v3_shoebox") else { return };
    let w = World::new(&mesh, &[]);
    let (lis, src) = (v(2.0, 2.0, 1.5), v(6.0, 4.0, 1.5));
    let b = Budget { rays: 2048, bounces: 4, k: 0.06, ema: 0.35, nee: 0, late_rays: 512, late_ema: 0.15, late_nee: 0, diffraction: false, ..Budget::default() };
    let (mut tc, mut tg) = (Tracer::new(7), Tracer::new(8));
    // the early field per tick: summed tap energy (direct excluded) and
    // the energy-weighted mean direction, compared as medians over ticks
    let early = |f: &SourceField| {
        f.taps.iter().filter(|t| t.key != KEY_DIRECT).fold((0.0f32, [0.0f32; 3]), |(e, d), t| {
            let p = t.gains[1] * t.gains[1];
            (e + p, [d[0] + t.dir[0] * p, d[1] + t.dir[1] * p, d[2] + t.dir[2] * p])
        })
    };
    let (mut ec, mut eg, mut rc, mut rg) = (vec![], vec![], vec![], vec![]);
    let (mut fc, mut fg) = (None, None);
    for tick in 0..24 {
        let c = cpu_tick(&mut tc, &w, lis, &one(0, src), b).remove(0);
        let g1 = gpu_tick(&mut g, &mut tg, &w, lis, &one(0, src), b).remove(0);
        if tick >= 4 {
            ec.push(early(&c).0);
            eg.push(early(&g1).0);
            rc.push(c.reflected[1]);
            rg.push(g1.reflected[1]);
        }
        fc = Some(c);
        fg = Some(g1);
    }
    let (fc, fg) = (fc.unwrap(), fg.unwrap());
    eprintln!("shoebox rt60 cpu {:?} gpu {:?}; level cpu {:?} gpu {:?}", fc.reverb.rt60, fg.reverb.rt60, fc.reverb.level, fg.reverb.level);
    for band in 0..3 {
        let r = fg.reverb.rt60[band] / fc.reverb.rt60[band];
        assert!((0.85..1.18).contains(&r), "band {band} rt60 gpu/cpu {r:.3}");
        let dl = 20.0 * (fg.reverb.level[band] / fc.reverb.level[band]).log10();
        assert!(dl.abs() < 1.0, "band {band} late level {dl:.2} dB");
    }
    let med = |mut v: Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let (ec, eg, rc, rg) = (med(ec), med(eg), med(rc), med(rg));
    eprintln!(
        "shoebox early taps: cpu {:.2} dB gpu {:.2} dB ({} vs {} taps); reflected/tick cpu {:.2} dB gpu {:.2} dB",
        db_e(ec),
        db_e(eg),
        fc.taps.len(),
        fg.taps.len(),
        db_e(rc),
        db_e(rg)
    );
    assert!((db_e(eg) - db_e(ec)).abs() < 1.0, "early tap energy cpu {ec:.3e} gpu {eg:.3e}");
    assert!((db_e(rg) - db_e(rc)).abs() < 0.5, "reflected per tick cpu {rc:.3e} gpu {rg:.3e}");
    assert_eq!(fg.taps.len(), fc.taps.len(), "both fill the tap budget");
    let (_, dg) = early(&fg);
    let (_, dc) = early(&fc);
    let cos = (dg[0] * dc[0] + dg[1] * dc[1] + dg[2] * dc[2])
        / ((dg[0] * dg[0] + dg[1] * dg[1] + dg[2] * dg[2]).sqrt() * (dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]).sqrt());
    eprintln!("shoebox early mean direction agreement cos {cos:.3}");
}

#[test]
fn v3_shadowed_facade_fill_matches_cpu() {
    let mut mb = MeshBuilder::new();
    let barrier = mb.material(flat_opaque(0.1, 0.1));
    let facade = mb.material(flat_opaque(0.1, 0.8));
    mb.abox(v(-40.0, 10.0, 0.0), v(40.0, 10.5, 6.0), barrier);
    mb.abox(v(-40.0, -2.0, 0.0), v(40.0, 0.0, 25.0), facade);
    let mesh = mb.build();
    let Some(mut g) = gpu(&mesh, "v3_facade") else { return };
    let w = World::new(&mesh, &[]);
    let (lis, src) = (v(0.0, 5.0, 1.6), v(0.0, 200.0, 25.0));
    let (c, gg) = median_reflected(&mut g, &w, lis, &one(0, src), shot(8192, 2, 0, 0), 7);
    let free = 1.0 / (200.0f32 * 200.0);
    eprintln!("facade fill: cpu {:.2} dB gpu {:.2} dB re free field", db_e(c[0][1] / free), db_e(gg[0][1] / free));
    assert!(gg[0][1] > 0.01 * free, "the lit facade must reach the shadow");
    assert!((db_e(gg[0][1]) - db_e(c[0][1])).abs() < 1.0);
    let f = gpu_tick(&mut g, &mut Tracer::new(3), &w, lis, &one(0, src), shot(8192, 2, 0, 0)).remove(0);
    // the fill's energy by direction (a single tap can be a discrete
    // echo between barrier and facade, arriving from the barrier)
    let mut d = [0.0f32; 3];
    for t in f.taps.iter().filter(|t| t.key != KEY_DIRECT) {
        for k in 0..3 {
            d[k] += t.dir[k] * t.gains[1] * t.gains[1];
        }
    }
    assert!(d[1] < 0.0 && d[2] > 0.0, "from the facade behind and above: {d:?}");
}

#[test]
fn v3_thick_wall_pays_the_mass_law_on_both() {
    // every reflected path from source to listener crosses the 0.3 m
    // wall: NEE segments from the floor through two paired faces
    let mut mb = MeshBuilder::new();
    let floor = mb.material(flat_opaque(0.1, 0.9));
    let wall = mb.material(flat_transmissive(0.1, 0.3, [0.1, 0.03, 0.01]));
    mb.abox(v(-60.0, -60.0, -1.0), v(60.0, 60.0, 0.0), floor);
    mb.abox(v(4.85, -60.0, 0.0), v(5.15, 60.0, 30.0), wall);
    let mesh = mb.build();
    let Some(mut g) = gpu(&mesh, "v3_thick_wall") else { return };
    let w = World::new(&mesh, &[]);
    let (lis, src) = (v(0.0, 0.0, 1.6), v(10.0, 0.0, 1.6));
    // a tick scatters ~2.8 dB here: the median of 21 is within ~0.6 dB
    // (of 7 it was ~1 dB — as wide as the tolerance)
    let (c, gg) = median_reflected(&mut g, &w, lis, &one(0, src), shot(16_384, 2, 0, 0), 21);
    for band in 0..3 {
        let d = db_e(gg[0][band]) - db_e(c[0][band]);
        eprintln!("thick wall band {band}: cpu {:.2} dB gpu {:.2} dB", db_e(c[0][band]), db_e(gg[0][band]));
        assert!(d.abs() < 1.0, "band {band}: {d:.2} dB");
    }
    // the mass law shows: highs lose far more than lows
    assert!(gg[0][2] < gg[0][0] * 0.1);
}

/// A shell room with a doorway hole in its x = 8 wall.
fn holed_room() -> Mesh {
    let mut mb = MeshBuilder::new();
    let m = mb.material(flat_opaque(0.1, 0.5));
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

#[test]
fn v3_panel_over_a_doorway_matches_cpu() {
    let mesh = holed_room();
    let Some(mut g) = gpu(&mesh, "v3_panel") else { return };
    let leaf = Material { transmission: [0.05, 0.02, 0.01], ..Material::WOOD_PANEL };
    let panels = [(v(7.95, 2.5, 0.0), v(8.05, 3.5, 2.1), leaf)];
    let (lis, src) = (v(4.0, 3.0, 1.6), v(14.0, 3.0, 1.6));
    let b = shot(8192, 3, 0, 0);
    let open = World::new(&mesh, &[]);
    let shut = World::new(&mesh, &panels);
    // behind the leaf the field is bimodal: a steady base (NEE through
    // the leaf) plus rare heavy rays that transmitted on the importance-
    // floored branch — its median flips between the modes from tick to
    // tick. Only ~28 % of ticks sit in the base (200 seeds, CPU: 57 and
    // 58 of 200 before and after the per-ray random streams): the lower
    // quartile sits on the edge between the modes and flips with any
    // draw; the lowest decile is inside the base
    let (co, go) = median_reflected(&mut g, &open, lis, &one(0, src), b, 21);
    let (cs, gs) = quantile_reflected(&mut g, &shut, lis, &one(0, src), b, 61, 0.1);
    eprintln!(
        "doorway: open cpu {:.2} gpu {:.2} dB; leaf cpu {:.2} gpu {:.2} dB",
        db_e(co[0][1]),
        db_e(go[0][1]),
        db_e(cs[0][1]),
        db_e(gs[0][1])
    );
    assert!((db_e(go[0][1]) - db_e(co[0][1])).abs() < 1.0, "open doorway");
    assert!((db_e(gs[0][1]) - db_e(cs[0][1])).abs() < 1.0, "leaf in the doorway");
    assert!(gs[0][1] < go[0][1] * 0.3, "the leaf must shut the doorway: {:.3e} vs {:.3e}", gs[0][1], go[0][1]);
    // the sky: through the doorway only, on both
    let sky = |w: &World, gpu: Option<&mut GpuV3Tracer>| {
        let mut t = Tracer::new(77);
        match gpu {
            Some(g) => {
                gpu_tick(g, &mut t, w, lis, &one(0, src), b);
            }
            None => {
                cpu_tick(&mut t, w, lis, &one(0, src), b);
            }
        }
        t.sky().iter().map(|(_, e)| e[1]).sum::<f32>()
    };
    let (sc, sg) = (sky(&open, None), sky(&open, Some(&mut g)));
    eprintln!("doorway sky: cpu {sc:.4} gpu {sg:.4}");
    assert!((db_e(sg) - db_e(sc)).abs() < 1.0);
}

#[test]
fn v3_earth_swallows_what_leaves_a_bunker_on_gpu() {
    let mut mb = MeshBuilder::new();
    let wall = mb.material(flat_transmissive(0.05, 0.2, [0.04, 0.005, 0.0006]));
    common::shell(&mut mb, v(0.0, 0.0, -3.0), v(6.0, 8.0, -0.8), [Some(wall), Some(wall), Some(wall), Some(wall), None, Some(wall)]);
    let ground = mb.material(flat_transmissive(0.1, 0.3, [0.1, 0.02, 0.005]));
    mb.quad(v(-50.0, -50.0, 0.0), v(50.0, -50.0, 0.0), v(50.0, 50.0, 0.0), v(-50.0, 50.0, 0.0), ground);
    let mesh = mb.build();
    let Some(mut g) = gpu(&mesh, "v3_bunker") else { return };
    let pockets = [(v(0.0, 0.0, -3.0), v(6.0, 8.0, -0.8))];
    let lis = v(3.0, 4.0, -1.4);
    let src = one(0, v(3.0, 30.0, 1.6));
    let b = shot(4096, 3, 0, 256);
    let sky_of = |w: &World, g: &mut GpuV3Tracer| {
        let mut t = Tracer::new(4);
        gpu_tick(g, &mut t, w, lis, &src, b);
        let s = t.sky().iter().map(|(_, e)| e[1]).sum::<f32>();
        let mut tc = Tracer::new(4);
        cpu_tick(&mut tc, w, lis, &src, b);
        (tc.sky().iter().map(|(_, e)| e[1]).sum::<f32>(), s)
    };
    let (oc, og) = sky_of(&World::new(&mesh, &[]), &mut g);
    let earthed = World { grade: Some(0.0), pockets: &pockets, ..World::new(&mesh, &[]) };
    let (ec, eg) = sky_of(&earthed, &mut g);
    eprintln!("bunker sky: open cpu {oc:.4} gpu {og:.4}; earthed cpu {ec:.2e} gpu {eg:.2e}");
    assert!(og > 0.1 && (db_e(og) - db_e(oc)).abs() < 1.0);
    assert!(eg < 1e-3, "earthed gpu sky {eg}");
    // reflections reaching the bunker pay the earth/roof like the CPU's
    let (c, gg) = median_reflected(&mut g, &earthed, lis, &src, b, 7);
    eprintln!("bunker reflected: cpu {:.2} dB gpu {:.2} dB", db_e(c[0][0]), db_e(gg[0][0]));
    assert!((db_e(gg[0][0]) - db_e(c[0][0])).abs() < 1.5);
}

#[test]
fn v3_sampled_connections_stay_unbiased_on_gpu() {
    let mesh = floor_slab(flat_opaque(0.0, 1.0));
    let Some(mut g) = gpu(&mesh, "v3_sampled") else { return };
    let w = World::new(&mesh, &[]);
    let lis = v(0.0, 0.0, 2.0);
    let srcs: Vec<Source> = (0..8)
        .map(|i| Source { id: i, pos: v(3.0 * i as f32 - 10.0, 4.0, 1.0 + 0.3 * i as f32), weight: 1.0 + i as f32 })
        .collect();
    let mut all = vec![0.0f32; 8];
    let mut two = vec![0.0f32; 8];
    for k in 0..4u64 {
        let a = cpu_tick(&mut Tracer::new(k), &w, lis, &srcs, shot(20_000, 1, 0, 0));
        let s = gpu_tick(&mut g, &mut Tracer::new(50 + k), &w, lis, &srcs, shot(40_000, 1, 2, 0));
        for i in 0..8 {
            all[i] += a[i].reflected[0] / 4.0;
            two[i] += s[i].reflected[0] / 4.0;
        }
    }
    for i in 0..8 {
        let r = two[i] / all[i];
        assert!((r - 1.0).abs() < 0.15, "source {i}: gpu sampled {:.4e} vs cpu all {:.4e} (×{r:.3})", two[i], all[i]);
    }
}

#[test]
fn v3_bends_over_a_barrier_match_cpu() {
    // the shadowed-facade street with diffraction ON: rays from the
    // listener bend over the barrier's top edge and come back off the
    // facade (and vice versa) — the walk's own bends, gated by what the
    // straight line from the bend point already carries
    let mut mb = MeshBuilder::new();
    let ground = mb.material(flat_opaque(0.1, 0.5));
    let barrier = mb.material(flat_opaque(0.1, 0.1));
    let facade = mb.material(flat_opaque(0.1, 0.8));
    mb.quad(v(-60.0, -20.0, 0.0), v(60.0, -20.0, 0.0), v(60.0, 80.0, 0.0), v(-60.0, 80.0, 0.0), ground);
    mb.abox(v(-40.0, 10.0, 0.0), v(40.0, 10.5, 6.0), barrier);
    mb.abox(v(-40.0, -2.0, 0.0), v(40.0, 0.0, 25.0), facade);
    let mesh = mb.build();
    let Some(mut g) = gpu(&mesh, "v3_bends") else { return };
    let w = World { grade: Some(0.0), ..World::new(&mesh, &[]) };
    let (lis, src) = (v(0.0, 5.0, 1.6), v(0.0, 40.0, 3.0));
    let on = Budget { rays: 8192, bounces: 3, k: 0.06, ema: 1.0, nee: 0, late_rays: 0, late_ema: 1.0, late_nee: 0, diffraction: true, ..Budget::default() };
    let off = Budget { diffraction: false, ..on };
    let (c_on, g_on) = median_reflected(&mut g, &w, lis, &one(0, src), on, 9);
    let (c_off, g_off) = median_reflected(&mut g, &w, lis, &one(0, src), off, 9);
    for band in 0..3 {
        eprintln!(
            "bends band {band}: diffraction on cpu {:.2} gpu {:.2} dB | off cpu {:.2} gpu {:.2} dB",
            db_e(c_on[0][band]),
            db_e(g_on[0][band]),
            db_e(c_off[0][band]),
            db_e(g_off[0][band])
        );
        assert!((db_e(g_on[0][band]) - db_e(c_on[0][band])).abs() < 1.0, "band {band} with bends");
    }
    // the bends add (bass most), on both
    assert!(c_on[0][0] > c_off[0][0] && g_on[0][0] > g_off[0][0]);
}

#[test]
fn v3_knife_edge_shadow_matches_cpu() {
    // fx_v3_tracer's absorbing half-plane: in the shadow the field is the
    // edge next-event path (CPU-side on both) plus the walk's multi-bend
    // paths (GPU-side) — the taps must agree band by band
    let mut mb = MeshBuilder::new();
    let m = mb.material(flat_opaque(1.0, 0.0));
    mb.quad(v(0.0, -300.0, -300.0), v(0.0, 300.0, -300.0), v(0.0, 300.0, 0.0), v(0.0, -300.0, 0.0), m);
    let mesh = mb.build();
    let Some(mut g) = gpu(&mesh, "v3_knife_edge") else { return };
    let w = World::new(&mesh, &[]);
    let src = one(0, v(-20.0, 0.0, -1.0));
    let b = Budget { rays: 8192, bounces: 1, k: 0.06, ema: 1.0, nee: 0, late_rays: 0, late_ema: 1.0, late_nee: 0, diffraction: true, ..Budget::default() };
    let energy = |f: &SourceField| {
        f.taps.iter().fold([0.0f32; 3], |e, t| core::array::from_fn(|k| e[k] + t.gains[k] * t.gains[k]))
    };
    for zl in [-2.0f32, -5.0, -10.0] {
        let lis = v(20.0, 0.0, zl);
        let mut ec = [0.0f32; 3];
        let mut eg = [0.0f32; 3];
        for k in 0..4u64 {
            let c = energy(&cpu_tick(&mut Tracer::new(10 + k), &w, lis, &src, b)[0]);
            let gg = energy(&gpu_tick(&mut g, &mut Tracer::new(60 + k), &w, lis, &src, b)[0]);
            for band in 0..3 {
                ec[band] += c[band] / 4.0;
                eg[band] += gg[band] / 4.0;
            }
        }
        for band in 0..3 {
            let d = db_e(eg[band]) - db_e(ec[band]);
            eprintln!("knife edge z {zl} band {band}: cpu {:.2} gpu {:.2} dB", db_e(ec[band]), db_e(eg[band]));
            assert!(d.abs() < 1.0, "z {zl} band {band}: {d:+.2} dB");
        }
    }
}

#[test]
fn v3_demo_world_fields_match_cpu() {
    // the real thing: the demo world, earth and all, the tier budget with
    // diffraction, several sources; per-source tap energy and late level
    // after the EMA has settled
    let rooms = omg_scene::walkthrough::rooms();
    let doors = omg_scene::walkthrough::doors();
    let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
    let Some(mut g) = gpu(&mesh, "v3_demo_world") else { return };
    let w = World { grade: Some(0.0), ..World::new(&mesh, &[]) };
    let lis = v(22.5, 31.0, 1.6);
    let srcs: Vec<Source> = (0..6)
        .map(|i| {
            let a = i as f32 * 1.1;
            let r = 5.0 + 6.0 * i as f32;
            Source { id: i, pos: v(22.5 + r * a.cos(), 31.0 + r * a.sin(), 1.6), weight: 1.0 }
        })
        .collect();
    let b = Budget { rays: 2048, bounces: 6, k: 0.06, ema: 0.35, nee: 4, late_rays: 1024, late_ema: 0.1, late_nee: 1, diffraction: true, ..Budget::default() };
    let (mut tc, mut tg) = (Tracer::new(1), Tracer::new(2));
    let n = 30;
    let mut ec = vec![Vec::new(); srcs.len()];
    let mut eg = vec![Vec::new(); srcs.len()];
    let (mut fc, mut fg) = (Vec::new(), Vec::new());
    for tick in 0..n {
        fc = cpu_tick(&mut tc, &w, lis, &srcs, b);
        fg = gpu_tick(&mut g, &mut tg, &w, lis, &srcs, b);
        if tick >= 10 {
            for i in 0..srcs.len() {
                let e = |f: &SourceField| f.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum::<f32>();
                ec[i].push(e(&fc[i]));
                eg[i].push(e(&fg[i]));
            }
        }
    }
    let med = |v: &mut Vec<f32>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    for i in 0..srcs.len() {
        let (c, gg) = (med(&mut ec[i]), med(&mut eg[i]));
        let (lc, lg) = (fc[i].reverb.level[1], fg[i].reverb.level[1]);
        eprintln!(
            "demo source {i}: taps cpu {:.2} gpu {:.2} dB | late level cpu {:.2} gpu {:.2} dB | rt60 cpu {:.2} gpu {:.2} s",
            db_e(c),
            db_e(gg),
            20.0 * lc.max(1e-9).log10(),
            20.0 * lg.max(1e-9).log10(),
            fc[i].reverb.rt60[1],
            fg[i].reverb.rt60[1]
        );
        // the strong sources pin to 1 dB; a far, walled-off one's few
        // early bins carry the Monte Carlo spread of its rare paths (two
        // CPU seeds differ by up to 2 dB there too)
        if db_e(c) < -40.0 {
            continue; // −40 dB re a unit source at 1 m: printed, not pinned
        }
        let tol = if db_e(c) > -20.0 { 1.0 } else { 2.5 };
        assert!((db_e(gg) - db_e(c)).abs() < tol, "source {i} tap energy");
        // the tail of a far, walled-off source is a handful of sampled
        // late connections per tick: only the well-sampled ones pin
        if 20.0 * lc.log10() > -40.0 {
            assert!((20.0 * (lg / lc).log10()).abs() < 1.5, "source {i} late level");
        }
    }
}

/// Discrete echoes on the GPU path: the kernel skips the chains the CPU
/// solves exactly, the CPU names them with its own mirror rays — the
/// flutter between two bare parallel walls comes out the same train.
#[test]
fn v3_flutter_train_matches_cpu() {
    let mut mb = MeshBuilder::new();
    let hard = mb.material(Material::CONCRETE);
    let tile = mb.material(Material::ACOUSTIC_TILE);
    let carpet = mb.material(Material::CARPET);
    common::shell(
        &mut mb,
        v(0.0, 0.0, 0.0),
        v(4.0, 6.0, 2.7),
        [Some(hard), Some(hard), Some(tile), Some(tile), Some(carpet), Some(tile)],
    );
    let mesh = mb.build();
    let Some(mut g) = gpu(&mesh, "v3_flutter") else { return };
    let w = World::new(&mesh, &[]);
    let lis = v(1.3, 3.0, 1.6);
    let src = one(0, lis + v(0.2, 0.15, -0.3));
    let b = Budget::default();
    let train = |f: &SourceField| -> Vec<i32> {
        let t0 = f.taps.iter().map(|t| t.delay_s).fold(f32::MAX, f32::min);
        let mut v: Vec<i32> =
            f.taps.iter().filter(|t| t.delay_s > t0 + 0.08).map(|t| (t.delay_s * 2000.0).round() as i32).collect();
        v.sort();
        v
    };
    let (mut tc, mut tg) = (Tracer::new(5), Tracer::new(5));
    let (mut fc, mut fg) = (None, None);
    for _ in 0..80 {
        fc = Some(cpu_tick(&mut tc, &w, lis, &src, b).remove(0));
        fg = Some(gpu_tick(&mut g, &mut tg, &w, lis, &src, b).remove(0));
    }
    let (c, gg) = (train(&fc.unwrap()), train(&fg.unwrap()));
    eprintln!("flutter train past 80 ms: cpu {} taps, gpu {} taps", c.len(), gg.len());
    assert!(gg.len() >= 15, "the GPU path flutters too: {gg:?}");
    // chains are found at random (and then kept): either path may be a
    // few orders further along, past a gap in its train — over the span
    // both trains cover without one (two flutter periods), what both
    // found must be the same
    let gapless = |v: &[i32]| -> i32 {
        let period = (2.0 * 4.0 / 343.0 * 2000.0) as i32;
        v.windows(2).find(|w| w[1] - w[0] > 2 * period).map_or(*v.last().unwrap_or(&0), |w| w[0])
    };
    let span = gapless(&c).min(gapless(&gg));
    let (cs, gs): (Vec<i32>, Vec<i32>) = (c.iter().copied().filter(|&x| x <= span).collect(), gg.iter().copied().filter(|&x| x <= span).collect());
    let shared = gs.iter().filter(|x| cs.contains(x)).count();
    assert!(shared * 10 >= gs.len().min(cs.len()) * 9, "same arrivals (±0.25 ms) up to {span}: cpu {c:?} gpu {gg:?}");
}

#[test]
fn v3_walk_speed_probe() {
    // the demo world, the listener at the club doorway, 16 sources at
    // the tier budget (High: 1024 early × 6 reflections + 128 late, with
    // diffraction): CPU walk vs the GPU walk at 8× — synchronously (wall
    // clock incl. readback) and as the app runs it, asynchronously (one
    // walk in flight, ticks 50 ms apart: the sim thread only pays for the
    // direct + edge paths, the job and the fold)
    use omg_scene::v3walk::V3Walker;
    let rooms = omg_scene::walkthrough::rooms();
    let doors = omg_scene::walkthrough::doors();
    let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
    let Some(mut g) = gpu(&mesh, "v3_speed") else { return };
    let w = World { grade: Some(0.0), ..World::new(&mesh, &[]) };
    let lis = v(22.5, 31.0, 1.6);
    let srcs: Vec<Source> = (0..16)
        .map(|i| {
            let a = i as f32 * 0.7;
            Source { id: i, pos: v(22.5 + (8.0 + 3.0 * i as f32) * a.cos(), 31.0 + (8.0 + 3.0 * i as f32) * a.sin(), 1.6), weight: 1.0 }
        })
        .collect();
    let cpu_b = Budget { rays: 1024, bounces: 6, late_rays: 128, ema: 0.35, late_ema: 0.1, ..Budget::default() };
    let gpu_b = Budget { rays: 8192, late_rays: 1024, ..cpu_b };
    let n = 6;
    let time = |f: &mut dyn FnMut()| {
        f(); // warm up
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            f();
        }
        t0.elapsed().as_secs_f32() * 1e3 / n as f32
    };
    let mut tc = Tracer::new(1);
    let cpu_ms = time(&mut || {
        cpu_tick(&mut tc, &w, lis, &srcs, cpu_b);
    });
    let mut tg = Tracer::new(2);
    let gpu1_ms = time(&mut || {
        gpu_tick(&mut g, &mut tg, &w, lis, &srcs, cpu_b);
    });
    let mut walk_ms = 0.0;
    let gpu8_ms = time(&mut || {
        gpu_tick(&mut g, &mut tg, &w, lis, &srcs, gpu_b);
        walk_ms = g.last_ms;
    });
    // as the app runs: ticks 50 ms apart. The gap is spun, not slept:
    // macOS demotes a thread that sleeps 80 % of the time to an
    // efficiency core, which stretches ANY 2 ms burst to ~11 ms (measured
    // without a GPU at all) — that is the scheduler, not the walk.
    let ticks = 20;
    let cadence = |f: &mut dyn FnMut() -> bool| {
        let (mut ms, mut fresh) = (0.0f32, 0u32);
        for k in 0..ticks + 2 {
            let t0 = std::time::Instant::now();
            let got = f();
            if k >= 2 {
                ms += t0.elapsed().as_secs_f32() * 1e3 / ticks as f32;
                fresh += got as u32;
            }
            while t0.elapsed().as_millis() < 50 {
                std::hint::spin_loop();
            }
        }
        (ms, fresh)
    };
    let mut tc2 = Tracer::new(4);
    let (cpu_tick_ms, _) = cadence(&mut || {
        cpu_tick(&mut tc2, &w, lis, &srcs, cpu_b);
        true
    });
    drop(g);
    // the live walker unpaced (all of the GPU): the async path's latency
    let mut walker = omg_gpu::GpuV3Walker::live(&mesh, 1.0).expect("a GPU above");
    let mut ta = Tracer::new(3);
    let (sim_ms, fresh) = cadence(&mut || {
        let mut got = false;
        ta.trace_walked(&w, lis, &srcs, gpu_b, Some(cpu_b.late_rays), &mut |job| {
            let r = walker.walk(job);
            got = matches!(r, Walk::Fresh(_));
            r
        })
        .unwrap();
        got
    });
    eprintln!(
        "v3 tick, demo world, 16 sources, diffraction on —\n  back to back: CPU walk 1024+128 rays {cpu_ms:.1} ms | GPU sync 1024+128 {gpu1_ms:.1} ms | GPU sync 8192+1024 {gpu8_ms:.1} ms (walk {walk_ms:.1})\n  at 20 Hz: CPU walk 1024+128 {cpu_tick_ms:.1} ms/tick | GPU async 8192+1024 {sim_ms:.1} ms/tick on the sim thread, fresh {fresh}/{ticks}"
    );
    assert!(gpu8_ms < 250.0, "GPU v3 tick {gpu8_ms:.1} ms");
    assert!(fresh >= ticks / 2, "async walks must land most ticks: {fresh}/{ticks}");
}

/// Engine v4 on the GPU: the same dense response as the CPU walk — total
/// scattered energy within 1.5 dB, the same decay (the energy past 100 ms
/// in the same share within 1.5 dB) — and the same coherent arrivals (the
/// CPU solves the mirror chains for both).
#[test]
fn v4_dense_response_matches_cpu() {
    let mesh = shoebox(0.08, 0.3);
    let Some(mut g) = gpu(&mesh, "v4_dense_response_matches_cpu") else { return };
    let panels = [];
    let w = World::new(&mesh, &panels);
    let lis = v(2.0, 2.5, 1.6);
    let srcs = one(1, v(6.0, 3.5, 1.5));
    let b = Budget { rays: 1024, bounces: 4, late_rays: 256, ..Budget::default() };
    let (mut tc, mut tg) = (Tracer::new(11), Tracer::new(12));
    tc.v4 = true;
    tg.v4 = true;
    let (mut fc, mut fg) = (None, None);
    for _ in 0..30 {
        fc = Some(cpu_tick(&mut tc, &w, lis, &srcs, b).remove(0));
        fg = Some(gpu_tick(&mut g, &mut tg, &w, lis, &srcs, b).remove(0));
    }
    let (fc, fg) = (fc.unwrap(), fg.unwrap());
    let (dc, dg) = (fc.dense.as_ref().expect("cpu dense"), fg.dense.as_ref().expect("gpu dense"));
    let sum = |d: &omg_core::v3::DenseResponse, from: usize| d.bins.iter().skip(from).map(|x| x[1]).sum::<f32>();
    let late = (0.1 / dc.bin_s) as usize;
    let (tc_, tg_) = (sum(dc, 0), sum(dg, 0));
    let (lc, lg) = (sum(dc, late) / tc_, sum(dg, late) / tg_);
    eprintln!(
        "v4 dense: total cpu {:.2} dB gpu {:.2} dB · past 100 ms cpu {:.2} dB gpu {:.2} dB · taps cpu {} gpu {}",
        db_e(tc_), db_e(tg_), db_e(lc), db_e(lg), fc.taps.len(), fg.taps.len()
    );
    assert!((db_e(tc_) - db_e(tg_)).abs() < 1.5, "dense totals differ");
    assert!((db_e(lc) - db_e(lg)).abs() < 1.5, "decays differ");
    assert!(fg.taps.len() >= 3 && fc.taps.len().abs_diff(fg.taps.len()) <= fc.taps.len() / 3 + 2, "coherent taps differ");
}
