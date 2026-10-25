//! Effect: ENGINE V3 — the one listener tracer (`omg_core::v3`, see
//! ENGINE_V3.md). Rays leave the listener; every bounce connects to the
//! sources it can see (next-event estimation); specular chains count
//! when they run into a distance-scaled source sphere, priced from the
//! source's real position; the direct path is one exact segment.
//!
//! Claims guaranteed, against closed-form answers:
//! - free field: exactly one tap, the direct path, air/d;
//! - a wall in between: the direct path pays that wall's transmission;
//! - a mirror: the reflection converges to the image source's 1/L²,
//!   from below, whatever the sphere size k;
//! - a Lambertian floor under a coincident source and listener at
//!   height h: the scattered energy converges to 1/(2h²);
//! - a listener in a barrier's shadow still hears a lit facade behind
//!   them (the bell-in-the-square case);
//! - late RT60 in a closed room agrees with the v2 tracer.
//!
//! Combined in the demo: `early = 3` in WorldSim (V3.1).

mod common;

use common::{flat_opaque, flat_transmissive, v};
use omg_core::material::{air_attenuation, Material};
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::rng::Rng;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::v3::{Budget, Source, Tracer, World, KEY_DIRECT};
use omg_core::vec3::Vec3;

fn one(mesh: &Mesh, lis: Vec3, src: Vec3, budget: Budget, seed: u64) -> omg_core::v3::SourceField {
    let world = World { mesh, panels: &[] };
    let mut t = Tracer::new(seed);
    t.trace(&world, lis, &[Source { id: 0, pos: src, weight: 1.0 }], budget).remove(0)
}

/// Mean of `reflected` over several independent traces.
fn mean_reflected(mesh: &Mesh, lis: Vec3, src: Vec3, budget: Budget, runs: u64) -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    for seed in 0..runs {
        let f = one(mesh, lis, src, budget, 1000 + seed);
        for b in 0..3 {
            acc[b] += f.reflected[b] / runs as f32;
        }
    }
    acc
}

fn floor_slab(material: Material) -> Mesh {
    let mut mb = MeshBuilder::new();
    let m = mb.material(material);
    mb.abox(v(-2000.0, -2000.0, -1.0), v(2000.0, 2000.0, 0.0), m);
    mb.build()
}

#[test]
fn free_field_is_one_exact_direct_tap() {
    let mesh = MeshBuilder::new().build();
    let (lis, src) = (v(0.0, 0.0, 1.6), v(10.0, 0.0, 1.6));
    let f = one(&mesh, lis, src, Budget::default(), 1);
    assert_eq!(f.taps.len(), 1, "free field: direct only, got {:?}", f.taps);
    let t = &f.taps[0];
    assert_eq!(t.key, KEY_DIRECT);
    let air = air_attenuation(10.0);
    for b in 0..3 {
        assert!((t.gains[b] - air[b] / 10.0).abs() < 1e-6, "band {b}: {:?}", t.gains);
    }
    assert!((t.delay_s - 10.0 / omg_core::SPEED_OF_SOUND).abs() < 1e-6);
    assert!(t.dir[0] > 0.999);
    assert!(f.reflected.iter().all(|&e| e == 0.0));
}

#[test]
fn direct_pays_the_wall_between() {
    let mut mb = MeshBuilder::new();
    let tr = [0.1, 0.03, 0.01];
    let m = mb.material(flat_transmissive(0.2, 0.0, tr));
    mb.abox(v(4.9, -50.0, -50.0), v(5.1, 50.0, 50.0), m);
    let mesh = mb.build();
    let wall = mesh.materials[0].transmission_at(0.2);
    let f = one(&mesh, v(0.0, 0.0, 1.6), v(10.0, 0.0, 1.6), Budget { rays: 64, bounces: 1, ..Budget::default() }, 2);
    let d = f.taps.iter().find(|t| t.key == KEY_DIRECT).expect("direct tap");
    let air = air_attenuation(10.0);
    for b in 0..3 {
        let want = wall[b] * air[b] / 10.0;
        assert!((d.gains[b] - want).abs() < 1e-6 * want.max(1e-6) + 1e-9, "band {b}: {} vs {want}", d.gains[b]);
    }
}

#[test]
fn mirror_converges_to_the_image_source() {
    // perfect mirror floor: R = 1, specular only
    let mesh = floor_slab(flat_opaque(0.0, 0.0));
    let h = 2.0;
    let (lis, src) = (v(0.0, 0.0, h), v(6.0, 0.0, h));
    let l_img = (36.0f32 + 4.0 * h * h).sqrt();
    let air = air_attenuation(l_img);
    let want = air[1] * air[1] / (l_img * l_img);
    for k in [0.04f32, 0.1] {
        let got = mean_reflected(&mesh, lis, src, Budget { rays: 20_000, bounces: 1, k, ema: 1.0, nee: 0, late_rays: 0, late_ema: 1.0, late_nee: 0 }, 6);
        let ratio = got[1] / want;
        assert!((ratio - 1.0).abs() < 0.15, "k {k}: mirror energy {:.3e} vs image {:.3e} (×{ratio:.3})", got[1], want);
    }
    // and it arrives from below, at the image delay
    let f = one(&mesh, lis, src, Budget { rays: 20_000, bounces: 1, k: 0.06, ema: 1.0, nee: 0, late_rays: 0, late_ema: 1.0, late_nee: 0 }, 9);
    let refl = f.taps.iter().filter(|t| t.key != KEY_DIRECT).max_by(|a, b| a.gains[1].total_cmp(&b.gains[1])).expect("a reflection tap");
    assert!(refl.dir[2] < -0.2, "mirror reflection must come from below: {:?}", refl.dir);
    assert!((refl.delay_s - l_img / omg_core::SPEED_OF_SOUND).abs() < 0.002, "delay {} vs {}", refl.delay_s, l_img / omg_core::SPEED_OF_SOUND);
}

#[test]
fn early_tap_delays_do_not_jitter_between_ticks() {
    // a static scene traced tick after tick: every early tap that
    // persists must keep EXACTLY its delay (the renderer glides any
    // same-key delay change as motion — jitter is audible pitch warble)
    let mesh = floor_slab(flat_opaque(0.1, 0.6));
    let world = World { mesh: &mesh, panels: &[] };
    let mut t = Tracer::new(21);
    let src = [Source { id: 0, pos: v(8.0, 3.0, 2.0), weight: 1.0 }];
    let b = Budget { rays: 512, bounces: 2, k: 0.06, ema: 0.35, nee: 0, late_rays: 32, late_ema: 0.1, late_nee: 0 };
    let mut prev: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for tick in 0..10 {
        let f = t.trace(&world, v(0.0, 0.0, 1.6), &src, b).remove(0);
        for tap in &f.taps {
            if let Some(d) = prev.get(&tap.key) {
                assert_eq!(*d, tap.delay_s, "tick {tick}: key {} moved {} -> {}", tap.key, d, tap.delay_s);
            }
            prev.insert(tap.key, tap.delay_s);
        }
    }
}

#[test]
fn lambertian_floor_converges_to_one_over_two_h_squared() {
    // white Lambertian floor, source and listener coincident at height h:
    // ∫ (cosθ/ρ²)(1/π)(cosθ/ρ²) dA = 1/(2h²)
    let mesh = floor_slab(flat_opaque(0.0, 1.0));
    let h = 2.0;
    let p = v(0.0, 0.0, h);
    let got = mean_reflected(&mesh, p, p, Budget { rays: 20_000, bounces: 1, k: 0.06, ema: 1.0, nee: 0, late_rays: 0, late_ema: 1.0, late_nee: 0 }, 6);
    let want = 1.0 / (2.0 * h * h);
    // low band: air absorption is negligible over these paths
    let ratio = got[0] / want;
    assert!((ratio - 1.0).abs() < 0.1, "diffuse floor {:.4} vs analytic {:.4} (×{ratio:.3})", got[0], want);
}

#[test]
fn shadowed_listener_hears_a_lit_facade_behind_them() {
    // source far north and high; an opaque 6 m barrier shadows the
    // listener; a tall diffuse facade behind the listener rises into the
    // light and sends the source back
    let mut mb = MeshBuilder::new();
    let barrier = mb.material(flat_opaque(0.1, 0.1));
    let facade = mb.material(flat_opaque(0.1, 0.8));
    mb.abox(v(-40.0, 10.0, 0.0), v(40.0, 10.5, 6.0), barrier);
    mb.abox(v(-40.0, -2.0, 0.0), v(40.0, 0.0, 25.0), facade);
    let mesh = mb.build();
    let (lis, src) = (v(0.0, 5.0, 1.6), v(0.0, 200.0, 25.0));
    let f = one(&mesh, lis, src, Budget { rays: 8192, bounces: 2, k: 0.06, ema: 1.0, nee: 0, late_rays: 0, late_ema: 1.0, late_nee: 0 }, 3);
    let direct = f.taps.iter().find(|t| t.key == KEY_DIRECT);
    assert!(direct.is_none() || direct.unwrap().gains[1] < 1e-6, "the barrier must block the direct path");
    let free = 1.0 / (200.0f32 * 200.0);
    let e = f.reflected[1];
    eprintln!("shadow fill: {:.1} dB re free field", 10.0 * (e / free).log10());
    assert!(e > 0.01 * free, "the lit facade must reach the shadowed listener: {e:.3e} vs free {free:.3e}");
    // it comes from behind (−y) and above
    let best = f.taps.iter().filter(|t| t.key != KEY_DIRECT).max_by(|a, b| a.gains[1].total_cmp(&b.gains[1])).unwrap();
    assert!(best.dir[1] < 0.0 && best.dir[2] > 0.0, "from the facade behind and above: {:?}", best.dir);
}

#[test]
fn late_rt60_agrees_with_the_v2_tracer() {
    let mut mb = MeshBuilder::new();
    let m = mb.material(flat_opaque(0.25, 0.5));
    common::shell(&mut mb, v(0.0, 0.0, 0.0), v(8.0, 6.0, 3.0), [Some(0); 6].map(|_| Some(m)));
    let mesh = mb.build();
    let (lis, src) = (v(2.0, 2.0, 1.5), v(6.0, 4.0, 1.5));
    // v2: source-launched stochastic trace
    let mut echo = Echogram::new();
    trace(&mesh, src, lis, 20_000, [1.0; 3], &mut Rng::new(5), &mut echo);
    let v2 = estimate_reverb(&echo);
    // v3: a few early bounces + the late family carrying the tail
    let mut t = Tracer::new(7);
    let world = World { mesh: &mesh, panels: &[] };
    let mut f = None;
    for _ in 0..8 {
        f = Some(t.trace(&world, lis, &[Source { id: 0, pos: src, weight: 1.0 }], Budget { rays: 1024, bounces: 4, k: 0.06, ema: 0.5, nee: 0, late_rays: 1024, late_ema: 0.5, late_nee: 0 }).remove(0));
    }
    let v3 = f.unwrap().reverb;
    eprintln!("rt60 v2 {:?} v3 {:?}; level v2 {:?} v3 {:?}", v2.rt60, v3.rt60, v2.level, v3.level);
    let r = v3.rt60[1] / v2.rt60[1];
    assert!((r - 1.0).abs() < 0.2, "mid RT60: v3 {} vs v2 {}", v3.rt60[1], v2.rt60[1]);
    let l = v3.level[1] / v2.level[1];
    assert!((0.6..1.6).contains(&l), "late level: v3 {} vs v2 {}", v3.level[1], v2.level[1]);
}

#[test]
fn sampled_connections_stay_unbiased() {
    // eight sources over a diffuse floor: connecting to 2 sampled
    // sources per bounce must converge to connecting to all eight
    let mesh = floor_slab(flat_opaque(0.0, 1.0));
    let lis = v(0.0, 0.0, 2.0);
    let srcs: Vec<Source> = (0..8)
        .map(|i| Source { id: i, pos: v(3.0 * i as f32 - 10.0, 4.0, 1.0 + 0.3 * i as f32), weight: 1.0 + i as f32 })
        .collect();
    let world = World { mesh: &mesh, panels: &[] };
    let run = |nee: usize, seed: u64| -> Vec<f32> {
        let mut t = Tracer::new(seed);
        t.trace(&world, lis, &srcs, Budget { rays: 20_000, bounces: 1, k: 0.06, ema: 1.0, nee, late_rays: 0, late_ema: 1.0, late_nee: 0 })
            .iter()
            .map(|f| f.reflected[0])
            .collect()
    };
    let mut all = vec![0.0f32; 8];
    let mut two = vec![0.0f32; 8];
    for seed in 0..4 {
        for (i, (a, b)) in run(0, seed).iter().zip(run(2, 100 + seed)).enumerate() {
            all[i] += a / 4.0;
            two[i] += b / 4.0;
        }
    }
    for i in 0..8 {
        let r = two[i] / all[i];
        assert!((r - 1.0).abs() < 0.2, "source {i}: sampled {:.4e} vs all {:.4e} (×{r:.3})", two[i], all[i]);
    }
}

#[test]
fn live_room_tail_is_traced_to_the_end() {
    // the v2 tracer's 64-bounce cap cut this room's tail to ~1.1 s
    // (fx_late_rt60::live_room_rt60_is_not_truncated); the late family
    // bounces until the path has lost 60 dB, so Eyring holds
    let mut mb = MeshBuilder::new();
    let alpha = 0.04f32;
    let m = mb.material(flat_opaque(alpha, 0.5));
    let (x, y, z) = (8.0f32, 6.0f32, 3.0f32);
    common::shell(&mut mb, v(0.0, 0.0, 0.0), v(x, y, z), [Some(m); 6]);
    let mesh = mb.build();
    let world = World { mesh: &mesh, panels: &[] };
    let mut t = Tracer::new(11);
    let src = [Source { id: 0, pos: v(6.0, 4.0, 1.5), weight: 1.0 }];
    let mut f = None;
    for _ in 0..8 {
        f = Some(t.trace(&world, v(2.0, 2.0, 1.5), &src, Budget { rays: 512, bounces: 4, k: 0.06, ema: 0.5, nee: 0, late_rays: 1024, late_ema: 0.5, late_nee: 0 }).remove(0));
    }
    let rt = f.unwrap().reverb.rt60;
    let (vol, area) = (x * y * z, 2.0 * (x * y + x * z + y * z));
    let eyring = 0.161 * vol / (-area * (1.0 - alpha).ln());
    eprintln!("live room RT60 v3 {rt:?} vs Eyring {eyring:.2} s");
    // low band: air absorption is negligible, Eyring is the reference
    let r = rt[0] / eyring;
    assert!((r - 1.0).abs() < 0.25, "low-band RT60 {:.2} s vs Eyring {eyring:.2} s", rt[0]);
    // air absorption shortens the highs
    assert!(rt[2] < rt[0], "air must shorten the high band: {rt:?}");
}
