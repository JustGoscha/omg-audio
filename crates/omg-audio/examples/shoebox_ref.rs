//! A shoebox against its image sources: what each engine traces — the
//! direct sound, the early part (its first 80 ms) and the late part —
//! against the image-source method's exact sum of arrivals (Allen &
//! Berkley; `omg_core::ism`, to order 40: beyond it lies at most
//! (1 − α)⁴⁰ of the energy — 1.5 % in the low band at α 0.1). Per band,
//! in dB against the image sources.
//!
//! The image sources are exact for mirror walls. With scattering (the
//! engines' walls scatter a share of each reflection, `SCATTER`), the
//! energy a reflection keeps is the same and arrives later on average:
//! the totals stay the reference, the early/late split shifts towards
//! late.
//!
//!     cargo run --release -p omg-audio --example shoebox_ref
//!     SCATTER=0..1 (default 0)  SIZE=x,y,z (10,7,4)  ALPHA=low,mid,high
//!     (0.1,0.2,0.3)  SRC=x,y,z  LIS=x,y,z  TICKS=n (60)  MODES=v6.2,v6.1,v5
//!     K=k: the source sphere's radius per metre of path
//!     CLEARANCE=1: sphere hits by their reachable share (`Config::clearance`)
//!     PROFILE=1: v5's response per 100 ms against the images'
//!     FLOOR=1: a floor alone (an infinite plane at z = 0): the direct
//!     sound and its one image

use omg_audio::{Config, Geometry, Material, Quality, Simulation, SourceDesc};
use omg_core::ism::image_source_taps;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};

const EARLY: f32 = 0.08;

fn env3(name: &str, d: [f32; 3]) -> [f32; 3] {
    std::env::var(name)
        .ok()
        .and_then(|v| {
            let p: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (p.len() == 3).then(|| [p[0], p[1], p[2]])
        })
        .unwrap_or(d)
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-30).log10()
}

/// (direct, early, late) energy per band.
type Parts = [[f64; NBANDS]; 3];

fn main() {
    let size = env3("SIZE", [10.0, 7.0, 4.0]);
    let alpha = env3("ALPHA", [0.1, 0.2, 0.3]);
    let sp = env3("SRC", [2.5, 2.0, 1.5]);
    let lp = env3("LIS", [7.0, 4.5, 1.6]);
    let scatter: f32 = std::env::var("SCATTER").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    let modes = std::env::var("MODES").unwrap_or_else(|_| "v6.2,v6.1,v5".into());
    let mat = Material::new(alpha, scatter, [0.0; NBANDS]);

    // the image sources
    let t_direct = {
        let d = [sp[0] - lp[0], sp[1] - lp[1], sp[2] - lp[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() / SPEED_OF_SOUND
    };
    let floor = std::env::var_os("FLOOR").is_some();
    let room = Shoebox::new(Vec3::new(size[0], size[1], size[2]), [mat; 6]);
    let mut taps = Vec::new();
    if floor {
        // the direct sound and the floor's image
        for (z, refl) in [(sp[2], false), (-sp[2], true)] {
            let d = [z - lp[2], sp[1] - lp[1], sp[0] - lp[0]];
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let air = omg_core::material::air_attenuation(l);
            let r = mat.reflection_amplitude();
            taps.push(omg_core::params::Tap { key: 0, delay_s: l / SPEED_OF_SOUND, dir: [0.0; 3], gains: core::array::from_fn(|b| air[b] / l * if refl { r[b] } else { 1.0 }) });
        }
    } else {
        image_source_taps(&room, Vec3::new(sp[0], sp[1], sp[2]), Vec3::new(lp[0], lp[1], lp[2]), 40, &mut taps);
    }
    let mut ism: Parts = [[0.0; NBANDS]; 3];
    for t in &taps {
        let part = if (t.delay_s - t_direct).abs() < 1e-5 { 0 } else if t.delay_s - t_direct < EARLY { 1 } else { 2 };
        for b in 0..NBANDS {
            ism[part][b] += (t.gains[b] as f64).powi(2);
        }
    }
    println!(
        "shoebox {:?} m, α {:?}, scattering {scatter}; source {:?}, ears {:?} ({:.1} m); {} image sources",
        size,
        alpha,
        sp,
        lp,
        t_direct * SPEED_OF_SOUND,
        taps.len()
    );
    let line = |name: &str, p: &Parts, r: &Parts| {
        let f = |k: usize| -> String {
            (0..NBANDS).map(|b| format!("{:+5.1}", db(p[k][b]) - db(r[k][b]))).collect::<Vec<_>>().join(" ")
        };
        println!("   {name:6} direct {}   early {}   late {}   (dB from the image sources)", f(0), f(1), f(2));
    };
    let abs = |p: &Parts| {
        let f = |k: usize| (0..NBANDS).map(|b| format!("{:6.1}", db(p[k][b]))).collect::<Vec<_>>().join(" ");
        format!("direct {}   early {}   late {}", f(0), f(1), f(2))
    };
    println!("   images {}", abs(&ism));

    for mode in modes.split(',') {
        let mut g = Geometry::new();
        let m = g.material(mat);
        let t = 0.3;
        let (lx, ly, lz) = (size[0], size[1], size[2]);
        if floor {
            g.add_box([-500.0, -500.0, -t], [500.0, 500.0, 0.0], m);
        } else {
            g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
            g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
            g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
            g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
            g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
            g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
        }
        // K=k: the sphere radius per metre of path (the rung's otherwise)
        let quality = match std::env::var("K").ok().and_then(|v| v.parse::<f32>().ok()) {
            Some(k) => Quality::Custom(omg_core::v3::Budget { k, ..Quality::High.budget(false) }),
            None => Quality::High,
        };
        let cfg = Config {
            quality,
            v4: mode == "v5",
            v5: mode == "v5",
            v6: mode.starts_with("v6"),
            v61: mode == "v6.1" || mode == "v6.2",
            v62: mode == "v6.2",
            clearance: std::env::var("CLEARANCE").as_deref() == Ok("1"),
            kernel_thread: false,
            ..Config::default()
        };
        let mut sim = Simulation::new(cfg);
        sim.set_geometry(g);
        let src = sim.add_source(SourceDesc::at(sp));
        sim.set_listener(lp, 0.0, 0.0, 0.0);
        let mut f = sim.tick();
        for _ in 1..ticks {
            f = sim.tick();
        }
        let pb = f.block(src).expect("the source's block");
        let mut p: Parts = [[0.0; NBANDS]; 3];
        for tp in &pb.taps {
            let part = if tp.key == omg_core::v3::KEY_DIRECT { 0 } else if tp.delay_s - pb.onset_s < EARLY { 1 } else { 2 };
            for b in 0..NBANDS {
                p[part][b] += (tp.gains[b] as f64).powi(2);
            }
        }
        for b in 0..NBANDS {
            p[1][b] += (pb.early.level[b] as f64).powi(2);
        }
        match sim.dense_response(src) {
            Some(d) => {
                for (i, x) in d.bins.iter().enumerate() {
                    let part = if (i as f32 + 0.5) * d.bin_s < EARLY { 1 } else { 2 };
                    for b in 0..NBANDS {
                        p[part][b] += x[b].max(0.0) as f64;
                    }
                }
            }
            None => {
                for b in 0..NBANDS {
                    p[2][b] += (pb.reverb.level[b] as f64).powi(2) + pb.remote.map_or(0.0, |r| (r.send[b] as f64).powi(2));
                }
            }
        }
        println!("   {mode:6} {}", abs(&p));
        line(mode, &p, &ism);
        // PROFILE=1 (v5, whose traced response is kept in bins): the mid
        // band per 100 ms after the direct sound, engine against images
        if std::env::var_os("PROFILE").is_some() {
            if let Some(d) = sim.dense_response(src) {
                let mut eng = [0.0f64; 30];
                for (i, x) in d.bins.iter().enumerate() {
                    let w = (((i as f32 + 0.5) * d.bin_s) / 0.1) as usize;
                    if w < 30 {
                        eng[w] += x[1].max(0.0) as f64;
                    }
                }
                for tp in pb.taps.iter().filter(|t| t.key != omg_core::v3::KEY_DIRECT) {
                    let w = ((tp.delay_s - pb.onset_s) / 0.1) as usize;
                    if w < 30 {
                        eng[w] += (tp.gains[1] as f64).powi(2);
                    }
                }
                let mut img = [0.0f64; 30];
                for t in &taps {
                    if (t.delay_s - t_direct).abs() < 1e-5 {
                        continue;
                    }
                    let w = ((t.delay_s - t_direct) / 0.1) as usize;
                    if w < 30 {
                        img[w] += (t.gains[1] as f64).powi(2);
                    }
                }
                let row: Vec<String> = (0..12).map(|w| format!("{:+.1}", db(eng[w]) - db(img[w]))).collect();
                println!("   {mode:6} mid band per 100 ms, dB from the images: {}", row.join(" "));
            }
        }
    }
}
