//! One wall (50 m long, `H` tall, scattering `SCAT`) and the ground
//! (scattering `GSCAT`; `NOGROUND=1`: none), a source and the ears as on
//! the town's square: the engine's taps against the image sources — the
//! ground's reflection, and the wall's echo with its twin off the ground
//! (within 2 ms), mid band, dB under the direct sound.
//!
//!     cargo run --release -p omg-audio --example wall_echo
//!     K=k (0.06) RAYS=n (1024) SEED=n NODIFF=1 (no edge bends) MODE=v6.2
//!     LIST=1: the taps of the last tick
use omg_audio::{Config, Geometry, Material, Quality, Simulation, SourceDesc};
use omg_core::SPEED_OF_SOUND;

fn main() {
    let h: f32 = std::env::var("H").ok().and_then(|v| v.parse().ok()).unwrap_or(3.1);
    let scat: f32 = std::env::var("SCAT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.10);
    let gscat: f32 = std::env::var("GSCAT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
    let mode = std::env::var("MODE").unwrap_or_else(|_| "v6.2".into());
    let noground = std::env::var_os("NOGROUND").is_some();
    let wall_m = Material::new([0.02; 3], scat, [0.0; 3]);
    let ground_m = Material::new([0.04; 3], gscat, [0.0; 3]);
    let (sp, lp) = ([5.0f32, 40.0, 1.2], [15.0f32, 32.0, 1.6]);
    let share = |m: &Material| (1.0 - m.absorption[1] - m.transmission[1] * m.transmission[1]) * (1.0 - m.scattering);
    let d = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let dl = d(sp, lp);
    // images: the ground (z → −z), the wall (y = 24: y → 48 − y)
    let imgs = [("ground", [sp[0], sp[1], -sp[2]], share(&ground_m)), ("wall", [sp[0], 48.0 - sp[1], sp[2]], share(&wall_m)), ("wall+ground", [sp[0], 48.0 - sp[1], -sp[2]], share(&wall_m) * share(&ground_m))];
    let v6 = mode.starts_with("v6");
    let rays: u32 = std::env::var("RAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let kk: f32 = std::env::var("K").ok().and_then(|v| v.parse().ok()).unwrap_or(0.06);
    let quality = Quality::Custom(omg_core::v3::Budget { rays, k: kk, diffraction: std::env::var_os("NODIFF").is_none(), ..Quality::High.budget(false) });
    let cfg = Config { quality, v4: !v6, v5: !v6, v6, v61: v6, v62: mode == "v6.2", kernel_thread: false, seed: std::env::var("SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(1), ..Config::default() };
    let mut g = Geometry::new();
    let (wm, gm) = (g.material(wall_m), g.material(ground_m));
    g.add_box([-10.0, 23.7, 0.0], [40.0, 24.0, h], wm);
    if !noground {
        g.ground_plane(0.0, gm);
    }
    let mut sim = Simulation::new(cfg);
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at(sp));
    sim.set_listener(lp, 0.0, 0.0, 0.0);
    let ticks = 80;
    let mut acc = vec![0.0f64; 4];
    let mut early_field = 0.0f64;
    for k in 0..ticks {
        let f = sim.tick();
        if k < 20 {
            continue;
        }
        let pb = f.block(src).expect("block");
        let dg = pb.taps.iter().find(|t| t.key == omg_core::v3::KEY_DIRECT).expect("direct").gains[1] as f64;
        // the ground's (≤ 2 ms), the wall's echo with its twin (within 2 ms)
        let at_w = (d(imgs[1].1, lp) / SPEED_OF_SOUND) as f64;
        for t in &pb.taps {
            let rel = t.delay_s as f64;
            let e = (t.gains[1] as f64 / dg).powi(2);
            if t.key == omg_core::v3::KEY_DIRECT {
                continue;
            }
            if rel - pb.onset_s as f64 <= 0.002 {
                acc[0] += e;
            } else if (rel - at_w).abs() <= 0.002 {
                acc[1] += e;
            } else {
                acc[2] += e;
            }
        }
        early_field += (pb.early.level[1] as f64 / dg).powi(2);
        if k == ticks - 1 && std::env::var_os("LIST").is_some() {
            let mut v: Vec<_> = pb.taps.iter().collect();
            v.sort_by(|a, b| a.delay_s.total_cmp(&b.delay_s));
            for t in v {
                println!("      {:+8.3} ms (echo at {:+.3})  {:+6.1} dB  key {:#x}", (t.delay_s - pb.onset_s) * 1e3, (at_w - pb.onset_s as f64) * 1e3, 20.0 * (t.gains[1] as f64 / dg).log10(), t.key);
            }
        }
    }
    let n = (ticks - 20) as f64;
    let db = |x: f64| 10.0 * x.max(1e-30).log10();
    let rel = |im: [f32; 3], s: f32| (s as f64) * (dl as f64 / d(im, lp) as f64).powi(2);
    let want_g = rel(imgs[0].1, imgs[0].2);
    let want_w = rel(imgs[1].1, imgs[1].2) + if noground { 0.0 } else { rel(imgs[2].1, imgs[2].2) };
    println!("{mode} wall {h} m (scat {scat}), ground {} (scat {gscat}): k·l at the echo {:.2} m", if noground { "none" } else { "yes" }, 0.06 * d(imgs[1].1, lp));
    println!("   ground refl: engine {:+.2} dB, images {:+.2} dB", db(acc[0] / n), db(want_g));
    println!("   wall echo (+twin) within 2 ms: engine {:+.2} dB, images {:+.2} dB  → {:+.2} dB", db(acc[1] / n), db(want_w), db(acc[1] / n) - db(want_w));
    println!("   other taps {:+.2} dB, the dense early field {:+.2} dB", db(acc[2] / n), db(early_field / n));
}
