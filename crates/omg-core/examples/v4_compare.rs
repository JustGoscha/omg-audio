//! Engine v3 vs v4, offline: the same scenes traced standing still, and
//! what each engine would render — v3: early taps + chain echoes + a
//! parametric tail (RT60 and level from 80 ms); v4: coherent taps + the
//! traced dense response. Prints each one's distinct arrivals (≥ 6 dB
//! above the energy around them) and decay, and writes mono impulse
//! responses to listen to.
//!
//!   cargo run --release -p omg-core --example v4_compare [out_dir]

use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::rng::Rng;
use omg_core::tracer::BIN_DT;
use omg_core::v3::{Budget, Source, SourceField, Tracer, World, EARLY_S};
use omg_core::vec3::Vec3;
use omg_core::SPEED_OF_SOUND;

const SR: f32 = 48000.0;
const ETC_DT: f32 = 0.001;
const ETC_LEN: usize = 6000;
/// estimate_reverb's late level is in the echogram's legacy units
const LEGACY_UNITS: f32 = 1.0 / 16.0;

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// An axis box's six faces, materials per face: [x0, x1, y0, y1, z0, z1].
fn shoebox(mb: &mut MeshBuilder, mn: Vec3, mx: Vec3, m: [Material; 6]) {
    let ids: Vec<u16> = m.iter().map(|&x| mb.material(x)).collect();
    let (a, b) = (mn, mx);
    mb.begin_surface();
    mb.quad(v(a.x, a.y, a.z), v(a.x, b.y, a.z), v(a.x, b.y, b.z), v(a.x, a.y, b.z), ids[0]);
    mb.begin_surface();
    mb.quad(v(b.x, a.y, a.z), v(b.x, b.y, a.z), v(b.x, b.y, b.z), v(b.x, a.y, b.z), ids[1]);
    mb.begin_surface();
    mb.quad(v(a.x, a.y, a.z), v(b.x, a.y, a.z), v(b.x, a.y, b.z), v(a.x, a.y, b.z), ids[2]);
    mb.begin_surface();
    mb.quad(v(a.x, b.y, a.z), v(b.x, b.y, a.z), v(b.x, b.y, b.z), v(a.x, b.y, b.z), ids[3]);
    mb.begin_surface();
    mb.quad(v(a.x, a.y, a.z), v(b.x, a.y, a.z), v(b.x, b.y, a.z), v(a.x, b.y, a.z), ids[4]);
    mb.begin_surface();
    mb.quad(v(a.x, a.y, b.z), v(b.x, a.y, b.z), v(b.x, b.y, b.z), v(a.x, b.y, b.z), ids[5]);
}

struct Scene {
    name: &'static str,
    mesh: Mesh,
    lis: Vec3,
    src: Vec3,
}

fn scenes() -> Vec<Scene> {
    let mut out = Vec::new();
    // 1. canyon: two rock cliffs 40 m apart, 24 m high, a rock floor, sky
    {
        let mut mb = MeshBuilder::new();
        let r = mb.material(Material::ROCK);
        mb.begin_surface();
        mb.quad(v(-20.0, -25.0, 0.0), v(-20.0, 25.0, 0.0), v(-20.0, 25.0, 24.0), v(-20.0, -25.0, 24.0), r);
        mb.begin_surface();
        mb.quad(v(20.0, -25.0, 0.0), v(20.0, 25.0, 0.0), v(20.0, 25.0, 24.0), v(20.0, -25.0, 24.0), r);
        mb.begin_surface();
        mb.quad(v(-40.0, -60.0, 0.0), v(40.0, -60.0, 0.0), v(40.0, 60.0, 0.0), v(-40.0, 60.0, 0.0), r);
        out.push(Scene { name: "canyon (cliffs 40 m apart), clap at the centre", mesh: mb.build(), lis: v(0.0, 0.0, 1.6), src: v(0.4, 0.0, 1.15) });
    }
    // 2. a bare 4 x 6 x 2.7 m room: two parallel concrete walls, the rest
    // absorptive (carpet floor, tiled ceiling, curtained ends) — flutter
    {
        let mut mb = MeshBuilder::new();
        let c = Material::CONCRETE;
        shoebox(&mut mb, v(0.0, 0.0, 0.0), v(4.0, 6.0, 2.7), [c, c, Material::CURTAIN, Material::CURTAIN, Material::CARPET, Material::ACOUSTIC_TILE]);
        out.push(Scene { name: "bare room 4 m between concrete walls, clap", mesh: mb.build(), lis: v(1.4, 3.0, 1.6), src: v(1.8, 3.0, 1.15) });
    }
    // 3. a furnished-like living room: brick walls, carpet, tiled ceiling
    {
        let mut mb = MeshBuilder::new();
        let b = Material::BRICK;
        shoebox(&mut mb, v(0.0, 0.0, 0.0), v(5.0, 4.0, 2.6), [b, b, b, Material::CURTAIN, Material::CARPET, Material::ACOUSTIC_TILE]);
        out.push(Scene { name: "living room (brick, carpet, curtain, tiles)", mesh: mb.build(), lis: v(1.5, 2.0, 1.6), src: v(3.8, 2.6, 1.2) });
    }
    // 4. a stone hall 16 x 22 x 14 m, all stone (the cathedral's size)
    {
        let mut mb = MeshBuilder::new();
        let s = Material::STONE;
        shoebox(&mut mb, v(0.0, 0.0, 0.0), v(16.0, 22.0, 14.0), [s; 6]);
        out.push(Scene { name: "stone hall 16 x 22 x 14 m", mesh: mb.build(), lis: v(8.0, 6.0, 1.6), src: v(8.0, 16.0, 1.6) });
    }
    out
}

/// Trace `ticks` ticks standing still; the last field.
fn trace(scene: &Scene, v4: bool, ticks: usize) -> (SourceField, f32) {
    let panels = [];
    let world = World::new(&scene.mesh, &panels);
    let mut tr = Tracer::new(0x0e3_5eed);
    tr.v4 = v4;
    let budget = Budget { rays: 1024, bounces: 6, late_rays: 128, late_ema: 0.1, ..Budget::default() };
    let src = [Source { id: 1, pos: scene.src, weight: 1.0 }];
    let t0 = std::time::Instant::now();
    let mut f = None;
    for _ in 0..ticks {
        f = Some(tr.trace(&world, scene.lis, &src, budget).remove(0));
    }
    let ms = t0.elapsed().as_secs_f32() * 1000.0 / ticks as f32;
    (f.expect("ticks > 0"), ms)
}

/// Energy per ETC_DT after emission (mid band), and the free-field time.
fn etc(f: &SourceField, t_free: f32, v4: bool) -> Vec<f32> {
    let mut e = vec![0.0f32; ETC_LEN];
    let put = |e: &mut Vec<f32>, t: f32, x: f32| {
        let i = (t / ETC_DT) as usize;
        if i < e.len() {
            e[i] += x;
        }
    };
    for tap in &f.taps {
        put(&mut e, tap.delay_s, tap.gains[1] * tap.gains[1]);
    }
    if v4 {
        let d = f.dense.as_ref().expect("v4 has a dense response");
        for (i, b) in d.bins.iter().enumerate() {
            // spread each bin evenly over its milliseconds
            let n = (d.bin_s / ETC_DT).round().max(1.0) as usize;
            for k in 0..n {
                put(&mut e, t_free + i as f32 * d.bin_s + k as f32 * ETC_DT, b[1] / n as f32);
            }
        }
    } else {
        // the parametric tail as v3 renders it: exponential decay at the
        // mid RT60 from 80 ms after the free field, holding the late energy
        let rt = f.reverb.rt60[1].max(0.05);
        let late = f.reverb.level[1] * f.reverb.level[1] / LEGACY_UNITS;
        let k = 13.8155 / rt; // ln(10^6)
        let norm = late * k * ETC_DT; // ∫ A e^{-kt} = A/k
        let mut t = 0.0;
        while t < 6.0 {
            put(&mut e, t_free + EARLY_S + t, norm * (-k * t).exp());
            t += ETC_DT;
        }
    }
    e
}

/// Distinct arrivals: ms bins at least 6 dB above the mean of the ±20 ms
/// around them (themselves excluded), after the direct sound; and the
/// decay (T20 of the Schroeder curve, extrapolated to 60 dB).
fn analyse(e: &[f32], t_free: f32) -> (Vec<(f32, f32)>, f32) {
    let start = (t_free / ETC_DT) as usize + 2;
    let direct = e[..start].iter().copied().fold(0.0f32, f32::max).max(1e-30);
    let mut peaks = Vec::new();
    for i in start..e.len() {
        if e[i] <= 0.0 || e[i] < direct * 1e-7 {
            continue;
        }
        let lo = i.saturating_sub(20).max(start);
        let hi = (i + 21).min(e.len());
        let around: f32 = (lo..hi).filter(|&j| j != i).map(|j| e[j]).sum::<f32>() / (hi - lo - 1).max(1) as f32;
        let local_max = (lo..i).all(|j| e[j] < e[i]) && (i + 1..hi).all(|j| e[j] <= e[i]);
        if local_max && e[i] >= 4.0 * around {
            peaks.push(((i as f32 * ETC_DT - t_free) * 1000.0, 10.0 * (e[i] / direct).log10()));
        }
    }
    // Schroeder decay, from just after the direct sound
    let mut sch = vec![0.0f64; e.len()];
    let mut acc = 0.0f64;
    for i in (start..e.len()).rev() {
        acc += e[i] as f64;
        sch[i] = acc;
    }
    let tot = sch[start].max(1e-30);
    let db = |i: usize| 10.0 * (sch[i] / tot).max(1e-30).log10();
    let i5 = (start..e.len()).find(|&i| db(i) <= -5.0);
    let i25 = (start..e.len()).find(|&i| db(i) <= -25.0);
    let rt = match (i5, i25) {
        (Some(a), Some(b)) if b > a => 3.0 * (b - a) as f32 * ETC_DT,
        _ => f32::NAN,
    };
    (peaks, rt)
}

/// A mono impulse response to listen to: taps as impulses, the dense or
/// parametric energy as noise with that envelope.
fn impulse_response(e_taps: &SourceField, e: &[f32], v4: bool, t_free: f32) -> Vec<f32> {
    let n = (e.len() as f32 * ETC_DT * SR) as usize;
    let mut ir = vec![0.0f32; n];
    let mut rng = Rng::new(7);
    // the smooth part: everything that is not a tap
    let mut smooth = e.to_vec();
    for tap in &e_taps.taps {
        let i = (tap.delay_s / ETC_DT) as usize;
        if i < smooth.len() {
            smooth[i] = (smooth[i] - tap.gains[1] * tap.gains[1]).max(0.0);
        }
        let s = (tap.delay_s * SR) as usize;
        if s < n {
            ir[s] += tap.gains[1];
        }
    }
    let per = ETC_DT * SR;
    for (i, &x) in smooth.iter().enumerate() {
        if x <= 0.0 {
            continue;
        }
        let a = (x / per).sqrt();
        for k in 0..per as usize {
            let s = (i as f32 * per) as usize + k;
            if s < n {
                ir[s] += a * (rng.next_f32() * 2.0 - 1.0) * 1.732;
            }
        }
    }
    let _ = (v4, t_free);
    let peak = ir.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-9);
    ir.iter().map(|x| 0.5 * x / peak).collect()
}

fn write_wav(path: &std::path::Path, x: &[f32]) {
    let mut b: Vec<u8> = Vec::new();
    let n = x.len() as u32;
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + 2 * n).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&(SR as u32).to_le_bytes());
    b.extend_from_slice(&(SR as u32 * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(2 * n).to_le_bytes());
    for &s in x {
        b.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, b).expect("write wav");
}

fn main() {
    let out = std::env::args().nth(1).map(std::path::PathBuf::from);
    if let Some(d) = &out {
        std::fs::create_dir_all(d).expect("out dir");
    }
    let ticks = 60;
    let _ = BIN_DT;
    for (si, sc) in scenes().iter().enumerate() {
        let t_free = (sc.src - sc.lis).length() / SPEED_OF_SOUND;
        println!("\n== {} ==", sc.name);
        for v4 in [false, true] {
            let (f, ms) = trace(sc, v4, ticks);
            let e = etc(&f, t_free, v4);
            let (peaks, rt) = analyse(&e, t_free);
            let label = if v4 { "v4" } else { "v3" };
            let dense_len = f.dense.as_ref().map_or(0.0, |d| d.bins.len() as f32 * d.bin_s);
            println!(
                "{label}: {:.1} ms/tick · {} taps · decay (T20→60) {:.2} s{} · {} distinct arrivals after the direct sound:",
                ms,
                f.taps.len(),
                rt,
                if v4 { format!(" · dense response {:.2} s long", dense_len) } else { String::new() },
                peaks.len()
            );
            let shown: Vec<String> = peaks.iter().take(16).map(|(t, db)| format!("{t:.1} ms {db:+.0} dB")).collect();
            println!("    {}", if shown.is_empty() { "(none)".to_string() } else { shown.join(" · ") });
            if let Some(d) = &out {
                let ir = impulse_response(&f, &e, v4, t_free);
                write_wav(&d.join(format!("{}_{}.wav", si + 1, label)), &ir);
            }
        }
    }
}
