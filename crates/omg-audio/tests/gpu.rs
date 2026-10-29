//! `Compute::Gpu` (feature `gpu`): the v3 walk on the GPU through the
//! core `V3Walker` seam. The same claims as the CPU tests, on the GPU's
//! budget (8× the rays) with its one-tick-late results. Self-skips
//! without an adapter (the simulation falls back to the CPU then).
#![cfg(feature = "gpu")]

use omg_audio::{Compute, Config, Geometry, Material, ParamBlock, Simulation, SourceDesc};

fn energy(pb: &ParamBlock) -> [f32; 3] {
    core::array::from_fn(|b| {
        pb.taps.iter().map(|t| t.gains[b] * t.gains[b]).sum::<f32>()
            + pb.reverb.level[b] * pb.reverb.level[b]
            + pb.remote.map_or(0.0, |r| r.send[b] * r.send[b])
    })
}

fn shoebox() -> Geometry {
    let mut g = Geometry::new();
    let m = g.material(Material::new([0.1; 3], 0.3, [0.0; 3]));
    let (lx, ly, lz, t) = (8.0, 6.0, 3.0, 0.2);
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
    g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
    g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
    g
}

#[test]
fn gpu_walk_matches_the_cpu_walk() {
    let run = |compute: Compute| {
        let mut sim = Simulation::new(Config { compute, ..Config::default() });
        sim.set_geometry(shoebox());
        let src = sim.add_source(SourceDesc::at([2.0, 2.0, 1.5]));
        sim.set_listener([5.5, 4.0, 1.6], 0.0, 0.0, 0.0);
        let mut f = sim.tick();
        for _ in 0..40 {
            f = sim.tick();
        }
        (sim.gpu_active(), f.block(src).unwrap().clone())
    };
    let (gpu_on, gpu) = run(Compute::GpuSync);
    if !gpu_on {
        eprintln!("SKIP: no wgpu adapter (the simulation ran on the CPU)");
        return;
    }
    let (_, cpu) = run(Compute::Cpu);
    let (eg, ec) = (energy(&gpu), energy(&cpu));
    eprintln!("gpu rt60 {:?} energy {eg:?}\ncpu rt60 {:?} energy {ec:?}", gpu.reverb.rt60, cpu.reverb.rt60);
    assert!((gpu.reverb.rt60[1] / cpu.reverb.rt60[1] - 1.0).abs() < 0.2, "rt60 gpu vs cpu");
    assert!((eg[1] / ec[1] - 1.0).abs() < 0.3, "energy gpu vs cpu");
    let d = |pb: &ParamBlock| pb.taps.iter().find(|t| t.key == omg_core::v3::KEY_DIRECT).map(|t| t.gains);
    assert_eq!(d(&gpu), d(&cpu), "the direct path is exact on both");
}

#[test]
fn async_gpu_walks_arrive_a_tick_late() {
    // the real-time mode: a tick never waits for the GPU; paced like a
    // host's 20 Hz loop, the walks land and the late field builds up
    let mut sim = Simulation::new(Config { compute: Compute::Gpu, ..Config::default() });
    sim.set_geometry(shoebox());
    if !sim.gpu_active() {
        eprintln!("SKIP: no wgpu adapter");
        return;
    }
    let src = sim.add_source(SourceDesc::at([2.0, 2.0, 1.5]));
    sim.set_listener([5.5, 4.0, 1.6], 0.0, 0.0, 0.0);
    let first = sim.tick();
    let direct_only = first.block(src).unwrap().taps.len();
    let mut f = first;
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        f = sim.tick();
    }
    let pb = f.block(src).unwrap();
    // the traced response's own decay (v4); the block's rt60 is its
    // parametric stand-in, set by the response's centre time (~0.7 s here
    // on the CPU too)
    let rt = sim.dense_response(src).map(|d| d.rt60()[1]).unwrap_or(0.0);
    eprintln!("async gpu: first tick {direct_only} taps, later {} taps, traced rt60 {rt:.2} s (stand-in {:?})", pb.taps.len(), pb.reverb.rt60);
    assert!(pb.taps.len() > direct_only, "reflections arrive");
    assert!(rt > 0.7, "a traced tail, not the fallback: {rt:.2} s");
}

/// A closed 6×5×3 room (20 cm walls) with a 1 × 2.1 m doorway in its +x
/// wall (`fx_simulation`'s), its leaf covering `1 − open` from y = 2, the
/// doorway the leaf's aperture (v6.2); the room's source heard by `ears`.
fn doorway_scene(mode: &str, compute: Compute, ears: [f32; 3], open: f32) -> (Simulation, omg_audio::SourceId, omg_audio::ObjectId) {
    let mut g = Geometry::new();
    let w = g.material(Material::CONCRETE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [6.0 + t, 5.0 + t, 0.0], w);
    g.add_box([-t, -t, 3.0], [6.0 + t, 5.0 + t, 3.0 + t], w);
    g.add_box([-t, -t, 0.0], [0.0, 5.0 + t, 3.0], w);
    g.add_box([-t, -t, 0.0], [6.0 + t, 0.0, 3.0], w);
    g.add_box([-t, 5.0, 0.0], [6.0 + t, 5.0 + t, 3.0], w);
    g.add_box([6.0, 0.0, 0.0], [6.0 + t, 2.0, 3.0], w);
    g.add_box([6.0, 3.0, 0.0], [6.0 + t, 5.0, 3.0], w);
    g.add_box([6.0, 2.0, 2.1], [6.0 + t, 3.0, 3.0], w);
    let ground = g.material(Material::GRASS);
    g.ground_plane(-t, ground);
    let mut sim = Simulation::new(Config { compute, v6: true, v61: true, v62: mode == "v6.2", kernel_thread: false, ..Config::default() });
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at([2.0, 1.5, 1.5]));
    sim.set_listener(ears, 0.0, 0.0, 0.0);
    let door = sim.add_object([6.08, 2.0, 0.0], [6.12, 3.0, 2.1], Material::DOOR);
    sim.set_object_aperture(door, [6.08, 2.0, 0.0], [6.12, 3.0, 2.1]);
    set_leaf(&mut sim, door, open);
    (sim, src, door)
}

fn set_leaf(sim: &mut Simulation, door: omg_audio::ObjectId, open: f32) {
    sim.set_object(door, [6.08, 2.0, 0.0], [6.12, 2.0 + (1.0 - open).max(1e-3), 2.1]);
    sim.set_object_enabled(door, open < 0.999);
}

/// The door swinging shut (openness e^(−7 t)) on the live GPU walk, paced
/// as a host's 20 Hz ticks with exact frames at 120 Hz between, against
/// the door standing at each openness (the CPU's v6.2, averaged: the
/// truth the walks estimate). rms difference, dB.
fn gpu_swing(mode: &str, ears: [f32; 3]) -> Option<f32> {
    let db = |e: [f32; 3]| 10.0 * e[1].max(1e-20).log10();
    let opens = [1.0f32, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0];
    let truth: Vec<f32> = opens
        .iter()
        .map(|&o| {
            let (mut sim, src, _) = doorway_scene("v6.2", Compute::Cpu, ears, o);
            for _ in 0..30 {
                sim.tick();
            }
            let mut acc = 0.0f32;
            for _ in 0..80 {
                acc += energy(sim.tick().block(src).unwrap())[1] / 80.0;
            }
            10.0 * acc.max(1e-20).log10()
        })
        .collect();
    let standing = |o: f32| -> f32 {
        let i = opens.iter().position(|&x| x <= o).unwrap_or(opens.len() - 1).max(1);
        let (a, b) = (opens[i - 1], opens[i]);
        let f = if b > 0.0 { (o.ln() - a.ln()) / (b.ln() - a.ln()) } else { (a - o) / a };
        truth[i - 1] + f.clamp(0.0, 1.0) * (truth[i] - truth[i - 1])
    };
    let (mut sim, src, door) = doorway_scene(mode, Compute::Gpu, ears, 1.0);
    if !sim.gpu_active() {
        return None;
    }
    let frame = 1.0 / 120.0;
    let mut t = 0.0f64;
    let step = |sim: &mut Simulation, k: usize, t: f64| {
        sim.set_time(t);
        if k % 6 == 0 {
            // a host's tick: the walk it queued last time has landed
            std::thread::sleep(std::time::Duration::from_millis(12));
            Some(sim.tick())
        } else {
            sim.tick_exact()
        }
    };
    for k in 0..240 {
        step(&mut sim, k, t);
        t += frame;
    }
    let (mut sq, mut n, mut level) = (0.0f32, 0, 0.0f32);
    for k in 0..150 {
        let mut open = (-7.0 * k as f32 * frame as f32).exp();
        if open < 0.001 {
            open = 0.0;
        }
        set_leaf(&mut sim, door, open);
        if let Some(f) = step(&mut sim, k, t) {
            level = db(energy(f.block(src).unwrap()));
        }
        let d = level - standing(open);
        sq += d * d;
        n += 1;
        t += frame;
    }
    Some((sq / n as f32).sqrt())
}

#[test]
fn a_door_swinging_shut_is_heard_where_it_is_on_the_gpu() {
    // v6.2 on the GPU walk: the kernel records no door paths, so while a
    // door moves the tick walks on the CPU (`Tracer::doors_moving`)
    let ears = [[7.5, 2.8, 1.6], [8.0, 6.0, 1.6]];
    let (mut e61, mut e62) = (0.0f32, 0.0f32);
    for e in ears {
        let (Some(r61), Some(r62)) = (gpu_swing("v6.1", e), gpu_swing("v6.2", e)) else {
            eprintln!("SKIP: no wgpu adapter");
            return;
        };
        eprintln!("ears {e:?}: on the GPU walk, the swing against the door standing: v6.1 rms {r61:.1} dB, v6.2 {r62:.1} dB");
        e61 += r61 / ears.len() as f32;
        e62 += r62 / ears.len() as f32;
    }
    eprintln!("mean rms: v6.1 {e61:.1} dB, v6.2 {e62:.1} dB");
    assert!(e62 < e61 - 1.0, "v6.2 follows the door closer than v6.1 on the GPU: {e62:.1} vs {e61:.1} dB");
    assert!(e62 < 4.0, "v6.2 within 4 dB rms of the door standing: {e62:.1} dB");
}
