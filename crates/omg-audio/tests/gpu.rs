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
    eprintln!("async gpu: first tick {direct_only} taps, later {} taps, rt60 {:?}", pb.taps.len(), pb.reverb.rt60);
    assert!(pb.taps.len() > direct_only, "reflections arrive");
    assert!(pb.reverb.rt60[1] > 0.7, "a traced tail, not the fallback");
}
