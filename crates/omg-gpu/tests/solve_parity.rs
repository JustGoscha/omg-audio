//! C7a kernel K4 acceptance. (1) Parity: the batched GPU solve must
//! return the SAME records as the CPU `mesh_record` for every
//! (source × chain) pair over the real walkthrough world — same
//! construction, so agreement is checked per slot to float tolerance,
//! with a tiny mismatch budget for traversal-order ties on coincident
//! planes. (2) Speed: 100 sources against the full chain list must
//! solve in one dispatch far under what the CPU pays for today's
//! audible set — that is the 30-source wall falling. Self-skips
//! without an adapter.

use omg_core::material::Material;
use omg_core::pt::Aabb;
use omg_core::pt_mesh::{mesh_chains, mesh_record, MChain, SurfaceTable, NO_SURF};
use omg_core::vec3::Vec3;

struct World {
    mesh: omg_core::mesh::Mesh,
    table: SurfaceTable,
    furn: Vec<(Vec3, Vec3, Material)>,
    extras: Vec<Aabb>,
}

/// The walkthrough world exactly as the app wires it: mesh, the
/// SurfaceTable with significant furniture appended, and the extras
/// set (all furniture boxes + a couple of shut door leaves).
fn world() -> World {
    let rooms = omg_scene::walkthrough::rooms();
    let doors = omg_scene::walkthrough::doors();
    let (mesh, _) = omg_scene::dome::build_world_mesh(&rooms, &doors);
    let mut furn: Vec<(Vec3, Vec3, Material)> = Vec::new();
    let mut extras: Vec<Aabb> = Vec::new();
    for (ri, r) in rooms.iter().enumerate() {
        let o = Vec3::new(r.min.0, r.min.1, r.floor_z);
        for (a, m) in omg_scene::walkthrough::furniture(ri)
            .iter()
            .zip(omg_scene::walkthrough::furniture_mats(ri))
        {
            extras.push(Aabb { min: a.min + o, max: a.max + o, transmission: a.transmission });
            let d = a.max - a.min;
            if d.x * d.y * d.z > omg_scene::walkthrough::FURN_REFLECTOR_MIN_VOL {
                furn.push((a.min + o, a.max + o, *m));
            }
        }
    }
    // two shut leaves as transient blockers, like the sim's extras_buf
    let leaf = Material::WOOD_PANEL.transmission;
    extras.push(Aabb {
        min: Vec3::new(6.9, 24.9, 0.0),
        max: Vec3::new(7.5, 25.1, 2.1),
        transmission: leaf,
    });
    extras.push(Aabb {
        min: Vec3::new(3.9, 7.9, 0.0),
        max: Vec3::new(4.9, 8.1, 2.1),
        transmission: leaf,
    });
    let mut table = SurfaceTable::build(&mesh);
    for (mn, mx, m) in &furn {
        table.append_box(*mn, *mx, m);
    }
    World { mesh, table, furn, extras }
}

/// Direct-first chain list from CPU discovery at `lis` — the same
/// shape WorldEarly::chain_list uploads.
fn chain_list(w: &World, lis: Vec3, cap: usize) -> Vec<MChain> {
    let boxes: Vec<(Vec3, Vec3)> = w.furn.iter().map(|(mn, mx, _)| (*mn, *mx)).collect();
    let mut cs: Vec<MChain> = Vec::new();
    for rot in 0..6 {
        mesh_chains(&w.mesh, &boxes, w.table.base_overlay, lis, 768, rot, &mut cs);
    }
    cs.sort();
    cs.dedup();
    cs.truncate(cap - 1);
    let mut list = vec![([NO_SURF; 3], 0u8)];
    list.extend(cs);
    list
}

fn db(v: f32) -> f32 {
    20.0 * v.max(1e-9).log10()
}

#[test]
fn batched_solve_matches_cpu_records() {
    let w = world();
    let Some(gpu) = omg_gpu::GpuMeshTracer::with_solve(
        &w.mesh,
        &w.furn.iter().map(|(mn, mx, _)| (*mn, *mx)).collect::<Vec<_>>(),
        &w.table,
    ) else {
        eprintln!("SKIP batched_solve_matches_cpu_records: no wgpu adapter");
        return;
    };
    // three listener contexts: hall interior, corridor near a door,
    // outside on the square — different chain populations
    let spots = [
        Vec3::new(10.5, 20.5, 1.6),
        Vec3::new(4.4, 7.5, 1.6),
        Vec3::new(7.0, 27.0, 1.6),
    ];
    // a spread of sources: real emitter spots + heights, some through
    // walls, some co-located with the listener's room
    let srcs: Vec<(u16, Vec3)> = vec![
        (8, Vec3::new(10.5, 20.5, 1.2)),
        (16, Vec3::new(2.0, 2.0, 1.0)),
        (24, Vec3::new(31.5, 18.0, 1.4)),
        (32, Vec3::new(7.0, 30.0, 1.7)),
        (40, Vec3::new(18.0, 9.0, 2.2)),
        (48, Vec3::new(4.0, 25.0, 0.8)),
    ];
    let mut seg_buf = Vec::new();
    let mut total = 0usize;
    let mut cpu_some = 0usize;
    let mut mismatch = 0usize;
    for lis in spots {
        let chains = chain_list(&w, lis, 400);
        let mut out: Vec<Option<omg_core::pt_mesh::MeshRecord>> = Vec::new();
        assert!(
            gpu.solve_batch(&srcs, &chains, lis, &w.extras, &mut out),
            "solve dispatch refused"
        );
        assert_eq!(out.len(), srcs.len() * chains.len());
        for (si, &(id, sp)) in srcs.iter().enumerate() {
            for (ci, &(chain, order)) in chains.iter().enumerate() {
                let cpu = mesh_record(
                    &w.mesh,
                    &w.table,
                    &chain[..order as usize],
                    id,
                    sp,
                    lis,
                    &w.extras,
                    &mut seg_buf,
                );
                let g = &out[si * chains.len() + ci];
                total += 1;
                cpu_some += cpu.is_some() as usize;
                match (&cpu, g) {
                    (None, None) => {}
                    (Some(c), Some(gr)) => {
                        let ddir = c.dir[0] * gr.dir[0]
                            + c.dir[1] * gr.dir[1]
                            + c.dir[2] * gr.dir[2];
                        let mut ok = (c.delay_s - gr.delay_s).abs() < 1e-5 && ddir > 0.99999;
                        for b in 0..3 {
                            let (a, bb) = (c.gains[b], gr.gains[b]);
                            ok &= (a - bb).abs() < 2e-6 || (a - bb).abs() < 0.02 * a.max(bb);
                        }
                        if !ok {
                            mismatch += 1;
                            if mismatch < 8 {
                                eprintln!(
                                    "value drift chain {:?}×{order} src {id} lis {:?}: cpu {:?} gpu {:?} (delay c {:.6} g {:.6})",
                                    &chain[..order as usize],
                                    (lis.x, lis.y),
                                    c.gains,
                                    gr.gains,
                                    c.delay_s,
                                    gr.delay_s
                                );
                            }
                        }
                    }
                    (a, b) => {
                        mismatch += 1;
                        if mismatch < 8 {
                            eprintln!(
                                "presence mismatch chain {:?}×{order} src {id} lis {:?}: cpu {} gpu {} (cpu gains {:?})",
                                &chain[..order as usize],
                                (lis.x, lis.y),
                                a.is_some(),
                                b.is_some(),
                                a.map(|r| r.gains.map(db)),
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "batched solve parity: {total} pairs, {cpu_some} solved on CPU, {mismatch} mismatched"
    );
    assert!(cpu_some > 200, "test degenerate: CPU solved only {cpu_some} pairs");
    // machine-exact construction on both sides; the budget covers
    // traversal-order ties on coincident planes and f32 re-association
    assert!(
        (mismatch as f64) < (total as f64) * 0.005,
        "{mismatch}/{total} pair mismatches — the batch is not a replayable solve"
    );
}

#[test]
fn hundred_source_batch_beats_the_wall() {
    let w = world();
    let Some(gpu) = omg_gpu::GpuMeshTracer::with_solve(
        &w.mesh,
        &w.furn.iter().map(|(mn, mx, _)| (*mn, *mx)).collect::<Vec<_>>(),
        &w.table,
    ) else {
        eprintln!("SKIP hundred_source_batch_beats_the_wall: no wgpu adapter");
        return;
    };
    let lis = Vec3::new(7.0, 27.0, 1.6); // the square — worst chain mix
    let chains = chain_list(&w, lis, 400);
    // 100 sources scattered over the block, varied heights
    let srcs: Vec<(u16, Vec3)> = (0..100u16)
        .map(|i| {
            let f = i as f32;
            (
                i * 8,
                Vec3::new(
                    2.0 + (f * 0.83) % 38.0,
                    2.0 + (f * 1.37) % 30.0,
                    0.8 + (f * 0.11) % 2.0,
                ),
            )
        })
        .collect();
    let mut out = Vec::new();
    assert!(gpu.solve_batch(&srcs, &chains, lis, &w.extras, &mut out), "dispatch refused");
    let solved = out.iter().flatten().count();
    // warm; now measure
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        gpu.solve_batch(&srcs, &chains, lis, &w.extras, &mut out);
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 100.0;
    // CPU reference: ONE source over the same chain list (what the sim
    // pays per audible source today)
    let mut seg_buf = Vec::new();
    let t1 = std::time::Instant::now();
    let mut cpu_n = 0usize;
    for _ in 0..10 {
        for &(chain, order) in &chains {
            cpu_n += mesh_record(
                &w.mesh,
                &w.table,
                &chain[..order as usize],
                8,
                srcs[1].1,
                lis,
                &w.extras,
                &mut seg_buf,
            )
            .is_some() as usize;
        }
    }
    let cpu_one_ms = t1.elapsed().as_secs_f64() * 100.0;
    eprintln!(
        "batched solve: 100 src × {} chains = {} records in {gpu_ms:.2} ms/dispatch · CPU pays {cpu_one_ms:.2} ms for ONE source ({cpu_n} recs)",
        chains.len(),
        solved,
    );
    assert!(solved > 100, "degenerate scene: only {solved} records");
    // the gate: the WHOLE 100-source batch must cost less than a
    // handful of CPU source-solves — per-source sim cost ~zero
    assert!(
        gpu_ms < (cpu_one_ms * 8.0).max(10.0),
        "batch too slow: {gpu_ms:.2} ms vs CPU {cpu_one_ms:.2} ms/source"
    );
}
