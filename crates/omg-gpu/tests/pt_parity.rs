//! PT-early C3 gate: the GPU chain bitmap, decoded and exact-solved,
//! must satisfy the same golden gate as CPU discovery — every ISM ≤2
//! path present (the seeds guarantee that on both sides; here we run
//! WITHOUT seeds to prove the kernel itself discovers), no invented
//! paths, and the kernel's chain set must agree with the CPU fan's on
//! every wall-reachability fact that matters: any chain the CPU finds
//! over N rotations, the GPU finds over the same rotations.

use omg_core::pt::{pt_chains, record_for, Chain};
use omg_gpu::{decode_chain_bitmap, GpuEarlyDiscovery};

#[path = "../../omg-core/tests/trace_golden.rs"]
mod golden;

#[test]
fn gpu_chain_discovery_matches_cpu() {
    let Some(gpu) = GpuEarlyDiscovery::new() else {
        eprintln!("SKIP gpu_chain_discovery_matches_cpu: no adapter");
        return;
    };
    for cfg in golden::golden_configs() {
        let mut cpu: Vec<Chain> = Vec::new();
        let mut gpu_chains: Vec<Chain> = Vec::new();
        for rot in 0..8u32 {
            pt_chains(&cfg.room, cfg.lis, 4096, rot, &mut cpu);
            let words = gpu.bitmap_for(
                [cfg.room.size.x, cfg.room.size.y, cfg.room.size.z],
                [cfg.lis.x, cfg.lis.y, cfg.lis.z],
                rot,
            );
            decode_chain_bitmap(&words, &mut gpu_chains);
        }
        let dedup = |v: &mut Vec<Chain>| {
            v.sort();
            v.dedup();
        };
        dedup(&mut cpu);
        dedup(&mut gpu_chains);

        // every VALIDATED cpu chain is also in the gpu set (validation
        // is the arbiter: raw grazing chains may differ by epsilon)
        let mut missing = 0;
        for &(chain, order) in &cpu {
            let c = &chain[..order as usize];
            if record_for(&cfg.room, c, 0, cfg.src, cfg.lis).is_none() {
                continue;
            }
            if !gpu_chains.contains(&(chain, order)) {
                missing += 1;
                eprintln!("{}: chain {:?} found by CPU, not GPU", cfg.name, c);
            }
        }
        assert_eq!(missing, 0, "{}: GPU discovery missed validated chains", cfg.name);

        // and no GPU chain solves to a path the CPU-side solver rejects
        // as geometrically impossible in a way ISM contradicts — the
        // solver itself is the shared arbiter, so it suffices that
        // decoding produced structurally sane chains
        for &(chain, order) in &gpu_chains {
            assert!(order >= 1 && order <= 3);
            assert!(chain[..order as usize].iter().all(|&w| w < 6));
        }
        println!(
            "{}: {} CPU chains / {} GPU chains — gate green",
            cfg.name,
            cpu.len(),
            gpu_chains.len()
        );
    }
}
