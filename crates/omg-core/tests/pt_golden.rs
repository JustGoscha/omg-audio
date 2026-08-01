//! PT-early C1 gate (GPU_PLAN.md Track C): in an empty shoebox the
//! discovered-and-solved path set must equal the image-source method's
//! — ISM is the oracle. Both directions are asserted:
//!  · completeness: every ISM path of order ≤ 2 is discovered, with
//!    delay ≤ 0.5 ms, direction and level ≤ 1 dB agreement (order-3
//!    coverage is allowed to converge over more rotations);
//!  · soundness: every PT record matches SOME ISM ≤3 path — discovery
//!    plus the exact mirror solve can never invent an impossible path.

use omg_core::ism::image_source_taps;
use omg_core::pt::{pt_discover, PathRecord};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};

#[path = "trace_golden.rs"]
mod golden;

fn db(a: f32) -> f32 {
    20.0 * a.max(1e-9).log10()
}

fn matches(r: &PathRecord, tap: &omg_core::params::Tap) -> bool {
    (r.delay_s - tap.delay_s).abs() * 1000.0 <= 0.5
        && (r.dir[0] * tap.dir[0] + r.dir[1] * tap.dir[1] + r.dir[2] * tap.dir[2]) > 0.995
        && (0..NBANDS).all(|b| (db(r.gains[b]) - db(tap.gains[b])).abs() <= 1.0)
}

#[test]
fn pt_matches_ism_in_empty_shoeboxes() {
    for cfg in golden::golden_configs() {
        let mut ism = Vec::new();
        image_source_taps(&cfg.room, cfg.src, cfg.lis, 3, &mut ism);
        let mut ism2 = Vec::new();
        image_source_taps(&cfg.room, cfg.src, cfg.lis, 2, &mut ism2);

        // accumulate discovery over rotations, like the cache does
        let mut records: Vec<PathRecord> = Vec::new();
        let mut tick = Vec::new();
        for rot in 0..12u32 {
            pt_discover(&cfg.room, &[cfg.src], cfg.lis, 4096, rot, &mut tick);
            for r in &tick {
                if !records.iter().any(|q| q.key() == r.key()) {
                    records.push(*r);
                }
            }
        }

        // completeness at order ≤ 2
        let mut missed = 0;
        for tap in &ism2 {
            if !records.iter().any(|r| matches(r, tap)) {
                missed += 1;
                eprintln!(
                    "{}: ISM path delay {:.1} ms dir ({:.2},{:.2},{:.2}) unmatched",
                    cfg.name,
                    tap.delay_s * 1000.0,
                    tap.dir[0],
                    tap.dir[1],
                    tap.dir[2]
                );
            }
        }
        assert_eq!(
            missed, 0,
            "{}: {missed}/{} ISM ≤2-order paths undiscovered",
            cfg.name,
            ism2.len()
        );

        // soundness at ≤ 3
        for r in &records {
            assert!(
                ism.iter().any(|tap| matches(r, tap)),
                "{}: PT invented a path — chain {:?} delay {:.2} ms ({}m)",
                cfg.name,
                &r.chain[..r.order as usize],
                r.delay_s * 1000.0,
                r.delay_s * SPEED_OF_SOUND
            );
        }

        // sanity: the direct path is there and exact
        let direct = records.iter().find(|r| r.order == 0).expect("direct path");
        let true_delay = (cfg.src - cfg.lis).length() / SPEED_OF_SOUND;
        assert!((direct.delay_s - true_delay).abs() < 1e-5);

        println!(
            "{}: {} PT records vs {} ISM ≤3 taps ({} at ≤2) — gate green",
            cfg.name,
            records.len(),
            ism.len(),
            ism2.len()
        );
    }
}

#[test]
fn pt_solver_rejects_impossible_chains() {
    let cfg = &golden::golden_configs()[0];
    // a chain that repeats the same wall twice in a row is geometrically
    // impossible off a plane — the validator must refuse it
    assert!(omg_core::pt::solve_chain(&cfg.room, &[0, 0], cfg.src, cfg.lis).is_none());
}

// silence dead-code warnings from the shared golden module
#[allow(dead_code)]
fn _use_golden() {
    let _ = golden::golden_configs();
    let _ = Shoebox::new(Vec3::new(1.0, 1.0, 1.0), [omg_core::material::Material::CONCRETE; 6]);
}
