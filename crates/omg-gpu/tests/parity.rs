//! Phase 1 acceptance: the GPU kernel must land inside the SAME golden
//! tolerances as a fresh CPU seed does (statistical parity, Phase 0).
//! Self-skips with a printed notice when no adapter exists — CI may be
//! headless; a machine with a GPU is the real gate.
//!
//! The golden configs/constants live in omg-core's test tree; they are
//! included here as a module so there is exactly one source of truth.

use omg_core::tracer::Echogram;

#[path = "../../omg-core/tests/trace_golden.rs"]
mod golden;

#[test]
fn gpu_tracer_within_goldens() {
    let Some(gpu) = omg_gpu::GpuTracer::new() else {
        eprintln!("SKIP gpu_tracer_within_goldens: no wgpu adapter available");
        return;
    };
    for cfg in golden::golden_configs() {
        let mut avg = Echogram::new();
        let mut echo = Echogram::new();
        for k in 0..golden::GOLDEN_AVG {
            gpu.trace(
                &cfg.room,
                cfg.src,
                cfg.lis,
                golden::GOLDEN_RAYS,
                [1.0; omg_core::NBANDS],
                0xD15EA5E ^ (k as u32).wrapping_mul(0x85EB_CA6B),
                &mut echo,
            );
            avg.ema(&echo, 1.0 / (k + 1) as f32);
        }
        let m = golden::metrics_of(&avg);
        golden::assert_golden(cfg.name, &m, &cfg.expect);
        println!(
            "{:>10} GPU: rt60 [{:.2} {:.2} {:.2}] level_db [{:.1} {:.1} {:.1}] aniso {:.3}",
            cfg.name, m[0], m[1], m[2], m[3], m[4], m[5], m[6]
        );
    }
}
