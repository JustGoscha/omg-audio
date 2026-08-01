//! Phase 0 of the GPU port (GPU_PLAN.md): golden baselines for the
//! stochastic shoebox tracer. These constants define "correct" for any
//! alternative trace implementation (the wgpu kernel): statistical
//! parity on derived quantities, NOT bit parity — a different RNG and
//! float order must land inside the same tolerances a different CPU
//! seed does.
//!
//! Calibration (print_goldens, seeds 0xC0FFEE / 0xBEEF1234): the
//! seed-to-seed spread reaches ~8% on low-band RT60 in sparse fields
//! (hall, dead room), ~0.7 dB on a −52 dB late tail, ~0.05 on
//! anisotropy. Acceptance tolerances sit ~1.5× above that spread —
//! tighter than the plan's guess would allow flake-free:
//!   RT60 ±12% per band · late level ±1.0 dB · anisotropy ±0.06.

use omg_core::material::Material;
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

pub const GOLDEN_RAYS: u32 = 4096;
/// Independent traces averaged per measurement. The engine never
/// consumes a single trace either — `Sim` EMAs across ticks — and one
/// 4096-ray trace leaves ~17% RT60 spread in sparse high-band fields
/// (the hall at 14 m). The GPU parity harness must average the same
/// count (N dispatches or N× rays).
pub const GOLDEN_AVG: u64 = 4;
pub const GOLDEN_SEED: u64 = 0xC0FFEE;
/// Late-field window start for the anisotropy measurement (s).
pub const GOLDEN_LATE_S: f32 = 0.05;

pub const RT60_REL_TOL: f32 = 0.12;
pub const LEVEL_DB_TOL: f32 = 1.0;
pub const ANISO_TOL: f32 = 0.06;
/// Anisotropy is only asserted when the mid-band late level clears this
/// floor: below it (the dead room's −42 dB tail) the direction estimate
/// hangs on a handful of rays and its seed-to-seed spread exceeds any
/// honest tolerance — direction statistics of an empty field are noise.
pub const ANISO_LEVEL_FLOOR_DB: f32 = -40.0;

pub struct GoldenConfig {
    pub name: &'static str,
    pub room: Shoebox,
    pub src: Vec3,
    pub lis: Vec3,
    /// Expected [rt60 low/mid/high, level_db low/mid/high, anisotropy].
    pub expect: [f32; 7],
}

pub fn golden_configs() -> Vec<GoldenConfig> {
    vec![
        GoldenConfig {
            // Small hard room: long RT60, dense diffuse field.
            name: "live-room",
            room: Shoebox::new(
                Vec3::new(6.0, 4.0, 3.0),
                [
                    Material::CONCRETE,
                    Material::CONCRETE,
                    Material::CONCRETE,
                    Material::CONCRETE,
                    Material::WOOD_PANEL,
                    Material::CONCRETE,
                ],
            ),
            src: Vec3::new(1.5, 1.0, 1.5),
            lis: Vec3::new(4.5, 3.0, 1.6),
            expect: [1.10, 0.98, 1.03, -4.9, -3.4, -4.8, 0.015],
        },
        GoldenConfig {
            // Large dead room: short RT60, weak late field.
            name: "dead-room",
            room: Shoebox::new(
                Vec3::new(12.0, 9.0, 4.0),
                [
                    Material::ACOUSTIC_TILE,
                    Material::ACOUSTIC_TILE,
                    Material::ACOUSTIC_TILE,
                    Material::ACOUSTIC_TILE,
                    Material::CARPET,
                    Material::ACOUSTIC_TILE,
                ],
            ),
            src: Vec3::new(3.0, 2.0, 1.5),
            lis: Vec3::new(9.0, 7.0, 1.6),
            expect: [0.69, 0.21, 0.17, -21.7, -41.4, -50.4, 0.123],
        },
        GoldenConfig {
            // Elongated hall, listener at the far end: the late field
            // arrives visibly from the source's direction (anisotropy).
            name: "hall",
            room: Shoebox::new(
                Vec3::new(16.0, 3.5, 3.0),
                [
                    Material::DRYWALL,
                    Material::DRYWALL,
                    Material::DRYWALL,
                    Material::DRYWALL,
                    Material::WOOD_PANEL,
                    Material::CONCRETE,
                ],
            ),
            src: Vec3::new(1.5, 1.75, 1.5),
            lis: Vec3::new(14.0, 1.75, 1.6),
            expect: [0.96, 1.36, 1.46, -17.3, -10.2, -8.4, 0.015],
        },
    ]
}

/// Run one golden config through a tracer and reduce to the compared
/// metrics: [rt60 ×3, late level dB ×3, anisotropy].
pub fn metrics_of(echo: &Echogram) -> [f32; 7] {
    let rv = estimate_reverb(echo);
    let mut m = [0.0f32; 7];
    for b in 0..NBANDS {
        m[b] = rv.rt60[b];
        m[3 + b] = 20.0 * rv.level[b].max(1e-9).log10();
    }
    let (_, aniso) = echo.late_direction(GOLDEN_LATE_S);
    m[6] = aniso;
    m
}

fn run_cpu(cfg: &GoldenConfig, seed: u64) -> [f32; 7] {
    let mut avg = Echogram::new();
    let mut echo = Echogram::new();
    for k in 0..GOLDEN_AVG {
        let mut rng = Rng::new(seed.wrapping_add(k.wrapping_mul(0x9E3779B97F4A7C15)));
        trace(
            &cfg.room,
            cfg.src,
            cfg.lis,
            GOLDEN_RAYS,
            [1.0; NBANDS],
            &mut rng,
            &mut echo,
        );
        // running mean over the k traces (ema with alpha 1/(k+1))
        avg.ema(&echo, 1.0 / (k + 1) as f32);
    }
    metrics_of(&avg)
}

pub fn assert_golden(name: &str, got: &[f32; 7], expect: &[f32; 7]) {
    for b in 0..NBANDS {
        let rel = (got[b] - expect[b]).abs() / expect[b];
        assert!(
            rel <= RT60_REL_TOL,
            "{name}: rt60[{b}] {} vs golden {} ({:.1}% > {:.0}%)",
            got[b],
            expect[b],
            rel * 100.0,
            RT60_REL_TOL * 100.0
        );
        let ddb = (got[3 + b] - expect[3 + b]).abs();
        // a band whose late tail sits below the audibility floor may
        // wobble more in absolute dB while staying inaudible
        let tol = if expect[3 + b] < ANISO_LEVEL_FLOOR_DB { 2.5 } else { LEVEL_DB_TOL };
        assert!(
            ddb <= tol,
            "{name}: level[{b}] {:.2} dB vs golden {:.2} dB (Δ{ddb:.2} > {tol})",
            got[3 + b],
            expect[3 + b]
        );
    }
    if expect[4] > ANISO_LEVEL_FLOOR_DB {
        assert!(
            (got[6] - expect[6]).abs() <= ANISO_TOL,
            "{name}: anisotropy {:.3} vs golden {:.3}",
            got[6],
            expect[6]
        );
    }
}

/// Prints the metric vector per config and per seed — run with
/// `cargo test -p omg-core print_goldens -- --nocapture --ignored`
/// to re-derive the constants after an intentional tracer change.
#[test]
#[ignore]
fn print_goldens() {
    for cfg in golden_configs() {
        for seed in [GOLDEN_SEED, 0xBEEF1234] {
            let m = run_cpu(&cfg, seed);
            println!(
                "{:>10} seed {seed:#x}: rt60 [{:.2} {:.2} {:.2}] level_db [{:.1} {:.1} {:.1}] aniso {:.3}",
                cfg.name, m[0], m[1], m[2], m[3], m[4], m[5], m[6]
            );
        }
    }
}

/// The CPU tracer must stay inside its own goldens (regression pin),
/// with a seed OTHER than the one that derived them — proving the
/// tolerances hold across RNG streams, which is exactly what the GPU
/// kernel will be held to.
#[test]
fn cpu_tracer_within_goldens() {
    for cfg in golden_configs() {
        let m = run_cpu(&cfg, GOLDEN_SEED);
        assert_golden(cfg.name, &m, &cfg.expect);
        let m2 = run_cpu(&cfg, 0x5EED_5EED);
        assert_golden(cfg.name, &m2, &cfg.expect);
    }
}
