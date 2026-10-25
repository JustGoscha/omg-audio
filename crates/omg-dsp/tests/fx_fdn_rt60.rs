//! Effect: own-room late reverb, 8-line Jot FDN (fdn.rs).
//!
//! Claim: `set_rt60(mid, high)` sets the decay of the late field: the
//! measured T60 (Schroeder backward integration of the impulse
//! response, T20 fit −5…−25 dB) in the MID band tracks `rt60_mid`, and
//! the in-loop one-pole damping makes the HIGH band decay at
//! `rt60_high` — highs die faster when the traced room says so. Also:
//! the network is stable (no growth) at the longest supported RT60.
//!
//! And at the renderer: the own-room late field is spread over the
//! sphere (lines at cube corners) — DIFFUSE, with no preferred side.
//!
//! Isolation: `Fdn` driven directly with a unit impulse; each of the 8
//! line outputs band-filtered (RBJ bandpass, 1 kHz / 6 kHz) and their
//! energies summed. The diffuseness check drives a bus-only
//! `Renderer::new` with no taps (own late field only). No scene.
//! Combined in the demo: omg-scene tests/late_world_gate.rs
//! `mesh_late_field_matches_shoebox` pins the traced RT60 this consumes.

mod common;
use common::*;
use omg_core::params::{ParamBlock, ReverbParams};
use omg_dsp::ambi::{StereoDecoder, NCH};
use omg_dsp::fdn::{Fdn, NLINES};
use omg_dsp::Renderer;

/// Impulse response energy per sample in a band, after the parameter
/// smoothers (100–150 ms) have settled.
fn band_ir_energy(rt60_mid: f32, rt60_high: f32, band_hz: f32, secs: f32) -> Vec<f64> {
    band_ir_energy_q(rt60_mid, rt60_high, band_hz, 1.4, secs)
}

fn band_ir_energy_q(rt60_mid: f32, rt60_high: f32, band_hz: f32, q: f32, secs: f32) -> Vec<f64> {
    let mut fdn = Fdn::new(SR);
    fdn.set_rt60(rt60_mid, rt60_high);
    let mut outs = [0.0f32; NLINES];
    for _ in 0..(SR as usize) {
        fdn.process(0.0, &mut outs); // settle gains; state is all zero
    }
    let mut bp = [Biquad::bandpass(band_hz, q); NLINES];
    let n = (secs * SR) as usize;
    let mut e = Vec::with_capacity(n);
    for k in 0..n {
        fdn.process(if k == 0 { 1.0 } else { 0.0 }, &mut outs);
        let mut s = 0.0f64;
        for i in 0..NLINES {
            let v = bp[i].tick(outs[i]) as f64;
            s += v * v;
        }
        e.push(s);
    }
    e
}

/// Mid band follows rt60_mid within 10 % over a short room, a living
/// room and a hall (high = mid: no extra damping).
#[test]
fn mid_band_t60_tracks_set_rt60() {
    for rt in [0.4f32, 1.0, 2.5] {
        let e = band_ir_energy(rt, rt, 1000.0, (1.2 * rt).max(0.8));
        let t = schroeder_t60(&e, -5.0, -25.0);
        assert!((t / rt - 1.0).abs() < 0.10, "set {rt} s, measured {t:.3} s");
    }
}

/// rt60_high < rt60_mid: at the damping design frequency (6 kHz, where
/// the in-loop one-pole is fitted to |H| = g_high/g_mid) the decay
/// follows rt60_high within 20 %, the mid band stays within 10 % of
/// rt60_mid, and highs clearly die first. Measured in a narrow (Q 5)
/// band at 6 kHz: a wide band smears in the slower-decaying lower
/// frequencies.
#[test]
fn high_band_decays_at_rt60_high() {
    for (mid, high) in [(2.0f32, 1.0f32), (1.2, 0.5), (0.6, 0.3)] {
        let tm = schroeder_t60(&band_ir_energy(mid, high, 1000.0, 1.2 * mid), -5.0, -25.0);
        let th = schroeder_t60(&band_ir_energy_q(mid, high, 6000.0, 5.0, 1.2 * mid), -5.0, -25.0);
        assert!((tm / mid - 1.0).abs() < 0.10, "mid: set {mid}, measured {tm:.3}");
        assert!((th / high - 1.0).abs() < 0.20, "high: set {high}, measured {th:.3}");
        assert!(th < 0.75 * tm, "highs must die faster: {th:.3} vs {tm:.3}");
    }
}

/// Strong HF damping must not steal MID decay: with rt60_high = 0.3 ×
/// rt60_mid the loop one-pole (fc ≈ 1.7 kHz) also attenuates 1 kHz by
/// ≈ 0.86 per pass, so the mid T60 comes out ≈ 0.84 s for a set 1.0 s
/// (≈ −16 %). Jot-style designs normalize the damping filter to unity at
/// the mid reference (or use a shelf) so the bands are independent.
#[test]
#[ignore = "BUG: FDN in-loop lowpass is not unity at mid — rt60_high=0.3·mid shortens measured mid T60 ~16% (1.0 s → 0.84 s)"]
fn mid_t60_independent_of_hf_damping() {
    for (mid, high) in [(1.0f32, 0.3f32), (2.0, 0.5)] {
        let tm = schroeder_t60(&band_ir_energy(mid, high, 1000.0, 1.2 * mid), -5.0, -25.0);
        assert!((tm / mid - 1.0).abs() < 0.10, "mid: set {mid} (high {high}), measured {tm:.3}");
    }
}

/// Longest tail the scene can ask for (gain clamps at 0.9999): energy
/// keeps falling, never grows.
#[test]
fn long_tail_is_stable() {
    let mut fdn = Fdn::new(SR);
    fdn.set_rt60(8.0, 8.0);
    let mut outs = [0.0f32; NLINES];
    for _ in 0..(SR as usize) {
        fdn.process(0.0, &mut outs);
    }
    let mut blocks = Vec::new();
    for b in 0..20 {
        let mut s = 0.0f64;
        for k in 0..9600 {
            fdn.process(if b == 0 && k == 0 { 1.0 } else { 0.0 }, &mut outs);
            s += outs.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
        }
        assert!(s.is_finite());
        blocks.push(s);
    }
    // 0.2 s blocks of an 8 s tail: −1.5 dB per block, monotone after the
    // first fill (~80 ms)
    for w in blocks[1..].windows(2) {
        assert!(w[1] < w[0], "tail must decay: {blocks:?}");
    }
}

/// Energy-weighted direction E[W·(Y, Z)]/E[W²] of the renderer's own
/// late field, and its cardioid-decoded L/R balance in dB.
fn own_field_lean(rt60: f32) -> (f64, f64, f64) {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock {
        taps: vec![],
        reverb: ReverbParams { rt60: [rt60; 3], level: [0.3; 3] },
        remote: None,
        version: 1,
    });
    let mut src = Noise::new(42);
    let dec = StereoDecoder::new();
    let (mut ew, mut ey, mut ez, mut el, mut er) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 0..4 * SR as usize {
        let mut b = [0.0f32; NCH];
        r.process(0.5 * src.next(), &mut b);
        if k >= SR as usize {
            let w = b[0] as f64;
            ew += w * w;
            ey += w * b[1] as f64;
            ez += w * b[2] as f64;
            let (l, rr) = dec.decode(&b);
            el += (l as f64).powi(2);
            er += (rr as f64).powi(2);
        }
    }
    (ey / ew, ez / ew, db_pow(el / er))
}

/// Long rooms are near-diffuse already (lean ≲ 0.12, L/R ≈ 1 dB).
#[test]
fn own_late_field_is_near_diffuse_for_long_rooms() {
    let (y, z, lr) = own_field_lean(2.0);
    assert!(y.abs() < 0.15 && z.abs() < 0.15 && lr.abs() < 1.5, "rt 2.0: y {y:.3} z {z:.3} L/R {lr:.2} dB");
}

/// The diffuse claim must hold for SHORT rooms too. It doesn't: line
/// gains fall with line length (g = 10^(−3·len/RT60)), and the two
/// shortest — hence loudest — lines (23.4 ms, 32.5 ms) both sit at
/// +y,+z cube corners. At RT60 0.3 s (a furnished living room) the late
/// field leans left-and-up: E[W·Y]/E[W²] ≈ 0.28, E[W·Z]/E[W²] ≈ 0.21,
/// cardioid L/R ≈ +3.1 dB (seed-independent). Fix: assign corners so
/// line energy balances per axis (e.g. consecutive lines antipodal).
#[test]
#[ignore = "BUG: own late field leans left/up in short rooms — RT60 0.3 s gives L/R +3.1 dB, E[WY]/E[W²]=0.28 (loud short FDN lines share +y,+z corners)"]
fn own_late_field_is_diffuse_for_short_rooms() {
    for rt in [0.3f32, 0.6, 1.0] {
        let (y, z, lr) = own_field_lean(rt);
        assert!(
            y.abs() < 0.1 && z.abs() < 0.1 && lr.abs() < 1.0,
            "rt {rt}: y {y:.3} z {z:.3} L/R {lr:.2} dB"
        );
    }
}
