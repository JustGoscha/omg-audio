//! Effect: coupled-room ("remote") reverb (renderer.rs, ParamBlock.remote).
//!
//! Claim: when the source sits in another room, its room's wet field is
//! heard THROUGH the doorway: a second FDN fed from the remote send,
//! decaying with the SOURCE room's RT60, emitted from the doorway
//! direction but spread over an aperture arc (±~30°, not a mono point —
//! a mono-summed FDN combs). A wall-leaked send (lows ≫ mids) is
//! lowpassed. With `remote: None` nothing is fed: silence.
//!
//! Isolation: bus-only `Renderer::new`, no taps, own-room reverb level 0
//! so only the remote path can sound; raw SH bus frames, plus the
//! cardioid `StereoDecoder` for an L/R check.
//! Combined in the demo: omg-scene tests/late_world_gate.rs
//! `doorway_wet_is_measured_directional_and_door_sensitive` and
//! tests/club_boom.rs (club wet leaking bass-first) assert the remote
//! parameters this path renders.

mod common;
use common::*;
use omg_core::params::{ParamBlock, RemoteReverb, ReverbParams};
use omg_dsp::ambi::{encode_gains, StereoDecoder, NCH};
use omg_dsp::Renderer;

const ON: usize = 72_000; // 1.5 s of noise, then silence

fn remote(rt60: f32, send: [f32; 3], door: [f32; 3]) -> RemoteReverb {
    RemoteReverb { rt60: [rt60; 3], send, sh: encode_gains(door) }
}

/// Bus frames for 1.5 s of noise + `tail` samples of silence.
fn render(remote: Option<RemoteReverb>, own_level: f32, tail: usize) -> Vec<[f32; NCH]> {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock {
        taps: vec![],
        reverb: ReverbParams { rt60: [0.3; 3], level: [own_level; 3] },
        remote,
        version: 1,
    });
    let mut src = Noise::new(42);
    (0..ON + tail)
        .map(|k| {
            let mut bus = [0.0f32; NCH];
            r.process(if k < ON { 0.5 * src.next() } else { 0.0 }, &mut bus);
            bus
        })
        .collect()
}

fn ch(frames: &[[f32; NCH]], c: usize) -> Vec<f32> {
    frames.iter().map(|f| f[c]).collect()
}

/// The remote send produces wet output whose free decay follows the
/// SOURCE room's RT60 (1.2 s), not the listener room's (0.3 s).
#[test]
fn remote_send_rings_with_the_source_rooms_decay() {
    let f = render(Some(remote(1.2, [0.3; 3], [0.0, 1.0, 0.0])), 0.0, 72_000);
    let w = ch(&f, 0);
    assert!(rms(&w[48_000..ON]) > 0.01, "remote wet must be audible: {}", rms(&w[48_000..ON]));
    let decay: Vec<f64> = Biquad::bandpass(1000.0, 1.4)
        .run(&w[ON..])
        .iter()
        .map(|v| (*v as f64).powi(2))
        .collect();
    let t = schroeder_t60(&decay, -5.0, -25.0);
    assert!((t / 1.2 - 1.0).abs() < 0.15, "remote tail T60 {t:.3} s, source room 1.2 s");
}

/// Directional at the doorway (left), but spread: the energy-weighted
/// direction vector E[W·(X,Y,Z)]/E[W²] points at the door with
/// magnitude < 1 (a point emitter is exactly 1), centered on it.
/// Decoded: clearly louder on the left.
#[test]
fn remote_wet_comes_from_the_doorway_with_aperture_spread() {
    let f = render(Some(remote(1.2, [0.3; 3], [0.0, 1.0, 0.0])), 0.0, 0);
    let s = &f[48_000..ON];
    let ew: f64 = s.iter().map(|b| (b[0] as f64).powi(2)).sum();
    let dir = |c: usize| s.iter().map(|b| b[0] as f64 * b[c] as f64).sum::<f64>() / ew;
    let (y, z, x) = (dir(1), dir(2), dir(3));
    assert!(y > 0.85 && y < 0.98, "toward the door but spread: rV_y = {y:.3}");
    // centered on the door within ~10° (the 8 lines carry unequal energy,
    // so the arc's centroid sits slightly off the nominal center)
    let off_deg = x.atan2(y).to_degrees();
    assert!(off_deg.abs() < 10.0 && z.abs() < 0.05, "arc centroid off the door by {off_deg:.1}°, z {z:.3}");

    let dec = StereoDecoder::new();
    let (mut el, mut er) = (0.0f64, 0.0f64);
    for b in s {
        let (l, r) = dec.decode(b);
        el += (l as f64).powi(2);
        er += (r as f64).powi(2);
    }
    assert!(db_pow(el / er) > 6.0, "door on the left must decode left: {:.1} dB", db_pow(el / er));

}

/// No remote → nothing at all, bit-exact (own level 0 here).
#[test]
fn no_remote_is_silent() {
    let f = render(None, 0.0, 4800);
    assert!(f.iter().all(|b| b.iter().all(|v| *v == 0.0)), "remote None must be silent");
}

/// A wall-leaked send (low/mid > 2) takes the continuous-lowpass branch:
/// much darker than a flat send of the same low-band level.
#[test]
fn wall_leaked_remote_is_lowpassed() {
    let hf_lf = |send: [f32; 3]| -> f64 {
        let f = render(Some(remote(1.0, send, [1.0, 0.0, 0.0])), 0.0, 0);
        let w = ch(&f[48_000..ON], 0);
        energy(&Biquad::bandpass(4000.0, 1.4).run(&w)) / energy(&Biquad::bandpass(150.0, 1.4).run(&w))
    };
    let flat = hf_lf([0.3, 0.3, 0.3]);
    let leaked = hf_lf([0.3, 0.06, 0.006]);
    assert!(leaked < 0.05 * flat, "leaked wet must be muffled: {leaked:.2e} vs flat {flat:.2e}");
}
