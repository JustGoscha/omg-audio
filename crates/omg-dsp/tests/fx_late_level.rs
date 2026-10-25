//! Effect: the late field's LEVEL is the field at the listener, applied
//! at the FDN's output (renderer.rs, own and remote FDN).
//!
//! Claims:
//! - when the level drops (the listener left the room: the traced late
//!   level collapses), the tail goes with it within the level smoother
//!   (τ 0.2 s) — it does not ring on outside for the room's RT60;
//! - when the SOURCE falls silent at a constant level, the tail rings out
//!   at the room's RT60, as a reverb must.
//!
//! Isolation: bus-only `Renderer::new`, no taps, one ParamBlock with a
//! 5 s RT60. Combined in the demo: flying out of the cathedral
//! (omg-core fx_v3_tracer `crossing_a_wall_drops_the_room_left_behind`
//! drops the traced level; this drops what you hear).

mod common;
use common::*;
use omg_core::params::{ParamBlock, ReverbParams};
use omg_dsp::ambi::NCH;
use omg_dsp::Renderer;

fn block(version: u64, level: f32) -> ParamBlock {
    ParamBlock { reverb: ReverbParams { rt60: [5.0; 3], level: [level; 3] }, version, ..Default::default() }
}

/// Render: noise until `stop_at` (samples), level `lv` until `drop_at`,
/// then level 0. Returns the W channel.
fn run(n: usize, stop_at: usize, drop_at: usize) -> Vec<f32> {
    let mut r = Renderer::new(SR);
    r.set_params(&block(1, 0.1));
    let x = noise(9, n, 0.5);
    let mut out = Vec::with_capacity(n);
    for (k, &s) in x.iter().enumerate() {
        if k == drop_at {
            r.set_params(&block(2, 0.0));
        }
        let mut bus = [0.0f32; NCH];
        r.process(if k < stop_at { s } else { 0.0 }, &mut bus);
        out.push(bus[0]);
    }
    out
}

fn db_window(y: &[f32], at: usize) -> f32 {
    db(rms(&y[at..at + 4800]))
}

#[test]
fn leaving_the_room_takes_its_tail_along() {
    let t0 = 96_000; // 2 s of steady field first
    let y = run(t0 + 96_000, usize::MAX, t0);
    let before = db_window(&y, t0 - 4800);
    let after = db_window(&y, t0 + 48_000 - 2400); // ~1 s later
    assert!(before - after > 35.0, "1 s after the level dropped: {:.1} dB below (ringing on outside?)", before - after);
}

#[test]
fn a_source_falling_silent_still_rings_out() {
    let t0 = 96_000;
    let y = run(t0 + 96_000, t0, usize::MAX);
    let before = db_window(&y, t0 - 4800);
    let after = db_window(&y, t0 + 48_000 - 2400);
    // RT60 5 s = 12 dB/s
    let fall = before - after;
    assert!(fall > 6.0 && fall < 20.0, "1 s after the source stopped: {fall:.1} dB below (want ~12)");
}
