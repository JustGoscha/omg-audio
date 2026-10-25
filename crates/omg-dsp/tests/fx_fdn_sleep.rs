//! Effect: FDN sleep when idle (renderer.rs FDN_IDLE_AFTER = 1024).
//!
//! Claim: an FDN whose input AND tail have been silent for 1024 samples
//! goes to sleep (adds nothing, costs nothing), without ever clipping an
//! audible tail, and wakes instantly on new input. A default ParamBlock
//! (reverb level 0 — what inactive sources ship) never feeds the FDN at
//! all: the "constant motor from somewhere" field report.
//!
//! Observable through the public API: there is no sleep flag, but an
//! awake FDN with a decaying tail keeps adding (tiny, non-zero) line
//! output for many seconds; a sleeping one adds exactly 0.0. So "the bus
//! becomes bit-exact zero shortly after the tail drops below −100 dB" is
//! the sleep signature.
//!
//! Isolation: bus-only `Renderer::new`, no taps, own and remote FDN.
//! Combined in the demo: omg-scene tests/car_drone.rs (a departing car's
//! wet bed dies with it) relies on the level → silence → sleep chain.

mod common;
use common::*;
use omg_core::params::{ParamBlock, RemoteReverb, ReverbParams};
use omg_dsp::ambi::{encode_gains, NCH};
use omg_dsp::Renderer;

fn own(level: f32, version: u64) -> ParamBlock {
    ParamBlock {
        taps: vec![],
        reverb: ReverbParams { rt60: [0.5; 3], level: [level; 3] },
        remote: None,
        version,
    }
}

fn run(r: &mut Renderer, input: impl Fn(usize) -> f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| {
            let mut bus = [0.0f32; NCH];
            r.process(input(k), &mut bus);
            bus.iter().fold(0.0f32, |m, v| m.max(v.abs()))
        })
        .collect()
}

/// Index after which `y` is bit-exact zero (None if it never settles).
fn silent_from(y: &[f32]) -> Option<usize> {
    let last = y.iter().rposition(|v| *v != 0.0)?;
    (last + 1 < y.len()).then_some(last + 1)
}

#[test]
fn default_block_never_feeds_the_fdn() {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock { version: 1, ..Default::default() });
    let n = noise(1, 48_000, 0.8);
    let y = run(&mut r, |k| n[k], 48_000);
    assert!(y.iter().all(|v| *v == 0.0), "level-0 source must be silent");
}

/// Burst → tail rings out → bit-exact silence within ~1.5 s (rt60 0.5 s),
/// and the last 50 ms before sleep sit below −100 dBFS absolute (the
/// sleep gate is an absolute energy floor) and > 60 dB under the burst —
/// the full RT60 rang out before the network stopped.
#[test]
fn own_fdn_sleeps_after_its_tail_and_never_clips_it() {
    let mut r = Renderer::new(SR);
    r.set_params(&own(0.2, 1));
    let n = noise(2, 14_400, 0.5);
    let y = run(&mut r, |k| if k < 14_400 { n[k] } else { 0.0 }, 14_400 + 4 * 48_000);
    let peak = y[..14_400].iter().cloned().fold(0.0, f32::max);
    assert!(peak > 1e-3, "burst must reach the late field: {peak}");
    assert!(y[14_400 + 14_400] > 0.0, "tail must still ring 0.3 s after the input stops");
    let s = silent_from(&y).expect("FDN never went to sleep");
    let after_stop = (s - 14_400) as f32 / SR;
    assert!(after_stop < 1.5, "sleep {after_stop:.2} s after the input stopped (rt60 0.5 s)");
    let pre = y[s - 2400..s].iter().cloned().fold(0.0, f32::max);
    assert!(
        pre < 1e-5 && pre < 1e-3 * peak,
        "tail clipped at sleep: {pre:.2e} vs burst {peak:.2e}"
    );
}

/// A sleeping FDN wakes on the next non-zero sample: output within one
/// shortest line length (23.4 ms), and the same response every wake.
#[test]
fn sleeping_fdn_wakes_instantly_and_identically() {
    let mut r = Renderer::new(SR);
    r.set_params(&own(0.2, 1));
    let n = noise(3, 9600, 0.5);
    let burst = |k: usize| if k < 9600 { n[k] } else { 0.0 };
    let _first = run(&mut r, burst, 3 * 48_000);
    let second = run(&mut r, burst, 3 * 48_000);
    let third = run(&mut r, burst, 3 * 48_000);
    assert!(second[2 * 48_000..].iter().all(|v| *v == 0.0), "should be asleep between bursts");
    let onset = second.iter().position(|v| *v > 0.0).unwrap();
    assert!(onset <= (0.0235 * SR) as usize + 2, "wake latency {onset} samples");
    let (a, b) = (rms(&second[..19_200]), rms(&third[..19_200]));
    assert!((db(a) - db(b)).abs() < 0.01, "wake response must repeat: {a} vs {b}");
}

/// The remote FDN sleeps too once the source re-enters our room
/// (remote → None): its tail rings out, then bit-exact silence.
#[test]
fn remote_fdn_sleeps_after_remote_goes_away() {
    let mut r = Renderer::new(SR);
    r.set_params(&ParamBlock {
        remote: Some(RemoteReverb { rt60: [0.6; 3], send: [0.3; 3], sh: encode_gains([1.0, 0.0, 0.0]) }),
        version: 1,
        ..Default::default()
    });
    let n = noise(4, 48_000, 0.5);
    let on = run(&mut r, |k| n[k], 48_000);
    assert!(rms(&on[24_000..]) > 1e-3, "remote must sound while present");
    r.set_params(&ParamBlock { version: 2, ..Default::default() });
    // source keeps playing — but nothing feeds the remote FDN any more
    let y = run(&mut r, |k| n[k % 48_000], 4 * 48_000);
    assert!(y[4800] > 0.0, "tail must ring out through the last door direction");
    let s = silent_from(&y).expect("remote FDN never went to sleep");
    assert!((s as f32 / SR) < 3.0, "remote sleep after {:.2} s", s as f32 / SR);
}
