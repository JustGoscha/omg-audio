//! The v4 dense response's reverberation time, measured as from an
//! impulse response: exact for an exponential decay, also one longer
//! than the traced window (a cathedral rings past `RESPONSE_MAX_S`).

use omg_core::v3::{DenseResponse, RESPONSE_MAX_S};

/// An exponential decay of `rt` seconds (−60 dB), starting after `delay`,
/// cut at the traced window.
fn decay(rt: f32, delay: f32) -> DenseResponse {
    let bin_s = 0.002;
    let n = (RESPONSE_MAX_S / bin_s) as usize;
    let k = -6.0 * std::f32::consts::LN_10 / rt; // ln E per second
    let bins = (0..n)
        .map(|i| {
            let t = i as f32 * bin_s;
            let e = if t < delay { 0.0 } else { (k * (t - delay)).exp() };
            [e, e, e * 0.5, 0.0, 0.0, 0.0]
        })
        .collect();
    DenseResponse { bin_s, bins }
}

#[test]
fn rt60_of_exponential_decays() {
    for rt in [0.4f32, 1.5, 4.0, 9.0, 15.0] {
        let got = decay(rt, 0.01).rt60();
        eprintln!("rt60 {rt} s → {got:?}");
        for b in 0..3 {
            assert!((got[b] - rt).abs() < 0.03 * rt, "RT60 {rt} s measured as {} s (band {b})", got[b]);
        }
    }
}

#[test]
fn a_response_that_ended_has_its_own_decay() {
    // a short room tail, then silence: no continuation invented
    let mut d = decay(0.5, 0.0);
    for v in d.bins.iter_mut().skip(1000) {
        *v = [0.0; 6];
    }
    let got = d.rt60();
    assert!((got[1] - 0.5).abs() < 0.03, "{got:?}");
    assert_eq!(DenseResponse { bin_s: 0.002, bins: Vec::new() }.rt60(), [0.0; 3]);
}
