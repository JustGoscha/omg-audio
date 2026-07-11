//! Import-loudness normalization: every clip entering the engine is
//! brought to one reference loudness, so how a sound was RECORDED stops
//! mattering — what it IS (its type's SPL calibration on the mixer)
//! decides the energy it emits, tweakable from there.

/// Reference clip loudness: gated RMS ≈ −18 dBFS. Chosen so existing
/// demo material lands near its previous levels.
pub const REF_CLIP_RMS: f32 = 0.12;

/// Normalize a clip in place to `target_rms`, measured as GATED RMS:
/// only samples above −30 dB of the clip's peak count, so speech pauses
/// and lead-in silence don't inflate the boost. Gain is capped at 24×
/// (a near-silent file shouldn't become a noise bomb). Samples may
/// exceed ±1 afterwards — the engine has headroom and a final tanh.
pub fn normalize_rms(samples: &mut [f32], target_rms: f32) {
    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if peak < 1e-6 {
        return;
    }
    let gate = peak * 0.0316; // −30 dB
    let mut energy = 0.0f64;
    let mut n = 0u64;
    for x in samples.iter() {
        if x.abs() > gate {
            energy += (*x as f64) * (*x as f64);
            n += 1;
        }
    }
    if n == 0 {
        return;
    }
    let rms = (energy / n as f64).sqrt() as f32;
    let g = (target_rms / rms.max(1e-9)).min(24.0);
    for x in samples.iter_mut() {
        *x *= g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gated_rms(s: &[f32]) -> f32 {
        let peak = s.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let gate = peak * 0.0316;
        let (mut e, mut n) = (0.0f64, 0u64);
        for x in s {
            if x.abs() > gate {
                e += (*x as f64) * (*x as f64);
                n += 1;
            }
        }
        (e / n.max(1) as f64).sqrt() as f32
    }

    /// The point of the feature: a quiet take and a loud take of the same
    /// material land at the same loudness.
    #[test]
    fn recording_level_stops_mattering() {
        let make = |amp: f32| -> Vec<f32> {
            (0..48_000)
                .map(|k| {
                    // "speech": bursts with pauses
                    let t = k as f32 / 48_000.0;
                    if (t * 3.0).fract() < 0.6 {
                        amp * (core::f32::consts::TAU * 220.0 * t).sin()
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let mut quiet = make(0.02);
        let mut loud = make(0.9);
        normalize_rms(&mut quiet, REF_CLIP_RMS);
        normalize_rms(&mut loud, REF_CLIP_RMS);
        let (rq, rl) = (gated_rms(&quiet), gated_rms(&loud));
        assert!(
            (rq / rl - 1.0).abs() < 0.02,
            "both takes should land at the same loudness: {rq} vs {rl}"
        );
        assert!((rq / REF_CLIP_RMS - 1.0).abs() < 0.05, "at the reference: {rq}");
    }

    /// Pauses don't inflate the boost (gated, not whole-file RMS).
    #[test]
    fn silence_padding_changes_nothing() {
        let tone: Vec<f32> =
            (0..24_000).map(|k| 0.3 * (0.03 * k as f32).sin()).collect();
        let mut padded = vec![0.0f32; 48_000];
        padded[..24_000].copy_from_slice(&tone);
        let mut bare = tone.clone();
        normalize_rms(&mut padded, REF_CLIP_RMS);
        normalize_rms(&mut bare, REF_CLIP_RMS);
        let ratio = gated_rms(&padded[..24_000].to_vec()) / gated_rms(&bare);
        assert!((ratio - 1.0).abs() < 0.02, "padding changed loudness: {ratio}");
    }

    #[test]
    fn near_silence_is_not_amplified_into_noise() {
        let mut hiss: Vec<f32> = (0..4800).map(|k| 1e-4 * ((k * 7919) % 97) as f32 / 97.0).collect();
        normalize_rms(&mut hiss, REF_CLIP_RMS);
        let peak = hiss.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(peak < 0.01, "boost should be capped: peak {peak}");
    }
}
