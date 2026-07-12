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

/// Beds must be BEDS. A field recording is not stationary — somewhere a
/// cricket chirps right next to the microphone for a few seconds — and
/// once `normalize_rms` sets the overall level, such a passage SURGES
/// out of the background (localized at whatever inlet happens to be
/// reading it: "a super loud ambience appears out of nowhere, then
/// fades"). This equalizes the slow loudness across the loop: windowed
/// RMS toward the median, gains clamped to −18/+12 dB and interpolated
/// per-sample (loop-circular), so second-scale passages level out while
/// chirp-scale texture inside a window survives untouched.
pub fn flatten_slow_loudness(samples: &mut [f32], channels: usize, sample_rate: f32) {
    // two scales: second-scale passages corrected deeply, then a gentler
    // sub-second pass so a single chirp burst cannot pop out of a quiet
    // room while ordinary texture keeps its shape
    flatten_pass(samples, channels, sample_rate, 1.0, 0.125, 4.0);
    flatten_pass(samples, channels, sample_rate, 0.35, 0.5, 2.0);
}

fn flatten_pass(
    samples: &mut [f32],
    channels: usize,
    sample_rate: f32,
    win_s: f32,
    g_min: f32,
    g_max: f32,
) {
    let ch = channels.max(1);
    let frames = samples.len() / ch;
    let win = (sample_rate * win_s) as usize;
    let hop = (win / 2).max(1);
    if frames < 3 * win {
        return; // too short for a slow-loudness profile
    }
    let n_win = frames / hop;
    let mut rms: Vec<f32> = (0..n_win)
        .map(|w| {
            let start = w * hop;
            let mut e = 0.0f64;
            let mut n = 0u64;
            let mut i = start;
            while i < (start + win).min(frames) {
                for c in 0..ch {
                    let v = samples[i * ch + c] as f64;
                    e += v * v;
                }
                n += ch as u64;
                i += 1;
            }
            ((e / n.max(1) as f64) as f32).sqrt().max(1e-6)
        })
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(f32::total_cmp);
    let target = sorted[sorted.len() / 2];
    // per-window gains, lightly smoothed (circular — the bed loops)
    let mut gains: Vec<f32> = rms.iter().map(|r| (target / r).clamp(g_min, g_max)).collect();
    let raw = gains.clone();
    for w in 0..n_win {
        let prev = raw[(w + n_win - 1) % n_win];
        let next = raw[(w + 1) % n_win];
        gains[w] = 0.25 * prev + 0.5 * raw[w] + 0.25 * next;
    }
    let _ = &mut rms;
    // apply: linear interpolation between window centers
    for f in 0..frames {
        let pos = f as f32 / hop as f32 - 0.5;
        let w0 = pos.floor();
        let t = pos - w0;
        let i0 = ((w0 as i64).rem_euclid(n_win as i64)) as usize;
        let i1 = (i0 + 1) % n_win;
        let g = gains[i0] * (1.0 - t) + gains[i1] * t;
        for c in 0..ch {
            samples[f * ch + c] *= g;
        }
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

    /// The bed guarantee: a loop with a few loud seconds (the cricket at
    /// the microphone) comes out with its slow loudness flat — no
    /// passage can surge out of the background anymore.
    #[test]
    fn hot_passages_are_flattened() {
        let sr = 48_000usize;
        let mut rng = 1u64;
        let mut noise = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / 2147483648.0) - 1.0
        };
        let n = sr * 20;
        let mut x: Vec<f32> = (0..n)
            .map(|i| {
                let hot = i >= sr * 8 && i < sr * 11; // 3 loud seconds
                noise() * if hot { 0.8 } else { 0.1 }
            })
            .collect();
        flatten_slow_loudness(&mut x, 1, sr as f32);
        let rms = |a: usize, b: usize| -> f32 {
            (x[a..b].iter().map(|v| (v * v) as f64).sum::<f64>() / (b - a) as f64).sqrt() as f32
        };
        let quiet = rms(2 * sr, 5 * sr);
        let was_hot = rms(sr * 9, sr * 10); // center of the hot region
        let ratio = was_hot / quiet;
        assert!(
            ratio < 1.6,
            "hot passage must flatten into the bed: ratio {ratio}"
        );
        // and a stationary signal passes through nearly untouched
        let mut flat: Vec<f32> = (0..n).map(|_| noise() * 0.1).collect();
        let before = flat.clone();
        flatten_slow_loudness(&mut flat, 1, sr as f32);
        let drift = flat
            .iter()
            .zip(before.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(drift < 0.03, "stationary input must be ~transparent: {drift}");
    }

    #[test]
    fn near_silence_is_not_amplified_into_noise() {
        let mut hiss: Vec<f32> = (0..4800).map(|k| 1e-4 * ((k * 7919) % 97) as f32 / 97.0).collect();
        normalize_rms(&mut hiss, REF_CLIP_RMS);
        let peak = hiss.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(peak < 0.01, "boost should be capped: peak {peak}");
    }
}
