//! Knife-edge diffraction: Kurze–Anderson insertion loss from the Fresnel
//! number. The standard engineering model for sound bending over/around a
//! barrier edge — frequency-correct (long waves bend, short ones don't)
//! with no tuned constants, asymptotically consistent with rigorous
//! solutions for a thin screen. Full UTD wedge coefficients (which also
//! model the wedge's interior angle and reflected-boundary terms) remain
//! future work; for building corners and roof lines the knife edge is the
//! dominant behavior.

use crate::{NBANDS, SPEED_OF_SOUND};

/// Geometric band centers of the three simulation bands
/// (<250 Hz, 250–2500 Hz, >2500 Hz).
pub const BAND_CENTER_HZ: [f32; NBANDS] = [125.0, 790.0, 5600.0];

/// Amplitude factor for one diffracting edge given the SIGNED detour `δ`:
/// positive = shadow zone (the bent path via the edge is δ longer than the
/// blocked straight line), negative = illuminated side (line of sight
/// exists; |δ| is the path difference via the edge — edge-proximity loss).
///
/// Kurze–Anderson, both branches of N = 2δ/λ:
///   N ≥ 0:        A = 5 + 20·log10(√(2πN)/tanh(√(2πN)))   (shadow)
///   −0.19 < N < 0: A = 5 + 20·log10(√(2π|N|)/tan(√(2π|N|))) (lit, → 0 dB)
///   N ≤ −0.19:    A = 0                                     (fully clear)
/// Continuous through the shadow boundary at −5 dB (N = 0).
pub fn knife_edge_amp(detour_m: f32, freq_hz: f32) -> f32 {
    let n = 2.0 * detour_m * freq_hz / SPEED_OF_SOUND;
    if n <= -0.1916 {
        return 1.0;
    }
    let x = (2.0 * core::f32::consts::PI * n.abs()).sqrt();
    let ratio = if x < 1e-3 {
        1.0
    } else if n >= 0.0 {
        x / x.tanh()
    } else {
        x / x.tan()
    };
    let a_db = 5.0 + 20.0 * ratio.log10();
    10f32.powf(-a_db / 20.0)
}

/// Per-band amplitude factors for one edge.
pub fn knife_edge_bands(detour_m: f32) -> [f32; NBANDS] {
    core::array::from_fn(|b| knife_edge_amp(detour_m, BAND_CENTER_HZ[b]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grazing_edge_is_5db() {
        let a = knife_edge_amp(1e-9, 1000.0);
        assert!((20.0 * a.log10() + 5.0).abs() < 0.1, "grazing ≈ −5 dB, got {}", 20.0 * a.log10());
    }

    #[test]
    fn lows_bend_more_than_highs() {
        let bands = knife_edge_bands(0.5);
        assert!(bands[0] > bands[1] && bands[1] > bands[2]);
        // half-meter detour: bass survives (> −12 dB), treble strongly shadowed (< −20 dB)
        assert!(20.0 * bands[0].log10() > -12.0);
        assert!(20.0 * bands[2].log10() < -20.0);
    }

    #[test]
    fn monotonic_in_detour() {
        let mut prev = 1.0;
        for i in 1..40 {
            let a = knife_edge_amp(i as f32 * 0.1, 790.0);
            assert!(a < prev);
            prev = a;
        }
    }

    #[test]
    fn illuminated_side_recovers_to_unity() {
        // far into the lit zone: no loss (cutoff N ≤ −0.19 ⇒ δ ≈ −0.041 m @ 790 Hz)
        assert!((knife_edge_amp(-0.05, 790.0) - 1.0).abs() < 1e-6);
        // inside the transition: partial, between clear and the −5 dB boundary
        let mid = knife_edge_amp(-0.02, 790.0);
        assert!(mid < 1.0 && mid > 0.56, "transition value {mid}");
        // monotonic through the shadow boundary into the shadow zone
        let mut prev = knife_edge_amp(-0.04, 790.0);
        for i in 0..20 {
            let a = knife_edge_amp(-0.04 + i as f32 * 0.005, 790.0);
            assert!(a <= prev + 1e-6, "not monotonic at step {i}");
            prev = a;
        }
    }
}
