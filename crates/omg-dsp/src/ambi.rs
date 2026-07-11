//! Order-2 ambisonics bus, ACN channel order, SN3D normalization.
//! Axes: +x forward, +y left, +z up. All spatialized signals sum into this
//! bus; the binaural (or speaker) decode is a single fixed-cost stage at the
//! end, independent of source count. Head rotation would be a 9×9 matrix on
//! the bus — currently unnecessary because encode directions are already
//! listener-relative.

/// Channel count for order 2: (N+1)² = 9.
pub const NCH: usize = 9;

const SQRT3: f32 = 1.732_050_8;

/// SN3D/ACN real spherical harmonics up to degree 2 for a unit direction.
#[inline]
pub fn encode_gains(dir: [f32; 3]) -> [f32; NCH] {
    let [x, y, z] = dir;
    [
        1.0,                        // W
        y,                          // Y
        z,                          // Z
        x,                          // X
        SQRT3 * x * y,              // V
        SQRT3 * y * z,              // T
        0.5 * (3.0 * z * z - 1.0),  // R
        SQRT3 * x * z,              // S
        0.5 * SQRT3 * (x * x - y * y), // U
    ]
}

/// Rotate the SH field about z by head yaw ψ (positive = head turns left).
/// Sources move opposite to the head: dir' = Rz(-ψ)·dir, which on ACN/SN3D
/// channels is: W,Z,R unchanged; (X,Y) and (S,T) rotate by ψ; (U,V) by 2ψ.
#[inline]
pub fn rotate_z(sh: &mut [f32; NCH], psi: f32) {
    let (s, c) = psi.sin_cos();
    let (s2, c2) = (2.0 * psi).sin_cos();
    let (y, x) = (sh[1], sh[3]);
    sh[3] = c * x + s * y;
    sh[1] = -s * x + c * y;
    let (t, sx) = (sh[5], sh[7]);
    sh[7] = c * sx + s * t;
    sh[5] = -s * sx + c * t;
    let (v, u) = (sh[4], sh[8]);
    sh[8] = c2 * u + s2 * v;
    sh[4] = -s2 * u + c2 * v;
}

/// Virtual-cardioid stereo decode of the degree-0/1 channels — the fallback
/// when no HRIR asset is available, and the eventual arbitrary-speaker path.
pub struct StereoDecoder {
    cos_l: f32,
    sin_l: f32,
}

impl StereoDecoder {
    pub fn new() -> Self {
        let th = 60.0f32.to_radians();
        Self { cos_l: th.cos(), sin_l: th.sin() }
    }

    #[inline]
    pub fn decode(&self, sh: &[f32; NCH]) -> (f32, f32) {
        let (w, y, x) = (sh[0], sh[1], sh[3]);
        let l = 0.5 * (w + x * self.cos_l + y * self.sin_l);
        let r = 0.5 * (w + x * self.cos_l - y * self.sin_l);
        (l, r)
    }
}

impl Default for StereoDecoder {
    fn default() -> Self {
        Self::new()
    }
}
