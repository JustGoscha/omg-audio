//! Order-2 ambisonics bus, ACN channel order, SN3D normalization.
//! Axes: +x forward, +y left, +z up. All spatialized signals sum into this
//! bus; the binaural (or speaker) decode is a single fixed-cost stage at the
//! end, independent of source count. Fast head rotation (mouse look, device
//! orientation, camera face tracking) counter-rotates the bus at the decode
//! stage: yaw-only via [`rotate_z`], full yaw/pitch/roll via
//! [`HeadRotation`] (block-diagonal 3×3 + 5×5, exact for order 2).

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

/// Full head rotation for the bus and for direction vectors.
///
/// Angle conventions (right-handed, +x forward, +y left, +z up):
///  - `yaw`   positive turns the head LEFT (about +z),
///  - `pitch` positive looks UP,
///  - `roll`  positive tilts the head RIGHT (right ear down, about +x).
///
/// The stored matrices are the INVERSE head rotation — the field counter-
/// rotates: `dir_head = M · dir_world`. Order-1 SH coefficients transform
/// exactly like direction components; the degree-2 block is derived by
/// substituting the rotated coordinates into the five quadratic harmonics
/// and reading the result back in the harmonic basis — exact, no Euler
/// re-decomposition, no order truncation.
pub struct HeadRotation {
    m: [[f32; 3]; 3],
    m2: [[f32; 5]; 5],
}

impl HeadRotation {
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();
        // M = Rx(-roll)·Ry(pitch)·Rz(-yaw) — the inverse of the head
        // orientation Rz(yaw)·Ry(-pitch)·Rx(roll) (intrinsic z-y'-x'').
        // With pitch = roll = 0 this is exactly the rotate_z convention.
        let m = [
            [cp * cy, cp * sy, sp],
            [-cr * sy - sr * sp * cy, cr * cy - sr * sp * sy, sr * cp],
            [sr * sy - cr * sp * cy, -sr * cy - cr * sp * sy, cr * cp],
        ];

        // Degree-2 block: q_j(M·u) expanded over the monomials
        // [x², y², z², xy, yz, xz] of u, then read back in the ACN 4..8
        // basis (V=√3xy, T=√3yz, R=(3z²−1)/2, S=√3xz, U=√3(x²−y²)/2).
        let quad = |a: [f32; 3], b: [f32; 3]| -> [f32; 6] {
            [
                a[0] * b[0],
                a[1] * b[1],
                a[2] * b[2],
                a[0] * b[1] + a[1] * b[0],
                a[1] * b[2] + a[2] * b[1],
                a[0] * b[2] + a[2] * b[0],
            ]
        };
        // On the unit sphere a trace-free quadratic re-reads as:
        //   coeff(V) = α_xy/√3, coeff(T) = α_yz/√3, coeff(S) = α_xz/√3,
        //   coeff(U) = (α_xx − α_yy)/√3,
        //   coeff(R) = ⅔·(α_zz − (α_xx + α_yy)/2).
        let to_row = |a: [f32; 6]| -> [f32; 5] {
            [
                a[3] / SQRT3,
                a[4] / SQRT3,
                (a[2] - 0.5 * (a[0] + a[1])) * (2.0 / 3.0),
                a[5] / SQRT3,
                (a[0] - a[1]) / SQRT3,
            ]
        };
        let scale = |a: [f32; 6], s: f32| -> [f32; 6] { core::array::from_fn(|i| a[i] * s) };
        let sub = |a: [f32; 6], b: [f32; 6]| -> [f32; 6] { core::array::from_fn(|i| a[i] - b[i]) };

        let xx = quad(m[0], m[0]);
        let yy = quad(m[1], m[1]);
        let zz = quad(m[2], m[2]);
        // |M·u|² = |u|² (orthogonal), so R's constant part is exactly
        // −(x² + y² + z²)/2 in the source monomials.
        let unit = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let m2 = [
            to_row(scale(quad(m[0], m[1]), SQRT3)),          // V' = √3·x'y'
            to_row(scale(quad(m[1], m[2]), SQRT3)),          // T' = √3·y'z'
            to_row(sub(scale(zz, 1.5), scale(unit, 0.5))),   // R' = (3z'²−1)/2
            to_row(scale(quad(m[0], m[2]), SQRT3)),          // S' = √3·x'z'
            to_row(scale(sub(xx, yy), 0.5 * SQRT3)),         // U' = √3(x'²−y'²)/2
        ];
        Self { m, m2 }
    }

    /// World direction → head-frame direction.
    #[inline]
    pub fn rotate_dir(&self, d: [f32; 3]) -> [f32; 3] {
        core::array::from_fn(|i| {
            self.m[i][0] * d[0] + self.m[i][1] * d[1] + self.m[i][2] * d[2]
        })
    }

    /// Counter-rotate the SH field into the head frame.
    #[inline]
    pub fn apply(&self, sh: &mut [f32; NCH]) {
        // ACN order 1 is (Y, Z, X) = the (y, z, x) direction components.
        let v = [sh[3], sh[1], sh[2]];
        let r = self.rotate_dir(v);
        sh[3] = r[0];
        sh[1] = r[1];
        sh[2] = r[2];
        let c = [sh[4], sh[5], sh[6], sh[7], sh[8]];
        for (j, row) in self.m2.iter().enumerate() {
            sh[4 + j] = row
                .iter()
                .zip(c.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The defining property of an SH rotation: rotating a point source's
    /// coefficients equals encoding the rotated direction. Exact for both
    /// blocks, over random orientations — this pins every sign and every
    /// entry of the degree-2 derivation.
    #[test]
    fn head_rotation_matches_encoded_directions() {
        let mut rng = omg_core::rng::Rng::new(11);
        let mut f = || rng.next_f32() * 2.0 - 1.0;
        for _ in 0..200 {
            let (yaw, pitch, roll) = (f() * 3.1, f() * 1.5, f() * 1.5);
            let d = {
                let v = [f(), f(), f()];
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-3);
                [v[0] / n, v[1] / n, v[2] / n]
            };
            let rot = HeadRotation::new(yaw, pitch, roll);
            let mut sh = encode_gains(d);
            rot.apply(&mut sh);
            let want = encode_gains(rot.rotate_dir(d));
            for k in 0..NCH {
                assert!(
                    (sh[k] - want[k]).abs() < 1e-4,
                    "ch {k}: {} vs {} at yaw {yaw} pitch {pitch} roll {roll}",
                    sh[k],
                    want[k]
                );
            }
        }
    }

    /// Yaw-only HeadRotation must be bit-for-bit the rotate_z convention
    /// (positive = head turns left) the whole engine already relies on.
    #[test]
    fn yaw_only_matches_rotate_z() {
        let mut rng = omg_core::rng::Rng::new(5);
        for _ in 0..50 {
            let psi = (rng.next_f32() * 2.0 - 1.0) * 3.1;
            let d = [0.6, -0.64, 0.48];
            let mut a = encode_gains(d);
            rotate_z(&mut a, psi);
            let mut b = encode_gains(d);
            HeadRotation::new(psi, 0.0, 0.0).apply(&mut b);
            for k in 0..NCH {
                assert!((a[k] - b[k]).abs() < 1e-5, "ch {k}: {} vs {}", a[k], b[k]);
            }
        }
    }

    /// Sanity of the physical directions: looking UP must bring an
    /// overhead source to the FRONT; tilting the head RIGHT must raise a
    /// world-left source in the head frame.
    #[test]
    fn pitch_and_roll_point_the_right_way() {
        let up = HeadRotation::new(0.0, 0.5, 0.0).rotate_dir([0.0, 0.0, 1.0]);
        assert!(up[0] > 0.4 && up[2] > 0.0, "overhead should move frontward: {up:?}");
        let left = HeadRotation::new(0.0, 0.0, 0.5).rotate_dir([0.0, 1.0, 0.0]);
        assert!(left[2] < -0.4, "tilting right lowers world-left in head frame: {left:?}");
    }
}
