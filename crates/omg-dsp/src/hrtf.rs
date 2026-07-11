//! Binaural rendering, two tiers:
//!
//! 1. `BinauralDecoder` — the ambisonic bus decode: order-2 SH → virtual
//!    speakers (icosahedron/dodecahedron) → measured HRIR per speaker.
//!    Fixed cost, order-limited sharpness: right for reflections + reverb.
//! 2. `HrirGrid` + `PointConv` — nearest-HRIR point rendering from the
//!    dense measurement grid (~710 directions): full sharpness, used for
//!    direct paths where localization actually happens. Index changes
//!    crossfade over ~10 ms; ITD lives in the measured pairs.
//!
//! HRIRs: MIT KEMAR (Gardner & Martin), 128 taps, resampled to 48 kHz and
//! packed by tools/make_hrir.py (flat little-endian, see that file).

use crate::ambi::{encode_gains, NCH};

/// max-rE degree weights for order 2 (P_l(cos 39.3°)).
const MAXRE: [f32; 3] = [1.0, 0.775, 0.4];
/// Empirical loudness match vs. the cardioid fallback decoder.
const MASTER: f32 = 1.4;

fn parse_bin(bytes: &[u8]) -> (usize, Vec<([f32; 3], Vec<f32>, Vec<f32>)>) {
    let mut off = 0usize;
    let u32_at = |o: &mut usize| {
        let v = u32::from_le_bytes(bytes[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let count = u32_at(&mut off) as usize;
    let taps = u32_at(&mut off) as usize;
    let f32s = |o: &mut usize, n: usize| -> Vec<f32> {
        let v = bytes[*o..*o + 4 * n]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        *o += 4 * n;
        v
    };
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let d = f32s(&mut off, 3);
        let l = f32s(&mut off, taps);
        let r = f32s(&mut off, taps);
        records.push(([d[0], d[1], d[2]], l, r));
    }
    (taps, records)
}

// ------------------------------------------------- ambisonic bus decoder

struct Speaker {
    matrix: [f32; NCH],
    hrir_l: Vec<f32>,
    hrir_r: Vec<f32>,
    hist: Vec<f32>,
}

pub struct BinauralDecoder {
    speakers: Vec<Speaker>,
    taps: usize,
    pos: usize,
}

impl BinauralDecoder {
    pub fn load(path: &str) -> std::io::Result<Self> {
        Ok(Self::from_bytes(&std::fs::read(path)?))
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let (taps, records) = parse_bin(bytes);
        let k = records.len() as f32;
        let speakers = records
            .into_iter()
            .map(|(d, mut hrir_l, mut hrir_r)| {
                let enc = encode_gains(d);
                let mut matrix = [0.0f32; NCH];
                for (n, m) in matrix.iter_mut().enumerate() {
                    let l = if n == 0 { 0 } else if n < 4 { 1 } else { 2 };
                    *m = (2 * l + 1) as f32 / k * MAXRE[l] * enc[n] * MASTER;
                }
                // stored reversed: see convolve_ring
                hrir_l.reverse();
                hrir_r.reverse();
                Speaker { matrix, hrir_l, hrir_r, hist: vec![0.0; taps] }
            })
            .collect();
        Self { speakers, taps, pos: 0 }
    }

    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    /// One ambisonic frame in → binaural stereo out.
    #[inline]
    pub fn process(&mut self, sh: &[f32; NCH]) -> (f32, f32) {
        let taps = self.taps;
        self.pos = (self.pos + 1) % taps;
        let pos = self.pos;
        let mut l = 0.0f32;
        let mut r = 0.0f32;

        for spk in &mut self.speakers {
            let mut s = 0.0f32;
            for n in 0..NCH {
                s += spk.matrix[n] * sh[n];
            }
            spk.hist[pos] = s;
            let (al, ar) = convolve_ring(&spk.hist, pos, &spk.hrir_l, &spk.hrir_r);
            l += al;
            r += ar;
        }
        (l, r)
    }
}

/// Dot products of `hist` against two kernels at once, with explicit 4-lane
/// accumulation: float reductions don't auto-vectorize under strict FP
/// semantics, so the reassociation is spelled out and the compiler keeps the
/// lanes in NEON / wasm simd128 registers.
#[inline]
fn dot2(hist: &[f32], kl: &[f32], kr: &[f32]) -> (f32, f32) {
    let mut al = [0.0f32; 4];
    let mut ar = [0.0f32; 4];
    let mut h4 = hist.chunks_exact(4);
    let mut l4 = kl.chunks_exact(4);
    let mut r4 = kr.chunks_exact(4);
    for ((h, l), r) in (&mut h4).zip(&mut l4).zip(&mut r4) {
        for k in 0..4 {
            al[k] += h[k] * l[k];
            ar[k] += h[k] * r[k];
        }
    }
    let mut sl = (al[0] + al[1]) + (al[2] + al[3]);
    let mut sr = (ar[0] + ar[1]) + (ar[2] + ar[3]);
    for ((h, l), r) in h4.remainder().iter().zip(l4.remainder()).zip(r4.remainder()) {
        sl += h * l;
        sr += h * r;
    }
    (sl, sr)
}

/// Ring convolution (newest sample at `pos`) with a PRE-REVERSED kernel
/// pair: with krev[i] = k[n−1−i], Σᵢ hist[(pos−i) mod n]·k[i] becomes two
/// contiguous forward dot products — the form the SIMD kernel needs.
#[inline]
fn convolve_ring(hist: &[f32], pos: usize, hl_rev: &[f32], hr_rev: &[f32]) -> (f32, f32) {
    let s = hist.len() - 1 - pos;
    let (l1, r1) = dot2(&hist[..=pos], &hl_rev[s..], &hr_rev[s..]);
    let (l2, r2) = dot2(&hist[pos + 1..], &hl_rev[..s], &hr_rev[..s]);
    (l1 + l2, r1 + r2)
}

// ------------------------------------------------- dense grid + point conv

pub struct HrirGrid {
    dirs: Vec<[f32; 3]>,
    hrir_l: Vec<Vec<f32>>,
    hrir_r: Vec<Vec<f32>>,
    pub taps: usize,
}

impl HrirGrid {
    pub fn load(path: &str) -> std::io::Result<Self> {
        Ok(Self::from_bytes(&std::fs::read(path)?))
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let (taps, records) = parse_bin(bytes);
        let mut g = Self {
            dirs: Vec::with_capacity(records.len()),
            hrir_l: Vec::with_capacity(records.len()),
            hrir_r: Vec::with_capacity(records.len()),
            taps,
        };
        for (d, mut l, mut r) in records {
            // stored reversed: see convolve_ring
            l.reverse();
            r.reverse();
            g.dirs.push(d);
            g.hrir_l.push(l);
            g.hrir_r.push(r);
        }
        g
    }

    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }

    pub fn nearest(&self, dir: [f32; 3]) -> usize {
        let mut best = 0;
        let mut best_dot = f32::MIN;
        for (i, d) in self.dirs.iter().enumerate() {
            let dot = d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2];
            if dot > best_dot {
                best_dot = dot;
                best = i;
            }
        }
        best
    }
}

/// Per-emitter nearest-HRIR convolver with ~10 ms crossfade on index change.
pub struct PointConv {
    ring: Vec<f32>,
    pos: usize,
    cur: usize,
    prev: usize,
    xfade: f32, // 1 → 0, blend from prev to cur
    xstep: f32,
}

impl PointConv {
    pub fn new(taps: usize, sample_rate: f32) -> Self {
        Self {
            ring: vec![0.0; taps],
            pos: 0,
            cur: 0,
            prev: 0,
            xfade: 0.0,
            xstep: 1.0 / (0.010 * sample_rate),
        }
    }

    pub fn reset(&mut self, idx: usize) {
        self.ring.iter_mut().for_each(|s| *s = 0.0);
        self.cur = idx;
        self.prev = idx;
        self.xfade = 0.0;
    }

    pub fn set_dir(&mut self, grid: &HrirGrid, dir: [f32; 3]) {
        let idx = grid.nearest(dir);
        if idx != self.cur {
            self.prev = self.cur;
            self.cur = idx;
            self.xfade = 1.0;
        }
    }

    #[inline]
    pub fn process(&mut self, grid: &HrirGrid, s: f32) -> (f32, f32) {
        let taps = self.ring.len();
        self.pos = (self.pos + 1) % taps;
        self.ring[self.pos] = s;
        let (cl, cr) = convolve_ring(&self.ring, self.pos, &grid.hrir_l[self.cur], &grid.hrir_r[self.cur]);
        if self.xfade > 0.0 {
            let (pl, pr) =
                convolve_ring(&self.ring, self.pos, &grid.hrir_l[self.prev], &grid.hrir_r[self.prev]);
            let f = self.xfade;
            self.xfade = (self.xfade - self.xstep).max(0.0);
            (cl + f * (pl - cl), cr + f * (pr - cr))
        } else {
            (cl, cr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The segmented/pre-reversed kernel must match the naive ring
    /// convolution Σᵢ hist[(pos−i) mod n]·k[i] at every ring position.
    #[test]
    fn convolve_ring_matches_naive() {
        let n = 128;
        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let hist: Vec<f32> = (0..n).map(|_| next()).collect();
        let kl: Vec<f32> = (0..n).map(|_| next()).collect();
        let kr: Vec<f32> = (0..n).map(|_| next()).collect();
        let (mut kl_rev, mut kr_rev) = (kl.clone(), kr.clone());
        kl_rev.reverse();
        kr_rev.reverse();

        for pos in 0..n {
            let (mut nl, mut nr) = (0.0f32, 0.0f32);
            for i in 0..n {
                let h = hist[(pos + n - i) % n];
                nl += h * kl[i];
                nr += h * kr[i];
            }
            let (sl, sr) = convolve_ring(&hist, pos, &kl_rev, &kr_rev);
            assert!((sl - nl).abs() < 1e-4 && (sr - nr).abs() < 1e-4,
                "pos {pos}: ({sl}, {sr}) vs naive ({nl}, {nr})");
        }
    }
}
