//! Binaural rendering, two tiers:
//!
//! 1. `BinauralDecoder` — the ambisonic bus decode, either straight from
//!    the SH channels (a decoder file, "SH2 ": per channel a filter per
//!    ear, fitted so a plane wave decodes to its HRTF — least squares low,
//!    magnitude least squares high; `tools/make_hrir_sofa.py`) or via
//!    virtual speakers (icosahedron/dodecahedron) and a measured HRIR per
//!    speaker. Fixed cost, order-limited sharpness: right for reflections
//!    + reverb.
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
        if bytes.starts_with(b"SH2 ") {
            return Self::from_sh(bytes);
        }
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

    /// An SH decoder file: each channel a "speaker" taking just that
    /// channel, its filters the two "HRIRs" — the convolution as for
    /// speakers. Its gain is the HRTFs' own (a plane wave decodes to its
    /// HRTF): no loudness match to tune.
    fn from_sh(bytes: &[u8]) -> Self {
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let (channels, taps) = (u32_at(4), u32_at(8));
        let f32s = |o: usize| -> Vec<f32> {
            bytes[o..o + 4 * taps].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        };
        let speakers = (0..channels.min(NCH))
            .map(|n| {
                let o = 12 + n * 8 * taps;
                let (mut hrir_l, mut hrir_r) = (f32s(o), f32s(o + 4 * taps));
                // stored reversed: see convolve_ring
                hrir_l.reverse();
                hrir_r.reverse();
                let mut matrix = [0.0f32; NCH];
                matrix[n] = 1.0;
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

/// Dot products of `hist` against two kernels at once (over the shortest
/// of the three). Float reductions don't auto-vectorize under strict FP
/// semantics — and the spelled-out 4-lane form below did not either on
/// aarch64 — so NEON and wasm simd128 get explicit kernels: 8 lanes per
/// kernel in two accumulators each, so consecutive adds overlap.
#[inline]
fn dot2(hist: &[f32], kl: &[f32], kr: &[f32]) -> (f32, f32) {
    let n = hist.len().min(kl.len()).min(kr.len());
    let (hist, kl, kr) = (&hist[..n], &kl[..n], &kr[..n]);
    #[cfg(target_arch = "aarch64")]
    {
        dot2_neon(hist, kl, kr)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        dot2_simd128(hist, kl, kr)
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        dot2_portable(hist, kl, kr)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot2_neon(hist: &[f32], kl: &[f32], kr: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::*;
    let n = hist.len();
    let n8 = n & !7;
    // SAFETY: NEON is baseline on aarch64; every load reads [i, i + 8)
    // with i + 8 ≤ n8 ≤ n, the common length of all three slices.
    let (mut sl, mut sr) = unsafe {
        let (h, l, r) = (hist.as_ptr(), kl.as_ptr(), kr.as_ptr());
        let z = vdupq_n_f32(0.0);
        let (mut l0, mut l1, mut r0, mut r1) = (z, z, z, z);
        let mut i = 0;
        while i < n8 {
            let h0 = vld1q_f32(h.add(i));
            let h1 = vld1q_f32(h.add(i + 4));
            l0 = vfmaq_f32(l0, h0, vld1q_f32(l.add(i)));
            l1 = vfmaq_f32(l1, h1, vld1q_f32(l.add(i + 4)));
            r0 = vfmaq_f32(r0, h0, vld1q_f32(r.add(i)));
            r1 = vfmaq_f32(r1, h1, vld1q_f32(r.add(i + 4)));
            i += 8;
        }
        (vaddvq_f32(vaddq_f32(l0, l1)), vaddvq_f32(vaddq_f32(r0, r1)))
    };
    for i in n8..n {
        sl += hist[i] * kl[i];
        sr += hist[i] * kr[i];
    }
    (sl, sr)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
fn dot2_simd128(hist: &[f32], kl: &[f32], kr: &[f32]) -> (f32, f32) {
    use core::arch::wasm32::*;
    let n = hist.len();
    let n8 = n & !7;
    let sum = |v: v128| (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v)) + (f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v));
    // SAFETY: simd128 is enabled for this build; every load reads
    // [i, i + 8) with i + 8 ≤ n8 ≤ n, the common length of all three
    // slices (v128_load has no alignment requirement).
    let (mut sl, mut sr) = unsafe {
        let (h, l, r) = (hist.as_ptr(), kl.as_ptr(), kr.as_ptr());
        let z = f32x4_splat(0.0);
        let (mut l0, mut l1, mut r0, mut r1) = (z, z, z, z);
        let mut i = 0;
        while i < n8 {
            let h0 = v128_load(h.add(i) as *const v128);
            let h1 = v128_load(h.add(i + 4) as *const v128);
            l0 = f32x4_add(l0, f32x4_mul(h0, v128_load(l.add(i) as *const v128)));
            l1 = f32x4_add(l1, f32x4_mul(h1, v128_load(l.add(i + 4) as *const v128)));
            r0 = f32x4_add(r0, f32x4_mul(h0, v128_load(r.add(i) as *const v128)));
            r1 = f32x4_add(r1, f32x4_mul(h1, v128_load(r.add(i + 4) as *const v128)));
            i += 8;
        }
        (sum(f32x4_add(l0, l1)), sum(f32x4_add(r0, r1)))
    };
    for i in n8..n {
        sl += hist[i] * kl[i];
        sr += hist[i] * kr[i];
    }
    (sl, sr)
}

#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
#[inline]
fn dot2_portable(hist: &[f32], kl: &[f32], kr: &[f32]) -> (f32, f32) {
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

    /// The HRIR pair at `index` (left, right; stored time-reversed — the
    /// order the convolution reads them in).
    pub fn hrir(&self, index: usize) -> (&[f32], &[f32]) {
        (&self.hrir_l[index], &self.hrir_r[index])
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

/// Longest history + block `PointConv::process_block` handles in one
/// linear buffer (128-tap HRIRs, 128-sample blocks).
const CONV_SPAN: usize = 512;

/// Per-emitter nearest-HRIR convolver with ~10 ms crossfade on index
/// change. A change asked for during a fade waits for it to finish (the
/// latest one wins): starting over mid-fade dropped the half-faded filter
/// at once — a click every time a fast turn (a flick of the view, a source
/// swept past close by) changed the HRIR within 10 ms.
pub struct PointConv {
    ring: Vec<f32>,
    pos: usize,
    cur: usize,
    prev: usize,
    /// The direction's HRIR asked for last: faded to when the fade under
    /// way has finished.
    next: usize,
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
            next: 0,
            xfade: 0.0,
            xstep: 1.0 / (0.010 * sample_rate),
        }
    }

    pub fn reset(&mut self, idx: usize) {
        self.ring.iter_mut().for_each(|s| *s = 0.0);
        self.cur = idx;
        self.prev = idx;
        self.next = idx;
        self.xfade = 0.0;
    }

    pub fn set_dir(&mut self, grid: &HrirGrid, dir: [f32; 3]) {
        self.next = grid.nearest(dir);
        self.advance();
    }

    /// No fade under way and another HRIR asked for: fade to it.
    #[inline]
    fn advance(&mut self) {
        if self.xfade <= 0.0 && self.next != self.cur {
            self.prev = self.cur;
            self.cur = self.next;
            self.xfade = 1.0;
        }
    }

    /// `x.len()` samples at once, ADDED onto `l` and `r`: what `process`
    /// returns sample by sample (up to float rounding). The history and
    /// the block sit in one linear buffer, so every output is one
    /// contiguous dot product (no ring split).
    pub fn process_block(&mut self, grid: &HrirGrid, x: &[f32], l: &mut [f32], r: &mut [f32]) {
        let n = self.ring.len();
        let m = x.len().min(l.len()).min(r.len());
        if n + m > CONV_SPAN || n != grid.taps {
            for j in 0..m {
                let (a, b) = self.process(grid, x[j]);
                l[j] += a;
                r[j] += b;
            }
            return;
        }
        // buf = history oldest → newest, then the block
        let mut buf = [0.0f32; CONV_SPAN];
        let old = (self.pos + 1) % n;
        buf[..n - old].copy_from_slice(&self.ring[old..]);
        buf[n - old..n].copy_from_slice(&self.ring[..old]);
        buf[n..n + m].copy_from_slice(&x[..m]);
        for j in 0..m {
            // a change asked for during a fade starts where it ends (as
            // `process` does, sample for sample)
            self.advance();
            // window ending at sample j (newest last), kernels pre-reversed
            let w = &buf[j + 1..j + 1 + n];
            let (mut a, mut b) = dot2(w, &grid.hrir_l[self.cur], &grid.hrir_r[self.cur]);
            if self.xfade > 0.0 {
                let (pa, pb) = dot2(w, &grid.hrir_l[self.prev], &grid.hrir_r[self.prev]);
                let f = self.xfade;
                self.xfade = (self.xfade - self.xstep).max(0.0);
                a += f * (pa - a);
                b += f * (pb - b);
            }
            l[j] += a;
            r[j] += b;
        }
        // the ring keeps the newest n, oldest first
        self.ring.copy_from_slice(&buf[m..m + n]);
        self.pos = n - 1;
    }

    #[inline]
    pub fn process(&mut self, grid: &HrirGrid, s: f32) -> (f32, f32) {
        self.advance();
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

    /// A path turning fast round the head (a flick of the view, a source
    /// swept past close by) changes its nearest HRIR more often than the
    /// 10 ms crossfade lasts: every change must still fade — a pure tone
    /// through it stays free of clicks (through any one filter it is a
    /// pure tone; only a step between filters makes treble).
    #[test]
    fn a_fast_turn_crossfades_every_hrir_change() {
        let grid = HrirGrid::from_bytes(include_bytes!("../../omg-audio/hrir/hrir_grid.bin"));
        let sr = 48_000.0f32;
        let mut conv = PointConv::new(grid.taps, sr);
        let n = (0.5 * sr) as usize;
        let block = 128usize;
        let deg_s = 1500.0f32;
        let (mut l, mut r) = (vec![0.0f32; n], vec![0.0f32; n]);
        let x: Vec<f32> = (0..n).map(|k| (std::f32::consts::TAU * 500.0 * k as f32 / sr).sin()).collect();
        for b in (0..n).step_by(block) {
            let a = (deg_s * b as f32 / sr).to_radians();
            conv.set_dir(&grid, [a.cos(), a.sin(), 0.0]);
            let e = (b + block).min(n);
            conv.process_block(&grid, &x[b..e], &mut l[b..e], &mut r[b..e]);
        }
        // the treble (a triple difference: 500 Hz −54 dB under what it
        // passes at 4 kHz) under the whole, per 5 ms
        let w = (0.005 * sr) as usize;
        let mut worst = f32::MIN;
        for ch in [&l, &r] {
            for k in (w..n - w).step_by(w) {
                let hf: f32 = (k..k + w).map(|i| (ch[i] - 3.0 * ch[i - 1] + 3.0 * ch[i - 2] - ch[i - 3]).powi(2)).sum();
                let all: f32 = (k..k + w).map(|i| ch[i] * ch[i]).sum();
                worst = worst.max(10.0 * (hf / all.max(1e-30)).max(1e-30).log10());
            }
        }
        eprintln!("turning {deg_s}°/s: treble at worst {worst:.1} dB under the tone");
        assert!(worst < -50.0, "a step between HRIRs: treble {worst:.1} dB under the tone");
    }
}
