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
    let mut u32_at = |o: &mut usize| {
        let v = u32::from_le_bytes(bytes[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let count = u32_at(&mut off) as usize;
    let taps = u32_at(&mut off) as usize;
    let mut f32s = |o: &mut usize, n: usize| -> Vec<f32> {
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
            .map(|(d, hrir_l, hrir_r)| {
                let enc = encode_gains(d);
                let mut matrix = [0.0f32; NCH];
                for (n, m) in matrix.iter_mut().enumerate() {
                    let l = if n == 0 { 0 } else if n < 4 { 1 } else { 2 };
                    *m = (2 * l + 1) as f32 / k * MAXRE[l] * enc[n] * MASTER;
                }
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

#[inline]
fn convolve_ring(hist: &[f32], pos: usize, hl: &[f32], hr: &[f32]) -> (f32, f32) {
    let taps = hist.len();
    let (mut al, mut ar) = (0.0f32, 0.0f32);
    let mut j = pos;
    for i in 0..taps {
        let h = hist[j];
        al += h * hl[i];
        ar += h * hr[i];
        j = if j == 0 { taps - 1 } else { j - 1 };
    }
    (al, ar)
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
        for (d, l, r) in records {
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
