//! Shared measurement helpers for the isolated per-effect tests
//! (`tests/fx_*.rs`). Signal generators, level/frequency/decay
//! estimators and synthetic HRIR byte builders. Deterministic only:
//! every noise source takes an explicit seed.
#![allow(dead_code)]

use omg_core::rng::Rng;
use std::f32::consts::TAU;

pub const SR: f32 = 48_000.0;

// ------------------------------------------------------------ generators

/// `n` samples of a sine at `freq` Hz, amplitude `amp`.
pub fn sine(freq: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n).map(|k| amp * (TAU * freq * k as f32 / SR).sin()).collect()
}

/// Sample `k` of a sine at `freq` Hz (phase-continuous by index; f64
/// phase so long runs stay exact).
pub fn sine_at(freq: f32, k: usize) -> f32 {
    ((std::f64::consts::TAU * freq as f64 * k as f64 / SR as f64).sin()) as f32
}

/// Uniform white noise in [-amp, amp) with a fixed seed.
pub fn noise(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| amp * (rng.next_f32() * 2.0 - 1.0)).collect()
}

/// Streaming white-noise source (same distribution as [`noise`]).
pub struct Noise(Rng);
impl Noise {
    pub fn new(seed: u64) -> Self {
        Noise(Rng::new(seed))
    }
    pub fn next(&mut self) -> f32 {
        self.0.next_f32() * 2.0 - 1.0
    }
}

// ------------------------------------------------------------ levels

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    ((x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()) / x.len() as f64).sqrt() as f32
}

pub fn energy(x: &[f32]) -> f64 {
    x.iter().map(|v| (*v as f64) * (*v as f64)).sum()
}

pub fn db(a: f32) -> f32 {
    20.0 * a.max(1e-12).log10()
}

pub fn db_pow(p: f64) -> f64 {
    10.0 * p.max(1e-30).log10()
}

/// Largest sample-to-sample step |x[n] − x[n−1]|.
pub fn max_step(x: &[f32]) -> f32 {
    x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
}

/// Energy of the second difference — a crude broadband-transient
/// detector: a click (step or kink) lights it up, a smooth low tone
/// barely registers.
pub fn d2_energy(x: &[f32]) -> f64 {
    x.windows(3)
        .map(|w| {
            let d = (w[2] - 2.0 * w[1] + w[0]) as f64;
            d * d
        })
        .sum()
}

// ------------------------------------------------------------ frequency

/// Single-bin DFT amplitude of `x` at `freq` (2|X|/N): the amplitude of
/// a steady sinusoid at that frequency (exact over whole cycles).
pub fn tone_amp(x: &[f32], freq: f32) -> f32 {
    let w = std::f64::consts::TAU * freq as f64 / SR as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, v) in x.iter().enumerate() {
        let ph = w * n as f64;
        re += *v as f64 * ph.cos();
        im -= *v as f64 * ph.sin();
    }
    (2.0 * (re * re + im * im).sqrt() / x.len() as f64) as f32
}

/// Frequency from linearly interpolated rising zero crossings: (count−1)
/// periods between the first and last crossing. Exact for a clean tone,
/// and the right "instantaneous pitch" average for a glide.
pub fn zc_freq(x: &[f32]) -> f32 {
    let mut first = None;
    let mut last = 0.0f64;
    let mut count = 0usize;
    for n in 1..x.len() {
        let (a, b) = (x[n - 1], x[n]);
        if a < 0.0 && b >= 0.0 {
            let t = (n - 1) as f64 + (-a / (b - a)) as f64;
            if first.is_none() {
                first = Some(t);
            }
            last = t;
            count += 1;
        }
    }
    match first {
        Some(f) if count >= 2 => ((count - 1) as f64 / (last - f) * SR as f64) as f32,
        _ => 0.0,
    }
}

// ------------------------------------------------------------ filters

/// RBJ biquad (bandpass, constant 0 dB peak gain).
#[derive(Clone, Copy)]
pub struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    z: [f64; 2],
}

impl Biquad {
    pub fn bandpass(fc: f32, q: f32) -> Self {
        let w0 = std::f64::consts::TAU * fc as f64 / SR as f64;
        let alpha = w0.sin() / (2.0 * q as f64);
        let a0 = 1.0 + alpha;
        Self {
            b: [alpha / a0, 0.0, -alpha / a0],
            a: [-2.0 * w0.cos() / a0, (1.0 - alpha) / a0],
            z: [0.0; 2],
        }
    }

    pub fn tick(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y as f32
    }

    pub fn run(mut self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|v| self.tick(*v)).collect()
    }
}

// ------------------------------------------------------------ decay

/// T60 from a squared impulse/decay response via Schroeder backward
/// integration: least-squares slope of the energy-decay curve between
/// `hi_db` and `lo_db` (e.g. −5 → −25 dB = T20), extrapolated to 60 dB.
pub fn schroeder_t60(e: &[f64], hi_db: f64, lo_db: f64) -> f32 {
    let mut edc = vec![0.0f64; e.len()];
    let mut acc = 0.0;
    for n in (0..e.len()).rev() {
        acc += e[n];
        edc[n] = acc;
    }
    let e0 = edc[0].max(1e-300);
    let (mut sx, mut sy, mut sxx, mut sxy, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (n, v) in edc.iter().enumerate() {
        let d = 10.0 * (v / e0).max(1e-300).log10();
        if d <= hi_db && d >= lo_db {
            let t = n as f64 / SR as f64;
            sx += t;
            sy += d;
            sxx += t * t;
            sxy += t * d;
            cnt += 1.0;
        }
    }
    assert!(cnt > 10.0, "EDC never spans {hi_db}..{lo_db} dB");
    let slope = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx); // dB/s
    (-60.0 / slope) as f32
}

// ------------------------------------------------------------ correlation

/// Lag (in samples, within ±max_lag) maximizing Σ a[n]·b[n+lag].
/// Positive lag ⇒ `b` is a delayed copy of `a` (a leads).
pub fn xcorr_peak_lag(a: &[f32], b: &[f32], max_lag: i32) -> i32 {
    let n = a.len().min(b.len()) as i32;
    let mut best = (f64::MIN, 0);
    for lag in -max_lag..=max_lag {
        let mut s = 0.0f64;
        for i in max_lag..(n - max_lag) {
            s += a[i as usize] as f64 * b[(i + lag) as usize] as f64;
        }
        if s > best.0 {
            best = (s, lag);
        }
    }
    best.1
}

// ------------------------------------------------------------ HRIR bytes

/// One HRIR record: direction (x fwd, y left, z up) + left/right kernels.
pub struct HrirRec {
    pub dir: [f32; 3],
    pub l: Vec<f32>,
    pub r: Vec<f32>,
}

/// Pack records in the asset format `hrtf::parse_bin` reads (flat
/// little-endian: u32 count, u32 taps, then per record dir[3], l[taps],
/// r[taps] as f32).
pub fn hrir_bytes(taps: usize, recs: &[HrirRec]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(taps as u32).to_le_bytes());
    for r in recs {
        assert_eq!(r.l.len(), taps);
        assert_eq!(r.r.len(), taps);
        for v in r.dir.iter().chain(r.l.iter()).chain(r.r.iter()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// A kernel of `taps` zeros with `gain` at sample `at`.
pub fn impulse(taps: usize, at: usize, gain: f32) -> Vec<f32> {
    let mut h = vec![0.0; taps];
    h[at] = gain;
    h
}

/// Direct linear convolution, output length = x.len() (causal, truncated).
pub fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    (0..x.len())
        .map(|n| {
            let mut s = 0.0f32;
            for (k, hk) in h.iter().enumerate() {
                if k > n {
                    break;
                }
                s += hk * x[n - k];
            }
            s
        })
        .collect()
}

pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// The 12 vertices of an icosahedron (a spherical 5-design — enough for
/// a clean order-2 decode).
pub fn icosahedron() -> Vec<[f32; 3]> {
    let p = (1.0 + 5f32.sqrt()) / 2.0;
    let mut v = Vec::new();
    for &a in &[-1.0f32, 1.0] {
        for &b in &[-p, p] {
            v.push(normalize([0.0, a, b]));
            v.push(normalize([a, b, 0.0]));
            v.push(normalize([b, 0.0, a]));
        }
    }
    v
}
