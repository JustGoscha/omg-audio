//! 8-line feedback delay network (Jot-style) for the late field.
//! Per-line gain sets the mid-band RT60; a one-pole lowpass in each loop
//! makes high frequencies decay faster, matched to the traced rt60[high].
//! An FDN is preferred over convolving a traced impulse response because
//! its parameters can change smoothly in real time with zero artifacts.

use crate::delay::DelayLine;
use crate::filter::OnePoleLp;
use crate::smooth::Smoothed;

pub const NLINES: usize = 8;

/// Mutually prime-ish lengths, ~23–81 ms at 48 kHz, scaled for actual rate.
const LEN_MS: [f32; NLINES] = [23.4, 32.5, 39.7, 46.9, 55.9, 62.9, 72.0, 81.4];
/// A second network's (`Fdn::second`): other lengths, and injection signs
/// orthogonal to the first's — two networks fed the same input play two
/// uncorrelated fields. (Identical networks play copies: summed in the
/// bus they add coherently, the one-sided share of a field heard +3 dB.)
const LEN_MS_B: [f32; NLINES] = [25.1, 29.9, 37.3, 43.9, 51.7, 59.3, 67.9, 77.3];
/// Band centres the loop filter is fitted at (omg-dsp bands: 125 / 790 /
/// 5600 Hz).
const MID_HZ: f32 = 790.0;
const HIGH_HZ: f32 = 5_600.0;

/// The bands' upper crossover (omg-dsp bands: 2.1 kHz).
const XOVER_HZ: f32 = 2_100.0;

/// |H| at `f` of the loop's high shelf `y = lp + s·(x − lp)`, `lp` a
/// one-pole lowpass with coefficient `c`: 1 at DC, `s` towards Nyquist.
fn shelf_gain(s: f32, c: f32, f: f32, sample_rate: f32) -> f32 {
    // H = s + (1 − s)·Hlp, Hlp = c / (1 − (1 − c)·e^{−jw})
    let w = core::f32::consts::TAU * f / sample_rate;
    let a = 1.0 - c;
    let (dr, di) = (1.0 - a * w.cos(), a * w.sin());
    let d2 = dr * dr + di * di;
    let (hr, hi) = (c * dr / d2, -c * di / d2);
    let (re, im) = (s + (1.0 - s) * hr, (1.0 - s) * hi);
    (re * re + im * im).sqrt()
}

/// The shelf gain whose loss at the high centre over the mid centre is
/// `ratio` (bisection: that loss grows as the shelf goes down).
fn shelf_for(ratio: f32, c: f32, sample_rate: f32) -> f32 {
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..40 {
        let s = 0.5 * (lo + hi);
        if shelf_gain(s, c, HIGH_HZ, sample_rate) / shelf_gain(s, c, MID_HZ, sample_rate) > ratio {
            hi = s;
        } else {
            lo = s;
        }
    }
    0.5 * (lo + hi)
}

/// The decay that makes the network's output arrive, on average, `ts`
/// seconds after its input — the energy's centre time (its first moment).
/// The lines hand their content on every L̄ (their mean length) keeping
/// ḡ² of it: pass n's energy ∝ ḡ²ⁿ at n·L̄, so the centre time is
/// L̄ / (1 − ḡ²), and ḡ² = 10^(−6·L̄/rt60). A response arriving sooner
/// on average than the network can (its first pass): the fastest decay.
pub fn rt60_for_centre_time(ts: f32) -> f32 {
    let lbar = LEN_MS.iter().sum::<f32>() / NLINES as f32 * 1e-3;
    let q = 1.0 - lbar / ts.max(1e-6);
    if q <= 1e-6 {
        return MIN_RT60;
    }
    (-6.0 * lbar / q.log10()).max(MIN_RT60)
}

/// The fastest decay `set_rt60` plays.
const MIN_RT60: f32 = 0.05;

/// Frequencies the band averages are taken at (log-spaced, 20 Hz to
/// Nyquist).
const BAND_POINTS: usize = 48;

/// The tracer's bands (`bands::BandSplit`) at log-spaced points: per
/// point its width Δf and each band's complex response, and the
/// frequency.
#[derive(Clone, Copy)]
struct BandPoint {
    f: f32,
    df: f32,
    s: [(f32, f32); 3],
}

fn band_points(sample_rate: f32) -> [BandPoint; BAND_POINTS] {
    let (f0, f1) = (20.0f32, 0.5 * sample_rate);
    let at = |k: f32| f0 * (f1 / f0).powf(k / (BAND_POINTS - 1) as f32);
    core::array::from_fn(|k| {
        let f = at(k as f32);
        let df = at(k as f32 + 0.5) - at(k as f32 - 0.5);
        BandPoint { f, df, s: crate::bands::responses(sample_rate, f.min(0.499 * sample_rate)) }
    })
}

/// Input share written into each line (`energy_norm` accounts for it),
/// with alternating signs: the lines start incoherent, as a diffuse field
/// is — the same sign everywhere would sum coherently in the decode.
const INJECT: f32 = 0.35;
const INJECT_SIGN: [f32; NLINES] = [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
const INJECT_SIGN_B: [f32; NLINES] = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

pub struct Fdn {
    /// The input's diffusion (`Diffuser`) before the lines.
    diffuser: Diffuser,
    lines: Vec<DelayLine>,
    lens: [f32; NLINES],
    damps: [OnePoleLp; NLINES],
    gains: [Smoothed; NLINES],
    /// Per line: the loop's high-shelf gain (1 = no filter).
    damp_coef: [Smoothed; NLINES],
    /// The shelf's crossover lowpass coefficient (`XOVER_HZ`).
    xover: f32,
    sample_rate: f32,
    /// The tracer's bands at the points `band_make_up` integrates over.
    bands: [BandPoint; BAND_POINTS],
    /// Per line: the sign its input is written with.
    sign: [f32; NLINES],
}

impl Fdn {
    pub fn new(sample_rate: f32) -> Self {
        Self::with(sample_rate, LEN_MS, INJECT_SIGN)
    }

    /// A network uncorrelated with `new`'s, fed the same input (the
    /// renderer's one-sided field beside its diffuse one).
    pub fn second(sample_rate: f32) -> Self {
        Self::with(sample_rate, LEN_MS_B, INJECT_SIGN_B)
    }

    /// `new` (or `second`) of its own: line lengths moved by up to ±6 %
    /// and injection signs flipped, both by `seed` — networks of different
    /// seeds fed the same input play uncorrelated fields, as the
    /// reverberant fields of two sources playing one signal are (and as
    /// their kernels, of different noise seeds, are). Identical networks
    /// play copies, which add coherently: four speakers of one club, each
    /// a source with the same network, came out up to 6 dB too loud.
    /// Seed 0: `new` / `second` themselves.
    pub fn seeded(sample_rate: f32, seed: u32, second: bool) -> Self {
        let (len, sign) = if second { (LEN_MS_B, INJECT_SIGN_B) } else { (LEN_MS, INJECT_SIGN) };
        if seed == 0 {
            return Self::with(sample_rate, len, sign);
        }
        let mut h = seed.wrapping_mul(0x9e37_79b9) ^ if second { 0x85eb_ca6b } else { 0 };
        let mut next = || {
            h ^= h << 13;
            h ^= h >> 17;
            h ^= h << 5;
            h as f32 / u32::MAX as f32
        };
        let len = core::array::from_fn(|i| len[i] * (1.0 + 0.12 * (next() - 0.5)));
        let sign = core::array::from_fn(|i| if next() < 0.5 { -sign[i] } else { sign[i] });
        Self::with(sample_rate, len, sign)
    }

    fn with(sample_rate: f32, len_ms: [f32; NLINES], sign: [f32; NLINES]) -> Self {
        // whole samples: a fractional read interpolates, and linear
        // interpolation lowpasses every pass of the loop
        let lens: [f32; NLINES] = core::array::from_fn(|i| (len_ms[i] * 1e-3 * sample_rate).round());
        Self {
            diffuser: Diffuser::new(sample_rate),
            lines: (0..NLINES)
                .map(|i| DelayLine::new(lens[i] as usize + 4))
                .collect(),
            lens,
            damps: [OnePoleLp::default(); NLINES],
            gains: [Smoothed::new(0.7, 0.1, sample_rate); NLINES],
            damp_coef: [Smoothed::new(1.0, 0.15, sample_rate); NLINES],
            xover: OnePoleLp::coef(XOVER_HZ, sample_rate),
            sample_rate,
            bands: band_points(sample_rate),
            sign,
        }
    }

    /// Drop everything ringing (the field it held no longer reaches the
    /// listener; see the renderer's output-scaled late level).
    pub fn clear(&mut self) {
        for l in &mut self.lines {
            l.clear();
        }
        self.damps = [OnePoleLp::default(); NLINES];
        self.diffuser.clear();
    }

    /// Jump the decay gains and loop filters to their targets — for a
    /// network holding nothing (asleep): its smoothers only move while it
    /// runs, and the next sound's tail must start at the decay set now,
    /// not glide there from whatever was set before.
    pub fn settle(&mut self) {
        for g in self.gains.iter_mut().chain(self.damp_coef.iter_mut()) {
            if g.current() != g.target_val() {
                g.snap(g.target_val());
            }
        }
        // asleep, nothing rings in the diffuser either (what is left is
        // under the idle floor)
        self.diffuser.clear();
    }

    /// Configure decay from traced per-band RT60 (bands: low/mid/high).
    /// Each line's loop holds a first-order high shelf at the bands' upper
    /// crossover — the high band decays at its own rate, as traced —
    /// whose loss at the high band's centre against the mid band's is
    /// exactly g_high / g_mid per pass; the line gain makes up its loss at
    /// the mid centre, so the mids decay at `rt60_mid` exactly.
    pub fn set_rt60(&mut self, rt60_mid: f32, rt60_high: f32) {
        for i in 0..NLINES {
            let len_s = self.lens[i] / self.sample_rate;
            let g_mid = 10f32.powf(-3.0 * len_s / rt60_mid.max(MIN_RT60));
            let g_high = 10f32.powf(-3.0 * len_s / rt60_high.max(MIN_RT60));
            let ratio = (g_high / g_mid).clamp(0.0, 1.0);
            let shelf = if ratio >= 0.999 { 1.0 } else { shelf_for(ratio, self.xover, self.sample_rate) };
            let make_up = shelf_gain(shelf, self.xover, MID_HZ, self.sample_rate);
            self.gains[i].set((g_mid / make_up).min(0.9999));
            self.damp_coef[i].set(shelf);
        }
    }

    /// Samples of silent input and output after which the network holds
    /// nothing any more: its longest line has run through. Sooner, a
    /// short sound (a clap) can still be in the lines with nothing come
    /// out yet — sleeping then would drop its reverb.
    pub fn idle_after(&self) -> u32 {
        self.lens.iter().fold(0.0f32, |a, &b| a.max(b)).ceil() as u32 + 1 + self.diffuser.longest()
    }

    /// The factor that makes the lines' output energy for a unit impulse
    /// exactly 1 at the decay set by `set_rt60` (mid band): an impulse
    /// puts `NLINES · INJECT²` into the lines; the mixing is orthogonal, so
    /// each pass keeps the mean of the lines' mid-band gains² (ḡ², the
    /// loop shelf included) of it, and the outputs sum to that ·
    /// ḡ² / (1 − ḡ²). A level `L` times it plays energy L² — at any RT60,
    /// like a path of gain L.
    ///
    /// From the decay the lines have NOW (their smoothed gains): a new
    /// RT60 glides in, and so does the normalization with it. Applied to
    /// the input, it scales what goes in; what already rings keeps its
    /// energy and decays at the new rate — a decay that changes never
    /// rescales the tail in place (a step at every new RT60).
    pub fn energy_norm(&self) -> f32 {
        let g2 = self
            .gains
            .iter()
            .zip(&self.damp_coef)
            .map(|(g, s)| {
                let g = g.current() * shelf_gain(s.current(), self.xover, MID_HZ, self.sample_rate);
                g * g
            })
            .sum::<f32>()
            / NLINES as f32;
        let e = NLINES as f32 * INJECT * INJECT * g2 / (1.0 - g2).max(1e-6);
        1.0 / e.max(1e-12).sqrt()
    }

    /// The sends that make the network play each band (the tracer's
    /// one-pole bands, as a click shaped at `level` per band holds them —
    /// the kernel's own synthesis) with that band's energy: per band a
    /// factor on its send, the mid's on the level. The tail's energy at a
    /// frequency grows with its decay there (ḡ² / (1 − ḡ²), the loop
    /// shelf included; the level is calibrated at the mid centre,
    /// `energy_norm`), the bands overlap — a band's skirt lands where
    /// another decays — so the three are solved together: each band's
    /// energy integrated over frequency, the factors iterated until every
    /// band holds what it is given. From the targets `set_rt60` left.
    pub fn band_make_up(&self, level: [f32; 3]) -> [f32; 3] {
        // the lines mix every pass: the network keeps the mean ḡ² of
        // what it holds per pass (as `energy_norm` counts it)
        let tail = |f: f32| -> f32 {
            let g2 = self
                .gains
                .iter()
                .zip(&self.damp_coef)
                .map(|(g, s)| {
                    let g = g.target_val() * shelf_gain(s.target_val(), self.xover, f, self.sample_rate);
                    g * g
                })
                .sum::<f32>()
                / NLINES as f32;
            g2 / (1.0 - g2).max(1e-6)
        };
        let mid = tail(MID_HZ).max(1e-12);
        let rel: [f32; BAND_POINTS] = core::array::from_fn(|k| tail(self.bands[k].f) / mid);
        // band b's energy of a click shaped at gains `a`, through the
        // network's tail (`t`: true) or dry (the target)
        let energy = |a: [f32; 3], t: bool| -> [f32; 3] {
            let mut e = [0.0f32; 3];
            for (k, p) in self.bands.iter().enumerate() {
                let (mut re, mut im) = (0.0f32, 0.0f32);
                for c in 0..3 {
                    re += a[c] * p.s[c].0;
                    im += a[c] * p.s[c].1;
                }
                let y = (re * re + im * im) * p.df * if t { rel[k] } else { 1.0 };
                for (b, eb) in e.iter_mut().enumerate() {
                    *eb += y * (p.s[b].0 * p.s[b].0 + p.s[b].1 * p.s[b].1);
                }
            }
            e
        };
        let want = energy(level, false);
        let mut m = [1.0f32; 3];
        for _ in 0..8 {
            let got = energy(core::array::from_fn(|c| level[c] * m[c]), true);
            for b in 0..3 {
                if want[b] > 0.0 && got[b] > 0.0 {
                    m[b] *= (want[b] / got[b]).sqrt();
                }
            }
        }
        m
    }

    /// Most samples `process_block` takes at once: a little under the
    /// shortest loop, so a block's reads (two neighbours each) all come
    /// from before its writes.
    pub fn max_block(&self) -> usize {
        self.lens.iter().fold(f32::MAX, |a, &b| a.min(b)) as usize - 2
    }

    /// `input.len()` (≤ `max_block()`, ≤ N) samples at once: `outs[i][j]`
    /// is line i's output at sample j — what `process` returns sample by
    /// sample. Reads for the whole block first (they precede its writes),
    /// the eight damping loops side by side, the mixing and writes after.
    pub fn process_block<const N: usize>(&mut self, input: &[f32], outs: &mut [[f32; N]; NLINES]) {
        let m = input.len().min(N).min(self.max_block());
        for (i, o) in outs.iter_mut().enumerate() {
            self.lines[i].read_ahead_block(self.lens[i], &mut o[..m]);
        }
        let mut damp: [f32; NLINES] = core::array::from_fn(|i| self.damps[i].state);
        // the loop filters' and gains' smoothers as plain lanes (`tick`
        // inlined), so the eight loops vectorize side by side
        let lanes = |g: &[Smoothed; NLINES]| -> [[f32; NLINES]; 3] {
            [
                core::array::from_fn(|i| g[i].current()),
                core::array::from_fn(|i| g[i].target_val()),
                core::array::from_fn(|i| g[i].coef()),
            ]
        };
        let [mut dc, dt, dk] = lanes(&self.damp_coef);
        let [mut gc, gt, gk] = lanes(&self.gains);
        let xc = self.xover;
        for j in 0..m {
            for i in 0..NLINES {
                dc[i] = dt[i] + (dc[i] - dt[i]) * dk[i];
                gc[i] = gt[i] + (gc[i] - gt[i]) * gk[i];
                // the high shelf: the crossover's lowpass, the rest × shelf
                let x = outs[i][j];
                damp[i] += xc * (x - damp[i]);
                outs[i][j] = (damp[i] + dc[i] * (x - damp[i])) * gc[i];
            }
        }
        for i in 0..NLINES {
            self.damp_coef[i].set_current(dc[i]);
            self.gains[i].set_current(gc[i]);
            self.damps[i].state = damp[i];
        }
        // 8×8 Hadamard per sample (as in `process`), vectorized over samples
        let mut v = *outs;
        for stride in [1usize, 2, 4] {
            for lo in 0..NLINES {
                if lo & stride != 0 {
                    continue;
                }
                let hi = lo | stride;
                for j in 0..m {
                    let (a, b) = (v[lo][j], v[hi][j]);
                    v[lo][j] = a + b;
                    v[hi][j] = a - b;
                }
            }
        }
        let norm = 1.0 / (NLINES as f32).sqrt();
        let mut dx = [0.0f32; N];
        for (d, &x) in dx[..m].iter_mut().zip(&input[..m]) {
            *d = self.diffuser.tick(x);
        }
        for (i, v) in v.iter_mut().enumerate() {
            for (v, x) in v[..m].iter_mut().zip(&dx[..m]) {
                *v = x * INJECT * self.sign[i] + *v * norm;
            }
            self.lines[i].write_block(&v[..m]);
        }
    }

    /// One sample in → 8 decorrelated line outputs (for spatial spread).
    #[inline]
    pub fn process(&mut self, input: f32, outs: &mut [f32; NLINES]) {
        for i in 0..NLINES {
            let raw = self.lines[i].read(self.lens[i]);
            let lp = self.damps[i].tick(raw, self.xover);
            let shelved = lp + self.damp_coef[i].tick() * (raw - lp);
            outs[i] = shelved * self.gains[i].tick();
        }
        // Fast 8×8 Hadamard (butterfly), normalized → energy-preserving.
        let mut v = *outs;
        for stride in [1usize, 2, 4] {
            let mut w = [0.0f32; NLINES];
            for i in 0..NLINES {
                let j = i ^ stride;
                w[i] = if i & stride == 0 { v[i] + v[j] } else { v[j] - v[i] };
            }
            v = w;
        }
        let norm = 1.0 / (NLINES as f32).sqrt();
        let input = self.diffuser.tick(input);
        for i in 0..NLINES {
            self.lines[i].write(input * INJECT * self.sign[i] + v[i] * norm);
        }
    }
}

/// The network's input diffused before the lines: four Schroeder
/// allpasses in series (Dattorro's input diffusion — 4.8, 3.6, 12.7 and
/// 9.3 ms at gains 0.75, 0.75, 0.625, 0.625). Lossless and flat in
/// spectrum, they only spread each arrival in time: injected bare, an
/// impulse came out of the eight lines as eight impulses over the first
/// 80 ms, 64 over the next pass — a late field far sparser than a room's
/// diffuse one at that time, heard in the treble as a crackling, spiky
/// tail (kurtosis ~77 at a clap in the cathedral, where the traced
/// response convolved has ~10; noise is 3).
struct Diffuser {
    stages: [Allpass; 4],
}

const DIFFUSER_MS: [f32; 4] = [4.77, 3.60, 12.73, 9.30];
const DIFFUSER_G: [f32; 4] = [0.75, 0.75, 0.625, 0.625];

impl Diffuser {
    fn new(sample_rate: f32) -> Self {
        Self { stages: core::array::from_fn(|i| Allpass::new((DIFFUSER_MS[i] * 1e-3 * sample_rate).round() as usize, DIFFUSER_G[i])) }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        self.stages.iter_mut().fold(x, |x, a| a.tick(x))
    }

    fn clear(&mut self) {
        for a in &mut self.stages {
            a.buf.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Samples an arrival takes through it once (what rings on in it
    /// feeds the lines, whose output keeps the network awake).
    fn longest(&self) -> u32 {
        self.stages.iter().map(|a| a.buf.len() as u32).sum()
    }
}

/// v[n] = x[n] + g·v[n−D]; y[n] = v[n−D] − g·v[n].
struct Allpass {
    buf: Vec<f32>,
    at: usize,
    g: f32,
}

impl Allpass {
    fn new(delay: usize, g: f32) -> Self {
        Self { buf: vec![0.0; delay.max(1)], at: 0, g }
    }

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let old = self.buf[self.at];
        let v = x + self.g * old;
        self.buf[self.at] = v;
        self.at = if self.at + 1 == self.buf.len() { 0 } else { self.at + 1 };
        old - self.g * v
    }
}

#[cfg(test)]
mod energy_tests {
    use super::*;

    /// `energy_norm` makes an impulse's output energy 1 at any RT60, and
    /// the lines come out nearly incoherent (their omni sum ≈ their
    /// energies' sum), as a diffuse field is.
    #[test]
    fn an_impulse_comes_out_with_unit_energy() {
        let sr = 48_000.0;
        for rt in [0.3f32, 0.6, 1.2, 2.4, 4.8] {
            let mut f = Fdn::new(sr);
            f.set_rt60(rt, rt);
            f.settle();
            let k = f.energy_norm();
            let mut outs = [0.0f32; NLINES];
            let (mut e, mut omni) = (0.0f64, 0.0f64);
            for i in 0..((rt * 4.0 + 0.2) * sr) as usize {
                f.process(if i == 0 { 1.0 } else { 0.0 }, &mut outs);
                e += outs.iter().map(|x| ((x * k) as f64).powi(2)).sum::<f64>();
                omni += outs.iter().map(|x| (x * k) as f64).sum::<f64>().powi(2);
            }
            eprintln!("rt60 {rt}: impulse energy {e:.3}, omni / Σ {:.2}", omni / e);
            assert!((e - 1.0).abs() < 0.03, "RT60 {rt}: energy {e}");
            assert!((omni / e - 1.0).abs() < 0.3, "RT60 {rt}: lines coherent ({})", omni / e);
        }
    }
}
