//! Engine v4: a source's dense (scattered) response, rendered by
//! convolution — the traced energy envelope per band and its direction,
//! turned into a first-order ambisonic impulse response (noise shaped to
//! the envelope: many incoherent arrivals sum like noise), convolved with
//! the source and added to the renderer's SH bus.
//!
//! - `build_kernel` (off the audio thread: the simulation side) makes the
//!   impulse response and its FFT partitions. The noise comes from one
//!   shared bank (a fixed realization per source): a new response only
//!   reshapes the envelope, so swapping kernels doesn't click.
//! - `DenseConv` (per source, on the audio thread) is a two-stage
//!   uniformly partitioned convolution: the first `HEAD_LEN` samples in
//!   `HEAD_B`-sample blocks, the rest in `TAIL_B`-sample blocks (a long
//!   tail at a fraction of the cost). Latency `HEAD_B`, compensated by
//!   starting the kernel `HEAD_B` samples into the response.

use std::sync::{Arc, OnceLock};

use omg_core::v3::DenseResponse;

use crate::ambi::NCH;
use crate::renderer::BusBlock;
use crate::fft::{mul_acc, RealFft, C};

pub const HEAD_B: usize = 128;
const HEAD_PARTS: usize = 32;
const HEAD_LEN: usize = HEAD_B * HEAD_PARTS;
pub const TAIL_B: usize = HEAD_LEN;
/// First-order ambisonics: W, Y, Z, X (ACN, SN3D) — the bus's first four.
const FOA: usize = 4;
const BANK_LEN: usize = 1 << 17;
/// The one-pole crossovers of `bands::BandSplit` (the tracer's bands).
const XOVER_LO_HZ: f32 = 314.0;
const XOVER_HI_HZ: f32 = 2100.0;

/// Five independent unit-variance white noises (one coherent, four
/// diffuse: W, Y, Z, X), each split into the three bands (bands sum to
/// the white noise): `[seq][band][sample]`.
struct Bank {
    n: Vec<[Vec<f32>; 3]>,
}

fn bank(sample_rate: f32) -> &'static Bank {
    static B: OnceLock<Bank> = OnceLock::new();
    B.get_or_init(|| {
        let coef = |fc: f32| 1.0 - (-core::f32::consts::TAU * fc / sample_rate).exp();
        let (c_lo, c_hi) = (coef(XOVER_LO_HZ), coef(XOVER_HI_HZ));
        let mut state = 0x9e37_79b9u32;
        let mut rnd = || {
            // xorshift, uniform in [-1, 1), scaled to unit variance
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * 1.732_050_8
        };
        let n = (0..5)
            .map(|_| {
                let mut b: [Vec<f32>; 3] = core::array::from_fn(|_| vec![0.0; BANK_LEN]);
                let (mut lo, mut hi) = (0.0f32, 0.0f32);
                // run in once so the filters start settled (the bank wraps)
                for pass in 0..2 {
                    for i in 0..BANK_LEN {
                        let x = rnd();
                        lo += c_lo * (x - lo);
                        hi += c_hi * (x - hi);
                        if pass == 1 {
                            b[0][i] = lo;
                            b[1][i] = hi - lo;
                            b[2][i] = x - hi;
                        }
                    }
                }
                b
            })
            .collect();
        Bank { n }
    })
}

/// A dense response ready to convolve: FFT partitions per FOA channel.
pub struct DenseKernel {
    /// `head[part][channel]`: `HEAD_B + 1` bins (FFT size 2·HEAD_B).
    head: Vec<[Vec<C>; FOA]>,
    /// `tail[part][channel]`: `TAIL_B + 1` bins (FFT size 2·TAIL_B).
    tail: Vec<[Vec<C>; FOA]>,
    /// Impulse response length in samples (kernel time).
    pub len: usize,
}

impl core::fmt::Debug for DenseKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DenseKernel({} samples, {}+{} parts)", self.len, self.head.len(), self.tail.len())
    }
}

impl DenseKernel {
    pub fn is_empty(&self) -> bool {
        self.head.is_empty()
    }
}

/// The FOA impulse response of `resp` (relative to the free-field
/// arrival), scaled by `gain²` in energy, `HEAD_B` samples early (the
/// convolver's latency). `seed` picks the noise realization (per source,
/// so sources stay decorrelated).
pub fn impulse_response(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32) -> [Vec<f32>; FOA] {
    let extended = continue_tail(resp);
    let resp = extended.as_ref().unwrap_or(resp);
    let bank = bank(sample_rate);
    let spb = (resp.bin_s * sample_rate).max(1.0);
    let total = (resp.bins.len() as f32 * spb) as usize;
    let len = total.saturating_sub(HEAD_B);
    let mut out: [Vec<f32>; FOA] = core::array::from_fn(|_| vec![0.0; len]);
    if len == 0 {
        return out;
    }
    let off = (seed as usize).wrapping_mul(0x9e37_79b9) % BANK_LEN;
    let g2 = gain * gain;
    let diffuse_g = [1.0f32, 0.577_350_3, 0.577_350_3, 0.577_350_3];
    for s in 0..len {
        let t = s + HEAD_B;
        // the envelope, linear between bin centres
        let pos = t as f32 / spb - 0.5;
        let i0 = pos.floor().max(0.0) as usize;
        let i1 = (i0 + 1).min(resp.bins.len() - 1);
        let f = (pos - i0 as f32).clamp(0.0, 1.0);
        let b0 = &resp.bins[i0.min(resp.bins.len() - 1)];
        let b1 = &resp.bins[i1];
        let mix = |c: usize| b0[c] + f * (b1[c] - b0[c]);
        let e: [f32; 3] = core::array::from_fn(|b| mix(b).max(0.0) * g2 / spb);
        let v = [mix(3), mix(4), mix(5)];
        let vl = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let em = (e[1] * spb / g2.max(1e-30)).max(1e-30);
        let a = (vl / em).clamp(0.0, 1.0);
        let d = if vl > 1e-20 { [v[0] / vl, v[1] / vl, v[2] / vl] } else { [0.0, 0.0, 1.0] };
        let enc = [1.0, d[1], d[2], d[0]];
        let k = (off + s) % BANK_LEN;
        for b in 0..3 {
            let amp_dir = (e[b] * a).sqrt() * bank.n[0][b][k];
            let amp_dif = (e[b] * (1.0 - a)).sqrt();
            for c in 0..FOA {
                out[c][s] += amp_dir * enc[c] + amp_dif * diffuse_g[c] * bank.n[1 + c][b][k];
            }
        }
    }
    out
}

/// Longest kernel (s): a continued tail stops here even if it is still
/// above -60 dB (memory and cost: ~30 MB per source per 20 s).
pub const MAX_KERNEL_S: f32 = 10.0;

/// A response traced up to the end of the window while still ringing
/// (above 60 dB under its loudest bin) is continued along its own decay —
/// past the mixing time a diffuse field only decays, exponentially, at
/// the rate its last second shows (per band; the energy vector keeps its
/// last share) — until -60 dB or `MAX_KERNEL_S`. None: nothing to do.
fn continue_tail(resp: &DenseResponse) -> Option<DenseResponse> {
    let n = resp.bins.len();
    let window = omg_core::v3::RESPONSE_MAX_S;
    if n == 0 || (n as f32) * resp.bin_s < window - 0.05 {
        return None; // ended on its own
    }
    let peak = resp.bins.iter().map(|b| b[1]).fold(0.0f32, f32::max);
    let fit_from = n.saturating_sub((1.0 / resp.bin_s) as usize);
    // per band: least squares of ln E over the last second
    let mut slope = [0.0f32; 3];
    let mut end = [0.0f32; 3];
    for b in 0..3 {
        let (mut sx, mut sy, mut sxx, mut sxy, mut m) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, v) in resp.bins.iter().enumerate().skip(fit_from) {
            if v[b] > 0.0 {
                let (x, y) = (i as f64, (v[b] as f64).ln());
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
                m += 1.0;
            }
        }
        if m < 8.0 {
            return None;
        }
        let k = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        if !(k < 0.0) {
            return None; // not decaying: nothing honest to continue
        }
        slope[b] = k as f32;
        end[b] = ((sy - k * sx) / m + k * (n - 1) as f64).exp() as f32;
    }
    if end[1] < peak * 1e-6 {
        return None;
    }
    let last = resp.bins[n - 1];
    let share = if last[1] > 0.0 { [last[3] / last[1], last[4] / last[1], last[5] / last[1]] } else { [0.0; 3] };
    let max_bins = (MAX_KERNEL_S / resp.bin_s) as usize;
    let mut out = resp.clone();
    let mut i = 1usize;
    while n + i < max_bins {
        let e: [f32; 3] = core::array::from_fn(|b| end[b] * (slope[b] * i as f32).exp());
        if e[1] < peak * 1e-6 {
            break;
        }
        out.bins.push([e[0], e[1], e[2], share[0] * e[1], share[1] * e[1], share[2] * e[1]]);
        i += 1;
    }
    Some(out)
}

/// `impulse_response`, partitioned and transformed for `DenseConv`.
pub fn build_kernel(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32) -> DenseKernel {
    let ir = impulse_response(resp, sample_rate, gain, seed);
    let len = ir[0].len();
    let mut fh = RealFft::new(2 * HEAD_B);
    let mut ft = RealFft::new(2 * TAIL_B);
    let n_head = len.div_ceil(HEAD_B).min(HEAD_PARTS);
    let n_tail = len.saturating_sub(HEAD_LEN).div_ceil(TAIL_B);
    let part = |fft: &mut RealFft, b: usize, start: usize| -> [Vec<C>; FOA] {
        let mut buf = vec![0.0f32; 2 * b];
        core::array::from_fn(|c| {
            buf.iter_mut().for_each(|x| *x = 0.0);
            let end = (start + b).min(len);
            if start < end {
                buf[..end - start].copy_from_slice(&ir[c][start..end]);
            }
            let mut spec = vec![C::default(); b + 1];
            fft.forward(&buf, &mut spec);
            spec
        })
    };
    let head = (0..n_head).map(|p| part(&mut fh, HEAD_B, p * HEAD_B)).collect();
    let tail = (0..n_tail).map(|q| part(&mut ft, TAIL_B, HEAD_LEN + q * TAIL_B)).collect();
    DenseKernel { head, tail, len }
}

/// The per-source convolver (audio thread). Feed it the source as of
/// its free-field arrival; it adds the dense response to the bus's
/// first-order channels.
pub struct DenseConv {
    kernel: Option<Arc<DenseKernel>>,
    fh: RealFft,
    ft: RealFft,
    /// Input: the last two head blocks, and the block being filled.
    hist: Vec<f32>,
    fill: usize,
    /// Head frequency-domain delay line (newest at `head_at`).
    head_fdl: Vec<Vec<C>>,
    head_at: usize,
    /// Tail input (the last two tail blocks) and its delay line.
    tail_hist: Vec<f32>,
    tail_fill: usize,
    tail_fdl: Vec<Vec<C>>,
    tail_at: usize,
    /// Output being played (one head block per FOA channel) and the
    /// tail's contribution for the current tail period.
    out: [Vec<f32>; FOA],
    out_pos: usize,
    tail_out: [Vec<f32>; FOA],
    tail_out_pos: usize,
    acc: Vec<C>,
    acc_t: Vec<C>,
    time: Vec<f32>,
    time_t: Vec<f32>,
    spec: Vec<C>,
    spec_t: Vec<C>,
    /// Samples of silent input in a row (asleep after the kernel's length).
    quiet: usize,
    asleep: bool,
}

impl Default for DenseConv {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseConv {
    pub fn new() -> Self {
        Self {
            kernel: None,
            fh: RealFft::new(2 * HEAD_B),
            ft: RealFft::new(2 * TAIL_B),
            hist: vec![0.0; 2 * HEAD_B],
            fill: 0,
            head_fdl: vec![vec![C::default(); HEAD_B + 1]; HEAD_PARTS],
            head_at: 0,
            tail_hist: vec![0.0; 2 * TAIL_B],
            tail_fill: 0,
            tail_fdl: Vec::new(),
            tail_at: 0,
            out: core::array::from_fn(|_| vec![0.0; HEAD_B]),
            out_pos: HEAD_B,
            tail_out: core::array::from_fn(|_| vec![0.0; TAIL_B]),
            tail_out_pos: 0,
            acc: vec![C::default(); HEAD_B + 1],
            acc_t: vec![C::default(); TAIL_B + 1],
            time: vec![0.0; 2 * HEAD_B],
            time_t: vec![0.0; 2 * TAIL_B],
            spec: vec![C::default(); HEAD_B + 1],
            spec_t: vec![C::default(); TAIL_B + 1],
            quiet: 0,
            asleep: true,
        }
    }

    /// Swap in a kernel (None: silence). Takes effect from the next block;
    /// the input history is kept, so the response changes, not the noise.
    pub fn set_kernel(&mut self, k: Option<Arc<DenseKernel>>) {
        if let Some(k) = &k {
            if self.tail_fdl.len() < k.tail.len() {
                self.tail_fdl.resize(k.tail.len(), vec![C::default(); TAIL_B + 1]);
            }
        }
        self.kernel = k.filter(|k| !k.is_empty());
    }

    pub fn kernel_len(&self) -> usize {
        self.kernel.as_ref().map_or(0, |k| k.len + crate::dense::HEAD_B)
    }

    pub fn has_kernel(&self) -> bool {
        self.kernel.is_some()
    }

    /// Nothing to compute: no kernel, or silent for longer than it rings.
    pub fn idle(&self) -> bool {
        self.kernel.is_none() || self.asleep
    }

    /// `x`: the source (free-field arrival time), one sample per bus
    /// sample from `off`; adds into `bus[0..4][off..]`.
    pub fn process(&mut self, x: &[f32], bus: &mut BusBlock, off: usize) {
        let Some(k) = self.kernel.clone() else { return };
        for (i, &v) in x.iter().enumerate() {
            if v != 0.0 {
                self.quiet = 0;
                if self.asleep {
                    self.asleep = false;
                }
            }
            self.hist[HEAD_B + self.fill] = v;
            self.fill += 1;
            if self.out_pos < HEAD_B {
                for c in 0..FOA {
                    bus[c][off + i] += self.out[c][self.out_pos];
                }
                self.out_pos += 1;
            }
            if self.fill == HEAD_B {
                self.block(&k);
            }
        }
    }

    /// One head block of input is complete: its output for the next block.
    fn block(&mut self, k: &DenseKernel) {
        self.fill = 0;
        let silent = self.hist[HEAD_B..].iter().all(|&v| v == 0.0);
        if silent {
            self.quiet += HEAD_B;
        }
        if self.asleep || (silent && self.quiet > k.len + 2 * TAIL_B) {
            if !self.asleep {
                // ring-out done: forget the history (resuming must not
                // convolve stale spectra)
                self.asleep = true;
                for s in self.head_fdl.iter_mut().chain(self.tail_fdl.iter_mut()) {
                    s.iter_mut().for_each(|c| *c = C::default());
                }
                self.hist.iter_mut().for_each(|v| *v = 0.0);
                self.tail_hist.iter_mut().for_each(|v| *v = 0.0);
                self.tail_fill = 0;
                for c in 0..FOA {
                    self.tail_out[c].iter_mut().for_each(|v| *v = 0.0);
                }
            }
            for c in 0..FOA {
                self.out[c].iter_mut().for_each(|v| *v = 0.0);
            }
            self.out_pos = 0;
            self.hist.copy_within(HEAD_B.., 0);
            return;
        }
        // head: the newest input spectrum into the delay line
        self.fh.forward(&self.hist, &mut self.spec);
        self.head_at = (self.head_at + HEAD_PARTS - 1) % HEAD_PARTS;
        self.head_fdl[self.head_at].copy_from_slice(&self.spec);
        // tail input
        self.tail_hist[TAIL_B + self.tail_fill..TAIL_B + self.tail_fill + HEAD_B].copy_from_slice(&self.hist[HEAD_B..]);
        self.tail_fill += HEAD_B;
        self.hist.copy_within(HEAD_B.., 0);
        for c in 0..FOA {
            self.acc.iter_mut().for_each(|v| *v = C::default());
            for (p, part) in k.head.iter().enumerate() {
                let x = &self.head_fdl[(self.head_at + p) % HEAD_PARTS];
                mul_acc(&mut self.acc, x, &part[c]);
            }
            self.fh.inverse(&self.acc, &mut self.time);
            // overlap-save: the second half is valid
            for j in 0..HEAD_B {
                let t = if self.tail_out_pos + j < TAIL_B { self.tail_out[c][self.tail_out_pos + j] } else { 0.0 };
                self.out[c][j] = self.time[HEAD_B + j] + t;
            }
        }
        self.out_pos = 0;
        self.tail_out_pos += HEAD_B;
        if self.tail_fill == TAIL_B {
            self.tail_block(k);
        }
    }

    /// A tail block of input is complete: the tail's output for the next
    /// `TAIL_B` samples (the tail starts `TAIL_B` into the kernel, so it
    /// is due exactly then).
    fn tail_block(&mut self, k: &DenseKernel) {
        self.tail_fill = 0;
        let n = k.tail.len();
        if n > 0 {
            self.ft.forward(&self.tail_hist, &mut self.spec_t);
            let parts = self.tail_fdl.len();
            self.tail_at = (self.tail_at + parts - 1) % parts;
            self.tail_fdl[self.tail_at].copy_from_slice(&self.spec_t);
            for c in 0..FOA {
                self.acc_t.iter_mut().for_each(|v| *v = C::default());
                for (q, part) in k.tail.iter().enumerate() {
                    let x = &self.tail_fdl[(self.tail_at + q) % parts];
                    mul_acc(&mut self.acc_t, x, &part[c]);
                }
                self.ft.inverse(&self.acc_t, &mut self.time_t);
                self.tail_out[c].copy_from_slice(&self.time_t[TAIL_B..]);
            }
        } else {
            for c in 0..FOA {
                self.tail_out[c].iter_mut().for_each(|v| *v = 0.0);
            }
        }
        self.tail_out_pos = 0;
        self.tail_hist.copy_within(TAIL_B.., 0);
    }
}

const _: () = assert!(NCH >= FOA);

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(bins: Vec<[f32; 6]>) -> DenseResponse {
        DenseResponse { bin_s: 0.002, bins }
    }

    /// The convolver equals direct convolution with the kernel's impulse
    /// response (delayed by its `HEAD_B` latency), head and tail alike.
    #[test]
    fn partitioned_convolution_matches_direct() {
        let sr = 48_000.0;
        // 0.3 s: head and a few tail partitions, energy decaying
        let bins: Vec<[f32; 6]> = (0..150).map(|i| {
            let e = (-(i as f32) / 40.0).exp() * 1e-3;
            [e, e, e * 0.5, 0.3 * e, 0.0, 0.1 * e]
        }).collect();
        let r = resp(bins);
        let ir = impulse_response(&r, sr, 1.0, 3);
        let k = Arc::new(build_kernel(&r, sr, 1.0, 3));
        let mut conv = DenseConv::new();
        conv.set_kernel(Some(k));
        let n = 20_000;
        let x: Vec<f32> = (0..n).map(|i| if i % 997 == 0 || i == 5 { 1.0 } else { ((i * 31 % 17) as f32 - 8.0) * 0.01 }).collect();
        let mut got: [Vec<f32>; 4] = core::array::from_fn(|_| Vec::with_capacity(n));
        let mut i = 0;
        while i < n {
            let m = (n - i).min(77); // odd sub-blocks, like the renderer's
            let mut bus: BusBlock = [[0.0; 128]; NCH];
            conv.process(&x[i..i + m], &mut bus, 0);
            for c in 0..4 {
                got[c].extend_from_slice(&bus[c][..m]);
            }
            i += m;
        }
        for c in 0..4 {
            let mut err = 0.0f64;
            let mut sig = 0.0f64;
            for t in HEAD_B..n {
                let mut want = 0.0f64;
                let tt = t - HEAD_B;
                for (s, h) in ir[c].iter().enumerate() {
                    if s > tt {
                        break;
                    }
                    want += x[tt - s] as f64 * *h as f64;
                }
                err += (got[c][t] as f64 - want).powi(2);
                sig += want * want;
            }
            assert!(err < 1e-6 * sig.max(1e-12), "channel {c}: error {err:.3e} vs signal {sig:.3e}");
        }
    }

    /// A response cut by the traced window while still ringing goes on
    /// at its own decay until -60 dB; one that ended on its own doesn't.
    #[test]
    fn a_ringing_tail_continues_at_its_own_decay() {
        let window = omg_core::v3::RESPONSE_MAX_S;
        let n = (window / 0.002) as usize;
        // RT60 6 s: -60 dB per 6 s, so -48 dB at the window's end
        let k = (1e-6f32).ln() / (6.0 / 0.002);
        let bins: Vec<[f32; 6]> = (0..n).map(|i| {
            let e = (k * i as f32).exp();
            [e, e, e, 0.0, 0.0, 0.0]
        }).collect();
        let r = resp(bins);
        let c = continue_tail(&r).expect("continued");
        let len_s = c.bins.len() as f32 * 0.002;
        assert!((len_s - 6.0).abs() < 0.2, "continued to -60 dB at ~6 s, got {len_s:.2} s");
        let short = resp(r.bins[..1000].to_vec());
        assert!(continue_tail(&short).is_none(), "a tail that ended on its own stays");
    }

    /// The impulse response carries the traced energy per bin (W), and
    /// its direction: all of it on X for a field from +x.
    #[test]
    fn impulse_response_holds_the_energy_and_direction() {
        let sr = 48_000.0;
        let e = 1e-3;
        let r = resp(vec![[e, e, e, e, 0.0, 0.0]; 200]);
        let ir = impulse_response(&r, sr, 1.0, 9);
        let spb = (0.002 * sr) as usize;
        let (lo, hi) = (spb * 20, spb * 180);
        let w: f32 = ir[0][lo..hi].iter().map(|v| v * v).sum::<f32>() / (160.0 * 1.0);
        assert!((w / e - 1.0).abs() < 0.15, "W energy per bin {w:.3e} vs {e:.3e}");
        let x: f32 = ir[3][lo..hi].iter().map(|v| v * v).sum::<f32>() / 160.0;
        let y: f32 = ir[1][lo..hi].iter().map(|v| v * v).sum::<f32>() / 160.0;
        assert!((x / e - 1.0).abs() < 0.15 && y < 0.01 * e, "fully one-sided: X {x:.3e}, Y {y:.3e}");
    }
}
