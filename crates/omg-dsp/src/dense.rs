//! Engine v4: a source's dense (scattered) response, rendered by
//! convolution — the traced energy envelope per band and its direction,
//! turned into a first-order ambisonic impulse response (noise shaped to
//! the envelope: many incoherent arrivals sum like noise), convolved with
//! the source and added to the renderer's SH bus.
//!
//! - `build_kernel` (off the audio thread: the simulation side) makes the
//!   impulse response and its FFT partitions. The noise comes from one
//!   shared bank (a fixed realization per source): a new response only
//!   reshapes the envelope, so swapping kernels doesn't click. A kernel
//!   is one contiguous buffer (`to_flat` / `from_flat`: a web worker
//!   hands it to the audio worklet without a second copy).
//! - `DenseConv` (per source, on the audio thread) is a three-stage
//!   partitioned convolution: kernel samples [0, 4096) in 128-sample
//!   blocks, [4096, 8192) in 1024-sample blocks, the rest in 4096-sample
//!   blocks — the long tail at a fraction of the cost, and since the
//!   tail starts a whole tail block after its input is complete, its
//!   work (FFTs and partitions) is spread over the next 32 head blocks:
//!   no callback pays a tail block at once. Latency `HEAD_B`,
//!   compensated by starting the kernel `HEAD_B` samples into the
//!   response.

use std::sync::{Arc, OnceLock};

use omg_core::v3::DenseResponse;

use crate::ambi::NCH;
use crate::fft::{mul_acc, RealFft, C};
use crate::renderer::BusBlock;

pub const HEAD_B: usize = 128;
const HEAD_PARTS: usize = 32;
const HEAD_LEN: usize = HEAD_B * HEAD_PARTS;
const MID_B: usize = 1024;
const MID_PARTS: usize = 4;
const MID_END: usize = HEAD_LEN + MID_B * MID_PARTS;
pub const TAIL_B: usize = 4096;
/// Head blocks per tail block: the tail's work is spread over them.
const TAIL_STEPS: usize = TAIL_B / HEAD_B;
/// Ahead-of-time output (mid and tail land up to 2·TAIL_B ahead).
const RING: usize = 4 * TAIL_B;
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

/// A dense response ready to convolve: the FFT partitions of its FOA
/// channels, in one buffer — two header words (len, n_head; n_mid,
/// n_tail), then `[part][channel]` spectra of the head (`HEAD_B + 1`
/// bins), the mid (`MID_B + 1`) and the tail (`TAIL_B + 1`).
pub struct DenseKernel {
    data: Vec<C>,
    n_head: usize,
    n_mid: usize,
    n_tail: usize,
    /// Impulse response length in samples (kernel time).
    pub len: usize,
}

impl core::fmt::Debug for DenseKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DenseKernel({} samples, {}+{}+{} parts)", self.len, self.n_head, self.n_mid, self.n_tail)
    }
}

impl DenseKernel {
    pub fn is_empty(&self) -> bool {
        self.n_head == 0
    }

    fn head(&self, p: usize, c: usize) -> &[C] {
        let o = 2 + (p * FOA + c) * (HEAD_B + 1);
        &self.data[o..o + HEAD_B + 1]
    }

    fn mid(&self, p: usize, c: usize) -> &[C] {
        let o = 2 + self.n_head * FOA * (HEAD_B + 1) + (p * FOA + c) * (MID_B + 1);
        &self.data[o..o + MID_B + 1]
    }

    fn tail(&self, q: usize, c: usize) -> &[C] {
        let o = 2 + self.n_head * FOA * (HEAD_B + 1) + self.n_mid * FOA * (MID_B + 1) + (q * FOA + c) * (TAIL_B + 1);
        &self.data[o..o + TAIL_B + 1]
    }

    /// The kernel as flat f32 words (re, im pairs; the header first).
    pub fn to_flat(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.data.len() * 2);
        for c in &self.data {
            v.push(c.re);
            v.push(c.im);
        }
        v
    }

    /// Words `to_flat` makes for a kernel of these part counts.
    pub fn flat_len(n_head: usize, n_mid: usize, n_tail: usize) -> usize {
        2 * (2 + FOA * (n_head * (HEAD_B + 1) + n_mid * (MID_B + 1) + n_tail * (TAIL_B + 1)))
    }

    /// `to_flat`'s words back into a kernel — adopting the buffer, no
    /// copy (the web worklet's case: one allocation, filled from JS).
    /// None when the words are not a kernel.
    pub fn from_flat(words: Vec<f32>) -> Option<Self> {
        if words.len() < 4 {
            return None;
        }
        let (len, n_head, n_mid, n_tail) = (words[0] as usize, words[1] as usize, words[2] as usize, words[3] as usize);
        if words.len() != Self::flat_len(n_head, n_mid, n_tail) || n_head > HEAD_PARTS || n_mid > MID_PARTS {
            return None;
        }
        let data: Vec<C> = if words.capacity() % 2 == 0 {
            let mut w = core::mem::ManuallyDrop::new(words);
            // SAFETY: C is #[repr(C)] { re: f32, im: f32 }: the size of
            // and the alignment of [f32; 2]; the allocation's capacity is
            // even (checked), so it holds exactly capacity / 2 C's, and
            // its length (even, checked above) is len / 2 initialized C's.
            unsafe { Vec::from_raw_parts(w.as_mut_ptr() as *mut C, w.len() / 2, w.capacity() / 2) }
        } else {
            words.chunks_exact(2).map(|p| C::new(p[0], p[1])).collect()
        };
        Some(Self { data, n_head, n_mid, n_tail, len })
    }
}

/// The FOA impulse response of `resp` (relative to the free-field
/// arrival), scaled by `gain²` in energy, `HEAD_B` samples early (the
/// convolver's latency). `seed` picks the noise realization (per source,
/// so sources stay decorrelated).
pub fn impulse_response(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32) -> [Vec<f32>; FOA] {
    impulse_response_max(resp, sample_rate, gain, seed, MAX_KERNEL_S)
}

/// `impulse_response` at most `max_s` long: a longer response (or its
/// continued tail) fades out over its last 10 % (a cost cap: the web's
/// convolution budget), never cut.
pub fn impulse_response_max(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32, max_s: f32) -> [Vec<f32>; FOA] {
    let resp = prepare(resp, max_s);
    let len = ir_len(&resp, sample_rate);
    let mut out: [Vec<f32>; FOA] = core::array::from_fn(|_| vec![0.0; len]);
    synth(&resp, sample_rate, gain, seed, 0, len, &mut out);
    out
}

/// The response as the kernel renders it: continued past the traced
/// window (`continue_tail`), at most `max_s` long with a fade over its
/// last 10 %.
fn prepare(resp: &DenseResponse, max_s: f32) -> DenseResponse {
    let mut resp = continue_tail(resp, max_s).unwrap_or_else(|| resp.clone());
    let cap = (max_s / resp.bin_s.max(1e-6)) as usize;
    if resp.bins.len() > cap {
        resp.bins.truncate(cap);
        let fade = (cap / 10).max(1);
        for (k, b) in resp.bins[cap - fade..].iter_mut().enumerate() {
            let g = 0.5 + 0.5 * (core::f32::consts::PI * (k + 1) as f32 / fade as f32).cos();
            for v in b.iter_mut() {
                *v *= g;
            }
        }
    }
    resp
}

/// Kernel samples of a prepared response (it starts `HEAD_B` in).
fn ir_len(resp: &DenseResponse, sample_rate: f32) -> usize {
    let spb = (resp.bin_s * sample_rate).max(1.0);
    ((resp.bins.len() as f32 * spb) as usize).saturating_sub(HEAD_B)
}

/// Kernel samples `from..to` of a prepared response into `out[c][0..]`.
fn synth(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32, from: usize, to: usize, out: &mut [Vec<f32>; FOA]) {
    for o in out.iter_mut() {
        o[..to - from].iter_mut().for_each(|v| *v = 0.0);
    }
    if resp.bins.is_empty() || to <= from {
        return;
    }
    let bank = bank(sample_rate);
    let spb = (resp.bin_s * sample_rate).max(1.0);
    let off = (seed as usize).wrapping_mul(0x9e37_79b9) % BANK_LEN;
    let g2 = gain * gain;
    let diffuse_g = [1.0f32, 0.577_350_3, 0.577_350_3, 0.577_350_3];
    for s in from..to {
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
                out[c][s - from] += amp_dir * enc[c] + amp_dif * diffuse_g[c] * bank.n[1 + c][b][k];
            }
        }
    }
}

/// Longest kernel (s): a continued tail stops here even if it is still
/// above -60 dB (memory and cost: ~30 MB per source per 20 s).
pub const MAX_KERNEL_S: f32 = 10.0;

/// A response traced up to the end of the window while still ringing
/// (above 60 dB under its loudest bin) is continued along its own decay —
/// past the mixing time a diffuse field only decays, exponentially, at
/// the rate its last second shows (per band; the energy vector keeps its
/// last share) — until -60 dB or `MAX_KERNEL_S`. None: nothing to do.
fn continue_tail(resp: &DenseResponse, max_s: f32) -> Option<DenseResponse> {
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
    let max_bins = (max_s.min(MAX_KERNEL_S) / resp.bin_s) as usize;
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
    build_kernel_max(resp, sample_rate, gain, seed, MAX_KERNEL_S)
}

/// `build_kernel` at most `max_s` long (see `impulse_response_max`).
pub fn build_kernel_max(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32, max_s: f32) -> DenseKernel {
    let mut b = KernelBuilder::new(resp, sample_rate, gain, seed, max_s);
    while !b.step(usize::MAX) {}
    b.finish()
}

/// A kernel built a few partitions at a time (a host with no thread to
/// spare — the web worker, between its ticks): each `step` synthesizes
/// and transforms partitions worth at most a work budget (in samples of
/// impulse response: a tail partition is `TAIL_B`), `finish` hands over
/// the kernel. The same kernel `build_kernel` makes.
pub struct KernelBuilder {
    resp: DenseResponse,
    sample_rate: f32,
    gain: f32,
    seed: u32,
    kernel: DenseKernel,
    /// Next partition to build, counted over head, mid, tail.
    next: usize,
    /// Write position in `kernel.data`.
    at: usize,
    ir: [Vec<f32>; FOA],
    buf: Vec<f32>,
    ffts: [RealFft; 3],
}

impl KernelBuilder {
    pub fn new(resp: &DenseResponse, sample_rate: f32, gain: f32, seed: u32, max_s: f32) -> Self {
        let resp = prepare(resp, max_s);
        let len = ir_len(&resp, sample_rate);
        let n_head = len.div_ceil(HEAD_B).min(HEAD_PARTS);
        let n_mid = len.saturating_sub(HEAD_LEN).div_ceil(MID_B).min(MID_PARTS);
        let n_tail = len.saturating_sub(MID_END).div_ceil(TAIL_B);
        let mut data = vec![C::default(); DenseKernel::flat_len(n_head, n_mid, n_tail) / 2];
        data[0] = C::new(len as f32, n_head as f32);
        data[1] = C::new(n_mid as f32, n_tail as f32);
        Self {
            resp,
            sample_rate,
            gain,
            seed,
            kernel: DenseKernel { data, n_head, n_mid, n_tail, len },
            next: 0,
            at: 2,
            ir: core::array::from_fn(|_| vec![0.0; TAIL_B]),
            buf: vec![0.0; 2 * TAIL_B],
            ffts: [RealFft::new(2 * HEAD_B), RealFft::new(2 * MID_B), RealFft::new(2 * TAIL_B)],
        }
    }

    /// Build partitions worth up to `budget` impulse-response samples
    /// (at least one); true when the kernel is complete.
    pub fn step(&mut self, budget: usize) -> bool {
        let (n_head, n_mid, n_tail, len) = (self.kernel.n_head, self.kernel.n_mid, self.kernel.n_tail, self.kernel.len);
        let total = n_head + n_mid + n_tail;
        let mut spent = 0usize;
        while self.next < total && (spent == 0 || spent < budget) {
            let (stage, b, start) = if self.next < n_head {
                (0, HEAD_B, self.next * HEAD_B)
            } else if self.next < n_head + n_mid {
                (1, MID_B, HEAD_LEN + (self.next - n_head) * MID_B)
            } else {
                (2, TAIL_B, MID_END + (self.next - n_head - n_mid) * TAIL_B)
            };
            let end = (start + b).min(len);
            synth(&self.resp, self.sample_rate, self.gain, self.seed, start, end.max(start), &mut self.ir);
            for c in 0..FOA {
                let buf = &mut self.buf[..2 * b];
                buf.iter_mut().for_each(|x| *x = 0.0);
                buf[..end.saturating_sub(start)].copy_from_slice(&self.ir[c][..end.saturating_sub(start)]);
                let at = self.at;
                self.ffts[stage].forward(buf, &mut self.kernel.data[at..at + b + 1]);
                self.at += b + 1;
            }
            self.next += 1;
            spent += b;
        }
        self.next >= total
    }

    pub fn done(&self) -> bool {
        self.next >= self.kernel.n_head + self.kernel.n_mid + self.kernel.n_tail
    }

    pub fn finish(self) -> DenseKernel {
        debug_assert!(self.done(), "finish before the last step");
        self.kernel
    }
}

/// One stage's frequency-domain delay line: the input spectra, the
/// newest at `at`, older ones after it (circularly).
struct Fdl {
    spectra: Vec<Vec<C>>,
    at: usize,
    bins: usize,
}

impl Fdl {
    fn new(parts: usize, bins: usize) -> Self {
        Self { spectra: vec![vec![C::default(); bins]; parts], at: 0, bins }
    }

    /// Grow to `parts` slots; the new ones are the oldest (silent).
    fn ensure(&mut self, parts: usize) {
        if self.spectra.len() < parts {
            let add = parts - self.spectra.len();
            for _ in 0..add {
                self.spectra.insert(self.at, vec![C::default(); self.bins]);
            }
            self.at += add;
        }
    }

    fn push(&mut self, x: &[C]) {
        let n = self.spectra.len();
        self.at = (self.at + n - 1) % n;
        self.spectra[self.at].copy_from_slice(x);
    }

    /// The input spectrum `age` blocks old.
    fn get(&self, age: usize) -> &[C] {
        &self.spectra[(self.at + age) % self.spectra.len()]
    }

    fn clear(&mut self) {
        for s in &mut self.spectra {
            s.iter_mut().for_each(|c| *c = C::default());
        }
    }
}

/// The per-source convolver (audio thread). Feed it the source as of
/// its free-field arrival; it adds the dense response to the bus's
/// first-order channels.
pub struct DenseConv {
    kernel: Option<Arc<DenseKernel>>,
    fh: RealFft,
    fm: RealFft,
    ft: RealFft,
    /// Input: the last two head blocks (the second being filled).
    hist: Vec<f32>,
    fill: usize,
    /// Head blocks completed: block k covers convolution time
    /// [128k, 128k + 128) (in kernel time, latency compensated).
    blocks: u64,
    head_fdl: Fdl,
    mid_hist: Vec<f32>,
    mid_fdl: Fdl,
    tail_hist: Vec<f32>,
    tail_fdl: Fdl,
    /// The tail block in progress: its input, its partial spectra per
    /// channel, the step it is at (1..=TAIL_STEPS, 0 = none) and the
    /// convolution time its output starts at.
    tail_in: Vec<f32>,
    tail_acc: Vec<Vec<C>>,
    tail_step: usize,
    tail_at: u64,
    /// Output being played (one head block per FOA channel).
    out: [Vec<f32>; FOA],
    out_pos: usize,
    /// Mid and tail output landed ahead of time, by convolution time
    /// modulo RING.
    ring: [Vec<f32>; FOA],
    acc: Vec<C>,
    acc_m: Vec<C>,
    time: Vec<f32>,
    time_m: Vec<f32>,
    time_t: Vec<f32>,
    spec: Vec<C>,
    spec_m: Vec<C>,
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
            fm: RealFft::new(2 * MID_B),
            ft: RealFft::new(2 * TAIL_B),
            hist: vec![0.0; 2 * HEAD_B],
            fill: 0,
            blocks: 0,
            head_fdl: Fdl::new(HEAD_PARTS, HEAD_B + 1),
            mid_hist: vec![0.0; 2 * MID_B],
            mid_fdl: Fdl::new(MID_PARTS, MID_B + 1),
            tail_hist: vec![0.0; 2 * TAIL_B],
            tail_fdl: Fdl::new(1, TAIL_B + 1),
            tail_in: vec![0.0; 2 * TAIL_B],
            tail_acc: Vec::new(),
            tail_step: 0,
            tail_at: 0,
            out: core::array::from_fn(|_| vec![0.0; HEAD_B]),
            out_pos: HEAD_B,
            ring: core::array::from_fn(|_| vec![0.0; RING]),
            acc: vec![C::default(); HEAD_B + 1],
            acc_m: vec![C::default(); MID_B + 1],
            time: vec![0.0; 2 * HEAD_B],
            time_m: vec![0.0; 2 * MID_B],
            time_t: vec![0.0; 2 * TAIL_B],
            spec: vec![C::default(); HEAD_B + 1],
            spec_m: vec![C::default(); MID_B + 1],
            quiet: 0,
            asleep: true,
        }
    }

    /// Swap in a kernel (None: silence). Takes effect from the next block;
    /// the input history is kept, so the response changes, not the noise.
    pub fn set_kernel(&mut self, k: Option<Arc<DenseKernel>>) {
        if let Some(k) = &k {
            self.tail_fdl.ensure(k.n_tail.max(1));
            if self.tail_acc.is_empty() && k.n_tail > 0 {
                self.tail_acc = vec![vec![C::default(); TAIL_B + 1]; FOA];
            }
        }
        self.kernel = k.filter(|k| !k.is_empty());
    }

    pub fn kernel_len(&self) -> usize {
        self.kernel.as_ref().map_or(0, |k| k.len + HEAD_B)
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
                self.asleep = false;
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

    /// One head block of input is complete: its output for the next block,
    /// and the mid and tail stages' share of the work.
    fn block(&mut self, k: &DenseKernel) {
        self.fill = 0;
        let silent = self.hist[HEAD_B..].iter().all(|&v| v == 0.0);
        if silent {
            self.quiet += HEAD_B;
        }
        if self.asleep || (silent && self.quiet > k.len + 3 * TAIL_B) {
            if !self.asleep {
                // ring-out done: forget everything (resuming must not
                // convolve stale spectra)
                self.asleep = true;
                self.head_fdl.clear();
                self.mid_fdl.clear();
                self.tail_fdl.clear();
                self.hist.iter_mut().for_each(|v| *v = 0.0);
                self.mid_hist.iter_mut().for_each(|v| *v = 0.0);
                self.tail_hist.iter_mut().for_each(|v| *v = 0.0);
                for r in &mut self.ring {
                    r.iter_mut().for_each(|v| *v = 0.0);
                }
                self.tail_step = 0;
                self.blocks = 0;
            }
            for c in 0..FOA {
                self.out[c].iter_mut().for_each(|v| *v = 0.0);
            }
            self.out_pos = 0;
            self.hist.copy_within(HEAD_B.., 0);
            return;
        }
        let kblk = self.blocks;
        self.blocks += 1;
        // the new input into the mid and tail histories (their current
        // block is the second half)
        let mo = MID_B + (kblk as usize % (MID_B / HEAD_B)) * HEAD_B;
        self.mid_hist[mo..mo + HEAD_B].copy_from_slice(&self.hist[HEAD_B..]);
        let to = TAIL_B + (kblk as usize % TAIL_STEPS) * HEAD_B;
        self.tail_hist[to..to + HEAD_B].copy_from_slice(&self.hist[HEAD_B..]);
        // head: the newest input spectrum into the delay line
        self.fh.forward(&self.hist, &mut self.spec);
        self.head_fdl.push(&self.spec);
        self.hist.copy_within(HEAD_B.., 0);
        let t0 = (kblk as usize * HEAD_B) % RING;
        for c in 0..FOA {
            self.acc.iter_mut().for_each(|v| *v = C::default());
            for p in 0..k.n_head {
                mul_acc(&mut self.acc, self.head_fdl.get(p), k.head(p, c));
            }
            self.fh.inverse(&self.acc, &mut self.time);
            // overlap-save: the second half is valid; plus whatever the
            // mid and tail stages landed for this stretch
            let ring = &mut self.ring[c];
            for j in 0..HEAD_B {
                self.out[c][j] = self.time[HEAD_B + j] + ring[t0 + j];
                ring[t0 + j] = 0.0;
            }
        }
        self.out_pos = 0;
        let done = (kblk + 1) * HEAD_B as u64;
        // mid: a mid block of input complete → its output, due 3072
        // samples on (the mid kernel starts at HEAD_LEN), into the ring
        if done % MID_B as u64 == 0 {
            self.mid_block(k, done);
        }
        // tail: a tail block complete → start it; one step per head block
        if done % TAIL_B as u64 == 0 {
            if k.n_tail > 0 && !self.tail_acc.is_empty() {
                self.tail_in.copy_from_slice(&self.tail_hist);
                self.tail_step = 1;
                // its output: convolution times [done - TAIL_B + MID_END, + TAIL_B)
                self.tail_at = done - TAIL_B as u64 + MID_END as u64;
            }
            self.tail_hist.copy_within(TAIL_B.., 0);
        }
        if self.tail_step > 0 {
            self.tail_work(k);
        }
    }

    fn mid_block(&mut self, k: &DenseKernel, done: u64) {
        if k.n_mid > 0 {
            self.fm.forward(&self.mid_hist, &mut self.spec_m);
            self.mid_fdl.push(&self.spec_m);
            let start = (done - MID_B as u64 + HEAD_LEN as u64) as usize;
            for c in 0..FOA {
                self.acc_m.iter_mut().for_each(|v| *v = C::default());
                for p in 0..k.n_mid {
                    mul_acc(&mut self.acc_m, self.mid_fdl.get(p), k.mid(p, c));
                }
                self.fm.inverse(&self.acc_m, &mut self.time_m);
                let ring = &mut self.ring[c];
                for j in 0..MID_B {
                    ring[(start + j) % RING] += self.time_m[MID_B + j];
                }
            }
        }
        self.mid_hist.copy_within(MID_B.., 0);
    }

    /// One step of the tail block in progress: step 1 transforms its
    /// input, the next `TAIL_STEPS - 2 - FOA` accumulate the partitions
    /// (spread evenly), the last `FOA` transform one channel each back
    /// into the ring — all done 2 head blocks before its output is due
    /// (the tail kernel starts at MID_END: a whole tail block of slack).
    fn tail_work(&mut self, k: &DenseKernel) {
        let step = self.tail_step;
        let mac_steps = TAIL_STEPS - 2 - FOA;
        if step == 1 {
            let mut spec = core::mem::take(&mut self.tail_acc[0]);
            self.ft.forward(&self.tail_in, &mut spec);
            self.tail_fdl.push(&spec);
            self.tail_acc[0] = spec;
            for a in &mut self.tail_acc {
                a.iter_mut().for_each(|v| *v = C::default());
            }
        } else if step <= mac_steps + 1 {
            let n = k.n_tail.min(self.tail_fdl.spectra.len());
            let (lo, hi) = (n * (step - 2) / mac_steps, n * (step - 1) / mac_steps);
            for q in lo..hi {
                for c in 0..FOA {
                    mul_acc(&mut self.tail_acc[c], self.tail_fdl.get(q), k.tail(q, c));
                }
            }
        } else {
            let c = step - mac_steps - 2;
            self.ft.inverse(&self.tail_acc[c], &mut self.time_t);
            let ring = &mut self.ring[c];
            let at = self.tail_at as usize;
            for j in 0..TAIL_B {
                ring[(at + j) % RING] += self.time_t[TAIL_B + j];
            }
        }
        self.tail_step = if step >= mac_steps + 1 + FOA { 0 } else { step + 1 };
    }
}

const _: () = assert!(NCH >= FOA);

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(bins: Vec<[f32; 6]>) -> DenseResponse {
        DenseResponse { bin_s: 0.002, bins, ..Default::default() }
    }

    /// The convolver equals direct convolution with the kernel's impulse
    /// response (delayed by its `HEAD_B` latency), head and tail alike.
    #[test]
    fn partitioned_convolution_matches_direct() {
        let sr = 48_000.0;
        // 0.6 s: head, all four mid and several tail partitions
        let bins: Vec<[f32; 6]> = (0..300).map(|i| {
            let e = (-(i as f32) / 40.0).exp() * 1e-3;
            [e, e, e * 0.5, 0.3 * e, 0.0, 0.1 * e]
        }).collect();
        let r = resp(bins);
        let ir = impulse_response(&r, sr, 1.0, 3);
        let k = Arc::new(build_kernel(&r, sr, 1.0, 3));
        let mut conv = DenseConv::new();
        conv.set_kernel(Some(k));
        let n = 36_000;
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

    /// Built a few partitions at a time, the kernel is the one-shot one.
    #[test]
    fn a_kernel_built_in_steps_is_the_same_kernel() {
        let bins: Vec<[f32; 6]> = (0..600).map(|i| {
            let e = (-(i as f32) / 150.0).exp();
            [e, e * 0.6, e * 0.2, 0.1 * e, 0.0, -0.2 * e]
        }).collect();
        let r = resp(bins);
        let once = build_kernel_max(&r, 48_000.0, 1.3, 2, 2.5);
        let mut b = KernelBuilder::new(&r, 48_000.0, 1.3, 2, 2.5);
        let mut steps = 0;
        while !b.step(3 * TAIL_B) {
            steps += 1;
        }
        assert!(steps > 3, "several steps: {steps}");
        assert_eq!(b.finish().to_flat(), once.to_flat());
        assert!(once.len as f32 <= 2.5 * 48_000.0, "capped at 2.5 s: {}", once.len);
    }

    /// A kernel survives `to_flat` / `from_flat` bit for bit (the web
    /// worker → worklet hand-over).
    #[test]
    fn kernels_round_trip_through_flat_words() {
        let bins: Vec<[f32; 6]> = (0..400).map(|i| {
            let e = (-(i as f32) / 90.0).exp();
            [e, e * 0.7, e * 0.3, 0.2 * e, -0.1 * e, 0.0]
        }).collect();
        let k = build_kernel(&resp(bins), 48_000.0, 0.8, 5);
        assert!(k.n_head > 0 && k.n_mid > 0 && k.n_tail > 0, "{k:?}");
        let flat = k.to_flat();
        assert_eq!(flat.len(), DenseKernel::flat_len(k.n_head, k.n_mid, k.n_tail));
        let back = DenseKernel::from_flat(flat.clone()).expect("a kernel");
        assert_eq!(back.to_flat(), flat);
        assert_eq!((back.len, back.n_head, back.n_mid, back.n_tail), (k.len, k.n_head, k.n_mid, k.n_tail));
        assert!(DenseKernel::from_flat(flat[..flat.len() - 2].to_vec()).is_none(), "truncated words are refused");
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
        let c = continue_tail(&r, MAX_KERNEL_S).expect("continued");
        let len_s = c.bins.len() as f32 * 0.002;
        assert!((len_s - 6.0).abs() < 0.2, "continued to -60 dB at ~6 s, got {len_s:.2} s");
        let short = resp(r.bins[..1000].to_vec());
        assert!(continue_tail(&short, MAX_KERNEL_S).is_none(), "a tail that ended on its own stays");
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
