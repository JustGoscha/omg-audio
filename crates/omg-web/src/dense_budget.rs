//! Engine v4 on the web (the worker side): which sources' dense
//! responses get convolved in the worklet, and their kernels, built a
//! few partitions per tick (`omg_dsp::dense::KernelBuilder`) and handed
//! over as flat words.
//!
//! The budget, for wasm on the audio thread:
//! - `SOURCES` sources at most are convolved: the loudest at the ears
//!   (a convolving source counts double, so the set doesn't flicker
//!   between near-equal ones);
//! - kernels are at most `KERNEL_S` long (the native cap is 10 s);
//! - a kernel is handed over at most every `EVERY_TICKS` ticks (the
//!   worklet copies it in on the audio thread), built over several ticks
//!   at `STEP_SAMPLES` of impulse response per tick;
//! - a source's kernel is rebuilt when its response moved (the same
//!   criterion as omg-audio's `Simulation`: the mid-band envelope by 15 %,
//!   or the gain; a quarter of it after `REFRESH_TICKS`).
//! Every other source keeps the block's parametric reverb, from the same
//! trace — nothing goes dry.

use omg_core::v3::DenseResponse;
use omg_dsp::dense::KernelBuilder;

pub const SOURCES: usize = 4;
pub const KERNEL_S: f32 = 2.5;
pub const EVERY_TICKS: u64 = 10;
pub const STEP_SAMPLES: usize = 4 * 4096;
const CHANGE: f32 = 0.15;
const REFRESH_TICKS: u64 = 40;

struct Job {
    src: usize,
    builder: KernelBuilder,
    resp: DenseResponse,
    gain: f32,
}

pub struct DenseBudget {
    /// Sources being convolved (the worklet has, or is getting, a kernel).
    members: Vec<usize>,
    /// Per source: the response, gain and tick its sent kernel was built
    /// from (None: the worklet has none).
    built: Vec<Option<(DenseResponse, f32, u64)>>,
    job: Option<Job>,
    /// This tick's output: a finished kernel (source, flat words) and the
    /// sources whose kernel the worklet must drop.
    pub ready: Option<(u32, Vec<f32>)>,
    pub clears: Vec<u32>,
    tick: u64,
    last_sent: u64,
    pub sample_rate: f32,
}

impl DenseBudget {
    pub fn new(n_sources: usize) -> Self {
        Self {
            members: Vec::new(),
            built: vec![None; n_sources],
            job: None,
            ready: None,
            clears: Vec::new(),
            tick: 0,
            last_sent: 0,
            sample_rate: 48_000.0,
        }
    }

    /// Does the worklet convolve source `i` (its block's parametric reverb
    /// must then go: the kernel carries that energy)?
    pub fn convolving(&self, i: usize) -> bool {
        self.built.get(i).is_some_and(|b| b.is_some())
    }

    pub fn count(&self) -> usize {
        self.built.iter().filter(|b| b.is_some()).count()
    }

    /// One tick. `dense(i)`: source i's dense response (None outside v4);
    /// `loud[i]`: its energy at the ears; `gain[i]`: its source gain.
    pub fn tick(&mut self, dense: &dyn Fn(usize) -> Option<DenseResponse>, loud: &[f32], gain: &[f32]) {
        self.tick += 1;
        self.ready = None;
        self.clears.clear();
        let n = self.built.len().min(loud.len()).min(gain.len());
        // the set: the loudest, a member counting double
        let mut cand: Vec<(f32, usize)> = (0..n)
            .filter(|&i| loud[i] > 0.0 && dense(i).is_some_and(|d| !d.bins.is_empty()))
            .map(|i| (loud[i] * if self.members.contains(&i) { 2.0 } else { 1.0 }, i))
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0));
        let set: Vec<usize> = cand.iter().take(SOURCES).map(|c| c.1).collect();
        for &m in &self.members {
            if !set.contains(&m) {
                if self.built[m].take().is_some() {
                    self.clears.push(m as u32);
                }
                if self.job.as_ref().is_some_and(|j| j.src == m) {
                    self.job = None;
                }
            }
        }
        self.members = set;
        // the kernel being built: a step a tick
        if let Some(job) = &mut self.job {
            if job.builder.step(STEP_SAMPLES) {
                let job = self.job.take().expect("checked");
                let k = job.builder.finish();
                self.ready = Some((job.src as u32, k.to_flat()));
                self.built[job.src] = Some((job.resp, job.gain, self.tick));
                self.last_sent = self.tick;
            }
            return;
        }
        if self.tick < self.last_sent + EVERY_TICKS && self.last_sent > 0 {
            return;
        }
        // the member that needs one most: none yet, else the most changed
        let mut best: Option<(f32, usize)> = None;
        for &i in &self.members {
            let Some(d) = dense(i) else { continue };
            let need = match &self.built[i] {
                None => f32::INFINITY,
                Some((from, g, at)) => {
                    let dg = if *g > 0.0 { (gain[i] / g - 1.0).abs() } else { 1.0 };
                    let c = change(from, &d).max(dg);
                    if c > CHANGE || (c > CHANGE / 4.0 && self.tick > at + REFRESH_TICKS) {
                        c
                    } else {
                        continue;
                    }
                }
            };
            let score = need.min(1e6) * loud[i];
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, i));
            }
        }
        if let Some((_, i)) = best {
            let resp = dense(i).expect("listed above");
            let mut builder = KernelBuilder::new(&resp, self.sample_rate, gain[i], i as u32, KERNEL_S);
            if builder.step(STEP_SAMPLES) {
                self.ready = Some((i as u32, builder.finish().to_flat()));
                self.built[i] = Some((resp, gain[i], self.tick));
                self.last_sent = self.tick;
            } else {
                self.job = Some(Job { src: i, builder, resp, gain: gain[i] });
            }
        }
    }

    /// Leaving v4: every kernel goes.
    pub fn clear_all(&mut self) {
        self.ready = None;
        self.clears.clear();
        self.job = None;
        self.members.clear();
        for (i, b) in self.built.iter_mut().enumerate() {
            if b.take().is_some() {
                self.clears.push(i as u32);
            }
        }
    }
}

/// How far `b`'s mid-band envelope is from `a`'s (L1 over the larger
/// total; omg-audio's `dense_change`).
fn change(a: &DenseResponse, b: &DenseResponse) -> f32 {
    let n = a.bins.len().max(b.bins.len());
    let (mut diff, mut ta, mut tb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let x = a.bins.get(i).map_or(0.0, |v| v[1]);
        let y = b.bins.get(i).map_or(0.0, |v| v[1]);
        diff += (x - y).abs();
        ta += x;
        tb += y;
    }
    diff / ta.max(tb).max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(scale: f32) -> DenseResponse {
        DenseResponse {
            bin_s: 0.002,
            bins: (0..400).map(|i| {
                let e = scale * (-(i as f32) / 80.0).exp();
                [e, e, e, 0.0, 0.0, 0.0]
            }).collect(),
        }
    }

    /// The loudest `SOURCES` get kernels, one every `EVERY_TICKS` at most;
    /// a source falling out is cleared; an unchanged response isn't rebuilt.
    #[test]
    fn the_loudest_get_kernels_within_the_budget() {
        let n = 7;
        let mut b = DenseBudget::new(n);
        let mut loud: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect(); // 6 loudest
        let gain = vec![1.0; n];
        let dense = |_i: usize| Some(resp(1.0));
        let mut sent: Vec<(u64, u32)> = Vec::new();
        for t in 0..80u64 {
            b.tick(&dense, &loud, &gain);
            if let Some((s, words)) = &b.ready {
                assert!(!words.is_empty());
                sent.push((t, *s));
            }
        }
        let got: Vec<u32> = sent.iter().map(|x| x.1).collect();
        assert_eq!(got.len(), SOURCES, "each of the loudest once, no rebuilds: {sent:?}");
        for s in [6, 5, 4, 3] {
            assert!(got.contains(&s), "source {s} convolved: {got:?}");
        }
        for w in sent.windows(2) {
            assert!(w[1].0 - w[0].0 >= EVERY_TICKS, "one kernel per {EVERY_TICKS} ticks: {sent:?}");
        }
        assert_eq!(b.count(), SOURCES);
        // source 0 becomes the loudest by far: 3 drops out (cleared)
        loud[0] = 100.0;
        b.tick(&dense, &loud, &gain);
        assert_eq!(b.clears, vec![3], "the weakest member is cleared");
        assert!(!b.convolving(3));
    }
}
