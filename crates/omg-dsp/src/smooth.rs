/// One-pole parameter smoother — every simulation-supplied value goes
/// through one of these before touching audio, so 10–30 Hz param updates
/// never produce zipper noise or clicks.
#[derive(Clone, Copy)]
pub struct Smoothed {
    cur: f32,
    target: f32,
    coef: f32,
}

impl Smoothed {
    pub fn new(initial: f32, tau_s: f32, sample_rate: f32) -> Self {
        Self {
            cur: initial,
            target: initial,
            coef: (-1.0 / (tau_s * sample_rate)).exp(),
        }
    }

    pub fn set(&mut self, target: f32) {
        self.target = target;
    }

    /// Jump immediately (init / reset only, not during playback).
    pub fn snap(&mut self, v: f32) {
        self.cur = v;
        self.target = v;
    }

    pub fn current(&self) -> f32 {
        self.cur
    }

    pub fn target_val(&self) -> f32 {
        self.target
    }

    #[inline]
    pub fn tick(&mut self) -> f32 {
        self.cur = self.target + (self.cur - self.target) * self.coef;
        self.cur
    }

    /// Per-sample decay factor: each `tick` scales `cur − target` by it.
    pub fn coef(&self) -> f32 {
        self.coef
    }

    /// Where a block renderer that ticked this smoother's state by hand
    /// (see `coef`, `target_val`) left it.
    pub fn set_current(&mut self, cur: f32) {
        self.cur = cur;
    }

    /// Jump `n` ticks at once, exactly: `pow_n` = `coef()`ⁿ. Within a
    /// block, sample i (0-based) of those ticks is
    /// `target + (cur − target)·coef^(i+1)` — what block renderers use
    /// instead of ticking.
    #[inline]
    pub fn advance(&mut self, pow_n: f32) {
        self.cur = self.target + (self.cur - self.target) * pow_n;
    }
}

/// f64 twin of [`Smoothed`] for path delays. A 10 km path is ~1.4 M
/// samples, where an f32 steps in 1/8 samples: a slow glide (Doppler)
/// would stall between steps and then jump.
#[derive(Clone, Copy)]
pub struct Smoothed64 {
    cur: f64,
    target: f64,
    coef: f64,
}

impl Smoothed64 {
    pub fn new(initial: f64, tau_s: f32, sample_rate: f32) -> Self {
        Self {
            cur: initial,
            target: initial,
            coef: (-1.0 / (tau_s as f64 * sample_rate as f64)).exp(),
        }
    }

    pub fn set(&mut self, target: f64) {
        self.target = target;
    }

    pub fn snap(&mut self, v: f64) {
        self.cur = v;
        self.target = v;
    }

    pub fn current(&self) -> f64 {
        self.cur
    }

    pub fn target_val(&self) -> f64 {
        self.target
    }

    #[inline]
    pub fn tick(&mut self) -> f64 {
        self.cur = self.target + (self.cur - self.target) * self.coef;
        self.cur
    }
}

/// Follows a path's delay (samples) between simulation updates so that a
/// CONSTANT velocity is a CONSTANT Doppler pitch.
///
/// A one-pole glide toward each new target moves fast right after an
/// update and slows before the next: the delay's slope — the pitch —
/// then wobbles at the update rate by an amount proportional to speed
/// (±1 % walking, tens of % in a fast flyover: an audible warble).
/// Instead, each update gives a velocity, (new − previous target) / the
/// simulated time between them (or the average update interval), and the
/// follower moves at that velocity one update behind the targets, with a
/// gentle correction (over sixteen intervals) for drift. Without news for
/// an interval past its window it stops coasting and settles on the last
/// target.
///
/// All the paths of one source must move on ONE timeline: a path born
/// later must sit where the others put it. So the renderer times every
/// update on a shared, phase-locked update clock (`retarget_over`'s
/// `age`) instead of each update's raw arrival — a raw arrival that
/// jitters leaves every follower alive at that moment a different error
/// to work off, and paths born at different moments then drift apart:
/// the sum of a room's reflections phases ("warble" on a tone).
#[derive(Clone, Copy)]
pub struct PathDelay {
    cur: f64,
    target: f64,
    vel: f64,
    slope: f64,
    /// Samples into this update's window (negative: the window starts
    /// later — the update arrived early).
    age: f64,
    /// Samples the reference takes from the previous target to this one.
    window: f64,
    /// Update interval this motion was planned for (samples), and the
    /// per-sample constants derived from it (no divides per sample).
    interval: f64,
    inv_correct: f64,
    slope_k: f64,
}

impl PathDelay {
    pub fn new(v: f64) -> Self {
        let mut s = Self { cur: v, target: v, vel: 0.0, slope: 0.0, age: 0.0, window: 0.0, interval: 0.0, inv_correct: 0.0, slope_k: 0.0 };
        s.set_interval(2400.0);
        s.window = s.interval;
        s
    }

    /// Jump (a new path): no motion history.
    pub fn snap(&mut self, v: f64) {
        let interval = self.interval;
        *self = Self::new(v);
        self.set_interval(interval);
        self.window = interval;
    }

    fn set_interval(&mut self, interval: f64) {
        self.interval = interval;
        // drift is corrected gently (over ~16 intervals): the reference
        // is timed by update ARRIVAL, which jitters, while the velocity
        // comes from the targets themselves, which don't
        self.inv_correct = 1.0 / (16.0 * interval);
        // slope settles within ~a quarter interval: no pitch steps
        self.slope_k = (4.0 / interval).min(1.0);
    }

    /// A new target from an update arriving every `interval` samples on
    /// average.
    pub fn retarget(&mut self, target: f64, interval: f64) {
        self.retarget_over(target, interval, interval, 0.0);
    }

    /// A new target that is `motion` samples of simulated time after the
    /// previous one (when the host knows it: the velocity is then exact
    /// even when updates sample the motion irregularly), from updates
    /// arriving every `interval` samples on average. `age`: where this
    /// update's window starts on the shared update clock, as samples
    /// already into it (negative = it starts that much later: the update
    /// arrived early). The reference glides from the previous target to
    /// this one over the window: continuous at constant velocity however
    /// the updates arrive.
    pub fn retarget_over(&mut self, target: f64, motion: f64, interval: f64, age: f64) {
        let interval = interval.max(16.0);
        let motion = motion.max(16.0);
        self.vel = (target - self.target) / motion;
        self.target = target;
        self.window = motion;
        self.age = age;
        if interval != self.interval {
            self.set_interval(interval);
        }
    }

    /// A path born mid-motion: it joins `like`'s timeline (a path of the
    /// same source that glides on — most of a source's paths share its
    /// motion: the listener's and the source's) at the point that
    /// timeline puts `target`, already moving at its velocity. Snapped still at its
    /// target instead, it would stand for a window, and land wherever the
    /// update's arrival happened to put it: paths born at different
    /// moments would then sit at scrambled relative delays.
    pub fn join(&mut self, target: f64, like: &PathDelay) {
        *self = *like;
        self.target = target;
        let coasting = like.age < like.window + like.interval;
        let lag = if coasting { (like.window - like.age).max(0.0) } else { 0.0 };
        self.vel = if coasting { like.vel } else { 0.0 };
        self.cur = target - self.vel * lag;
        self.slope = self.vel;
    }

    pub fn current(&self) -> f64 {
        self.cur
    }

    pub fn target_val(&self) -> f64 {
        self.target
    }

    #[inline]
    pub fn tick(&mut self) -> f64 {
        let w = self.window;
        let coasting = self.age < w + self.interval;
        let vel = if coasting { self.vel } else { 0.0 };
        // the reference runs one window behind the newest target: at the
        // window's end (age = w) it has just reached it
        let lag = if coasting { (w - self.age).max(0.0) } else { 0.0 };
        let reference = self.target - vel * lag;
        let goal = vel + (reference - self.cur) * self.inv_correct;
        self.slope += (goal - self.slope) * self.slope_k;
        self.cur += self.slope;
        self.age += 1.0;
        self.cur
    }

    /// `out.len()` ticks of `L` followers side by side: `out[i][l]` is
    /// what follower l's i-th `tick` returns — the same per-sample motion
    /// (the slope is the Doppler pitch, so it stays sample-accurate), up
    /// to float rounding. Two passes: the part of each step that does not
    /// depend on the state (where the reference is, given the age) for
    /// the whole block, then the recursion itself with the followers'
    /// dependency chains interleaved — one follower alone is a ~20-cycle
    /// chain per sample; side by side they overlap and vectorize.
    #[inline]
    pub fn tick_lanes<const L: usize>(p: &mut [PathDelay; L], out: &mut [[f64; L]]) {
        // `tick` is  goal = A(age) − ic·cur,  slope += (goal − slope)·k,
        // cur += slope,  with
        //   A = vel + ic·(target − vel·max(window − age, 0))  while coasting
        //   A = ic·target                                       after.
        // Rearranged as  slope = (1 − k)·slope + k·A − k·ic·cur  the chain
        // from one sample to the next is a multiply, a subtract and the
        // add into cur (half of tick's).
        // pass 1: k·A(age) for the block
        for l in 0..L {
            let PathDelay { target, vel, age, window: w, interval: i, inv_correct: ic, slope_k: k, .. } = p[l];
            let coast = k * (vel + ic * target);
            let rest = k * (ic * target);
            let civ = k * (ic * vel);
            for (j, o) in out.iter_mut().enumerate() {
                let age = age + j as f64;
                let lag = w - age;
                // (plain compares: f64::max's NaN rules cost a select
                // chain in wasm)
                o[l] = if age >= w + i {
                    rest
                } else if lag > 0.0 {
                    coast - civ * lag
                } else {
                    coast
                };
            }
        }
        // pass 2: the recursion
        let a: [f64; L] = core::array::from_fn(|l| 1.0 - p[l].slope_k);
        let b: [f64; L] = core::array::from_fn(|l| p[l].slope_k * p[l].inv_correct);
        let mut cur: [f64; L] = core::array::from_fn(|l| p[l].cur);
        let mut slope: [f64; L] = core::array::from_fn(|l| p[l].slope);
        for o in out.iter_mut() {
            for l in 0..L {
                slope[l] = (a[l] * slope[l] + o[l]) - b[l] * cur[l];
                cur[l] += slope[l];
                o[l] = cur[l];
            }
        }
        let n = out.len() as f64;
        for l in 0..L {
            p[l].cur = cur[l];
            p[l].slope = slope[l];
            p[l].age += n;
        }
    }
}
