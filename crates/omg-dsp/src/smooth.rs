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
/// average update interval, and the follower moves at that velocity one
/// update behind the targets, with a gentle correction (over sixteen
/// intervals) for drift and update jitter. Without news for two
/// intervals it stops coasting and settles on the last target.
#[derive(Clone, Copy)]
pub struct PathDelay {
    cur: f64,
    target: f64,
    vel: f64,
    slope: f64,
    /// Samples since the last retarget.
    age: f64,
    /// Update interval this motion was planned for (samples).
    interval: f64,
}

impl PathDelay {
    pub fn new(v: f64) -> Self {
        Self { cur: v, target: v, vel: 0.0, slope: 0.0, age: 0.0, interval: 2400.0 }
    }

    /// Jump (a new path): no motion history.
    pub fn snap(&mut self, v: f64) {
        *self = Self { interval: self.interval, ..Self::new(v) };
    }

    /// A new target from an update arriving every `interval` samples on
    /// average.
    pub fn retarget(&mut self, target: f64, interval: f64) {
        let interval = interval.max(16.0);
        self.vel = (target - self.target) / interval;
        self.target = target;
        self.age = 0.0;
        self.interval = interval;
    }

    pub fn current(&self) -> f64 {
        self.cur
    }

    pub fn target_val(&self) -> f64 {
        self.target
    }

    #[inline]
    pub fn tick(&mut self) -> f64 {
        let i = self.interval;
        let coasting = self.age < 2.0 * i;
        let vel = if coasting { self.vel } else { 0.0 };
        // the reference runs one interval behind the newest target: at
        // the next update (age = i) it has just reached it
        let lag = if coasting { (i - self.age).max(0.0) } else { 0.0 };
        let reference = self.target - vel * lag;
        // drift is corrected gently (over ~16 intervals): the reference
        // is timed by update ARRIVAL, which jitters, while the velocity
        // comes from the targets themselves, which don't
        let goal = vel + (reference - self.cur) / (16.0 * i);
        // slope settles within ~a quarter interval: no pitch steps
        self.slope += (goal - self.slope) * (4.0 / i).min(1.0);
        self.cur += self.slope;
        self.age += 1.0;
        self.cur
    }
}
