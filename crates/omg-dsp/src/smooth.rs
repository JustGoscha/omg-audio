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
