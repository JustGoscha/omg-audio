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
