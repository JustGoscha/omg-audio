/// First-order one-pole lowpass; building block for shelves and damping.
#[derive(Clone, Copy, Default)]
pub struct OnePoleLp {
    pub state: f32,
}

impl OnePoleLp {
    pub fn coef(cutoff_hz: f32, sample_rate: f32) -> f32 {
        1.0 - (-core::f32::consts::TAU * cutoff_hz / sample_rate).exp()
    }

    #[inline]
    pub fn tick(&mut self, x: f32, coef: f32) -> f32 {
        self.state += coef * (x - self.state);
        self.state
    }
}

/// Per-tap 3-band tone shaping: broadband gain is applied elsewhere;
/// this applies low-band and high-band gains *relative to mid* using two
/// first-order shelves. Cheap enough to run per early-reflection tap.
#[derive(Clone, Copy, Default)]
pub struct TapEq {
    lp_low: OnePoleLp,
    lp_high: OnePoleLp,
}

impl TapEq {
    /// `g_low`, `g_high`: amplitude ratios vs. mid band.
    #[inline]
    pub fn tick(&mut self, x: f32, g_low: f32, g_high: f32, c_low: f32, c_high: f32) -> f32 {
        let low = self.lp_low.tick(x, c_low);
        let y = x + (g_low - 1.0) * low; // low shelf
        let high_part = y - self.lp_high.tick(y, c_high);
        y + (g_high - 1.0) * high_part // high shelf
    }
}
