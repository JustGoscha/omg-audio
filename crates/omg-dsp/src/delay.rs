/// Power-of-two circular buffer with linear-interpolated fractional reads.
/// A time-varying fractional delay is what produces physically correct
/// Doppler shift for free.
pub struct DelayLine {
    buf: Vec<f32>,
    mask: usize,
    write: usize,
}

impl DelayLine {
    pub fn new(min_len: usize) -> Self {
        let len = min_len.next_power_of_two();
        Self { buf: vec![0.0; len], mask: len - 1, write: 0 }
    }

    pub fn write(&mut self, x: f32) {
        self.write = (self.write + 1) & self.mask;
        self.buf[self.write] = x;
    }

    /// Read `delay` samples behind the last write. Fractional, linear interp.
    pub fn read(&self, delay: f32) -> f32 {
        let delay = delay.clamp(1.0, (self.buf.len() - 2) as f32);
        let i = delay as usize;
        let frac = delay - i as f32;
        let a = self.buf[(self.write + self.buf.len() - i) & self.mask];
        let b = self.buf[(self.write + self.buf.len() - i - 1) & self.mask];
        a + frac * (b - a)
    }
}
