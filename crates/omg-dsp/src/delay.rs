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

    /// A line that holds nothing (no allocation) — for renderers whose
    /// source is a `PlayLog`.
    pub fn empty() -> Self {
        Self { buf: Vec::new(), mask: 0, write: 0 }
    }

    /// Silence the whole line.
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
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

    /// What `read(delay)` will return after `ahead` more writes, read now
    /// — valid while those writes cannot reach it (`ahead` < `delay` − 1):
    /// a block of a feedback loop longer than the block reads its whole
    /// block before writing it.
    #[inline]
    pub fn read_ahead(&self, delay: f32, ahead: usize) -> f32 {
        let delay = delay.clamp(1.0, (self.buf.len() - 2) as f32);
        let i = delay as usize;
        let frac = delay - i as f32;
        let p = self.write + ahead + self.buf.len() - i;
        let a = self.buf[p & self.mask];
        let b = self.buf[(p - 1) & self.mask];
        a + frac * (b - a)
    }

    /// `out[j]` = `read_ahead(delay, j)` for a whole block (same values):
    /// the span is copied out of the ring once, then interpolated with
    /// plain offsets.
    pub fn read_ahead_block(&self, delay: f32, out: &mut [f32]) {
        let m = out.len();
        let len = self.buf.len();
        let n = m + 1; // the span: every sample's pair of neighbours
        if m == 0 || n > READ_SPAN || n + 1 > len {
            for (j, o) in out.iter_mut().enumerate() {
                *o = self.read_ahead(delay, j);
            }
            return;
        }
        let delay = delay.clamp(1.0, (len - 2) as f32);
        let i = delay as usize;
        let frac = delay - i as f32;
        // w[t] = buf[p0 − 1 + t]: the older neighbour of sample 0 first
        let start = (self.write + len - i - 1) & self.mask;
        let mut w = [0.0f32; READ_SPAN];
        let first = n.min(len - start);
        w[..first].copy_from_slice(&self.buf[start..start + first]);
        w[first..n].copy_from_slice(&self.buf[..n - first]);
        for (j, o) in out.iter_mut().enumerate() {
            let (b, a) = (w[j], w[j + 1]);
            *o = a + frac * (b - a);
        }
    }

    /// `write` each of `xs` in turn.
    pub fn write_block(&mut self, xs: &[f32]) {
        let len = self.buf.len();
        if xs.len() > len {
            for &x in xs {
                self.write(x);
            }
            return;
        }
        let start = (self.write + 1) & self.mask;
        let first = xs.len().min(len - start);
        self.buf[start..start + first].copy_from_slice(&xs[..first]);
        self.buf[..xs.len() - first].copy_from_slice(&xs[first..]);
        self.write = (self.write + xs.len()) & self.mask;
    }

    /// Read `delay` samples behind the write that was `back` writes ago
    /// (`back` < `READ_SLACK`): block renderers write a whole block, then
    /// read each sample's paths as of that sample. The clamp keeps every
    /// read inside the written past for any `back` < `READ_SLACK`, so
    /// `read_back(d, 0)` per sample and `read_back(d, back)` after a
    /// block write agree exactly.
    ///
    /// Cubic between samples (Hermite through the two either side): a
    /// path whose delay slides keeps its treble (`playlog::hermite`).
    #[inline]
    pub fn read_back(&self, delay: f32, back: usize) -> f32 {
        let len = self.buf.len();
        let delay = delay.clamp(1.0, (len - 3 - READ_SLACK) as f32);
        let i = delay as usize;
        let f = delay - i as f32;
        let p = self.write + len - back - i;
        let y1 = self.buf[p & self.mask];
        if f == 0.0 {
            return y1;
        }
        let (y0, y2, y3) = (self.buf[(p + 1) & self.mask], self.buf[(p - 1) & self.mask], self.buf[(p - 2) & self.mask]);
        let c1 = 0.5 * (y2 - y0);
        let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
        let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
        ((c3 * f + c2) * f + c1) * f + y1
    }
}

/// How far behind the newest write `read_back` may look (one render
/// block).
pub const READ_SLACK: usize = 128;
/// Longest block `read_ahead_block` interpolates from a local copy.
const READ_SPAN: usize = 257;
