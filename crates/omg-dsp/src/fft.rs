//! A small FFT for the dense-response convolver (engine v4): radix-2,
//! iterative, precomputed twiddles; real signals through a half-size
//! complex transform. No dependencies (the crate builds for wasm).

/// A complex number (f32). `repr(C)`: laid out as `[re, im]`, so a
/// spectrum is a flat f32 buffer (`dense::DenseKernel::from_flat`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct C {
    pub re: f32,
    pub im: f32,
}

impl C {
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    #[inline]
    pub fn mul(self, o: C) -> C {
        C::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

/// `acc += a · b` over whole spectra (the convolver's inner loop).
#[inline]
pub fn mul_acc(acc: &mut [C], a: &[C], b: &[C]) {
    for ((o, x), y) in acc.iter_mut().zip(a).zip(b) {
        o.re += x.re * y.re - x.im * y.im;
        o.im += x.re * y.im + x.im * y.re;
    }
}

/// Complex FFT of a power-of-two size.
pub struct Fft {
    n: usize,
    tw: Vec<C>,
    rev: Vec<u32>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let tw = (0..n / 2)
            .map(|k| {
                let a = -2.0 * core::f64::consts::PI * k as f64 / n as f64;
                C::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        Self { n, tw, rev }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// In place; `inverse` uses conjugate twiddles (unscaled).
    pub fn run(&self, x: &mut [C], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                x.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let mut w = self.tw[k * step];
                    if inverse {
                        w.im = -w.im;
                    }
                    let a = x[start + k];
                    let b = x[start + k + half].mul(w);
                    x[start + k] = C::new(a.re + b.re, a.im + b.im);
                    x[start + k + half] = C::new(a.re - b.re, a.im - b.im);
                }
            }
            len *= 2;
        }
    }
}

/// Real FFT of size `n` (a power of two ≥ 4) through an `n/2` complex
/// transform: `n/2 + 1` bins out; the inverse takes them back to `n`
/// real samples, scaled by `1/n` (forward then inverse is the identity).
pub struct RealFft {
    n: usize,
    half: Fft,
    w: Vec<C>,
    buf: Vec<C>,
}

impl RealFft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 4);
        let w = (0..n / 2)
            .map(|k| {
                let a = -2.0 * core::f64::consts::PI * k as f64 / n as f64;
                C::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        Self { n, half: Fft::new(n / 2), w, buf: vec![C::default(); n / 2] }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bins of the forward transform.
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    pub fn forward(&mut self, x: &[f32], out: &mut [C]) {
        let h = self.n / 2;
        for k in 0..h {
            self.buf[k] = C::new(x[2 * k], x[2 * k + 1]);
        }
        self.half.run(&mut self.buf, false);
        let z0 = self.buf[0];
        out[0] = C::new(z0.re + z0.im, 0.0);
        out[h] = C::new(z0.re - z0.im, 0.0);
        for k in 1..h {
            let a = self.buf[k];
            let b = self.buf[h - k];
            let b = C::new(b.re, -b.im);
            let even = C::new(0.5 * (a.re + b.re), 0.5 * (a.im + b.im));
            let odd = C::new(0.5 * (a.im - b.im), -0.5 * (a.re - b.re));
            let t = odd.mul(self.w[k]);
            out[k] = C::new(even.re + t.re, even.im + t.im);
        }
    }

    pub fn inverse(&mut self, spec: &[C], x: &mut [f32]) {
        let h = self.n / 2;
        for k in 0..h {
            let a = spec[k];
            let b = spec[h - k];
            let b = C::new(b.re, -b.im);
            let even = C::new(a.re + b.re, a.im + b.im);
            let w = self.w[k];
            let odd = C::new(a.re - b.re, a.im - b.im).mul(C::new(w.re, -w.im));
            // z = even + i·odd
            self.buf[k] = C::new(even.re - odd.im, even.im + odd.re);
        }
        self.half.run(&mut self.buf, true);
        let s = 1.0 / self.n as f32;
        for k in 0..h {
            x[2 * k] = self.buf[k].re * s;
            x[2 * k + 1] = self.buf[k].im * s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dft(x: &[f32]) -> Vec<C> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (t, &v) in x.iter().enumerate() {
                    let a = -2.0 * core::f64::consts::PI * (k * t) as f64 / n as f64;
                    re += v as f64 * a.cos();
                    im += v as f64 * a.sin();
                }
                C::new(re as f32, im as f32)
            })
            .collect()
    }

    #[test]
    fn real_fft_matches_the_dft_and_inverts() {
        for n in [4usize, 16, 256, 1024] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 7919 % 97) as f32 / 48.0 - 1.0) * (1.0 + (i % 5) as f32)).collect();
            let mut f = RealFft::new(n);
            let mut s = vec![C::default(); f.bins()];
            f.forward(&x, &mut s);
            let want = dft(&x);
            let scale = x.iter().map(|v| v.abs()).sum::<f32>();
            for (a, b) in s.iter().zip(&want) {
                assert!((a.re - b.re).abs() < 1e-4 * scale && (a.im - b.im).abs() < 1e-4 * scale, "n={n}: {a:?} vs {b:?}");
            }
            let mut y = vec![0.0f32; n];
            f.inverse(&s, &mut y);
            for (a, b) in x.iter().zip(&y) {
                assert!((a - b).abs() < 1e-4 * (1.0 + a.abs()), "n={n}: inverse {a} vs {b}");
            }
        }
    }
}
