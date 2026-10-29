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
    /// Both internal states below `eps` (fed silence, its output is
    /// negligible).
    pub fn quiet(&self, eps: f32) -> bool {
        self.lp_low.state.abs() < eps && self.lp_high.state.abs() < eps
    }

    /// `g_low`, `g_high`: amplitude ratios vs. mid band.
    #[inline]
    pub fn tick(&mut self, x: f32, g_low: f32, g_high: f32, c_low: f32, c_high: f32) -> f32 {
        let low = self.lp_low.tick(x, c_low);
        let y = x + (g_low - 1.0) * low; // low shelf
        let high_part = y - self.lp_high.tick(y, c_high);
        y + (g_high - 1.0) * high_part // high shelf
    }
}

/// A path's three band gains as one filter: a low shelf at the tracer's
/// lower band edge and a high shelf at its upper (second order, RBJ,
/// slope 1), at the low and high band's gains over the mid's — every band
/// centre within ~1.7 dB of its gain for steps up to ~20 dB between
/// neighbouring bands; a wall's whole span (bass 28 dB over the mids,
/// treble 20 under) comes within 2–4.3 dB, the shelves not quite turned
/// by the next centre (first-order shelves capped a −20 dB band near −8,
/// and the fitted low-pass left a wall's treble 28 dB high). The
/// coefficients follow the gains as they glide (`set`); unit gains are
/// the identity, bit for bit.
#[derive(Clone, Copy)]
pub struct Shelves {
    lo: Sos,
    hi: Sos,
    /// The gains the coefficients are for.
    at: (f32, f32),
}

#[derive(Clone, Copy)]
struct Sos {
    b: [f32; 3],
    a: [f32; 2],
    z: [f32; 2],
}

impl Sos {
    const IDENTITY: Self = Self { b: [1.0, 0.0, 0.0], a: [0.0, 0.0], z: [0.0, 0.0] };

    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.z[0];
        self.z[0] = self.b[1] * x - self.a[0] * y + self.z[1];
        self.z[1] = self.b[2] * x - self.a[1] * y;
        y
    }
}

/// The shelves' fixed parts at a sample rate: per edge cos ω₀ and
/// sin ω₀ / √2 (the slope-1 α, the same at any gain).
#[derive(Clone, Copy, Debug)]
pub struct ShelfEdges {
    lo: (f32, f32),
    hi: (f32, f32),
}

impl ShelfEdges {
    pub fn new(sample_rate: f32) -> Self {
        let edge = |f: f32| {
            let w = core::f32::consts::TAU * f / sample_rate;
            (w.cos(), w.sin() * core::f32::consts::FRAC_1_SQRT_2)
        };
        Self { lo: edge(crate::bands::XOVER_LO_HZ), hi: edge(crate::bands::XOVER_HI_HZ) }
    }
}

impl Default for Shelves {
    fn default() -> Self {
        Self { lo: Sos::IDENTITY, hi: Sos::IDENTITY, at: (1.0, 1.0) }
    }
}

/// A shelf's coefficients (RBJ, slope 1) for amplitude ratio `g`
/// (floored 80 dB down): A = 10^(dB/40) = √g. Unit gain: the identity.
fn shelf(sos: &mut Sos, g: f32, (c, al): (f32, f32), high: bool) {
    if g == 1.0 {
        sos.b = Sos::IDENTITY.b;
        sos.a = Sos::IDENTITY.a;
        return;
    }
    let a = g.max(1e-4).sqrt();
    let sq = 2.0 * a.sqrt() * al;
    let s = if high { -1.0 } else { 1.0 };
    let (ap, am) = (a + 1.0, a - 1.0);
    let a0 = ap + s * am * c + sq;
    let b = [a * (ap - s * am * c + sq), s * 2.0 * a * (am - s * ap * c), a * (ap - s * am * c - sq)];
    let den = [-s * 2.0 * (am + s * ap * c), ap + s * am * c - sq];
    sos.b = b.map(|v| v / a0);
    sos.a = den.map(|v| v / a0);
}

impl Shelves {
    /// The shelves for low and high band gains `g_low`, `g_high` over the
    /// mid's (amplitude ratios). The filters' state carries on.
    pub fn set(&mut self, g_low: f32, g_high: f32, k: &ShelfEdges) {
        if (g_low, g_high) == self.at {
            return;
        }
        self.at = (g_low, g_high);
        shelf(&mut self.lo, g_low, k.lo, false);
        shelf(&mut self.hi, g_high, k.hi, true);
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let y = self.lo.tick(x);
        self.hi.tick(y)
    }

    /// Both sections' state below `eps`.
    pub fn quiet(&self, eps: f32) -> bool {
        self.lo.z.iter().chain(&self.hi.z).all(|v| v.abs() < eps)
    }
}
