//! The render-load governor: walks the mixer's mix-wide budgets (taps,
//! point-HRTF paths, rendering sources) by how long rendering takes
//! against the audio it produced — shed at once when a 0.25 s window runs
//! over, earn density back after calm windows (the web worklet's rules).
//! The host measures (a wasm AudioWorklet has no clock of its own) and
//! reports; the governor sets the budgets.

use crate::mixer::Mixer;

/// Budget ladder (mix-wide taps, point-HRTF paths, rendering sources).
/// Rung 1 is the mixer's default.
pub const LADDER: [(usize, usize, usize); 5] = [(48, 4, 12), (96, 8, 24), (192, 16, 32), (320, 24, 48), (480, 32, 64)];

/// Render time over audio time: over this, shed straight to the floor.
const EMERGENCY: f32 = 0.80;
/// Over this, one rung down.
const HEAVY: f32 = 0.55;
/// Under this for `CALM_WINDOWS` windows in a row, one rung up.
const LIGHT: f32 = 0.30;
const CALM_WINDOWS: u32 = 4;

pub struct Governor {
    rung: usize,
    pinned: Option<usize>,
    load_s: f64,
    load_frames: usize,
    calm: u32,
    load: f32,
    sample_rate: f32,
}

impl Governor {
    pub fn new(sample_rate: f32) -> Self {
        Self { rung: 1, pinned: None, load_s: 0.0, load_frames: 0, calm: 0, load: 0.0, sample_rate }
    }

    /// Put the current rung's budgets on `mixer` (call once at start).
    pub fn apply(&self, mixer: &mut Mixer) {
        let (t, p, v) = LADDER[self.rung];
        mixer.set_budget(t, p, v);
    }

    fn set_rung(&mut self, mixer: &mut Mixer, r: usize) {
        self.rung = r;
        self.apply(mixer);
    }

    /// Rendering `frames` samples took `secs`: govern once a 0.25 s
    /// window is full (then true).
    pub fn report(&mut self, mixer: &mut Mixer, secs: f64, frames: usize) -> bool {
        self.load_s += secs;
        self.load_frames += frames;
        if (self.load_frames as f32) < self.sample_rate / 4.0 {
            return false;
        }
        let ratio = (self.load_s / (self.load_frames as f64 / self.sample_rate as f64)) as f32;
        self.load_s = 0.0;
        self.load_frames = 0;
        self.load = ratio;
        if self.pinned.is_some() {
            return true;
        }
        if ratio > EMERGENCY && self.rung > 0 {
            self.set_rung(mixer, 0);
            self.calm = 0;
        } else if ratio > HEAVY && self.rung > 0 {
            self.set_rung(mixer, self.rung - 1);
            self.calm = 0;
        } else if ratio < LIGHT {
            self.calm += 1;
            if self.calm >= CALM_WINDOWS && self.rung + 1 < LADDER.len() {
                self.calm = 0;
                self.set_rung(mixer, self.rung + 1);
            }
        } else {
            self.calm = 0;
        }
        true
    }

    /// The output missed its deadline (the audio thread inserted
    /// silence): shed to the floor now and start a fresh window.
    pub fn missed_deadline(&mut self, mixer: &mut Mixer) {
        if self.pinned.is_none() && self.rung > 0 {
            self.set_rung(mixer, 0);
        }
        self.calm = 0;
        self.load_s = 0.0;
        self.load_frames = 0;
    }

    /// Pin a rung (None hands it back to the governor).
    pub fn pin(&mut self, mixer: &mut Mixer, rung: Option<usize>) {
        self.pinned = rung.map(|r| r.min(LADDER.len() - 1));
        let r = self.pinned.unwrap_or(self.rung);
        self.set_rung(mixer, r);
    }

    pub fn rung(&self) -> usize {
        self.rung
    }

    /// The last window's render time over audio time.
    pub fn load(&self) -> f32 {
        self.load
    }

    /// The current rung's budgets (taps, point paths, voices).
    pub fn budget(&self) -> (usize, usize, usize) {
        LADDER[self.rung]
    }
}
