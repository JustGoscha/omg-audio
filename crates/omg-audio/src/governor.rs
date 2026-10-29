//! The render-load governor: walks the mixer's mix-wide budgets (taps,
//! point-HRTF paths, rendering sources, convolved sources) by how long
//! rendering takes
//! against the audio it produced — shed at once when a 0.25 s window runs
//! over, earn density back after calm windows (the web worklet's rules).
//! The host measures (a wasm AudioWorklet has no clock of its own) and
//! reports; the governor sets the budgets.

use crate::mixer::Mixer;

/// Budget ladder (mix-wide taps, point-HRTF paths, rendering sources,
/// convolved sources). Rung 1 is the mixer's default.
pub const LADDER: [(usize, usize, usize, usize); 5] =
    [(48, 4, 12, 2), (96, 8, 24, 4), (192, 16, 32, 8), (320, 24, 48, 16), (480, 32, 64, 64)];

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
        let (t, p, v, c) = LADDER[self.rung];
        mixer.set_budget(t, p, v);
        mixer.set_convolution_budget(c);
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

    /// The current rung's budgets (taps, point paths, voices, convolved).
    pub fn budget(&self) -> (usize, usize, usize, usize) {
        LADDER[self.rung]
    }
}

/// The simulation's quality level (`sim::LADDER`), walked by what a tick
/// costs against a target (the host's: 16 ms leaves a 60 Hz game its
/// frame — the tick shares the machine with it): starts at the lowest
/// level and earns its way up, one level after `CALM_TICKS` ticks under
/// `CLIMB` of the target with the audio thread under `LOAD`; sheds at
/// once when a tick runs over the target — as many levels as the overrun
/// asks, each about `sim::LEVEL_STEP`× the one below — and one level
/// after `HOT_TICKS` ticks with the audio thread over `LOAD`.
pub struct LevelGovernor {
    level: usize,
    top: usize,
    calm: u32,
    hot: u32,
}

const CLIMB: f32 = 0.6;
const LOAD: f32 = 0.8;
const CALM_TICKS: u32 = 40;
const HOT_TICKS: u32 = 20;

impl LevelGovernor {
    /// Levels 0 (lowest) … `top`.
    pub fn new(top: usize) -> Self {
        Self { level: 0, top, calm: 0, hot: 0 }
    }

    pub fn level(&self) -> usize {
        self.level
    }

    /// One tick took `tick_ms` against `target_ms`, with the audio thread
    /// at `load` (render time over audio time): the new level if it moves.
    pub fn report(&mut self, tick_ms: f32, target_ms: f32, load: f32) -> Option<usize> {
        if tick_ms > target_ms && self.level > 0 {
            let over = ((tick_ms / target_ms).ln() / crate::sim::LEVEL_STEP.ln()).ceil().max(1.0) as usize;
            self.level = self.level.saturating_sub(over);
            self.calm = 0;
            self.hot = 0;
            return Some(self.level);
        }
        if load > LOAD {
            self.calm = 0;
            self.hot += 1;
            if self.hot >= HOT_TICKS && self.level > 0 {
                self.hot = 0;
                self.level -= 1;
                return Some(self.level);
            }
            return None;
        }
        self.hot = 0;
        if tick_ms < CLIMB * target_ms && self.level < self.top {
            self.calm += 1;
            if self.calm >= CALM_TICKS {
                self.calm = 0;
                self.level += 1;
                return Some(self.level);
            }
        } else {
            self.calm = 0;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::LevelGovernor;

    #[test]
    fn the_level_climbs_when_calm_and_sheds_at_once() {
        let mut g = LevelGovernor::new(7);
        for l in 1..=4 {
            for _ in 0..39 {
                assert_eq!(g.report(5.0, 16.0, 0.2), None);
            }
            assert_eq!(g.report(5.0, 16.0, 0.2), Some(l));
        }
        // a tick just over the target: one level down, now
        assert_eq!(g.report(17.0, 16.0, 0.2), Some(3));
        // three times over: as many levels as that takes
        assert_eq!(g.report(48.0, 16.0, 0.2), Some(0));
        // the audio thread pressed for a second of ticks: no climb, and
        // at the floor nothing to shed
        for _ in 0..400 {
            assert_eq!(g.report(5.0, 16.0, 0.9), None);
        }
        assert_eq!(g.level(), 0);
    }
}
