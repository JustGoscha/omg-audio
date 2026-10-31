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

/// The part of a tick (ms) the quality ladder doesn't scale, from its
/// parts (`omg_core::v3::TickCosts::parts`: direct, one-edge, two-edge,
/// early, late, chains, fold, portals): the direct paths (never shed),
/// the exactly solved mirror chains (named off the faces and a fixed few
/// mirror rays up to the middle levels, every cached one solved each
/// tick) and the portals' source rays (a floor of them on the lower
/// levels). The rest — the rays, the edge searches, the fold of what they
/// found — counts as scaling (climbing, the safe side).
pub fn fixed_ms(parts: &[f32; 8]) -> f32 {
    parts[0] + parts[5] + parts[7]
}

/// The simulation's quality level (`sim::LADDER`), walked by what a tick
/// costs (the tick runs on a thread of its own every 50 ms): starts at
/// the lowest level and earns its way up.
/// - **Down:** a tick over `shed_ms` (30 ms) — a level down at once; that
///   level stays out of reach for `HOLD_S` (no flip-flop round it).
/// - **Up:** the window's mean tick (`WINDOW`: 2 s) under `climb_ms` (12)
///   with the game holding its frame rate and the audio thread under
///   `LOAD` — a level up, after `CALM_TICKS` at this one.
/// - The audio thread over `LOAD` for `HOT_TICKS`: a level down.
///
/// On the GPU a tick is 2–6 ms until a walk lands, 15–18 the tick it does:
/// judged tick by tick against a 16 ms target the governor sat at 0/10 on
/// a machine that plays level 8 at 17–19 ms a tick and 107 fps (Godot,
/// Oct 31). The mean keeps the landed walks' share; a tick past 30 ms is
/// one that matters.
pub struct LevelGovernor {
    level: usize,
    top: usize,
    calm: u32,
    hot: u32,
    /// The last ticks' costs (ms).
    recent: std::collections::VecDeque<f32>,
    /// A level shed from, and the ticks left before it may be climbed to.
    held: Option<(usize, u32)>,
}

const LOAD: f32 = 0.8;
/// Ticks at a level before it may climb (1 s at 20 Hz).
const CALM_TICKS: u32 = 20;
const HOT_TICKS: u32 = 20;
/// The window the mean is taken over (2 s at 20 Hz).
const WINDOW: usize = 40;
/// A level shed from stays out of reach this many ticks (10 s at 20 Hz).
const HOLD_TICKS: u32 = 200;

impl LevelGovernor {
    /// Levels 0 (lowest) … `top`.
    pub fn new(top: usize) -> Self {
        Self { level: 0, top, calm: 0, hot: 0, recent: std::collections::VecDeque::new(), held: None }
    }

    pub fn level(&self) -> usize {
        self.level
    }

    /// One tick took `tick_ms`; over `shed_ms` sheds, a window's mean
    /// under `climb_ms` with the game's frames held (`frames_ok`) and the
    /// audio thread at `load` (render time over audio time) under `LOAD`
    /// climbs: the new level if it moves.
    pub fn report(&mut self, tick_ms: f32, shed_ms: f32, climb_ms: f32, frames_ok: bool, load: f32) -> Option<usize> {
        self.recent.push_back(tick_ms);
        while self.recent.len() > WINDOW {
            self.recent.pop_front();
        }
        if let Some((l, left)) = self.held {
            self.held = (left > 1).then_some((l, left - 1));
        }
        if tick_ms > shed_ms && self.level > 0 {
            self.held = Some((self.level, HOLD_TICKS));
            return Some(self.moved(self.level - 1));
        }
        if load > LOAD {
            self.calm = 0;
            self.hot += 1;
            if self.hot >= HOT_TICKS && self.level > 0 {
                return Some(self.moved(self.level - 1));
            }
            return None;
        }
        self.hot = 0;
        if !frames_ok {
            self.calm = 0;
            return None;
        }
        self.calm += 1;
        let next = self.level + 1;
        if self.calm < CALM_TICKS || next > self.top || self.recent.len() < CALM_TICKS as usize || self.held.is_some_and(|(l, _)| next >= l) {
            return None;
        }
        let mean = self.recent.iter().sum::<f32>() / self.recent.len() as f32;
        (mean < climb_ms).then(|| self.moved(next))
    }

    /// Now at `level`: what the window held was another level's.
    fn moved(&mut self, level: usize) -> usize {
        self.level = level;
        self.calm = 0;
        self.hot = 0;
        self.recent.clear();
        level
    }
}

#[cfg(test)]
mod tests {
    use super::LevelGovernor;

    const SHED: f32 = 30.0;
    const CLIMB: f32 = 12.0;

    /// Calm ticks: a level a second; a tick over 30 ms: a level down at
    /// once, and not back up to the level shed from for 10 s.
    #[test]
    fn up_while_the_mean_is_low_down_at_a_slow_tick() {
        let mut g = LevelGovernor::new(7);
        for _ in 0..80 {
            g.report(5.0, SHED, CLIMB, true, 0.2);
        }
        assert_eq!(g.level(), 4, "a level a second while calm");
        assert_eq!(g.report(31.0, SHED, CLIMB, true, 0.2), Some(3), "a tick over 30 ms: down at once");
        for _ in 0..180 {
            g.report(5.0, SHED, CLIMB, true, 0.2);
        }
        assert_eq!(g.level(), 3, "not back to the level shed from within 10 s");
        for _ in 0..60 {
            g.report(5.0, SHED, CLIMB, true, 0.2);
        }
        assert!(g.level() >= 4, "after 10 s it may: {}", g.level());
    }

    /// The GPU's ticks: 3 ms until a walk lands, 17 the tick it does —
    /// every fourth (mean 6.5): it climbs, a level a second.
    #[test]
    fn the_gpus_landed_walks_do_not_hold_the_level_down() {
        let mut g = LevelGovernor::new(10);
        for i in 0..100 {
            g.report(if i % 4 == 3 { 17.0 } else { 3.0 }, SHED, CLIMB, true, 0.2);
        }
        assert!(g.level() >= 4, "the landed walks held auto at {}", g.level());
    }

    /// No climb while the game misses its frames or the audio thread is
    /// pressed (that one sheds, a level a second); a mean over 12 ms holds.
    #[test]
    fn frames_audio_and_a_high_mean_hold_it() {
        let mut g = LevelGovernor::new(7);
        for _ in 0..100 {
            g.report(5.0, SHED, CLIMB, false, 0.2);
        }
        assert_eq!(g.level(), 0, "the game under its frame rate: no climb");
        let mut h = LevelGovernor::new(7);
        for _ in 0..100 {
            h.report(14.0, SHED, CLIMB, true, 0.2);
        }
        assert_eq!(h.level(), 0, "a mean over 12 ms: no climb");
        for _ in 0..40 {
            g.report(5.0, SHED, CLIMB, true, 0.2);
        }
        let up = g.level();
        assert!(up >= 1);
        for _ in 0..40 {
            g.report(5.0, SHED, CLIMB, true, 0.9);
        }
        assert!(g.level() < up, "the audio thread pressed: down");
    }
}
