//! What a source PLAYED, kept as a log of events instead of audio.
//!
//! A path of delay d hears what the source emitted at t − d. For a source
//! built from recordings (a loop, one-shots started at known moments) that
//! emission is a pure function of time: which recording, started when, at
//! what level and rate, stopped when. So instead of a delay line holding
//! the last seconds of audio, the source keeps a few bytes per event and
//! every path reads the recordings directly at its own emission time.
//! Distance costs nothing: a bell 10 km away (29 s) needs the same memory
//! as one at 10 m, and the only limit left is the sound becoming too quiet
//! to keep a path at all. Generated or live audio has no recording to jump
//! back into and keeps the renderer's delay line (`Renderer::process`).
//!
//! Times are engine samples as f64: exact to well past a day at 48 kHz.

use std::sync::Arc;

/// A recording, shared between the fx bank and every log that plays it.
pub type Clip = Arc<[f32]>;

/// Most one-shots one source keeps on record (still playing, or still in
/// flight toward the listener); further plays are dropped.
pub const MAX_SHOTS: usize = 24;

struct LoopSeg {
    clip: Clip,
    /// Emission time this loop took over (its phase origin);
    /// −∞ for a source's first loop, which has always been playing.
    start: f64,
}

struct Shot {
    kind: u32,
    clip: Clip,
    start: f64,
    gain: f32,
    rate: f32,
    /// Emission time it was stopped (∞ = plays out).
    stop: f64,
}

impl Shot {
    /// First emission time after its last sample.
    fn end(&self) -> f64 {
        (self.start + self.clip.len() as f64 / self.rate as f64).min(self.stop)
    }
}

#[derive(Default)]
pub struct PlayLog {
    loops: Vec<LoopSeg>,
    shots: Vec<Shot>,
}

impl PlayLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loop `clip` from emission time `now` on. A source's first loop has
    /// always been playing, so a far source's bed is there from the start
    /// instead of arriving one travel time late.
    pub fn set_loop(&mut self, clip: Clip, now: f64) {
        let start = if self.loops.is_empty() { f64::NEG_INFINITY } else { now };
        self.loops.push(LoopSeg { clip, start });
    }

    /// Start a one-shot at emission time `now`. False when the log is full.
    pub fn play(&mut self, kind: u32, clip: Clip, now: f64, gain: f32, rate: f32) -> bool {
        if self.shots.len() >= MAX_SHOTS {
            return false;
        }
        let rate = if rate > 0.0 { rate } else { 1.0 };
        self.shots.push(Shot { kind, clip, start: now, gain, rate, stop: f64::INFINITY });
        true
    }

    /// Stop every one-shot of `kind` at emission time `now`. What was
    /// emitted before stays on record: paths still carrying it play it out.
    pub fn stop(&mut self, kind: u32, now: f64) {
        for s in &mut self.shots {
            if s.kind == kind && s.stop > now {
                s.stop = now;
            }
        }
    }

    /// One-shots on record.
    pub fn shots(&self) -> usize {
        self.shots.len()
    }

    /// One-shots that have not reached their end by emission time `t`.
    pub fn sounding_at(&self, t: f64) -> usize {
        self.shots.iter().filter(|s| t >= s.start && t < s.end()).count()
    }

    /// The source's emitted signal at emission time `t` (linear
    /// interpolation between samples; exact on whole samples).
    #[inline]
    pub fn at(&self, t: f64) -> f32 {
        let mut x = 0.0f32;
        let seg = match self.loops.as_slice() {
            [only] if only.start <= t => Some(only),
            [_] | [] => None,
            many => many.iter().rev().find(|s| s.start <= t),
        };
        if let Some(seg) = seg {
            let origin = if seg.start.is_finite() { seg.start } else { 0.0 };
            x += loop_sample(&seg.clip, t - origin);
        }
        for s in &self.shots {
            if t >= s.start && t < s.stop {
                x += clip_sample(&s.clip, (t - s.start) * s.rate as f64) * s.gain;
            }
        }
        x
    }

    /// Forget what no path can still hear: nothing will read before `t`.
    pub fn forget_before(&mut self, t: f64) {
        self.shots.retain(|s| s.end() > t);
        while self.loops.len() > 1 && self.loops[1].start <= t {
            self.loops.remove(0);
        }
    }
}

fn loop_sample(c: &[f32], p: f64) -> f32 {
    let n = c.len();
    if n == 0 {
        return 0.0;
    }
    // wrap with floor, not rem_euclid: in wasm f64 `%` is a libm call,
    // floor is one instruction — this runs per tap per sample
    let nf = n as f64;
    let p = p - (p / nf).floor() * nf;
    let i = (p as usize).min(n - 1);
    let f = (p - i as f64) as f32;
    let a = c[i];
    if f == 0.0 {
        return a;
    }
    a + (c[(i + 1) % n] - a) * f
}

fn clip_sample(c: &[f32], p: f64) -> f32 {
    let i = p as usize;
    if i >= c.len() {
        return 0.0;
    }
    let f = (p - i as f64) as f32;
    let a = c[i];
    if f == 0.0 {
        return a;
    }
    a + (c.get(i + 1).copied().unwrap_or(0.0) - a) * f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(v: &[f32]) -> Clip {
        v.into()
    }

    #[test]
    fn a_shot_plays_its_clip_from_its_start() {
        let mut log = PlayLog::new();
        log.play(0, clip(&[1.0, 2.0, 3.0]), 100.0, 0.5, 1.0);
        assert_eq!(log.at(99.0), 0.0);
        assert_eq!(log.at(100.0), 0.5);
        assert_eq!(log.at(101.5), 1.25);
        assert_eq!(log.at(102.0), 1.5);
        assert_eq!(log.at(103.0), 0.0);
    }

    #[test]
    fn rate_and_stop() {
        let mut log = PlayLog::new();
        log.play(0, clip(&[0.0, 1.0, 2.0, 3.0]), 0.0, 1.0, 2.0);
        assert_eq!(log.at(1.0), 2.0);
        assert_eq!(log.at(2.0), 0.0, "4 samples at 2x last 2");
        log.play(1, clip(&[1.0; 100]), 0.0, 1.0, 1.0);
        log.stop(1, 10.0);
        assert_eq!(log.at(9.0), 1.0, "emitted before the stop stays");
        assert_eq!(log.at(10.0), 0.0);
    }

    #[test]
    fn far_past_is_read_exactly() {
        // 10 km: the listener hears emission time now − 1.4 M samples
        let mut log = PlayLog::new();
        let now = 5.0e9;
        log.play(0, clip(&[0.25, 0.5]), now, 1.0, 1.0);
        let d = 1_399_416.0;
        assert_eq!(log.at(now + d - d), 0.25);
        assert_eq!(log.at(now + 1.0), 0.5);
    }

    #[test]
    fn loops_have_always_played_and_hand_over() {
        let mut log = PlayLog::new();
        log.set_loop(clip(&[1.0, 2.0, 3.0, 4.0]), 500.0);
        assert_eq!(log.at(-2.0), 3.0, "the first loop has no start");
        assert_eq!(log.at(5.0), 2.0);
        log.set_loop(clip(&[9.0, 8.0]), 1000.0);
        assert_eq!(log.at(999.0), 4.0, "before the swap: old loop");
        assert_eq!(log.at(1000.0), 9.0, "new loop from its own start");
        assert_eq!(log.at(1001.0), 8.0);
        log.forget_before(1000.0);
        assert_eq!(log.loops.len(), 1);
    }

    #[test]
    fn forgets_only_what_nobody_can_hear() {
        let mut log = PlayLog::new();
        log.play(0, clip(&[1.0; 10]), 0.0, 1.0, 1.0);
        log.forget_before(9.5);
        assert_eq!(log.shots(), 1);
        log.forget_before(10.0);
        assert_eq!(log.shots(), 0);
        for _ in 0..MAX_SHOTS + 3 {
            log.play(0, clip(&[1.0]), 0.0, 1.0, 1.0);
        }
        assert_eq!(log.shots(), MAX_SHOTS);
    }
}
