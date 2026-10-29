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

#[derive(Clone)]
struct LoopSeg {
    clip: Clip,
    /// Emission time this loop took over (its phase origin);
    /// −∞ for a source's first loop, which has always been playing.
    start: f64,
    gain: f32,
    rate: f32,
}

/// Loop segments kept preallocated (a few hand-overs within one horizon).
const LOOPS_RESERVED: usize = 4;

#[derive(Clone)]
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

/// One thing on record that sounds (`PlayLog::sounding`): `clip` at
/// `gain`, from emission time `start` (a one-shot's first sample; a loop's
/// hand-over, −∞ for a first loop) to `end`, read at `rate`.
pub struct Sounding<'a> {
    pub clip: &'a Clip,
    pub gain: f32,
    pub start: f64,
    pub end: f64,
    pub rate: f32,
    pub looped: bool,
}

pub struct PlayLog {
    loops: Vec<LoopSeg>,
    shots: Vec<Shot>,
    /// The emission times that are this emitter's: what the source emitted
    /// before `since` or from `until` on left from another place (a
    /// teleport: the old place's emissions play on in a ghost, `ghost`).
    since: f64,
    until: f64,
}

impl Default for PlayLog {
    fn default() -> Self {
        Self { loops: Vec::new(), shots: Vec::new(), since: f64::NEG_INFINITY, until: f64::INFINITY }
    }
}

impl PlayLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// A log with its event storage reserved up front: `play` and loop
    /// hand-overs then never allocate (real-time callers).
    pub fn preallocated() -> Self {
        Self { loops: Vec::with_capacity(LOOPS_RESERVED), shots: Vec::with_capacity(MAX_SHOTS), ..Self::default() }
    }

    /// Loop `clip` from emission time `now` on. A source's first loop has
    /// always been playing, so a far source's bed is there from the start
    /// instead of arriving one travel time late.
    pub fn set_loop(&mut self, clip: Clip, now: f64) {
        self.set_loop_with(clip, now, 1.0, 1.0);
    }

    /// `set_loop` at its own level and playback rate.
    pub fn set_loop_with(&mut self, clip: Clip, now: f64, gain: f32, rate: f32) {
        let start = if self.loops.is_empty() { f64::NEG_INFINITY } else { now };
        let rate = if rate > 0.0 { rate } else { 1.0 };
        self.loops.push(LoopSeg { clip, start, gain, rate });
    }

    /// End the loop at emission time `now` (paths still carrying what it
    /// emitted before play it out). `silence` is an empty clip the caller
    /// keeps around, so stopping never allocates.
    pub fn stop_loop(&mut self, now: f64, silence: Clip) {
        if self.loops.last().is_some_and(|l| !l.clip.is_empty()) {
            self.loops.push(LoopSeg { clip: silence, start: now, gain: 0.0, rate: 1.0 });
        }
    }

    /// What emitted before `t` is no longer this emitter's (the source
    /// jumped: it left from the old place — see `ghost`).
    pub fn emitted_from(&mut self, t: f64) {
        self.since = self.since.max(t);
    }

    /// The emitter the source was until `t` (it jumped then): the same
    /// record, ending at `t` — its paths from the old place play what it
    /// emitted there, until it has all arrived and rung out.
    pub fn ghost(&self, t: f64) -> PlayLog {
        PlayLog { loops: self.loops.clone(), shots: self.shots.clone(), since: self.since, until: t.min(self.until) }
    }

    /// Last emission time anything on record still sounds: +∞ while a
    /// loop plays, −∞ when nothing was ever played. A renderer whose
    /// paths and reverb have all moved past this has nothing left to say.
    pub fn sounds_until(&self) -> f64 {
        if self.loops.last().is_some_and(|l| !l.clip.is_empty() && l.gain != 0.0) {
            return self.until;
        }
        // a stopped loop sounds up to its stop
        let mut end = f64::NEG_INFINITY;
        if self.loops.len() > 1 {
            end = self.loops[self.loops.len() - 1].start;
        }
        for s in &self.shots {
            end = end.max(s.end());
        }
        end.min(self.until)
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

    /// Restart the one-shots of these kinds (still on record, not
    /// stopped) as if they began at emission time `now`: for a source
    /// heard for the first time, what it emitted while its paths were
    /// unknown plays from its beginning, late by the trace, instead of
    /// from the middle (or, a click shorter than a trace, not at all).
    pub fn restart(&mut self, kinds: &[u32], now: f64) {
        for s in &mut self.shots {
            if kinds.contains(&s.kind) && s.start < now && s.stop == f64::INFINITY {
                s.start = now;
            }
        }
    }

    /// Everything on record that sounds between emission times `t0` and
    /// `t1`: its clip, gain, the span it sounds [from, to) and its rate —
    /// a loop (`looped`: at any phase) or a one-shot (read from its
    /// beginning at `from`... see `Sounding`).
    pub fn sounding(&self, t0: f64, t1: f64, mut f: impl FnMut(Sounding<'_>)) {
        // only what is this emitter's
        let (t0, t1) = (t0.max(self.since), t1.min(self.until));
        if t1 <= t0 {
            return;
        }
        for (i, seg) in self.loops.iter().enumerate() {
            let until = self.loops.get(i + 1).map_or(f64::INFINITY, |n| n.start);
            if seg.clip.is_empty() || seg.gain == 0.0 || until <= t0 || seg.start >= t1 {
                continue;
            }
            f(Sounding { clip: &seg.clip, gain: seg.gain, start: seg.start.max(self.since), end: until.min(self.until), rate: seg.rate, looped: true });
        }
        for s in &self.shots {
            let end = s.end();
            if end <= t0 || s.start >= t1 {
                continue;
            }
            f(Sounding { clip: &s.clip, gain: s.gain, start: s.start, end: end.min(self.until), rate: s.rate, looped: false });
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
        if t < self.since || t >= self.until {
            return 0.0;
        }
        let mut x = 0.0f32;
        let seg = match self.loops.as_slice() {
            [only] if only.start <= t => Some(only),
            [_] | [] => None,
            many => many.iter().rev().find(|s| s.start <= t),
        };
        if let Some(seg) = seg {
            let origin = if seg.start.is_finite() { seg.start } else { 0.0 };
            x += if seg.rate == 1.0 {
                loop_sample(&seg.clip, t - origin)
            } else {
                loop_sample(&seg.clip, (t - origin) * seg.rate as f64)
            } * seg.gain;
        }
        for s in &self.shots {
            // (a read up to one sample before the start interpolates into
            // the first sample from the silence before it)
            if t > s.start - 1.0 / s.rate as f64 && t < s.stop {
                x += clip_sample(&s.clip, (t - s.start) * s.rate as f64) * s.gain;
            }
        }
        x
    }

    /// `out[i] = at(t[i]) · gain[i]` for a block of emission times — one
    /// path's reads over a render block. `t` moves by about one sample
    /// per entry (slowly varying delay), so the loop segment is looked up
    /// once and its read position stepped instead of re-wrapped; one-shots
    /// that cannot sound anywhere in the block are skipped whole.
    pub fn read_block(&self, t: &[f64], gain: &[f32], out: &mut [f32]) {
        let n = t.len().min(gain.len()).min(out.len());
        let (t, gain, out) = (&t[..n], &gain[..n], &mut out[..n]);
        if n == 0 {
            return;
        }
        // the block's emission-time span — only needed when a loop
        // hand-over or a one-shot could start or end inside it (plain
        // compares: f64::min/max's NaN rules cost a select chain in wasm)
        let always = matches!(self.loops.as_slice(), [only] if only.start == f64::NEG_INFINITY);
        let (lo, hi) = if always && self.shots.is_empty() {
            (t[0], t[0])
        } else {
            let (mut lo, mut hi) = (t[0], t[0]);
            for &x in t {
                if x < lo {
                    lo = x;
                }
                if x > hi {
                    hi = x;
                }
            }
            (lo, hi)
        };
        // loop part: one segment covering the whole block (the usual
        // case), else per sample
        let seg_of = |x: f64| -> Option<usize> {
            match self.loops.as_slice() {
                [only] if only.start <= x => Some(0),
                [_] | [] => None,
                many => many.iter().rposition(|s| s.start <= x),
            }
        };
        let (slo, shi) = (seg_of(lo), seg_of(hi));
        if slo == shi {
            match slo.map(|i| &self.loops[i]) {
                Some(seg) if !seg.clip.is_empty() => {
                    let origin = if seg.start.is_finite() { seg.start } else { 0.0 };
                    loop_block(&seg.clip, t, origin, seg.rate as f64, seg.gain, gain, out);
                }
                _ => out.fill(0.0),
            }
        } else {
            for ((o, &x), g) in out.iter_mut().zip(t).zip(gain) {
                *o = match seg_of(x).map(|i| &self.loops[i]) {
                    Some(seg) => {
                        let origin = if seg.start.is_finite() { seg.start } else { 0.0 };
                        let p = if seg.rate == 1.0 { x - origin } else { (x - origin) * seg.rate as f64 };
                        loop_sample(&seg.clip, p) * seg.gain * g
                    }
                    None => 0.0,
                };
            }
        }
        for s in &self.shots {
            let rate = s.rate as f64;
            let from = s.start - 1.0 / rate;
            if hi <= from || lo >= s.end() {
                continue;
            }
            for ((o, &x), g) in out.iter_mut().zip(t).zip(gain) {
                if x > from && x < s.stop {
                    *o += clip_sample(&s.clip, (x - s.start) * rate) * s.gain * g;
                }
            }
        }
        // only what is this emitter's (a ghost's, a jumped source's)
        if self.since > f64::NEG_INFINITY || self.until < f64::INFINITY {
            for (o, &x) in out.iter_mut().zip(t) {
                if x < self.since || x >= self.until {
                    *o = 0.0;
                }
            }
        }
    }

    /// Forget what no path can still hear: nothing will read before `t`.
    pub fn forget_before(&mut self, t: f64) {
        self.forget_before_keeping(t, &[]);
    }

    /// `forget_before`, but one-shots of the `keep` kinds stay on record
    /// (sounds waiting for their source's first paths, see `restart`).
    pub fn forget_before_keeping(&mut self, t: f64, keep: &[u32]) {
        self.shots.retain(|s| s.end() > t || keep.contains(&s.kind));
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

/// `out[i] = loop_sample(c, (t[i] − origin)·rate) · seg_gain · gain[i]`,
/// stepping the wrapped read position from `t[0]` instead of re-wrapping
/// every sample.
fn loop_block(c: &[f32], t: &[f64], origin: f64, rate: f64, seg_gain: f32, gain: &[f32], out: &mut [f32]) {
    let n = c.len();
    let nf = n as f64;
    let wrap = |p: f64| p - (p / nf).floor() * nf;
    let t0 = t[0];
    let p0 = wrap((t0 - origin) * rate);
    for ((o, &x), g) in out.iter_mut().zip(t).zip(gain) {
        let mut p = p0 + (x - t0) * rate;
        if p >= nf {
            p -= nf;
            if p >= nf {
                p = wrap(p);
            }
        } else if p < 0.0 {
            p += nf;
            if p < 0.0 {
                p = wrap(p);
            }
        }
        let i = (p as usize).min(n - 1);
        let f = (p - i as f64) as f32;
        let a = c[i];
        let b = c[if i + 1 == n { 0 } else { i + 1 }];
        *o = (a + (b - a) * f) * seg_gain * g;
    }
}

/// A clip's sample at fractional position `p`, linear between samples and
/// from the silence on either side: a read within a sample before its
/// start ramps into the first sample (a fractional delay must not lose
/// the onset — a click read at a phase of 0.9 kept 1 % of its energy).
fn clip_sample(c: &[f32], p: f64) -> f32 {
    if p < 0.0 {
        return if p > -1.0 { c.first().copied().unwrap_or(0.0) * (1.0 + p) as f32 } else { 0.0 };
    }
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
    fn loops_take_gain_rate_and_stop() {
        let mut log = PlayLog::preallocated();
        assert_eq!(log.sounds_until(), f64::NEG_INFINITY);
        log.set_loop_with(clip(&[1.0, 2.0, 3.0, 4.0]), 0.0, 0.5, 2.0);
        assert_eq!(log.at(0.0), 0.5);
        assert_eq!(log.at(1.0), 1.5, "rate 2 skips a sample");
        assert_eq!(log.sounds_until(), f64::INFINITY);
        log.stop_loop(10.0, clip(&[]));
        assert_eq!(log.at(9.0), 1.5, "before the stop: still looping (18 mod 4 = 2)");
        assert_eq!(log.at(10.0), 0.0);
        assert_eq!(log.sounds_until(), 10.0);
        log.play(0, clip(&[1.0; 5]), 20.0, 1.0, 1.0);
        assert_eq!(log.sounds_until(), 25.0);
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
