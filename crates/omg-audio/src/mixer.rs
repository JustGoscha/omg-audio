//! `Mixer`: the real-time half. Renders every source's paths (from the
//! latest `Frame`) to binaural stereo, one sample clock for everything.
//!
//! Sources are made of recordings: each keeps a *play log* (what clip
//! started when, at what level and rate) instead of audio, and every path
//! reads the log at its own emission time — a bell 2 km away needs no
//! more memory than one at 2 m, and Doppler falls out of the moving path
//! delays. A clip is mono f32 at the mixer's sample rate.
//!
//! Cost is bounded by mix-wide budgets, not by the number of sources:
//! after each frame every audible source's paths are ranked by loudness
//! at the ears; the strongest `mix_taps` render (each source keeps at
//! least its strongest), the strongest `mix_points` of them get their own
//! HRIR convolution, and only the `max_voices` loudest sources render at
//! all (the rest fade out and resume in time — their logs keep the
//! clock). A source with nothing playing and its reverb rung out costs
//! nothing.
//!
//! Real-time contract: `render` never allocates, locks or blocks. The
//! exceptions, all outside `render`: `apply` (re-targets the renderers;
//! omg-dsp's `Renderer::set_params` allocates small scratch vectors —
//! once per frame, 10–30 Hz), and creating the channel of a source seen
//! for the first time beyond the preallocated capacity (`with_capacity`).
//! `render` may drop the last reference to a finished clip (a free) —
//! keep your clips alive in a bank so it never does.

use crate::frame::{Frame, SourceId};
use omg_core::params::ParamBlock;
use omg_dsp::ambi::NCH;
use omg_dsp::hrtf::HrirGrid;
use omg_dsp::near::NearVoice;
use omg_dsp::output::OutputStage;
use omg_dsp::renderer::{BusBlock, BLOCK};
use omg_dsp::smooth::Smoothed;
use omg_dsp::{PlayLog, Renderer};
use std::collections::HashMap;
use std::sync::Arc;

/// A recording: mono f32 samples at the mixer's sample rate, shared
/// (cloning is a reference count).
pub type Clip = omg_dsp::playlog::Clip;

/// A clip from samples.
pub fn clip(samples: &[f32]) -> Clip {
    Clip::from(samples)
}

/// Normalize a recording's loudness (gated RMS → the engine's reference
/// level): how a clip was recorded stops mattering, the source's `gain`
/// sets its loudness. Optional; call before `clip`.
pub fn normalize(samples: &mut [f32]) {
    omg_dsp::level::normalize_rms(samples, omg_dsp::level::REF_CLIP_RMS);
}

/// Head-related impulse responses for the binaural decode.
#[derive(Clone, Default)]
pub struct Hrir {
    speakers: Option<Arc<[u8]>>,
    grid: Option<Arc<HrirGrid>>,
}

impl Hrir {
    /// No HRIRs: a virtual-cardioid stereo decode (works anywhere,
    /// localizes less sharply — fine for tests and speakers).
    pub fn none() -> Self {
        Self::default()
    }

    /// From the engine's HRIR files: `speakers` = a virtual-speaker set
    /// (`assets/hrir_dodeca20.bin`) for the ambisonic decode, `grid` = a
    /// dense direction grid (`assets/hrir_grid.bin`) for the strongest
    /// paths' own convolution. Either may be None.
    pub fn from_bytes(speakers: Option<&[u8]>, grid: Option<&[u8]>) -> Self {
        Self { speakers: speakers.map(Arc::from), grid: grid.map(|g| Arc::new(HrirGrid::from_bytes(g))) }
    }

    /// Load `hrir_dodeca20.bin` (else `hrir_ico12.bin`) and
    /// `hrir_grid.bin` from a directory; missing files are skipped.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_dir(dir: impl AsRef<std::path::Path>) -> Self {
        let d = dir.as_ref();
        let spk = std::fs::read(d.join("hrir_dodeca20.bin")).or_else(|_| std::fs::read(d.join("hrir_ico12.bin"))).ok();
        let grid = std::fs::read(d.join("hrir_grid.bin")).ok();
        Self::from_bytes(spk.as_deref(), grid.as_deref())
    }

    pub fn is_binaural(&self) -> bool {
        self.speakers.is_some()
    }
}

/// How to play a clip on a source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Play {
    /// Loop it (one loop per source: a new loop replaces the old one)
    /// or play it once (up to 24 one-shots in flight per source).
    pub looped: bool,
    /// Amplitude.
    pub gain: f32,
    /// Playback rate (1.03 = 3% faster and a third of a semitone higher).
    pub rate: f32,
}

impl Play {
    pub const ONCE: Play = Play { looped: false, gain: 1.0, rate: 1.0 };
    pub const LOOP: Play = Play { looped: true, gain: 1.0, rate: 1.0 };

    pub fn gain(self, gain: f32) -> Self {
        Self { gain, ..self }
    }

    pub fn rate(self, rate: f32) -> Self {
        Self { rate, ..self }
    }
}

/// A playing clip. `VoiceId::NONE` = the play was dropped (the source's
/// one-shot log was full, or all near-field voices busy).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VoiceId(pub u32);

impl VoiceId {
    pub const NONE: VoiceId = VoiceId(0);
}

/// A channel's level since the last `take_meter`: peak and RMS amplitude.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Meter {
    pub peak: f32,
    pub rms: f32,
}

#[derive(Clone, Copy, Default)]
struct MeterAcc {
    peak2: f32,
    sum2: f64,
    n: u32,
}

impl MeterAcc {
    fn add(&mut self, m2: f32) {
        self.peak2 = self.peak2.max(m2);
        self.sum2 += m2 as f64;
        self.n += 1;
    }

    fn take(&mut self) -> Meter {
        let m = Meter {
            peak: self.peak2.sqrt(),
            rms: if self.n > 0 { ((self.sum2 / self.n as f64) as f32).sqrt() } else { 0.0 },
        };
        *self = Self::default();
        m
    }
}

struct Slot {
    id: SourceId,
    used: bool,
    renderer: Renderer,
    log: PlayLog,
    /// User volume (amplitude), smoothed.
    fader: Smoothed,
    /// Voice virtualization ramp: 1 = among the `max_voices` loudest.
    audible: Smoothed,
    latest: ParamBlock,
    /// What the renderer gets: `latest` cut to this source's share of
    /// the tap budget, with hysteresis (see `select`).
    filtered: ParamBlock,
    /// Keys of the paths admitted last time (sorted), and scratch.
    admitted: Vec<u32>,
    admitted_next: Vec<u32>,
    /// `filtered` needs rebuilding (new block or new share).
    dirty: bool,
    /// The share changed: the renderer must re-apply even the same block.
    reapply: bool,
    has_block: bool,
    in_frame: bool,
    /// Host time of the frame `latest` came from, and the simulated time
    /// (samples) since the block before it — for the renderer's next
    /// `set_params_after` (see `Frame::time`).
    traced_at: Option<f64>,
    motion: Option<f64>,
    loop_voice: u32,
    share: (usize, usize),
    /// Loudness at the ears for the ranking (path gains × fader).
    loud: f32,
    /// `quiet_after`, cached: refreshed when the log or the paths change
    /// (play/stop, apply) and after each block it rendered — a channel
    /// that does not render costs nothing per block.
    quiet_at: f64,
    audible_next: bool,
    meter: MeterAcc,
}

impl Slot {
    /// Emission-clock time after which nothing it plays can still be
    /// heard: its last sound + the longest path + the reverb's ring.
    fn quiet_after(&self, sample_rate: f32) -> f64 {
        let end = self.log.sounds_until();
        if !end.is_finite() {
            return end;
        }
        let pb = &self.latest;
        let mut rt = pb.reverb.rt60.iter().fold(0.0f32, |a, &b| a.max(b));
        if let Some(r) = &pb.remote {
            rt = r.rt60.iter().fold(rt, |a, &b| a.max(b));
        }
        end + self.renderer.horizon() + (rt.min(20.0) as f64 * 1.5 + 0.1) * sample_rate as f64
    }

    /// Cut `latest` to `k` paths for the renderer, with hysteresis: the
    /// `k` strongest always pass, and a path admitted last time stays
    /// while it ranks within `2k`. Without it, paths near the cut flip in
    /// and out every frame (the tracer's estimates are noisy), and every
    /// flip costs a crossfade — the evicted slot keeps rendering while it
    /// fades out, which multiplies the real tap count.
    fn select(&mut self, k: usize, order: &mut Vec<(f32, u16)>) {
        let pb = &self.latest;
        self.filtered.taps.clear();
        self.admitted_next.clear();
        if pb.taps.len() <= k {
            self.filtered.taps.extend_from_slice(&pb.taps);
            self.admitted_next.extend(pb.taps.iter().map(|t| t.key));
        } else {
            order.clear();
            order.extend(pb.taps.iter().enumerate().map(|(i, t)| (t.gains[1], i as u16)));
            order.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            for (rank, &(_, i)) in order.iter().enumerate() {
                let t = pb.taps[i as usize];
                if rank < k || (rank < 2 * k && self.admitted.binary_search(&t.key).is_ok()) {
                    self.filtered.taps.push(t);
                    self.admitted_next.push(t.key);
                }
            }
        }
        self.admitted_next.sort_unstable();
        core::mem::swap(&mut self.admitted, &mut self.admitted_next);
        self.filtered.reverb = pb.reverb;
        self.filtered.remote = pb.remote;
        self.filtered.version = pb.version;
    }
}

struct NearPlay {
    voice: u32,
    clip: Clip,
    pos: usize,
    ear: NearVoice,
}

/// Default mix-wide budgets: worst case (every source playing, the
/// listener walking) ~1.3× real time on one Apple M1 core with HRIRs,
/// however many sources (examples/bench_sources.rs). Raise them on
/// faster machines with `set_budget`.
const MIX_TAPS: usize = 96;
const MIX_POINTS: usize = 8;
const MAX_VOICES: usize = 24;
const SLOT_CAPACITY: usize = 16;
/// Per-source ceilings while the mix-wide budget is off.
const SOURCE_TAPS: usize = 160;
const SOURCE_POINTS: usize = 8;

/// The real-time renderer for all sources.
pub struct Mixer {
    sample_rate: f32,
    grid: Option<Arc<HrirGrid>>,
    slots: Vec<Slot>,
    index: HashMap<u32, usize>,
    out: OutputStage,
    near: Vec<NearPlay>,
    master: Smoothed,
    master_meter: MeterAcc,
    clock: f64,
    next_voice: u32,
    silence: Clip,
    mix_taps: usize,
    mix_points: usize,
    max_voices: usize,
    head: [f32; 3],
    head_from_frames: bool,
    // scratch, preallocated
    active: Vec<u16>,
    ranked: Vec<(f32, u16)>,
    counts: Vec<(usize, usize)>,
    louds: Vec<f32>,
    order: Vec<(f32, u16)>,
}

impl Mixer {
    pub fn new(sample_rate: f32, hrir: Hrir) -> Self {
        Self::with_capacity(sample_rate, hrir, SLOT_CAPACITY)
    }

    /// Preallocate channels for `sources` sources (beyond it, a new
    /// source's channel allocates in `apply`/`play`).
    pub fn with_capacity(sample_rate: f32, hrir: Hrir, sources: usize) -> Self {
        let mut m = Self {
            sample_rate,
            grid: hrir.grid.clone(),
            slots: Vec::with_capacity(sources),
            index: HashMap::with_capacity(sources * 2),
            out: OutputStage::from_speaker_bytes(hrir.speakers.as_deref(), sample_rate),
            near: Vec::with_capacity(omg_dsp::near::MAX_VOICES),
            master: Smoothed::new(1.0, 0.02, sample_rate),
            master_meter: MeterAcc::default(),
            clock: 0.0,
            next_voice: 1,
            silence: Clip::from(&[][..]),
            mix_taps: MIX_TAPS,
            mix_points: MIX_POINTS,
            max_voices: MAX_VOICES,
            head: [0.0; 3],
            head_from_frames: true,
            active: Vec::with_capacity(sources),
            ranked: Vec::with_capacity(sources * 32),
            counts: Vec::with_capacity(sources),
            louds: Vec::with_capacity(sources),
            order: Vec::with_capacity(4 * SOURCE_TAPS),
        };
        for _ in 0..sources {
            let s = m.new_slot();
            m.slots.push(s);
        }
        m
    }

    fn new_slot(&self) -> Slot {
        let mut r = Renderer::with_grid(self.sample_rate, self.grid.clone()).without_line();
        r.set_point_budget(SOURCE_POINTS);
        // the mixer cuts each block to the source's share itself (`select`)
        r.set_tap_ceiling(omg_dsp::renderer::MAX_TAPS);
        r.set_head(self.head[0], self.head[1], self.head[2]);
        Slot {
            id: SourceId(u32::MAX),
            used: false,
            renderer: r,
            log: PlayLog::preallocated(),
            fader: Smoothed::new(1.0, 0.02, self.sample_rate),
            audible: Smoothed::new(1.0, 0.02, self.sample_rate),
            latest: ParamBlock::default(),
            filtered: ParamBlock { taps: Vec::with_capacity(2 * SOURCE_TAPS), ..Default::default() },
            admitted: Vec::with_capacity(2 * SOURCE_TAPS),
            admitted_next: Vec::with_capacity(2 * SOURCE_TAPS),
            dirty: false,
            reapply: false,
            has_block: false,
            in_frame: false,
            traced_at: None,
            motion: None,
            loop_voice: 0,
            share: (usize::MAX, usize::MAX),
            loud: 0.0,
            quiet_at: f64::NEG_INFINITY,
            audible_next: false,
            meter: MeterAcc::default(),
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Samples rendered so far (the mixer's clock).
    pub fn clock(&self) -> f64 {
        self.clock
    }

    fn slot_of(&mut self, id: SourceId) -> usize {
        if let Some(&i) = self.index.get(&id.0) {
            return i;
        }
        let i = match self.slots.iter().position(|s| !s.used) {
            Some(i) => {
                // a recycled channel: a fresh log, and its renderer
                // crossfades from whatever it last played (silence)
                let s = &mut self.slots[i];
                s.log = PlayLog::preallocated();
                s.fader.snap(1.0);
                s.audible.snap(1.0);
                s.renderer.invalidate_params();
                i
            }
            None => {
                let s = self.new_slot();
                self.slots.push(s);
                // keep render's scratch big enough (allocates here, not there)
                let n = self.slots.len();
                self.active.reserve(n.saturating_sub(self.active.len()));
                n - 1
            }
        };
        let s = &mut self.slots[i];
        s.id = id;
        s.used = true;
        s.has_block = false;
        s.in_frame = false;
        s.traced_at = None;
        s.motion = None;
        s.loop_voice = 0;
        s.loud = 0.0;
        s.meter = MeterAcc::default();
        s.admitted.clear();
        s.share = (usize::MAX, usize::MAX);
        self.index.insert(id.0, i);
        i
    }

    /// Mix-wide budgets: at most `taps` paths and `points` point-HRTF
    /// convolutions across ALL sources (0 taps = per-source ceilings of
    /// 160 / 8 instead), and at most `voices` sources rendering at once.
    pub fn set_budget(&mut self, taps: usize, points: usize, voices: usize) {
        self.mix_taps = taps;
        self.mix_points = points;
        self.max_voices = voices.max(1);
        for s in &mut self.slots {
            s.share = (usize::MAX, usize::MAX);
            s.dirty = true;
            if taps == 0 {
                s.renderer.set_point_budget(SOURCE_POINTS);
            }
        }
    }

    /// Take a simulation frame: every source's paths glide to their new
    /// values over the next frame interval. Sources missing from the
    /// frame fade out. Not real-time safe in the strict sense (see the
    /// module docs); call it between `render` calls.
    pub fn apply(&mut self, frame: &Frame) {
        for s in &mut self.slots {
            s.in_frame = false;
        }
        let sr = self.sample_rate as f64;
        for (id, pb) in &frame.blocks {
            let i = self.slot_of(*id);
            let s = &mut self.slots[i];
            s.in_frame = true;
            if !s.has_block || s.latest.version != pb.version {
                // traced anew: how much time passed since its last trace
                s.motion = match (frame.time, s.traced_at) {
                    (Some(t), Some(t0)) => Some((t - t0) * sr),
                    _ => None,
                };
                s.traced_at = frame.time;
                copy_block(&mut s.latest, pb);
                s.has_block = true;
                s.dirty = true;
            }
        }
        // gone from the simulation: fade out, free once silent
        for s in &mut self.slots {
            if !s.used || s.in_frame {
                continue;
            }
            if s.has_block && (!s.latest.taps.is_empty() || s.latest.reverb.level[1] > 0.0 || s.latest.remote.is_some()) {
                s.latest.taps.clear();
                s.latest.reverb.level = [0.0; 3];
                s.latest.remote = None;
                s.latest.version = s.latest.version.wrapping_add(1 << 40);
                s.dirty = true;
            }
            if s.quiet_at < self.clock {
                s.used = false;
                self.index.remove(&s.id.0);
            }
        }
        if self.head_from_frames {
            let l = &frame.listener;
            self.set_head_inner(l.yaw, l.pitch, l.roll);
        }
        self.rebudget();
        let per_source = if self.mix_taps == 0 { Some(SOURCE_TAPS) } else { None };
        for s in self.slots.iter_mut().filter(|s| s.used && s.has_block && s.dirty) {
            let k = per_source.unwrap_or(s.share.0).max(1);
            s.select(k, &mut self.order);
            if s.reapply {
                // a changed share takes effect now, not one frame later
                s.renderer.invalidate_params();
                s.reapply = false;
            }
            s.renderer.set_params_after(&s.filtered, s.motion.take());
            s.dirty = false;
        }
        let (now, sr) = (self.clock, self.sample_rate);
        for s in self.slots.iter_mut().filter(|s| s.used) {
            s.log.forget_before(now - s.renderer.horizon() - 1.0);
            s.quiet_at = s.quiet_after(sr);
        }
    }

    /// Rank every sounding source's paths by loudness at the ears and
    /// share the budgets (cf. omg-web `eng_params_commit`).
    fn rebudget(&mut self) {
        let now = self.clock;
        self.ranked.clear();
        self.counts.clear();
        self.counts.resize(self.slots.len(), (0, 0));
        let mut voices: usize = 0;
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.loud = 0.0;
            if !s.used || !s.has_block {
                continue;
            }
            let sounding = s.quiet_at >= now;
            let fader = s.fader.target_val().max(0.0);
            if !sounding || fader == 0.0 {
                continue;
            }
            let pb = &s.latest;
            let mut loud = pb.reverb.level[1] + pb.remote.as_ref().map_or(0.0, |r| r.send[1]);
            for t in &pb.taps {
                loud += t.gains[1];
            }
            s.loud = loud * fader + 1e-12;
            voices += 1;
            if self.mix_taps > 0 {
                for t in &pb.taps {
                    self.ranked.push((t.gains[1] * fader, i as u16));
                }
            }
        }
        // voice virtualization: only the loudest max_voices render
        let cut = if voices > self.max_voices {
            self.louds.clear();
            self.louds.extend(self.slots.iter().filter(|s| s.loud > 0.0).map(|s| s.loud));
            let k = self.max_voices;
            self.louds.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            self.louds[k - 1]
        } else {
            0.0
        };
        // hysteresis: a voice already rendering keeps going while it is
        // within 12 dB of the cut (up to 25% over the budget) — the
        // tracer's noise must not flip voices in and out every frame
        let mut n = 0usize;
        for s in &mut self.slots {
            let on = s.loud > 0.0 && s.loud >= cut;
            n += on as usize;
            s.audible_next = on;
        }
        let (hi, lo) = (self.max_voices + self.max_voices / 4, cut * 0.25);
        for s in &mut self.slots {
            if !s.audible_next && n < hi && s.audible.target_val() > 0.0 && s.loud > 0.0 && s.loud >= lo {
                s.audible_next = true;
                n += 1;
            }
            s.audible.set(if s.audible_next { 1.0 } else { 0.0 });
        }
        if self.mix_taps == 0 {
            return;
        }
        // drop the virtualized sources' paths from the ranking
        let slots = &self.slots;
        self.ranked.retain(|&(_, i)| slots[i as usize].audible.target_val() > 0.0);
        let count_top = |k: usize, points: bool, ranked: &mut Vec<(f32, u16)>, counts: &mut Vec<(usize, usize)>| {
            if k == 0 {
                return;
            }
            if ranked.len() > k {
                ranked.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            }
            for &(_, i) in ranked.iter().take(k) {
                let c = &mut counts[i as usize];
                if points {
                    c.1 += 1;
                } else {
                    c.0 += 1;
                }
            }
        };
        count_top(self.mix_taps, false, &mut self.ranked, &mut self.counts);
        count_top(self.mix_points, true, &mut self.ranked, &mut self.counts);
        for (i, s) in self.slots.iter_mut().enumerate() {
            if !s.used {
                continue;
            }
            let share = (self.counts[i].0.max(1), self.counts[i].1);
            if s.share != share {
                s.share = share;
                s.renderer.set_point_budget(share.1);
                s.dirty = true;
                s.reapply = true;
            }
        }
    }

    /// Play `clip` on a source (its channel is created if the source has
    /// not been in a frame yet). The clip starts sounding at the source
    /// now and reaches the listener one path delay later.
    pub fn play(&mut self, source: SourceId, clip: &Clip, play: Play) -> VoiceId {
        let v = self.alloc_voice();
        self.play_as(v, source, clip, play)
    }

    fn alloc_voice(&mut self) -> VoiceId {
        let v = self.next_voice;
        self.next_voice = self.next_voice.wrapping_add(1).max(1);
        VoiceId(v)
    }

    pub(crate) fn play_as(&mut self, voice: VoiceId, source: SourceId, clip: &Clip, play: Play) -> VoiceId {
        let i = self.slot_of(source);
        let (now, gain, rate) = (self.clock, play.gain.clamp(0.0, 16.0), play.rate.clamp(0.25, 4.0));
        let sr = self.sample_rate;
        let s = &mut self.slots[i];
        // audible at once (the next frame re-ranks the voices)
        s.audible.set(1.0);
        let v = if play.looped {
            s.log.set_loop_with(clip.clone(), now, gain, rate);
            s.loop_voice = voice.0;
            voice
        } else if s.log.play(voice.0, clip.clone(), now, gain, rate) {
            voice
        } else {
            VoiceId::NONE
        };
        s.quiet_at = s.quiet_after(sr);
        v
    }

    /// Stop a voice (paths still carrying what it already emitted play
    /// it out — a far explosion you stopped is still on its way).
    pub fn stop(&mut self, voice: VoiceId) {
        if voice == VoiceId::NONE {
            return;
        }
        let now = self.clock;
        let sr = self.sample_rate;
        for s in self.slots.iter_mut().filter(|s| s.used) {
            s.log.stop(voice.0, now);
            if s.loop_voice == voice.0 {
                s.log.stop_loop(now, self.silence.clone());
                s.loop_voice = 0;
            }
            s.quiet_at = s.quiet_after(sr);
        }
        self.near.retain(|n| n.voice != voice.0);
    }

    /// A near-field voice: `clip` played `dist_m` (2–60 cm) from one ear
    /// (`right` or left), straight into the ears — a whisper, a
    /// mosquito, a radio earpiece. `gain` is the near-ear amplitude at
    /// 6 cm. Not part of the scene: no paths, no reverb, not turned by
    /// the head, not ridden by the loudness stage. Up to 4 at once.
    pub fn play_near(&mut self, clip: &Clip, right: bool, dist_m: f32, gain: f32) -> VoiceId {
        let v = self.alloc_voice();
        self.play_near_as(v, clip, right, dist_m, gain)
    }

    pub(crate) fn play_near_as(&mut self, voice: VoiceId, clip: &Clip, right: bool, dist_m: f32, gain: f32) -> VoiceId {
        if self.near.len() >= omg_dsp::near::MAX_VOICES {
            return VoiceId::NONE;
        }
        self.near.push(NearPlay {
            voice: voice.0,
            clip: clip.clone(),
            pos: 0,
            ear: NearVoice::new(self.sample_rate, right, dist_m, gain),
        });
        voice
    }

    /// Head orientation (radians; see `Listener`), applied at audio rate.
    /// Calling it switches the mixer to manual head control: frames no
    /// longer set the head (`follow_frames_head` switches back).
    pub fn set_head(&mut self, yaw: f32, pitch: f32, roll: f32) {
        self.head_from_frames = false;
        self.set_head_inner(yaw, pitch, roll);
    }

    /// Take the head orientation from each applied frame's listener again.
    pub fn follow_frames_head(&mut self) {
        self.head_from_frames = true;
    }

    fn set_head_inner(&mut self, yaw: f32, pitch: f32, roll: f32) {
        if self.head == [yaw, pitch, roll] {
            return;
        }
        self.head = [yaw, pitch, roll];
        for s in &mut self.slots {
            s.renderer.set_head(yaw, pitch, roll);
        }
        self.out.set_head(yaw, pitch, roll);
    }

    /// A source's volume (amplitude, smoothed; 0 mutes and stops its
    /// rendering cost).
    pub fn set_source_gain(&mut self, source: SourceId, gain: f32) {
        let i = self.slot_of(source);
        self.slots[i].fader.set(gain.clamp(0.0, 256.0));
    }

    /// Master volume (amplitude, after the loudness stage).
    pub fn set_master(&mut self, gain: f32) {
        self.master.set(gain.clamp(0.0, 4.0));
    }

    /// A source's output level since the last call (peak, RMS).
    pub fn take_meter(&mut self, source: SourceId) -> Meter {
        match self.index.get(&source.0) {
            Some(&i) => self.slots[i].meter.take(),
            None => Meter::default(),
        }
    }

    /// The final output level since the last call.
    pub fn take_master_meter(&mut self) -> Meter {
        self.master_meter.take()
    }

    /// Sources rendering in the last `render` call.
    pub fn rendering(&self) -> usize {
        self.active.len()
    }

    /// Render occupancy: (sources rendering, live paths, point-HRTF
    /// paths). Live paths include ones still fading out — what actually
    /// costs CPU. Walks every channel: for HUDs, not per sample.
    pub fn occupancy(&self) -> (usize, usize, usize) {
        let (mut live, mut points) = (0, 0);
        for &i in &self.active {
            let (l, p, ..) = self.slots[i as usize].renderer.debug_stats();
            live += l as usize;
            points += p as usize;
        }
        (self.active.len(), live, points)
    }

    /// Render the next `l.len()` samples (stereo; `l` and `r` the same
    /// length). Real-time safe: no allocation, no locks.
    pub fn render(&mut self, l: &mut [f32], r: &mut [f32]) {
        let n = l.len().min(r.len());
        // who renders this block: used, not muted, not virtualized, and
        // still able to make a sound
        self.active.clear();
        for (i, s) in self.slots.iter_mut().enumerate() {
            let fader_on = s.fader.target_val() > 0.0 || s.fader.current() > 1e-4;
            let voice_on = s.audible.target_val() > 0.0 || s.audible.current() > 1e-3;
            let on = s.used
                && s.has_block
                && fader_on
                && voice_on
                && s.quiet_at >= self.clock
                && self.active.len() < self.active.capacity();
            if on {
                self.active.push(i as u16);
            } else {
                // silent anyway: its ramps jump to where they are headed
                s.fader.snap(s.fader.target_val());
                s.audible.snap(s.audible.target_val());
            }
        }
        // sources render a block at a time, path by path
        // (Renderer::process_log_block); the output stage per sample
        let mut done = 0;
        while done < n {
            let m = (n - done).min(BLOCK);
            let mut bus: BusBlock = [[0.0; BLOCK]; NCH];
            let (mut pl, mut pr) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            let now = self.clock;
            for &i in &self.active {
                let s = &mut self.slots[i as usize];
                let mut gain = [0.0f32; BLOCK];
                for g in &mut gain[..m] {
                    *g = s.fader.tick() * s.audible.tick();
                }
                let w0 = bus[0];
                let (mut a, mut b) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
                s.renderer.process_log_block(&s.log, now, &gain[..m], &mut bus, &mut a[..m], &mut b[..m]);
                for j in 0..m {
                    pl[j] += a[j];
                    pr[j] += b[j];
                    let dw = bus[0][j] - w0[j];
                    s.meter.add(0.5 * (a[j] * a[j] + b[j] * b[j]) + 2.0 * dw * dw);
                }
            }
            for j in 0..m {
                let k = done + j;
                let bus: [f32; NCH] = core::array::from_fn(|c| bus[c][j]);
                let (mut ol, mut or) = self.out.process(&bus, pl[j], pr[j]);
                // near-field voices: post-loudness-stage on purpose — a
                // whisper's identity is its absolute closeness
                for v in &mut self.near {
                    if v.pos < v.clip.len() {
                        let (a, b) = v.ear.process(v.clip[v.pos]);
                        v.pos += 1;
                        ol += a;
                        or += b;
                    }
                }
                let mg = self.master.tick();
                l[k] = (ol * mg).clamp(-1.0, 1.0);
                r[k] = (or * mg).clamp(-1.0, 1.0);
                self.master_meter.add(0.5 * (l[k] * l[k] + r[k] * r[k]));
                self.clock += 1.0;
            }
            done += m;
        }
        // forget what no path of a source can still reach (the others
        // forget at the next apply)
        for &i in &self.active {
            let s = &mut self.slots[i as usize];
            s.log.forget_before(self.clock - s.renderer.horizon() - 1.0);
            s.quiet_at = s.quiet_after(self.sample_rate);
        }
        self.near.retain(|v| v.pos < v.clip.len());
    }
}

/// `dst = src`, reusing `dst`'s tap storage.
fn copy_block(dst: &mut ParamBlock, src: &ParamBlock) {
    dst.taps.clear();
    dst.taps.extend_from_slice(&src.taps);
    dst.reverb = src.reverb;
    dst.remote = src.remote;
    dst.version = src.version;
}
