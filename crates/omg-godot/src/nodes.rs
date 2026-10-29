//! The Godot face of omg-audio: nodes a level drops in.
//!
//!  - `OmgWorld`     — the engine. Collects the level's acoustic geometry,
//!                     runs omg-audio's threaded Engine (simulation thread,
//!                     v3 on the GPU when there is one), renders on an
//!                     output thread and plays through its own
//!                     AudioStreamPlayer whose mix runs on Godot's audio
//!                     thread.
//!  - `OmgGeometry`  — tags everything below it with an acoustic material
//!                     (`material = "brick"`); `as_boxes` turns its meshes
//!                     into solid boxes (furniture, columns).
//!  - `OmgSource`    — an emitter: a looping file and/or one-shot clips.
//!  - `OmgListener`  — the ears (usually a child of the camera).
//!  - `OmgDoor`      — a leaf or a pane filling an opening (an omg-audio
//!                     object box, shrunk with `openness`).
//!
//! Material lookup for a MeshInstance3D surface: an EXPLICIT tag — the
//! mesh node's (or an ancestor's) `omg_material` meta, or the nearest
//! OmgGeometry ancestor with a non-empty `material` — wins; otherwise the
//! surface's visual material brings its own acoustics through its
//! `omg_material` meta (tag a Material resource once, every mesh using
//! it is acoustic); otherwise `OmgWorld.default_material` ("" = untagged
//! meshes are not acoustic). "none" opts a subtree out. A value is a
//! preset name (`OmgWorld.material_presets()`) or a Dictionary
//! {absorption: [lo, mid, hi], scattering, transmission: [..]}.
//!
//! Engine frame: x east, y north, z up = Godot (x, −z, y) (omg-audio's
//! `from_y_up`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};

use godot::classes::native::AudioFrame;
use godot::classes::{
    AudioServer, AudioStream, AudioStreamPlayback, AudioStreamPlayer, IAudioStream, IAudioStreamPlayback, INode3D,
    Mesh, MeshInstance3D, Node, Node3D,
};
use godot::meta::conv::RawPtr;
use godot::prelude::*;

use omg_audio::governor::LevelGovernor;
use omg_audio::sim::LEVELS;
use omg_audio::{from_y_up, to_y_up, Compute, Config, Geometry, Material, MaterialId, ObjectId, Play, Quality, SourceDesc, SourceId, VoiceId};

use crate::backend::clips::{self, Import};
use crate::backend::decode::ClipOptions;
use crate::backend::{materials, Host, OutCmd, OutputReader};

use omg_audio::{HRIR_GRID, HRIR_SPEAKERS};

/// The live engine (one OmgWorld at a time). Sources reach it directly
/// to start their clips.
static HOST: Mutex<Option<Host>> = Mutex::new(None);
static RATE: AtomicU32 = AtomicU32::new(0);

fn host() -> MutexGuard<'static, Option<Host>> {
    HOST.lock().unwrap_or_else(|e| e.into_inner())
}

fn engine_rate() -> f32 {
    match RATE.load(Ordering::Relaxed) {
        0 => AudioServer::singleton().get_mix_rate(),
        r => f32::from_bits(r),
    }
}

/// Godot position → engine frame.
fn eng(v: Vector3) -> [f32; 3] {
    from_y_up([v.x, v.y, v.z])
}

fn db(x: f32) -> f32 {
    10f32.powf(x / 20.0)
}

/// A clip file for the clip bank: natively a path on disk (from the
/// checkout); on the web and in exported builds its bytes, read through
/// Godot (res:// lives in the .pck, not on a disk).
#[cfg(not(target_os = "emscripten"))]
fn clip_source(p: &GString) -> clips::Source {
    let s = p.to_string();
    if s.starts_with("res://") || s.starts_with("user://") {
        let g = godot::classes::ProjectSettings::singleton().globalize_path(p).to_string();
        if std::path::Path::new(&g).exists() {
            return clips::Source::Path(g);
        }
        // an exported build: the file is packed in the .pck
        let bytes = godot::classes::FileAccess::get_file_as_bytes(p).to_vec();
        return clips::Source::Bytes { name: s, bytes };
    }
    clips::Source::Path(s)
}

#[cfg(target_os = "emscripten")]
fn clip_source(p: &GString) -> clips::Source {
    let bytes = godot::classes::FileAccess::get_file_as_bytes(p).to_vec();
    clips::Source::Bytes { name: p.to_string(), bytes }
}

fn band3(v: &Variant) -> Option<[f32; 3]> {
    let a: VarArray = v.try_to().ok()?;
    if a.len() != 3 {
        return None;
    }
    let f = |i: usize| a.at(i).try_to::<f32>().ok();
    Some([f(0)?, f(1)?, f(2)?])
}

/// A material value: preset name or {absorption, scattering, transmission}.
fn material_of(v: &Variant) -> Option<Material> {
    if let Ok(name) = v.try_to::<GString>() {
        return materials::get(&name.to_string());
    }
    let d: Dictionary<Variant, Variant> = v.try_to().ok()?;
    Some(Material::new(
        band3(&d.get("absorption")?)?,
        d.get("scattering").and_then(|s| s.try_to::<f32>().ok()).unwrap_or(0.1),
        d.get("transmission").as_ref().and_then(band3).unwrap_or([0.0; 3]),
    ))
}

fn options_of(d: &Dictionary<Variant, Variant>) -> Import {
    let f = |k: &str| d.get(k).and_then(|v| v.try_to::<f32>().ok()).unwrap_or(0.0);
    let b = |k: &str| d.get(k).and_then(|v| v.try_to::<bool>().ok()).unwrap_or(false);
    Import {
        opts: ClipOptions {
            stereo: b("stereo"),
            rate: f("rate"),
            peak: f("peak"),
            highpass_hz: f("highpass_hz"),
            trim_start: f("trim_start"),
            trim_end: f("trim_end"),
            max_seconds: f("max_seconds"),
            fade_out_s: f("fade_out"),
            fade_in_s: f("fade_in"),
        },
        normalize_rms: b("normalize"),
    }
}

// ===================================================================== tags

/// Everything below this node is acoustic geometry: of `material`, or
/// (empty) of whatever its surfaces' visual materials declare in their
/// `omg_material` meta.
#[derive(GodotClass)]
#[class(base=Node3D, init)]
pub struct OmgGeometry {
    /// Preset name (see OmgWorld.material_presets()); "none" = not
    /// acoustic; "" = from each surface material's `omg_material` meta.
    #[export]
    material: GString,
    /// Treat each mesh as its bounding box (furniture, columns, props).
    #[export]
    as_boxes: bool,
    base: Base<Node3D>,
}

/// The listener's ears. Without one, OmgWorld listens at the camera.
#[derive(GodotClass)]
#[class(base=Node3D, init)]
pub struct OmgListener {
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for OmgListener {
    fn ready(&mut self) {
        self.base_mut().add_to_group("omg_listener");
    }
}

/// A leaf or pane filling an opening: a box of `size` (local, centered)
/// covering `1 − openness` of its width from its local −x side.
#[derive(GodotClass)]
#[class(base=Node3D, init)]
pub struct OmgDoor {
    #[export]
    #[init(val = Vector3::new(1.1, 2.0, 0.08))]
    size: Vector3,
    /// 0 closed … 1 fully open (the swing is priced continuously).
    #[export]
    openness: f32,
    /// Preset: door, heavy_door, glass, heavy_glass, …
    #[export]
    #[init(val = GString::from("door"))]
    material: GString,
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for OmgDoor {
    fn ready(&mut self) {
        self.base_mut().add_to_group("omg_door");
    }
}

// =================================================================== source

enum Pending {
    Play(i64, f32, f32),
    Stop(i64),
    Loop(i64),
}

/// A sound emitter. `file` loops (when `playing`); clips from
/// `OmgWorld.load_clip` play as one-shots on top. Its level is a real
/// SPL: `level_db` at 1 m, the engine's 1.0 ≙ 90 dB SPL.
#[derive(GodotClass)]
#[class(base=Node3D, init)]
pub struct OmgSource {
    /// Looping recording (res://, user:// or absolute path; ogg/flac/wav).
    #[export]
    file: GString,
    #[export]
    #[init(val = 90.0)]
    level_db: f32,
    #[export]
    #[init(val = true)]
    playing: bool,
    /// Traced and heard at all? (an idle car slot: false costs nothing)
    #[export]
    #[init(val = true)]
    active: bool,
    /// A mixer fader over `level_db` (dB), for the listener's own mix:
    /// the scene's levels stay what the script sets.
    #[export]
    volume_db: f32,
    /// Muted on the mixer: not traced or heard (costs nothing), whatever
    /// `active` says.
    #[export]
    muted: bool,
    id: Option<SourceId>,
    /// Voices of one-shots in flight, by clip id (for stop_clip).
    voices: Vec<(i64, VoiceId)>,
    /// A loop set with set_loop_clip (wins over `file`).
    loop_clip: Option<i64>,
    pending: Vec<Pending>,
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for OmgSource {
    fn ready(&mut self) {
        self.base_mut().add_to_group("omg_source");
    }
}

#[godot_api]
impl OmgSource {
    #[func]
    fn play(&mut self) {
        self.playing = true;
    }

    #[func]
    fn stop(&mut self) {
        self.playing = false;
    }

    /// One-shot of clip `id` at linear `gain` and playback `rate`. False
    /// when the clip is still decoding (or unknown).
    #[func]
    fn play_clip(&mut self, id: i64, gain: f32, rate: f32) -> bool {
        if clips::get(id.max(0) as usize).is_none() {
            return false;
        }
        self.dispatch(Pending::Play(id, gain, rate));
        true
    }

    #[func]
    fn stop_clip(&mut self, id: i64) {
        self.dispatch(Pending::Stop(id));
    }

    /// Loop clip `id` instead of `file` (e.g. a per-spawn resampled motor).
    #[func]
    fn set_loop_clip(&mut self, id: i64) -> bool {
        if clips::get(id.max(0) as usize).is_none() {
            return false;
        }
        self.dispatch(Pending::Loop(id));
        self.playing = true;
        true
    }

    /// Engine source id (−1 until OmgWorld has picked the source up).
    #[func]
    fn source_id(&self) -> i64 {
        self.id.map_or(-1, |s| s.0 as i64)
    }
}

impl OmgSource {
    fn dispatch(&mut self, p: Pending) {
        match self.id {
            Some(id) => self.apply(id, p),
            None => self.pending.push(p),
        }
    }

    fn apply(&mut self, id: SourceId, p: Pending) {
        if let Pending::Loop(c) = p {
            self.loop_clip = Some(c);
            return;
        }
        let mut g = host();
        let Some(h) = g.as_mut() else { return };
        match p {
            Pending::Play(c, gain, rate) => {
                if let Some(clip) = clips::get(c.max(0) as usize) {
                    let v = h.engine.play(id, &clip, Play::ONCE.gain(gain).rate(rate));
                    if self.voices.len() >= 64 {
                        self.voices.remove(0);
                    }
                    self.voices.push((c, v));
                }
            }
            Pending::Stop(c) => {
                for (_, v) in self.voices.iter().filter(|(k, _)| *k == c) {
                    h.engine.stop(*v);
                }
                self.voices.retain(|(k, _)| *k != c);
            }
            Pending::Loop(_) => {}
        }
    }
}

// =================================================================== output

/// The engine's binaural output as an AudioStream: its playback's mix()
/// runs on Godot's AUDIO thread and copies frames the output thread
/// rendered ahead — the game thread never touches audio.
#[derive(GodotClass)]
#[class(base=AudioStream, init)]
pub struct OmgOutputStream {
    reader: Option<OutputReader>,
    base: Base<AudioStream>,
}

#[godot_api]
impl IAudioStream for OmgOutputStream {
    fn instantiate_playback(&self) -> Option<Gd<AudioStreamPlayback>> {
        let mut p = OmgOutputPlayback::new_gd();
        p.bind_mut().reader = self.reader.clone();
        Some(p.upcast())
    }

    fn get_length(&self) -> f64 {
        0.0
    }
}

#[derive(GodotClass)]
#[class(base=AudioStreamPlayback, init)]
pub struct OmgOutputPlayback {
    reader: Option<OutputReader>,
    playing: bool,
    scratch: Vec<(f32, f32)>,
    base: Base<AudioStreamPlayback>,
}

#[godot_api]
impl IAudioStreamPlayback for OmgOutputPlayback {
    fn start(&mut self, _from_pos: f64) {
        self.playing = true;
    }

    fn stop(&mut self) {
        self.playing = false;
    }

    fn is_playing(&self) -> bool {
        self.playing
    }

    fn get_playback_position(&self) -> f64 {
        0.0
    }

    unsafe fn mix_rawptr(&mut self, buffer: RawPtr<*mut AudioFrame>, _rate_scale: f32, frames: i32) -> i32 {
        let n = frames.max(0) as usize;
        let out = unsafe { std::slice::from_raw_parts_mut(buffer.ptr(), n) };
        self.scratch.clear();
        self.scratch.resize(n, (0.0, 0.0));
        if let (Some(r), true) = (&self.reader, self.playing) {
            let (t0, late0) = (std::time::Instant::now(), r.late());
            r.read(&mut self.scratch);
            // a recording (OmgWorld.start_recording): exactly what Godot plays
            crate::backend::record::push(&self.scratch, t0.elapsed().as_micros() as u32, (r.late() - late0) as u32);
        }
        for (o, (l, r)) in out.iter_mut().zip(&self.scratch) {
            o.left = *l;
            o.right = *r;
        }
        frames
    }
}

// ==================================================================== world

struct SourceRec {
    id: SourceId,
    level_db: f32,
    volume_db: f32,
    file: GString,
    file_clip: Option<usize>,
    /// What loops now: (clip id, voice).
    looping: Option<(usize, VoiceId)>,
    seen: u64,
}

struct DoorRec {
    id: ObjectId,
    min: [f32; 3],
    max: [f32; 3],
    enabled: bool,
    seen: u64,
}

/// The engine node. Put it in your level, tag geometry, add sources.
#[derive(GodotClass)]
#[class(base=Node3D, init)]
pub struct OmgWorld {
    /// Where to collect acoustic geometry (empty: this node's parent).
    #[export]
    geometry_root: NodePath,
    /// Material for meshes nothing tags ("" = they are not acoustic).
    #[export]
    default_material: GString,
    /// Open ground to the horizon at `ground_height` (solid earth below),
    /// for levels whose own floors do not cover the surroundings.
    #[export]
    ground_plane: bool,
    #[export]
    ground_height: f32,
    #[export]
    #[init(val = GString::from("grass"))]
    ground_material: GString,
    /// Trace on the GPU when there is one (falls back to the CPU).
    #[export]
    #[init(val = true)]
    use_gpu: bool,
    /// With `use_gpu`: the share of the GPU's time the sound may take
    /// (0..1) — the game's frames come first (omg-audio `Config::gpu_budget`).
    #[export]
    #[init(val = 0.2)]
    gpu_budget: f32,
    /// Render on a background thread and play through an
    /// AudioStreamPlayer (false: call render() yourself — tests).
    #[export]
    #[init(val = true)]
    realtime: bool,
    #[export]
    #[init(val = StringName::from("Master"))]
    bus: StringName,
    /// Outdoor ambience bed (stereo or mono file) and its trim (dB).
    #[export]
    ambience_file: GString,
    #[export]
    #[init(val = -16.0)]
    ambience_db: f32,
    /// Seconds of the ambience file to use (0 = all of it).
    #[export]
    #[init(val = 90.0)]
    ambience_seconds: f32,
    /// Raindrop bank (mono, uniform 150 ms slices), trim and intensity.
    #[export]
    rain_bank_file: GString,
    #[export]
    rain_db: f32,
    #[export]
    rain_intensity: f32,
    #[export]
    master_db: f32,
    /// Simulation quality: 0 low, 1 medium, 2 high, 3 ultra; −1 auto
    /// (omg-audio's level governor walks the whole quality ladder: starts
    /// at the bottom, climbs while ticks are cheap and the audio thread
    /// calm, sheds at once when a tick runs over `tick_target_ms`).
    #[export]
    #[init(val = -1)]
    quality: i32,
    /// Quality auto: what a simulation tick may cost (ms). It runs on a
    /// thread of its own, but shares the machine with the game: 16 ms is
    /// a 60 Hz frame.
    #[export]
    #[init(val = 16.0)]
    tick_target_ms: f32,
    /// Pin single levers of the tracer budget over the quality tier
    /// (0 = the tier's own): rays from the listener per tick, early
    /// reflections per ray, late (reverb) rays per tick.
    #[export]
    rays: i32,
    #[export]
    bounces: i32,
    #[export]
    late_rays: i32,
    /// Sound bends around edges (knife-edge diffraction).
    #[export]
    #[init(val = true)]
    diffraction: bool,
    /// Engine v4 (the traced response rendered as traced: taps at any
    /// delay, the scattered rest convolved) vs v3 (early taps + a
    /// parametric reverb).
    #[export]
    #[init(val = true)]
    engine_v4: bool,
    /// Engine v5 (on v4): the reverb's late paths reused across ticks,
    /// reconnected exactly to where the listener is — half the late
    /// tail's noise for ~2 % of the tick.
    #[export]
    #[init(val = true)]
    engine_v5: bool,
    /// Engine v6 (ENGINE_V6.md): v3's estimate and reverb, steady in
    /// motion, with v6's refinements — the default; on, v4/v5 are off.
    #[export]
    #[init(val = true)]
    engine_v6: bool,
    /// Record each tick's strongest traced paths for `paths()` (a debug
    /// view). Off: the tracer records nothing.
    #[export]
    debug_paths: bool,

    started: bool,
    sources: HashMap<InstanceId, SourceRec>,
    doors: HashMap<InstanceId, DoorRec>,
    frame: u64,
    sent: [f32; 15],
    #[init(val = LevelGovernor::new(LEVELS - 1))]
    auto_tier: LevelGovernor,
    auto_ticks: u32,
    amb_clip: Option<(usize, bool)>,
    rain_clip: Option<(usize, bool)>,
    scratch: Vec<(f32, f32)>,
    head: (f32, f32, f32),
    base: Base<Node3D>,
}

#[godot_api]
impl INode3D for OmgWorld {
    fn ready(&mut self) {
        self.start();
    }

    fn process(&mut self, _dt: f64) {
        self.update_now();
    }

    fn exit_tree(&mut self) {
        *host() = None;
    }
}

#[godot_api]
impl OmgWorld {
    /// Collect the level's acoustic geometry now (call after building or
    /// changing static geometry). Returns {triangles, boxes, meshes}.
    #[func]
    fn rebuild_geometry(&mut self) -> Dictionary<Variant, Variant> {
        let default = if self.default_material.is_empty() { None } else { Some(self.default_material.to_variant()) };
        let mut c = Collector { g: Geometry::new(), default, mats: HashMap::new(), meshes: 0, boxes: 0 };
        if self.ground_plane {
            if let Some(m) = materials::get(&self.ground_material.to_string()) {
                let id = c.material(m);
                c.g.ground_plane(self.ground_height, id);
            }
        }
        let root = if self.geometry_root.is_empty() {
            self.base().get_parent()
        } else {
            self.base().get_node_or_null(&self.geometry_root)
        };
        if let Some(root) = root {
            c.walk(root, None, false);
        }
        let mut d = Dictionary::new();
        d.set("triangles", c.g.triangle_count() as i64);
        d.set("boxes", c.boxes as i64);
        d.set("meshes", c.meshes as i64);
        if let Some(h) = host().as_mut() {
            h.engine.set_geometry(c.g);
        }
        d
    }

    /// Import a clip (decoded in the background; the id is valid at once).
    /// Options: stereo, rate (varispeed), peak, normalize (RMS), highpass_hz,
    /// trim_start, trim_end, max_seconds, fade_in, fade_out.
    #[func]
    fn load_clip(&self, path: GString, options: Dictionary<Variant, Variant>) -> i64 {
        clips::load(clip_source(&path), engine_rate(), options_of(&options)) as i64
    }

    /// A clip from samples made in a script (mono, engine rate).
    #[func]
    fn add_clip(&self, samples: PackedFloat32Array) -> i64 {
        clips::add(samples.to_vec()) as i64
    }

    /// 1 ready, 0 decoding, −1 failed.
    #[func]
    fn clip_state(&self, id: i64) -> i64 {
        let (s, e) = clips::state(id.max(0) as usize);
        if s < 0 {
            godot_warn!("omg: clip {id}: {e}");
        }
        s as i64
    }

    /// Clips decoded (or failed) and queued so far: a loading screen's
    /// progress (the ambience bed is left out: it fades in when ready).
    /// Sources queue their files on the first frames after they enter the
    /// tree.
    #[func]
    fn clips_progress(&self) -> Vector2i {
        let (done, total) = clips::progress();
        Vector2i::new(done as i32, total as i32)
    }

    /// Wait (blocking) until every queued clip has decoded.
    #[func]
    fn wait_for_clips(&self, timeout_s: f32) -> bool {
        clips::wait_all(timeout_s)
    }

    /// Play clip `id` `dist` m from one ear (near field: no room path).
    #[func]
    fn whisper(&self, id: i64, right: bool, dist: f32, gain: f32) -> bool {
        let Some(clip) = clips::get(id.max(0) as usize) else { return false };
        match host().as_mut() {
            Some(h) => h.engine.play_near(&clip, right, dist, gain) != VoiceId::NONE,
            None => false,
        }
    }

    /// Engine sample rate.
    #[func]
    fn sample_rate(&self) -> f32 {
        engine_rate()
    }

    /// Synchronous render (only when `realtime` is off).
    #[func]
    fn render(&mut self, frames: i64) -> PackedVector2Array {
        let mut g = host();
        let Some(h) = g.as_mut() else { return PackedVector2Array::new() };
        if h.output_running() {
            godot_warn!("omg: render() while the realtime output runs");
            return PackedVector2Array::new();
        }
        self.scratch.clear();
        h.render(frames.max(0) as usize, &mut self.scratch);
        self.scratch.iter().map(|(l, r)| Vector2::new(*l, *r)).collect()
    }

    /// Push listener, sources and doors to the engine now (done every
    /// frame in _process; call it when driving the engine by hand).
    #[func]
    fn update_now(&mut self) {
        if !self.started {
            return;
        }
        self.frame += 1;
        self.sync_listener();
        self.sync_sources();
        self.sync_doors();
        self.govern_tier();
        self.sync_mix();
    }

    /// Quality auto: one report per finished simulation tick.
    fn govern_tier(&mut self) {
        if self.quality >= 0 {
            return;
        }
        let g = host();
        let Some(h) = g.as_ref() else { return };
        let s = h.stats();
        if s.ticks == self.auto_ticks {
            return;
        }
        self.auto_ticks = s.ticks;
        // a door moved: the extra rays are the change's cost, not the level's
        if !s.tick_burst {
            self.auto_tier.report(s.tick_ms, self.tick_target_ms, s.load);
        }
    }

    /// The quality ladder's level in effect (0 … `LEVELS` − 1): the
    /// pinned tier's, or where quality auto stands.
    #[func]
    fn effective_quality(&self) -> i32 {
        self.sim_quality().level().unwrap_or(0) as i32
    }

    /// The quality in effect, for a screen: the tier's name, or auto's
    /// level on the ladder.
    #[func]
    fn quality_label(&self) -> GString {
        match self.quality {
            0..=3 => ["low", "medium", "high", "ultra"][self.quality as usize].into(),
            _ => format!("auto {}/{}", self.auto_tier.level(), LEVELS - 1).as_str().into(),
        }
    }

    /// Engine state: rt60 (s, mid band, loudest field), enclosure, sim
    /// tick_ms / ticks, gpu, load (render time / audio time), budget
    /// (taps, points, voices), rendering (sources), paths, sources,
    /// peak, late (frames the audio thread found missing and rendered
    /// itself), callback (largest mix callback, frames), ring (frames
    /// rendered ahead).
    #[func]
    fn stats(&self) -> Dictionary<Variant, Variant> {
        let mut d = Dictionary::new();
        if let Some(h) = host().as_ref() {
            let s = h.stats();
            d.set("rt60", s.rt60);
            d.set("enclosure", s.enclosure);
            d.set("tick_ms", s.tick_ms);
            d.set("tick_parts", &PackedFloat32Array::from(&s.tick_parts[..]));
            d.set("gpu_ms", s.gpu_ms);
            d.set("ticks", s.ticks as i64);
            d.set("gpu", h.gpu);
            d.set("load", s.load);
            d.set("taps", s.budget.0 as i64);
            d.set("points", s.budget.1 as i64);
            d.set("voices", s.budget.2 as i64);
            d.set("rendering", s.rendering as i64);
            d.set("paths", s.paths as i64);
            d.set("sources", self.sources.len() as i64);
            d.set("peak", s.peak);
            d.set("late", s.underruns as i64);
            d.set("callback", s.read_max as i64);
            d.set("ring", s.ring as i64);
        }
        d
    }

    /// Levels for a mixer view, refreshed ~10 times a second: source id
    /// (`OmgSource.source_id()`) → Vector2(peak, RMS) amplitude since the
    /// refresh before; "out" → the whole output, "ambience" and "rain" →
    /// the outdoor beds.
    #[func]
    fn meters(&self) -> Dictionary<Variant, Variant> {
        let mut d = Dictionary::new();
        if let Some(h) = host().as_ref() {
            for (id, peak, rms) in h.meters() {
                let v = Vector2::new(peak, rms);
                if id == u32::MAX {
                    d.set("out", v);
                } else if id == u32::MAX - 1 {
                    d.set("ambience", v);
                } else if id == u32::MAX - 2 {
                    d.set("rain", v);
                } else {
                    d.set(id as i64, v);
                }
            }
        }
        d
    }

    /// The traced paths (with `debug_paths` on; empty otherwise),
    /// strongest first: each {source: the OmgSource node (null if gone),
    /// energy: mid-band, relative, points: PackedVector3Array from the
    /// listener's ears through every reflection or edge bend to the
    /// source, in world coordinates, kind: 0 direct, 1 edge bend, 2
    /// mirror chain — exactly solved, the same path from tick to tick
    /// (`key` names it within its source and kind) — or 3 one of the
    /// last tick's strongest random samples of the walk}.
    #[func]
    fn paths(&self) -> VarArray {
        let mut out = VarArray::new();
        // copy out under the host lock, build the Variants after it
        let paths = match host().as_ref() {
            Some(h) => h.engine.debug_paths(),
            None => return out,
        };
        for p in paths {
            let node = self
                .sources
                .iter()
                .find(|(_, r)| r.id == p.source)
                .and_then(|(k, _)| Gd::<Node>::try_from_instance_id(*k).ok());
            let pts: PackedVector3Array = p
                .points
                .iter()
                .map(|q| {
                    let g = to_y_up(*q);
                    Vector3::new(g[0], g[1], g[2])
                })
                .collect();
            let mut d: Dictionary<Variant, Variant> = Dictionary::new();
            d.set("source", &node.map_or(Variant::nil(), |n| n.to_variant()));
            d.set("energy", p.energy);
            d.set("points", &pts);
            d.set("kind", p.kind as i64);
            d.set("key", p.key as i64);
            out.push(&d.to_variant());
        }
        out
    }

    /// The tracer budget in effect (tier + pinned levers): rays, bounces,
    /// late_rays, diffraction — what a settings screen shows.
    #[func]
    fn effective_budget(&self) -> Dictionary<Variant, Variant> {
        let b = self.sim_config().quality.budget(self.use_gpu);
        let mut d = Dictionary::new();
        d.set("rays", b.rays as i64);
        d.set("bounces", b.bounces as i64);
        d.set("late_rays", b.late_rays as i64);
        d.set("diffraction", b.diffraction);
        d
    }

    /// Acoustic material preset names.
    #[func]
    fn material_presets() -> PackedStringArray {
        materials::names().map(GString::from).collect()
    }

    /// Pin the render budget to a rung of the governor's ladder
    /// (0 lightest … 4 densest; 1 = omg-audio's default); −1 = automatic.
    #[func]
    fn set_render_budget(&self, rung: i64) {
        if let Some(h) = host().as_ref() {
            h.send(OutCmd::Budget((rung >= 0).then_some(rung as usize)));
        }
    }

    /// The ear (loud places duck, quiet ones come up, the reflex) on or
    /// off: off, the scene plays at its own level.
    /// Record what Godot plays — every frame its audio thread takes from
    /// the engine — to `path` (a float WAV at the mix rate), with each
    /// audio callback's timing in `path`.csv (when, frames, how long the
    /// take took, frames rendered late). True when it started.
    #[func]
    fn start_recording(&self, path: GString) -> bool {
        let sr = AudioServer::singleton().get_mix_rate() as u32;
        crate::backend::record::start(&path.to_string(), sr).is_ok()
    }

    /// Finish the recording; how many frames it lost (0 unless the disk
    /// fell behind).
    #[func]
    fn stop_recording(&self) -> i64 {
        crate::backend::record::stop() as i64
    }

    #[func]
    fn set_ear(&self, on: bool) {
        if let Some(h) = host().as_ref() {
            h.send(OutCmd::Ear(on));
        }
    }
}

impl OmgWorld {
    /// The pinned tier, or quality auto's level.
    fn sim_quality(&self) -> Quality {
        match self.quality {
            0 => Quality::Low,
            1 => Quality::Medium,
            2 => Quality::High,
            3 => Quality::Ultra,
            _ => Quality::Level(self.auto_tier.level()),
        }
    }

    fn sim_config(&self) -> Config {
        let tier = self.sim_quality();
        // pinned levers: the tier's budget with them overridden
        let pinned = self.rays > 0 || self.bounces > 0 || self.late_rays > 0 || !self.diffraction;
        let quality = if pinned {
            let mut b = tier.budget(self.use_gpu);
            if self.rays > 0 {
                b.rays = self.rays as u32;
            }
            if self.bounces > 0 {
                b.bounces = self.bounces as u32;
            }
            if self.late_rays > 0 {
                b.late_rays = self.late_rays as u32;
            }
            b.diffraction = self.diffraction;
            Quality::Custom(b)
        } else {
            tier
        };
        Config {
            quality,
            compute: if self.use_gpu { Compute::Gpu } else { Compute::Cpu },
            v4: self.engine_v4,
            v5: self.engine_v4 && self.engine_v5,
            v6: self.engine_v6,
            sample_rate: engine_rate(),
            gpu_budget: self.gpu_budget.clamp(0.02, 1.0),
            ..Config::default()
        }
    }

    fn start(&mut self) {
        let sr = AudioServer::singleton().get_mix_rate();
        RATE.store(sr.to_bits(), Ordering::Relaxed);
        let mut h = Host::new(sr, HRIR_GRID, HRIR_SPEAKERS, self.use_gpu);
        if self.realtime {
            // ~21 ms rendered ahead on the output thread; Godot's audio
            // thread pulls it through OmgOutputStream
            h.start_output(1024);
            let mut stream = OmgOutputStream::new_gd();
            stream.bind_mut().reader = Some(h.reader());
            let mut p = AudioStreamPlayer::new_alloc();
            p.set_stream(&stream);
            p.set_bus(&self.bus);
            self.base_mut().add_child(&p);
            p.play();
        }
        *host() = Some(h);
        self.started = true;
        self.sent = [f32::NAN; 15];
        if !self.ambience_file.is_empty() {
            let mut imp = Import::default();
            imp.opts.stereo = true;
            imp.opts.max_seconds = self.ambience_seconds;
            let id = clips::load(clip_source(&self.ambience_file), sr, imp);
            clips::set_background(id); // a long bed: it fades in when decoded
            self.amb_clip = Some((id, false));
        }
        if !self.rain_bank_file.is_empty() {
            self.rain_clip = Some((clips::load(clip_source(&self.rain_bank_file), sr, Import::default()), false));
        }
    }

    fn sync_listener(&mut self) {
        let base = self.base().clone();
        let tree = base.get_tree();
        let node = tree
            .get_first_node_in_group("omg_listener")
            .and_then(|n| n.try_cast::<Node3D>().ok())
            .or_else(|| base.get_viewport().and_then(|v| v.get_camera_3d()).map(|c| c.upcast()));
        let Some(node) = node else { return };
        let t = node.get_global_transform();
        let p = eng(t.origin);
        let f = eng(-t.basis.col_c());
        let r = eng(t.basis.col_a());
        let yaw = f[1].atan2(f[0]);
        let pitch = f[2].clamp(-1.0, 1.0).asin();
        let roll = (-r[2]).clamp(-1.0, 1.0).asin();
        let mut g = host();
        let Some(h) = g.as_mut() else { return };
        h.engine.set_listener(p, yaw, pitch, roll);
        if (yaw, pitch, roll) != self.head {
            self.head = (yaw, pitch, roll);
            h.set_head(yaw, pitch, roll);
        }
    }

    fn sync_sources(&mut self) {
        let tree = self.base().get_tree();
        let sr = engine_rate();
        let frame = self.frame;
        for n in tree.get_nodes_in_group("omg_source").iter_shared() {
            let Ok(mut src) = n.try_cast::<OmgSource>() else { continue };
            let key = src.instance_id();
            let pos = eng(src.get_global_position());
            let on = src.bind().active && !src.bind().muted && src.is_visible_in_tree();
            let (file, level_db, volume_db, playing) = {
                let b = src.bind();
                (b.file.clone(), b.level_db, b.volume_db, b.playing)
            };
            if !self.sources.contains_key(&key) {
                let id = {
                    let mut g = host();
                    let Some(h) = g.as_mut() else { return };
                    h.engine.add_source(SourceDesc { position: pos, gain: db(level_db - 90.0), active: on })
                };
                self.sources.insert(
                    key,
                    SourceRec { id, level_db, volume_db: 0.0, file: GString::new(), file_clip: None, looping: None, seen: frame },
                );
                // plays queued before the source was linked
                let pending = {
                    let mut b = src.bind_mut();
                    b.id = Some(id);
                    std::mem::take(&mut b.pending)
                };
                for p in pending {
                    src.bind_mut().apply(id, p);
                }
            }
            let loop_clip = src.bind().loop_clip;
            let mut g = host();
            let Some(h) = g.as_mut() else { return };
            let rec = self.sources.get_mut(&key).unwrap();
            rec.seen = frame;
            h.engine.set_source(rec.id, pos, on);
            if level_db != rec.level_db {
                rec.level_db = level_db;
                h.engine.set_source_loudness(rec.id, db(level_db - 90.0));
            }
            if volume_db != rec.volume_db {
                rec.volume_db = volume_db;
                h.engine.set_volume(rec.id, db(volume_db));
            }
            if file != rec.file {
                rec.file = file.clone();
                rec.file_clip = (!file.is_empty())
                    .then(|| clips::load(clip_source(&file), sr, Import { normalize_rms: true, ..Default::default() }));
            }
            // what should loop: the script's loop clip, else the file
            let want = if playing { loop_clip.map(|c| c as usize).or(rec.file_clip) } else { None };
            if want != rec.looping.map(|(c, _)| c) {
                let ready = want.map(|c| clips::get(c));
                if !matches!(ready, Some(None)) {
                    // (a clip still decoding keeps the old loop a frame longer)
                    if let Some((_, v)) = rec.looping.take() {
                        h.engine.stop(v);
                    }
                    if let (Some(c), Some(Some(clip))) = (want, ready) {
                        rec.looping = Some((c, h.engine.play(rec.id, &clip, Play::LOOP)));
                    }
                }
            }
        }
        // sources that left the tree
        let gone: Vec<InstanceId> = self.sources.iter().filter(|(_, r)| r.seen != frame).map(|(k, _)| *k).collect();
        let mut g = host();
        let Some(h) = g.as_mut() else { return };
        for k in gone {
            let r = self.sources.remove(&k).unwrap();
            if let Some((_, v)) = r.looping {
                h.engine.stop(v);
            }
            h.engine.remove_source(r.id);
        }
    }

    fn sync_doors(&mut self) {
        let tree = self.base().get_tree();
        let frame = self.frame;
        let mut g = host();
        let Some(h) = g.as_mut() else { return };
        for n in tree.get_nodes_in_group("omg_door").iter_shared() {
            let Ok(d) = n.try_cast::<OmgDoor>() else { continue };
            let (size, open, mat) = {
                let b = d.bind();
                (b.size, b.openness.clamp(0.0, 1.0), b.material.to_string())
            };
            let Some(m) = materials::get(&mat) else { continue };
            let t = d.get_global_transform();
            let (x0, x1) = (-0.5 * size.x, -0.5 * size.x + (1.0 - open) * size.x);
            let mut mn = [f32::MAX; 3];
            let mut mx = [f32::MIN; 3];
            for &x in &[x0, x1] {
                for &y in &[-0.5 * size.y, 0.5 * size.y] {
                    for &z in &[-0.5 * size.z, 0.5 * size.z] {
                        let p = eng(t * Vector3::new(x, y, z));
                        for k in 0..3 {
                            mn[k] = mn[k].min(p[k]);
                            mx[k] = mx[k].max(p[k]);
                        }
                    }
                }
            }
            let enabled = open < 0.999;
            let key = d.instance_id();
            match self.doors.get_mut(&key) {
                None => {
                    let id = h.engine.add_object(mn, mx, m);
                    if !enabled {
                        h.engine.set_object_enabled(id, false);
                    }
                    self.doors.insert(key, DoorRec { id, min: mn, max: mx, enabled, seen: frame });
                }
                Some(r) => {
                    r.seen = frame;
                    if enabled && (r.min != mn || r.max != mx) {
                        h.engine.set_object(r.id, mn, mx);
                        r.min = mn;
                        r.max = mx;
                    }
                    if enabled != r.enabled {
                        h.engine.set_object_enabled(r.id, enabled);
                        r.enabled = enabled;
                    }
                }
            }
        }
        let gone: Vec<InstanceId> = self.doors.iter().filter(|(_, r)| r.seen != frame).map(|(k, _)| *k).collect();
        for k in gone {
            let r = self.doors.remove(&k).unwrap();
            h.engine.remove_object(r.id);
        }
    }

    fn sync_mix(&mut self) {
        let vals = [
            self.ambience_db,
            self.rain_db,
            self.rain_intensity,
            self.master_db,
            self.effective_quality() as f32,
            self.use_gpu as i32 as f32,
            self.rays as f32,
            self.bounces as f32,
            self.late_rays as f32,
            self.diffraction as i32 as f32,
            self.debug_paths as i32 as f32,
            self.engine_v4 as i32 as f32,
            self.gpu_budget,
            self.engine_v5 as i32 as f32,
            self.engine_v6 as i32 as f32,
        ];
        let cfg = self.sim_config();
        let mut g = host();
        let Some(h) = g.as_mut() else { return };
        for (i, v) in vals.iter().enumerate() {
            if self.sent[i] == *v {
                continue;
            }
            self.sent[i] = *v;
            match i {
                0 => h.send(OutCmd::AmbienceGain(db(*v))),
                1 => h.send(OutCmd::RainGain(db(*v))),
                2 => h.send(OutCmd::Rain(v.clamp(0.0, 1.0))),
                3 => h.engine.set_master(db(*v)),
                10 => h.engine.set_debug_paths(*v > 0.5),
                _ => {
                    // quality, compute and the pinned levers: one config
                    h.engine.set_config(cfg);
                    h.gpu = self.use_gpu;
                }
            }
        }
        if let Some((id, false)) = self.amb_clip {
            if let Some(c) = clips::get(id) {
                h.set_ambience(c.to_vec(), true, engine_rate());
                self.amb_clip = Some((id, true));
            }
        }
        if let Some((id, false)) = self.rain_clip {
            if let Some(c) = clips::get(id) {
                h.send(OutCmd::RainBank(c.to_vec()));
                self.rain_clip = Some((id, true));
            }
        }
    }
}

/// Walks the level: tag resolution + triangles/boxes into the Geometry.
struct Collector {
    g: Geometry,
    default: Option<Variant>,
    mats: HashMap<[u32; 7], MaterialId>,
    meshes: usize,
    boxes: usize,
}

impl Collector {
    fn material(&mut self, m: Material) -> MaterialId {
        let key = [
            m.absorption[0].to_bits(),
            m.absorption[1].to_bits(),
            m.absorption[2].to_bits(),
            m.scattering.to_bits(),
            m.transmission[0].to_bits(),
            m.transmission[1].to_bits(),
            m.transmission[2].to_bits(),
        ];
        if let Some(&id) = self.mats.get(&key) {
            return id;
        }
        let id = self.g.material(m);
        self.mats.insert(key, id);
        id
    }

    fn walk(&mut self, node: Gd<Node>, tag: Option<Variant>, boxes: bool) {
        let (mut tag, mut boxes) = (tag, boxes);
        if let Ok(g) = node.clone().try_cast::<OmgGeometry>() {
            let b = g.bind();
            // "" = back to the surfaces' own declarations
            tag = (!b.material.is_empty()).then(|| b.material.to_variant());
            boxes = b.as_boxes;
        }
        if node.has_meta("omg_material") {
            tag = Some(node.get_meta("omg_material"));
        }
        if let Ok(mi) = node.clone().try_cast::<MeshInstance3D>() {
            if mi.is_visible_in_tree() {
                if let Some(mesh) = mi.get_mesh() {
                    self.add_mesh(&mi, &mesh, tag.as_ref(), boxes);
                }
            }
        }
        for c in node.get_children().iter_shared() {
            self.walk(c, tag.clone(), boxes);
        }
    }

    fn add_mesh(&mut self, mi: &Gd<MeshInstance3D>, mesh: &Gd<Mesh>, tag: Option<&Variant>, boxes: bool) {
        let t = mi.get_global_transform();
        // explicit tag, else the surface material's declaration, else default
        let surface = |s: i32| {
            mi.get_active_material(s).filter(|m| m.has_meta("omg_material")).map(|m| m.get_meta("omg_material"))
        };
        if boxes {
            let v = tag.cloned().or_else(|| surface(0)).or_else(|| self.default.clone());
            let Some(m) = v.as_ref().and_then(material_of) else { return };
            let a = mesh.get_aabb();
            let mut mn = [f32::MAX; 3];
            let mut mx = [f32::MIN; 3];
            for i in 0..8 {
                let c = a.position
                    + Vector3::new(
                        if i & 1 != 0 { a.size.x } else { 0.0 },
                        if i & 2 != 0 { a.size.y } else { 0.0 },
                        if i & 4 != 0 { a.size.z } else { 0.0 },
                    );
                let p = eng(t * c);
                for k in 0..3 {
                    mn[k] = mn[k].min(p[k]);
                    mx[k] = mx[k].max(p[k]);
                }
            }
            let id = self.material(m);
            self.g.add_box(mn, mx, id);
            self.boxes += 1;
            return;
        }
        for s in 0..mesh.get_surface_count() {
            let v = tag.cloned().or_else(|| surface(s)).or_else(|| self.default.clone());
            let Some(m) = v.as_ref().and_then(material_of) else { continue };
            if let Ok(am) = mesh.clone().try_cast::<godot::classes::ArrayMesh>() {
                if am.surface_get_primitive_type(s) != godot::classes::mesh::PrimitiveType::TRIANGLES {
                    continue;
                }
            }
            let arrays = mesh.surface_get_arrays(s);
            let Some(Ok(verts)) = arrays.get(0).map(|v| v.try_to::<PackedVector3Array>()) else { continue };
            let pos: Vec<[f32; 3]> = verts.as_slice().iter().map(|v| eng(t * *v)).collect();
            let idx: Vec<u32> = match arrays.get(12).and_then(|v| v.try_to::<PackedInt32Array>().ok()).filter(|i| !i.is_empty()) {
                Some(ix) => ix.as_slice().iter().map(|&i| i as u32).filter(|&i| (i as usize) < pos.len()).collect(),
                None => (0..pos.len() as u32).collect(),
            };
            let n = idx.len() / 3 * 3;
            let id = self.material(m);
            self.g.add_mesh(&pos, &idx[..n], id);
            self.meshes += 1;
        }
    }
}
