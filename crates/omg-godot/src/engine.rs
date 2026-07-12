//! Godot-facing facade over the omg-audio engine: the same two-clock
//! architecture as the native and web builds. A background thread runs
//! `WorldSim` at 20 Hz; the game thread pulls rendered stereo whenever it
//! wants to feed its `AudioStreamGenerator`. The two sides only speak
//! through the ParamBlock mailbox — exactly like the other frontends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use omg_core::params::ParamBlock;
use omg_dsp::ambi::NCH;
use omg_dsp::hrtf::HrirGrid;
use omg_dsp::output::OutputStage;
use omg_dsp::Renderer;
use omg_scene::walkthrough::DYN_SLOTS;
use omg_scene::world::WorldSim;

/// KEMAR HRIRs baked into the library — a game integration should not
/// depend on loose asset files next to the dylib.
const HRIR_GRID: &[u8] = include_bytes!("../../../assets/hrir_grid.bin");
const HRIR_SPEAKERS: &[u8] = include_bytes!("../../../assets/hrir_dodeca20.bin");

pub const NSRC: usize = 10;

#[derive(Clone, Copy, Default)]
struct Pose {
    x: f32,
    y: f32,
    yaw: f32,
    dynamics: [(f32, f32, f32, bool); DYN_SLOTS],
}

#[derive(Default)]
struct Mailbox {
    blocks: Option<Vec<ParamBlock>>,
    room: usize,
    rt60_mid: f32,
}

pub struct SpatialEngine {
    renderers: Vec<Renderer>,
    sources: Vec<(Vec<f32>, usize)>, // (loop samples, play position)
    out: OutputStage,
    pose: Arc<Mutex<Pose>>,
    mailbox: Arc<Mutex<Mailbox>>,
    stop: Arc<AtomicBool>,
    sim_thread: Option<JoinHandle<()>>,
    pub room: usize,
    pub rt60_mid: f32,
}

impl SpatialEngine {
    pub fn new(sample_rate: f32) -> Self {
        let grid = Arc::new(HrirGrid::from_bytes(HRIR_GRID));
        let renderers = (0..NSRC)
            .map(|_| Renderer::with_grid(sample_rate, Some(grid.clone())))
            .collect();
        let pose = Arc::new(Mutex::new(Pose::default()));
        let mailbox = Arc::new(Mutex::new(Mailbox::default()));
        let stop = Arc::new(AtomicBool::new(false));

        let sim_thread = {
            let (pose, mailbox, stop) = (pose.clone(), mailbox.clone(), stop.clone());
            Some(std::thread::spawn(move || {
                let mut world = WorldSim::new();
                while !stop.load(Ordering::Relaxed) {
                    let p = *pose.lock().unwrap();
                    for (slot, (x, y, z, active)) in p.dynamics.iter().enumerate() {
                        world.set_dynamic(slot, *x, *y, *z, *active);
                    }
                    let (blocks, info) = world.tick_at(p.x, p.y, p.yaw);
                    let mut mb = mailbox.lock().unwrap();
                    mb.blocks = Some(blocks);
                    mb.room = info.room;
                    mb.rt60_mid = info.rt60_mid;
                    drop(mb);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }))
        };

        Self {
            renderers,
            sources: (0..NSRC).map(|_| (Vec::new(), 0)).collect(),
            out: OutputStage::from_speaker_bytes(Some(HRIR_SPEAKERS), sample_rate),
            pose,
            mailbox,
            stop,
            sim_thread,
            room: 0,
            rt60_mid: 0.0,
        }
    }

    pub fn set_source_samples(&mut self, i: usize, mut samples: Vec<f32>) {
        omg_dsp::level::normalize_rms(&mut samples, omg_dsp::level::REF_CLIP_RMS);
        if let Some(s) = self.sources.get_mut(i) {
            *s = (samples, 0);
        }
    }

    pub fn set_listener(&self, x: f32, y: f32, yaw: f32) {
        let mut p = self.pose.lock().unwrap();
        p.x = x;
        p.y = y;
        p.yaw = yaw;
    }

    pub fn set_dynamic(&self, slot: usize, x: f32, y: f32, z: f32, active: bool) {
        if slot < DYN_SLOTS {
            self.pose.lock().unwrap().dynamics[slot] = (x, y, z, active);
        }
    }

    /// Fast head rotation (camera yaw), applied at the DSP without waiting
    /// for a simulation tick.
    pub fn set_head_yaw(&mut self, yaw: f32) {
        for r in &mut self.renderers {
            r.set_head_yaw(yaw);
        }
        self.out.set_head_yaw(yaw);
    }

    pub fn set_point_budget(&mut self, n: usize) {
        for r in &mut self.renderers {
            r.set_point_budget(n);
        }
    }

    pub fn agc_gain(&self) -> f32 {
        self.out.agc_gain()
    }

    /// Render `frames` stereo samples into `out` as interleaved (l, r).
    pub fn render(&mut self, frames: usize, out: &mut Vec<(f32, f32)>) {
        if let Some(blocks) = {
            let mut mb = self.mailbox.lock().unwrap();
            self.room = mb.room;
            self.rt60_mid = mb.rt60_mid;
            mb.blocks.take()
        } {
            for (r, pb) in self.renderers.iter_mut().zip(blocks.iter()) {
                r.set_params(pb);
            }
        }
        out.clear();
        out.reserve(frames);
        for _ in 0..frames {
            let mut bus = [0.0f32; NCH];
            let mut pl = 0.0f32;
            let mut pr = 0.0f32;
            for (ren, (data, pos)) in self.renderers.iter_mut().zip(self.sources.iter_mut()) {
                let x = if data.is_empty() {
                    0.0
                } else {
                    let s = data[*pos];
                    *pos = (*pos + 1) % data.len();
                    s
                };
                let (a, b) = ren.process(x, &mut bus);
                pl += a;
                pr += b;
            }
            out.push(self.out.process(&bus, pl, pr));
        }
    }
}

impl Drop for SpatialEngine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.sim_thread.take() {
            let _ = h.join();
        }
    }
}
