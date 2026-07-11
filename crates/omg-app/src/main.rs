//! Native demo app for the omg-audio engine.
//!
//!   omg-audio                                     live orbit demo (clapper)
//!   omg-audio --render out.wav [--secs 12]        offline orbit render
//!   omg-audio --input voice.wav ...               use a WAV as source signal
//!   omg-audio --walkthrough --render out.wav --json path.json
//!       scripted walkthrough: fixed music + voice sources, 4 rooms + outdoors
//!
//! Architecture: the simulation clock (20 Hz) and the audio clock
//! (per-sample) only communicate through ParamBlock. Live mode runs them on
//! separate threads exactly like the web build runs them on Worker +
//! AudioWorklet; offline mode interleaves them serially.

use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use omg_core::material::Material;
use omg_core::params::ParamBlock;
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_dsp::ambi::NCH;
use omg_dsp::output::OutputStage;
use omg_scene::sim::Sim;
use omg_scene::walkthrough;
use omg_scene::world::WorldSim;

const SIM_RATE_HZ: f32 = 20.0;
const ORBIT_PERIOD_S: f32 = 8.0;
const ORBIT_RADIUS: f32 = 2.2;

// -------------------------------------------------------------- orbit scene

struct Orbit {
    room: Shoebox,
    listener: Vec3,
}

impl Orbit {
    fn new() -> Self {
        Self {
            room: Shoebox::new(
                Vec3::new(8.0, 6.0, 3.0),
                [
                    Material::DRYWALL,
                    Material::WOOD_PANEL,
                    Material::DRYWALL,
                    Material::CONCRETE,
                    Material::CARPET,
                    Material::ACOUSTIC_TILE,
                ],
            ),
            listener: Vec3::new(4.0, 3.0, 1.5),
        }
    }

    fn source_at(&self, t: f32) -> Vec3 {
        let ang = core::f32::consts::TAU * t / ORBIT_PERIOD_S;
        Vec3::new(
            self.listener.x + ORBIT_RADIUS * ang.cos(),
            self.listener.y + ORBIT_RADIUS * ang.sin(),
            1.5,
        )
    }
}

// -------------------------------------------------------------- test signal

/// Percussive noise bursts (a "clapper") — transients make early reflections
/// and reverb tail clearly audible.
struct Exciter {
    rng: Rng,
    counter: u64,
    sample_rate: f32,
}

impl Exciter {
    fn new(sample_rate: f32) -> Self {
        Self { rng: Rng::new(0xBADA55), counter: 0, sample_rate }
    }

    #[inline]
    fn tick(&mut self) -> f32 {
        let period = (self.sample_rate * 0.6) as u64;
        let phase = self.counter % period;
        self.counter += 1;
        let burst_len = (self.sample_rate * 0.03) as u64;
        if phase < burst_len {
            let env = (-(phase as f32) / (self.sample_rate * 0.006)).exp();
            (self.rng.next_f32() * 2.0 - 1.0) * env * 0.9
        } else {
            0.0
        }
    }
}

enum Source {
    Clapper(Exciter),
    File { data: Vec<f32>, pos: usize },
}

impl Source {
    #[inline]
    fn tick(&mut self) -> f32 {
        match self {
            Source::Clapper(e) => e.tick(),
            Source::File { data, pos } => {
                let s = data[*pos];
                *pos = (*pos + 1) % data.len();
                s
            }
        }
    }
}

/// Load a WAV, mix to mono, linear-resample to `target_fs`, normalize.
fn load_wav_mono(path: &str, target_fs: f32) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open input wav");
    let spec = reader.spec();
    let ch = spec.channels as usize;

    let mono: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader
            .samples::<f32>()
            .map(|s| s.unwrap())
            .collect::<Vec<_>>()
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect(),
        (hound::SampleFormat::Int, bits) => {
            let scale = 1.0 / (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 * scale)
                .collect::<Vec<_>>()
                .chunks(ch)
                .map(|c| c.iter().sum::<f32>() / ch as f32)
                .collect()
        }
    };

    let ratio = spec.sample_rate as f32 / target_fs;
    let out_len = (mono.len() as f32 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let x = i as f32 * ratio;
        let j = x as usize;
        let f = x - j as f32;
        let a = mono[j.min(mono.len() - 1)];
        let b = mono[(j + 1).min(mono.len() - 1)];
        out.push(a + f * (b - a));
    }

    let peak = out.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak > 1e-6 {
        let g = 0.6 / peak;
        for s in &mut out {
            *s *= g;
        }
    }
    out
}

// --------------------------------------------------------------------- main

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let render_path = arg_value(&args, "--render");
    let secs_opt: Option<f32> = arg_value(&args, "--secs").and_then(|s| s.parse().ok());
    let input = arg_value(&args, "--input");
    let json_path = arg_value(&args, "--json");
    let walk = args.iter().any(|a| a == "--walkthrough");

    let make_source = |fs: f32| -> Source {
        match &input {
            Some(p) => Source::File { data: load_wav_mono(p, fs), pos: 0 },
            None => Source::Clapper(Exciter::new(fs)),
        }
    };

    match (render_path, walk) {
        (Some(path), true) => {
            let music = arg_value(&args, "--music").unwrap_or("assets/aria48.wav".into());
            let voice = arg_value(&args, "--voice").unwrap_or("assets/alice48.wav".into());
            let club = arg_value(&args, "--club").unwrap_or("assets/club48.wav".into());
            render_walkthrough(
                &path,
                secs_opt.unwrap_or(walkthrough::DURATION_S),
                &[music, voice, club],
                json_path.as_deref(),
            )
        }
        (Some(path), false) => render_orbit(&path, secs_opt.unwrap_or(12.0), make_source),
        (None, _) => run_live(secs_opt, make_source),
    }
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

// ------------------------------------------------------------ output setup

fn make_output(fs: f32) -> OutputStage {
    let bytes = ["assets/hrir_dodeca20.bin", "assets/hrir_ico12.bin"]
        .iter()
        .find_map(|p| std::fs::read(p).ok());
    let out = OutputStage::from_speaker_bytes(bytes.as_deref(), fs);
    if out.is_binaural() {
        println!(
            "binaural: MIT KEMAR HRIRs, {} virtual speakers, order-2 ambisonics",
            out.speaker_count()
        );
    } else {
        println!("binaural: no HRIR asset — cardioid stereo fallback");
    }
    out
}

fn load_grid() -> Option<Arc<omg_dsp::hrtf::HrirGrid>> {
    match omg_dsp::hrtf::HrirGrid::load("assets/hrir_grid.bin") {
        Ok(g) => {
            println!("point render: {} HRIR directions", g.len());
            Some(Arc::new(g))
        }
        Err(_) => None,
    }
}

fn make_renderer(fs: f32, grid: Option<Arc<omg_dsp::hrtf::HrirGrid>>) -> omg_dsp::Renderer {
    let mut r = omg_dsp::Renderer::with_grid(fs, grid);
    if let Some(n) = std::env::var("OMG_POINT_TAPS").ok().and_then(|v| v.parse().ok()) {
        r.set_point_budget(n);
    }
    r.mute_taps = std::env::var("OMG_MUTE_TAPS").is_ok();
    r.mute_own_fdn = std::env::var("OMG_MUTE_OWN").is_ok();
    r.mute_remote = std::env::var("OMG_MUTE_REMOTE").is_ok();
    r
}

// ------------------------------------------------------------ wav utilities

struct WavOut {
    writer: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    peak: f32,
    sum_sq: f64,
    n: usize,
}

impl WavOut {
    fn create(path: &str, fs: f32) -> Self {
        let writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 2,
                sample_rate: fs as u32,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .expect("create wav");
        Self { writer, peak: 0.0, sum_sq: 0.0, n: 0 }
    }

    #[inline]
    fn write(&mut self, l: f32, r: f32) {
        self.peak = self.peak.max(l.abs()).max(r.abs());
        self.sum_sq += (l * l + r * r) as f64 * 0.5;
        self.n += 1;
        self.writer.write_sample((l * 32767.0) as i16).unwrap();
        self.writer.write_sample((r * 32767.0) as i16).unwrap();
    }

    fn finish(self, label: &str) {
        let rms = (self.sum_sq / self.n as f64).sqrt();
        self.writer.finalize().unwrap();
        println!("{label}: peak {:.3}, RMS {:.4}", self.peak, rms);
    }
}

// ---------------------------------------------------------- offline renders

fn render_orbit(path: &str, secs: f32, make_source: impl Fn(f32) -> Source) {
    let fs = 48_000.0f32;
    let scene = Orbit::new();
    let mut sim = Sim::new();
    let mut renderer = make_renderer(fs, load_grid());
    let mut source = make_source(fs);
    let mut out = make_output(fs);

    let total = (secs * fs) as usize;
    let sim_interval = (fs / SIM_RATE_HZ) as usize;
    let mut wav = WavOut::create(path, fs);
    let started = Instant::now();

    for n in 0..total {
        if n % sim_interval == 0 {
            let t = n as f32 / fs;
            let pb = sim.update(&scene.room, scene.source_at(t), scene.listener, 0.0);
            if pb.version == 1 {
                println!(
                    "sim: {} taps | RT60 mid {:.2}s | late level {:.4}",
                    pb.taps.len(),
                    pb.reverb.rt60[1],
                    pb.reverb.level[1]
                );
            }
            renderer.set_params(&pb);
        }
        let mut bus = [0.0f32; NCH];
        let (pl, pr) = renderer.process(source.tick(), &mut bus);
        let (l, r) = out.process(&bus, pl, pr);
        wav.write(l, r);
    }
    wav.finish(&format!("rendered {secs}s to {path}"));
    println!("realtime factor: {:.1}×", secs / started.elapsed().as_secs_f32());
}

fn render_walkthrough(path: &str, secs: f32, inputs: &[String], json_path: Option<&str>) {
    let fs = 48_000.0f32;
    let mut world = WorldSim::new();
    let grid = load_grid();

    struct WalkSource {
        signal: Source,
        renderer: omg_dsp::Renderer,
    }
    let mut srcs: Vec<WalkSource> = inputs
        .iter()
        .map(|p| WalkSource {
            signal: Source::File { data: load_wav_mono(p, fs), pos: 0 },
            renderer: make_renderer(fs, grid.clone()),
        })
        .collect();
    assert!(srcs.len() <= world.defs.len()); // dynamic slot has no file

    let total = (secs * fs) as usize;
    let sim_interval = (fs / SIM_RATE_HZ) as usize;
    let mut wav = WavOut::create(path, fs);
    let mut out = make_output(fs);
    let mut ticks_json = String::new();
    let mut last_room = usize::MAX;
    let mut sim_time = Duration::ZERO;
    let mut n_ticks = 0u32;

    for n in 0..total {
        if n % sim_interval == 0 {
            let t = n as f32 / fs;
            let sim_started = Instant::now();
            let (blocks, info) = world.tick_scripted(t);
            for (ws, pb) in srcs.iter_mut().zip(blocks.iter()) {
                ws.renderer.set_params(pb);
            }
            sim_time += sim_started.elapsed();
            n_ticks += 1;

            if info.room != last_room {
                println!(
                    "t={t:5.1}s  entering {:12} | RT60 mid {:.2}s",
                    world.rooms[info.room].name, info.rt60_mid
                );
                last_room = info.room;
            }
            if !ticks_json.is_empty() {
                ticks_json.push(',');
            }
            let routes: Vec<String> = info
                .routes
                .iter()
                .map(|route| {
                    let pts: Vec<String> =
                        route.iter().map(|p| format!("[{:.2},{:.2}]", p.0, p.1)).collect();
                    format!("[{}]", pts.join(","))
                })
                .collect();
            ticks_json.push_str(&format!(
                r#"{{"t":{t:.3},"pos":[{:.3},{:.3}],"yaw":{:.4},"room":{},"rt60":{:.3},"routes":[{}]}}"#,
                info.listener.0,
                info.listener.1,
                info.yaw,
                info.room,
                info.rt60_mid,
                routes.join(",")
            ));
        }

        let mut bus = [0.0f32; NCH];
        let mut pl = 0.0f32;
        let mut pr = 0.0f32;
        for ws in &mut srcs {
            let x = ws.signal.tick();
            let (a, b) = ws.renderer.process(x, &mut bus);
            pl += a;
            pr += b;
        }
        let (l, r) = out.process(&bus, pl, pr);
        wav.write(l, r);
    }
    wav.finish(&format!("rendered {secs}s walkthrough to {path}"));
    println!(
        "simulation cost: {:.2?} total, {:.2?} avg/tick ({} ticks, {} sources)",
        sim_time,
        sim_time / n_ticks.max(1),
        n_ticks,
        srcs.len()
    );

    if let Some(jp) = json_path {
        let rooms_json: Vec<String> = world
            .rooms
            .iter()
            .map(|r| {
                format!(
                    r#"{{"name":"{}","min":[{},{}],"max":[{},{}]}}"#,
                    r.name, r.min.0, r.min.1, r.max.0, r.max.1
                )
            })
            .collect();
        let doors_json: Vec<String> = walkthrough::doors()
            .iter()
            .map(|d| format!(r#"{{"pos":[{},{}],"axis":{}}}"#, d.pos.0, d.pos.1, d.axis))
            .collect();
        let sources_json: Vec<String> = world
            .defs
            .iter()
            .map(|s| {
                format!(
                    r#"{{"name":"{}","pos":[{},{}],"room":{}}}"#,
                    s.name, s.pos.0, s.pos.1, s.room
                )
            })
            .collect();
        let json = format!(
            r#"{{"duration":{secs},"rooms":[{}],"doors":[{}],"sources":[{}],"ticks":[{}]}}"#,
            rooms_json.join(","),
            doors_json.join(","),
            sources_json.join(","),
            ticks_json
        );
        let mut f = std::fs::File::create(jp).expect("create json");
        f.write_all(json.as_bytes()).unwrap();
        println!("wrote path data to {jp}");
    }
}

// -------------------------------------------------------------------- live

fn run_live(secs: Option<f32>, make_source: impl Fn(f32) -> Source) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = host.default_output_device().expect("no output device");
    let config = device.default_output_config().expect("no output config");
    assert_eq!(
        config.sample_format(),
        cpal::SampleFormat::F32,
        "demo expects f32 output (macOS default)"
    );
    let fs = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;
    let stream_config: cpal::StreamConfig = config.into();

    println!("live: {} Hz, {} ch — source orbits every {ORBIT_PERIOD_S}s, Ctrl+C to quit", fs, channels);

    // The mailbox between the clocks. Audio thread uses try_lock and simply
    // keeps last params on contention — never blocks. (Web build: postMessage.)
    let mailbox: Arc<Mutex<Option<ParamBlock>>> = Arc::new(Mutex::new(None));

    {
        let mailbox = Arc::clone(&mailbox);
        std::thread::spawn(move || {
            let scene = Orbit::new();
            let mut sim = Sim::new();
            let started = Instant::now();
            loop {
                let t = started.elapsed().as_secs_f32();
                let pb = sim.update(&scene.room, scene.source_at(t), scene.listener, 0.0);
                *mailbox.lock().unwrap() = Some(pb);
                std::thread::sleep(Duration::from_secs_f32(1.0 / SIM_RATE_HZ));
            }
        });
    }

    let mut renderer = make_renderer(fs, load_grid());
    let mut source = make_source(fs);
    let mut out = make_output(fs);
    let mailbox_audio = Arc::clone(&mailbox);
    let stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                if let Ok(mut guard) = mailbox_audio.try_lock() {
                    if let Some(pb) = guard.take() {
                        renderer.set_params(&pb);
                    }
                }
                for frame in data.chunks_mut(channels) {
                    let mut bus = [0.0f32; NCH];
                    let (pl, pr) = renderer.process(source.tick(), &mut bus);
                    let (l, r) = out.process(&bus, pl, pr);
                    frame[0] = l;
                    if channels > 1 {
                        frame[1] = r;
                    }
                    for c in frame.iter_mut().skip(2) {
                        *c = 0.0;
                    }
                }
            },
            |err| eprintln!("stream error: {err}"),
            None,
        )
        .expect("build stream");
    stream.play().expect("play");

    match secs {
        Some(s) => std::thread::sleep(Duration::from_secs_f32(s)),
        None => loop {
            std::thread::sleep(Duration::from_secs(3600));
        },
    }
}
