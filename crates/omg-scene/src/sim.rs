//! Per-(source, room) simulation state: image sources + stochastic tracing
//! with temporal accumulation. Moved verbatim from the native app so the
//! web build shares it.

use omg_core::ism::image_source_taps;
use omg_core::material::Material;
use omg_core::params::ParamBlock;
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::vec3::Vec3;
use omg_core::{NBANDS, SPEED_OF_SOUND};

pub const ISM_ORDER: u32 = 3;
pub const N_RAYS: u32 = 4096;
/// Near-field radius of a doorway acting as a secondary source,
/// √(aperture area / 2π) for a ~2.3 m² door: within it the re-radiated
/// field is the incoming field; beyond it the 1/d spreading takes over.
const APERTURE_RERADIATION: f32 = 0.6;

/// Exterior building wall, for first-order outdoor reflections
/// (coordinates in the outdoor room's local frame).
pub struct Facade {
    pub axis: usize, // 0: x = plane, 1: y = plane
    pub plane: f32,
    pub lo: f32,
    pub hi: f32,
    pub refl: [f32; NBANDS],
}

/// Trace-skip gate: the late field is a slowly-varying STATISTIC. When
/// the trace inputs haven't meaningfully changed, another 4096 rays only
/// re-measure the same distribution — the EMA absorbs the variance either
/// way, so skipping trades nothing but staleness, and the stagger keeps
/// even that bounded (~2.5 Hz refresh). Motion and door swings re-trace
/// immediately.
struct TraceGate {
    last_src: Vec3,
    last_lis: Vec3,
    last_energy: [f32; NBANDS],
    age: u32,
}

impl TraceGate {
    fn new(phase: u32) -> Self {
        Self {
            last_src: Vec3::new(f32::MAX, 0.0, 0.0),
            last_lis: Vec3::new(f32::MAX, 0.0, 0.0),
            last_energy: [0.0; NBANDS],
            age: phase,
        }
    }

    fn should_trace(&mut self, src: Vec3, lis: Vec3, energy: [f32; NBANDS]) -> bool {
        self.age += 1;
        let d = |a: Vec3, b: Vec3| (a - b).length();
        let energy_moved = (0..NBANDS).any(|b| {
            let (e0, e1) = (self.last_energy[b], energy[b]);
            (e1 - e0).abs() > 0.1 * e0.max(e1).max(1e-6)
        });
        if self.age >= 8
            || d(src, self.last_src) > 0.25
            || d(lis, self.last_lis) > 0.25
            || energy_moved
        {
            self.last_src = src;
            self.last_lis = lis;
            self.last_energy = energy;
            self.age = 0;
            true
        } else {
            false
        }
    }
}

pub struct Sim {
    rng: Rng,
    gate: TraceGate,
    echo_avg: Echogram,
    echo_cur: Echogram,
    taps_buf: Vec<omg_core::params::Tap>,
    version: u64,
}

static SIM_COUNTER: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

impl Sim {
    pub fn new() -> Self {
        // stagger refresh phases across instances so idle re-traces don't
        // all land on the same tick
        let phase = SIM_COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed) % 8;
        Self {
            rng: Rng::new(0xC0FFEE),
            gate: TraceGate::new(phase),
            echo_avg: Echogram::new(),
            echo_cur: Echogram::new(),
            taps_buf: Vec::new(),
            version: 0,
        }
    }

    pub fn update(&mut self, room: &Shoebox, src: Vec3, listener: Vec3, yaw: f32) -> ParamBlock {
        self.update_routed(room, src, listener, yaw, 0.0, [1.0; NBANDS])
    }

    /// One tick: room-local source/listener positions + listener yaw
    /// (rotation about z; arrival directions are made listener-relative).
    /// `extra_dist`/`muffle` describe the pre-room path of a portal-routed
    /// source (0 / unity when source and listener share the room).
    pub fn update_routed(
        &mut self,
        room: &Shoebox,
        src: Vec3,
        listener: Vec3,
        yaw: f32,
        extra_dist: f32,
        muffle: [f32; NBANDS],
    ) -> ParamBlock {
        image_source_taps(room, src, listener, ISM_ORDER, &mut self.taps_buf);

        // World → listener frame (listener faces +x at yaw = 0), then fold
        // the pre-door path into each tap. The aperture is a Huygens
        // secondary source: the field arriving at the door (∝ 1/extra)
        // re-radiates into the room and decays with IN-ROOM distance —
        // the ISM's own 1/d — not with total path length. Folding extra
        // into each image's length instead (1/(d_room + extra)) held all
        // ~63 image taps at direct-path strength: a distant motor summed
        // to +18 dB inside the room and got LOUDER as it drove away.
        // True line-of-sight continuation through the opening is the
        // straight tap's job (world.rs, key 9000), not this path's.
        let (sin, cos) = yaw.sin_cos();
        let a_in = APERTURE_RERADIATION / extra_dist.max(APERTURE_RERADIATION);
        for t in &mut self.taps_buf {
            let [dx, dy, dz] = t.dir;
            t.dir = [cos * dx + sin * dy, -sin * dx + cos * dy, dz];
            if extra_dist > 0.0 {
                for b in 0..NBANDS {
                    t.gains[b] *= muffle[b] * a_in;
                }
                t.delay_s += extra_dist / SPEED_OF_SOUND;
            }
        }

        // Aperture decoherence: the field entering through a doorway is
        // spatially extended and partly diffuse, not a coherent point — the
        // listener-room image sources it excites are weaker and less
        // regular than a true point source's. Keep the direct tap, shave
        // the reflections (~4 dB) to kill the comb ring.
        if extra_dist > 0.0 {
            let direct = self
                .taps_buf
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.delay_s.total_cmp(&b.1.delay_s))
                .map(|(i, _)| i);
            for (i, t) in self.taps_buf.iter_mut().enumerate() {
                if Some(i) != direct {
                    for b in 0..NBANDS {
                        t.gains[b] *= 0.6;
                    }
                }
            }
        }

        // Energy entering the room through the portal, per band (heuristic
        // spreading loss over the pre-door path).
        let src_energy: [f32; NBANDS] =
            core::array::from_fn(|b| muffle[b] * muffle[b] / (1.0 + extra_dist * extra_dist));
        let traced = self.gate.should_trace(src, listener, src_energy);
        if traced {
            trace(room, src, listener, N_RAYS, src_energy, &mut self.rng, &mut self.echo_cur);
        }
        self.finish_update(traced)
    }

    /// Same-room speaker rig: identical signal from several emitters
    /// (power-split), one shared reverb estimate. Tap keys are offset per
    /// emitter (64 ISM slots each) so every path keeps a stable identity.
    pub fn update_multi(
        &mut self,
        room: &Shoebox,
        emitters: &[Vec3],
        listener: Vec3,
        yaw: f32,
    ) -> ParamBlock {
        self.taps_buf.clear();
        let amp = 1.0 / (emitters.len() as f32).sqrt();
        let (sin, cos) = yaw.sin_cos();
        let mut tmp = Vec::new();
        for (ei, src) in emitters.iter().enumerate() {
            // order 2 per emitter: a dense rig masks higher-order detail,
            // and the tap count is what breaks throttled devices
            image_source_taps(room, *src, listener, 2, &mut tmp);
            for t in &mut tmp {
                t.key += ei as u32 * 64;
                let [dx, dy, dz] = t.dir;
                t.dir = [cos * dx + sin * dy, -sin * dx + cos * dy, dz];
                for b in 0..NBANDS {
                    t.gains[b] *= amp;
                }
            }
            self.taps_buf.append(&mut tmp);
        }
        // Reverb: one trace from the rig centroid-ish (first emitter);
        // in-room late statistics barely depend on the exact position.
        let traced = self.gate.should_trace(emitters[0], listener, [1.0; NBANDS]);
        if traced {
            trace(room, emitters[0], listener, N_RAYS, [1.0; NBANDS], &mut self.rng, &mut self.echo_cur);
        }
        self.finish_update(traced)
    }

    /// Open air: direct path + ground reflection + first-order slap-back
    /// off building facades — no enclosing reverb.
    pub fn update_outdoor(
        &mut self,
        ground: &Material,
        facades: &[Facade],
        src: Vec3,
        listener: Vec3,
        yaw: f32,
        extra_dist: f32,
        muffle: [f32; NBANDS],
    ) -> ParamBlock {
        use omg_core::material::air_attenuation;
        use omg_core::params::Tap;

        self.taps_buf.clear();
        let (sin, cos) = yaw.sin_cos();
        let ground_refl = ground.reflection_amplitude();

        let mut push = |key: u32, img: Vec3, refl: Option<[f32; NBANDS]>| {
            let to = img - listener;
            let d = to.length().max(0.3);
            let dir = to.normalize();
            let total = d + extra_dist;
            let air = air_attenuation(total);
            let gains = core::array::from_fn(|b| {
                muffle[b] * air[b] / total * refl.map_or(1.0, |r| r[b])
            });
            self.taps_buf.push(Tap {
                key,
                delay_s: total / SPEED_OF_SOUND,
                dir: [cos * dir.x + sin * dir.y, -sin * dir.x + cos * dir.y, dir.z],
                gains,
            });
        };
        push(0, src, None);
        push(1, Vec3::new(src.x, src.y, -src.z), Some(ground_refl));

        // Facade mirrors: valid when source and listener share the outer
        // side of the wall and the reflection point lands on the wall.
        for (fi, f) in facades.iter().enumerate() {
            let (sp, lp2) = if f.axis == 0 {
                (src.x - f.plane, listener.x - f.plane)
            } else {
                (src.y - f.plane, listener.y - f.plane)
            };
            if sp * lp2 <= 0.01 {
                continue;
            }
            let img = if f.axis == 0 {
                Vec3::new(2.0 * f.plane - src.x, src.y, src.z)
            } else {
                Vec3::new(src.x, 2.0 * f.plane - src.y, src.z)
            };
            // reflection point along listener → image
            let (l0, i0) = if f.axis == 0 { (listener.x, img.x) } else { (listener.y, img.y) };
            let denom = i0 - l0;
            if denom.abs() < 1e-6 {
                continue;
            }
            let t = (f.plane - l0) / denom;
            if !(0.02..=0.98).contains(&t) {
                continue;
            }
            let along = if f.axis == 0 {
                listener.y + t * (img.y - listener.y)
            } else {
                listener.x + t * (img.x - listener.x)
            };
            if along < f.lo || along > f.hi {
                continue;
            }
            push(10 + fi as u32, img, Some(f.refl));
        }

        self.version += 1;
        ParamBlock {
            taps: self.taps_buf.clone(),
            reverb: omg_core::params::ReverbParams { rt60: [0.25; NBANDS], level: [0.0; NBANDS] },
            remote: None,
            version: self.version,
        }
    }

    fn finish_update(&mut self, traced: bool) -> ParamBlock {
        if traced {
            let alpha = if self.version == 0 { 1.0 } else { 0.3 };
            self.echo_avg.ema(&self.echo_cur, alpha);
        }
        let reverb = estimate_reverb(&self.echo_avg);

        self.version += 1;
        ParamBlock { taps: self.taps_buf.clone(), reverb, remote: None, version: self.version }
    }

    /// Reverb estimate only (no taps): used for the coupled-room wet field —
    /// how reverberant the source's room is at its exit doorway.
    pub fn reverb_only(
        &mut self,
        room: &Shoebox,
        src: Vec3,
        receiver: Vec3,
    ) -> omg_core::params::ReverbParams {
        if self.gate.should_trace(src, receiver, [1.0; NBANDS]) {
            trace(room, src, receiver, N_RAYS, [1.0; NBANDS], &mut self.rng, &mut self.echo_cur);
            let alpha = if self.version == 0 { 1.0 } else { 0.3 };
            self.echo_avg.ema(&self.echo_cur, alpha);
        }
        self.version += 1;
        estimate_reverb(&self.echo_avg)
    }
}

impl Default for Sim {
    fn default() -> Self {
        Self::new()
    }
}
