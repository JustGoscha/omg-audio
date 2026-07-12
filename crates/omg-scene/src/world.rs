//! WorldSim: one tick of the full demo-world simulation for all sources —
//! portal routing, room blending, coupled-room reverb — producing one
//! ParamBlock per source plus viz/telemetry info. Drives both the native
//! scripted walkthrough and the interactive web build.

use crate::dome::{door_panels, DomeProbe, DOME_ID_BASE};
use crate::environment::AcousticsGraph;
use crate::sim::{Facade, Sim};
use crate::walkthrough::{
    self, aperture_routes, blended_state_at, blended_state_for, route_source,
    straight_path_transmission, to_local, BlendedState, Door, RoomDef, SourceDef,
};
use omg_dsp::env::{EnvRoute, EnvWindow, Environment, MAX_ENV_ROUTES, MAX_ENV_WINDOWS};
use omg_core::ism::image_source_taps;
use omg_core::paths::{AutoPaths, PathBudget};
use omg_core::material::{air_attenuation, Material};
use omg_core::params::{ParamBlock, RemoteReverb, Tap};
use omg_core::SPEED_OF_SOUND;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

pub struct TickInfo {
    pub listener: (f32, f32),
    pub yaw: f32,
    pub room: usize,
    pub rt60_mid: f32,
    /// Apparent acoustic path per source (source → doors → listener).
    pub routes: Vec<Vec<(f32, f32)>>,
    /// How the outdoor field (ambience, rain) reaches this pose: aperture
    /// inlets, shell seep, roof exposure — geometry-priced, blend-smooth.
    pub env: Environment,
}

pub struct WorldSim {
    pub rooms: Vec<RoomDef>,
    pub doors: Vec<Door>,
    pub defs: Vec<SourceDef>,
    /// Per source, per room: simulation state (echogram accumulation).
    sims: Vec<Vec<Sim>>,
    /// Per source: source-room-at-exit-door state for coupled reverb.
    sim_remotes: Vec<Sim>,
    /// Per source emission height (dynamic sources fly).
    src_z: Vec<f32>,
    dynamic_active: [bool; walkthrough::DYN_SLOTS],
    /// Exterior building walls in outdoor-local coordinates.
    facades: Vec<Facade>,
    /// Mesh-emergent propagation paths (M4): jambs, building corners and
    /// roof lines are auto-extracted diffraction edges on the world mesh.
    auto: AutoPaths,
    paths_buf: Vec<omg_core::paths::FoundPath>,
    /// Room-coupling graph for the outdoor-field power balance.
    acoustics: AcousticsGraph,
    /// Ray-sampled ambient dome (the audio skybox).
    dome: DomeProbe,
    /// Occlusion-floor cache: floors vary smoothly in space, so per
    /// (exit, listener-cell) results are reused and refreshed round-robin
    /// — same philosophy as the trace gate (staleness bounded, no bias).
    bend_cache: std::collections::HashMap<(i32, i32), ([f32; NBANDS], (i32, i32))>,
    /// Per-source LOD: last produced block/route and its total level —
    /// inaudible sources refresh at 5 Hz instead of 20 (perceptually
    /// unimportant work goes first).
    last_blocks: Vec<ParamBlock>,
    last_routes: Vec<Vec<(f32, f32)>>,
    last_level: Vec<f32>,
    tick_no: u64,
}

/// Below this summed mid-band gain a source is treated as inaudible for
/// LOD purposes (well under the quietest thing the mix resolves).
const LOD_QUIET: f32 = 0.004;

/// Maps the dome's escaped-ray energy fraction to route amplitude,
/// calibrated so the open field lands at the loudness the old horizon
/// sectors had (Σ gains² ≈ 3 in the open).
const DOME_GAIN: f32 = 4.5;

impl WorldSim {
    pub fn new() -> Self {
        let rooms = walkthrough::rooms();
        let doors = walkthrough::doors();
        let defs: Vec<SourceDef> = walkthrough::sources().into();
        let dome = DomeProbe::new(&rooms, &doors);
        // generous edge budget: vertical building corners are short and lose
        // importance ranking to long roof lines, but they carry the alley
        // bends that keep shadows continuous
        let auto = AutoPaths::new(&dome.mesh, 512);
        Self {
            sims: defs
                .iter()
                .map(|_| (0..rooms.len()).map(|_| Sim::new()).collect())
                .collect(),
            sim_remotes: defs.iter().map(|_| Sim::new()).collect(),
            src_z: defs
                .iter()
                .map(|d| rooms[d.room].floor_z + walkthrough::SRC_HEIGHT)
                .collect(),
            dynamic_active: [false; walkthrough::DYN_SLOTS],
            facades: {
                let out = rooms.last().unwrap();
                let (ox, oy) = out.min;
                let mut f = Vec::new();
                for r in rooms.iter().filter(|r| !r.outdoor && r.floor_z + r.height > 0.2) {
                    let edges = [
                        (0usize, r.min.0, r.min.1, r.max.1, 0usize),
                        (0, r.max.0, r.min.1, r.max.1, 1),
                        (1, r.min.1, r.min.0, r.max.0, 2),
                        (1, r.max.1, r.min.0, r.max.0, 3),
                    ];
                    for (axis, plane, lo, hi, w) in edges {
                        f.push(Facade {
                            axis,
                            plane: plane - if axis == 0 { ox } else { oy },
                            lo: lo - if axis == 0 { oy } else { ox },
                            hi: hi - if axis == 0 { oy } else { ox },
                            refl: r.walls[w].reflection_amplitude(),
                        });
                    }
                }
                f
            },
            acoustics: AcousticsGraph::new(&rooms, &doors),
            dome,
            auto,
            paths_buf: Vec::new(),
            rooms,
            doors,
            bend_cache: std::collections::HashMap::new(),
            last_blocks: defs.iter().map(|_| ParamBlock::default()).collect(),
            last_routes: defs.iter().map(|_| Vec::new()).collect(),
            last_level: defs.iter().map(|_| f32::MAX).collect(),
            defs,
            tick_no: 0,
        }
    }

    /// Door panel openness 0 (closed) … 1 (open) — pass the ANIMATED leaf
    /// position each tick, so the swing sweeps every transmission filter
    /// continuously (glass panes ignore this).
    pub fn set_door(&mut self, i: usize, openness: f32) {
        if let Some(d) = self.doors.get_mut(i) {
            if !d.glass {
                d.openness = openness.clamp(0.0, 1.0);
            }
        }
    }

    /// Move dynamic source `slot` (0..DYN_SLOTS). Inactive → silent.
    pub fn set_dynamic(&mut self, slot: usize, x: f32, y: f32, z: f32, active: bool) {
        if slot >= walkthrough::DYN_SLOTS {
            return;
        }
        let i = self.defs.len() - walkthrough::DYN_SLOTS + slot;
        self.defs[i].pos = (x, y);
        self.defs[i].room = walkthrough::room_of_z(&self.rooms, x, y, z);
        self.src_z[i] = z.clamp(0.2, 12.0);
        self.dynamic_active[slot] = active;
    }

    /// Scripted-path tick (native walkthrough render).
    pub fn tick_scripted(&mut self, t: f32) -> (Vec<ParamBlock>, TickInfo) {
        let bs = blended_state_at(&self.rooms, t);
        self.tick_blended(bs)
    }

    /// Interactive tick (web build): explicit listener pose. `yaw` is the
    /// walk/course yaw baked into arrival directions; fast head rotation is
    /// applied downstream in the renderer.
    pub fn tick_at(&mut self, lx: f32, ly: f32, yaw: f32) -> (Vec<ParamBlock>, TickInfo) {
        self.tick_at_z(lx, ly, walkthrough::EYE_HEIGHT, yaw)
    }

    /// Interactive tick with explicit eye height (stacked storeys).
    pub fn tick_at_z(
        &mut self,
        lx: f32,
        ly: f32,
        lz: f32,
        yaw: f32,
    ) -> (Vec<ParamBlock>, TickInfo) {
        let bs = blended_state_for(&self.rooms, lx, ly, lz, yaw);
        self.tick_blended(bs)
    }

    fn tick_blended(&mut self, bs: BlendedState) -> (Vec<ParamBlock>, TickInfo) {
        self.tick_no += 1;
        let st = &bs.primary;
        let mut blocks = Vec::with_capacity(self.defs.len());
        let mut routes = Vec::with_capacity(self.defs.len());
        let mut rt60_mid = 0.0f32;

        // per-tick budget of fresh occlusion-floor evaluations; everything
        // beyond it reuses its last (spatially nearby) value one tick longer
        let mut bend_budget: i32 = 5;
        let n_static = self.defs.len() - walkthrough::DYN_SLOTS;
        for si in 0..self.defs.len() {
            let def = self.defs[si]; // Copy — frees `self` for &mut queries
            if si >= n_static && !self.dynamic_active[si - n_static] {
                let mut pb = ParamBlock::default();
                pb.version = self.tick_no;
                blocks.push(pb);
                routes.push(vec![]);
                continue;
            }
            // LOD: a source whose last block was inaudible refreshes at
            // 5 Hz, staggered; the engine's smoothers hold in between.
            if self.last_level[si] < LOD_QUIET && (self.tick_no + si as u64) % 4 != 0 {
                blocks.push(self.last_blocks[si].clone());
                routes.push(self.last_routes[si].clone());
                continue;
            }
            let src_z = self.src_z[si];

            // Bend floor (outdoors): for every point this source's energy
            // exits through (door-chain exit, each window, or the source
            // itself in the open), compute the best knife-edge path around
            // the blocking geometry. Occlusion below never drops under this
            // floor — losing sight of a radiator hands its energy (dry,
            // reflections AND coupled wet) to the bend instead of cutting
            // it, which is what makes walking behind a building continuous.
            let bend_floor: Vec<((f32, f32), [f32; NBANDS])> = {
                let mut m = Vec::new();
                if self.rooms[st.room].outdoor {
                    let mut pts: Vec<(f32, f32)> = Vec::new();
                    if def.room == st.room {
                        pts.push(def.pos);
                    } else {
                        pts.push(route_source(&def, st.room, st.listener_world, &self.doors).virt_world);
                        for ar in
                            aperture_routes(&def, st.room, st.listener_world, &self.doors)
                        {
                            pts.push(ar.virt_world);
                        }
                    }
                    for p in pts {
                        // Exits are EXTENDED apertures, not points: a
                        // point probe sees a cliff exactly where the
                        // pane's width physically smears it. Average the
                        // bent ENERGY across the opening's extent.
                        let probes: Vec<(f32, f32)> = match self.doors.iter().find(|d| {
                            let (dp, dl) = if d.axis == 0 {
                                (p.0 - d.pos.0, p.1 - d.pos.1)
                            } else {
                                (p.1 - d.pos.1, p.0 - d.pos.0)
                            };
                            dp.abs() < 0.1 && dl.abs() <= d.half + 0.1
                        }) {
                            Some(d) => {
                                let o = 0.7 * d.half;
                                if d.axis == 0 {
                                    vec![(d.pos.0, d.pos.1 - o), p, (d.pos.0, d.pos.1 + o)]
                                } else {
                                    vec![(d.pos.0 - o, d.pos.1), p, (d.pos.0 + o, d.pos.1)]
                                }
                            }
                            None => vec![p],
                        };
                        let key = ((p.0 * 50.0) as i32, (p.1 * 50.0) as i32);
                        let cell = (
                            (st.listener_world.0 * 4.0) as i32,
                            (st.listener_world.1 * 4.0) as i32,
                        );
                        let cached = self.bend_cache.get(&key).copied();
                        let fresh = match cached {
                            Some((f, c)) if c == cell || bend_budget <= 0 => {
                                let _ = f;
                                None
                            }
                            None if bend_budget <= 0 => Some([0.0; NBANDS]), // first sight: cheap guess, refined next ticks
                            _ => {
                                bend_budget -= 1;
                                let mut e = [0.0f32; NBANDS];
                                for q in &probes {
                                    let f = self.bend_factor(*q, st.listener_world, src_z);
                                    for b in 0..NBANDS {
                                        e[b] += f[b] * f[b];
                                    }
                                }
                                let n = probes.len() as f32;
                                Some(core::array::from_fn(|b| (e[b] / n).sqrt()))
                            }
                        };
                        let val = match (fresh, cached) {
                            (Some(f), _) => {
                                self.bend_cache.insert(key, (f, cell));
                                f
                            }
                            (None, Some((f, _))) => f,
                            (None, None) => [0.0; NBANDS],
                        };
                        m.push((p, val));
                    }
                }
                m
            };
            // nearest emitter within 1.5 m (bend points drift a little
            // between blend states; apertures are further apart than that)
            let floor_of = |p: (f32, f32)| -> [f32; NBANDS] {
                bend_floor
                    .iter()
                    .map(|(q, f)| {
                        ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2), f)
                    })
                    .filter(|(d2, _)| *d2 < 2.25)
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map_or([0.0; NBANDS], |(_, f)| *f)
            };

            let rooms = &self.rooms;
            let doors_l = &self.doors;
            let facades = &self.facades;
            let sims = &mut self.sims[si];

            // One (room-state, weight) simulation → weighted ParamBlock.
            let mut sim_in_room = |state: &walkthrough::WalkState,
                                   w: f32|
             -> (ParamBlock, Vec<(f32, f32)>) {
                let routed = route_source(&def, state.room, state.listener_world, doors_l);
                let margin = if routed.extra_dist > 0.0 { 0.06 } else { 0.3 };
                // Speaker rigs only at (near-)full weight: during a portal
                // blend both room states render at once, and the doubled
                // tap load is what makes throttled CPUs glitch at doorways.
                let rig_ok = w > 0.72;
                let virt = walkthrough::to_local_margin(
                    &rooms[state.room],
                    routed.virt_world.0,
                    routed.virt_world.1,
                    if routed.extra_dist > 0.0 { walkthrough::SRC_HEIGHT } else { src_z },
                    margin,
                );
                let sim = &mut sims[state.room];
                let mut pb = if state.room == def.room && def.emitters.len() > 1 && rig_ok {
                    // Speaker rig in this room: per-emitter image sources.
                    let ems: Vec<Vec3> = def
                        .emitters
                        .iter()
                        .map(|e| to_local(&rooms[state.room], e.0, e.1, src_z))
                        .collect();
                    sim.update_multi(&state.shoebox, &ems, state.listener_local, state.yaw)
                } else if rooms[state.room].outdoor {
                    sim.update_outdoor(
                        &Material::GRASS,
                        facades,
                        virt,
                        state.listener_local,
                        state.yaw,
                        routed.extra_dist,
                        routed.muffle,
                    )
                } else {
                    sim.update_routed(
                        &state.shoebox,
                        virt,
                        state.listener_local,
                        state.yaw,
                        routed.extra_dist,
                        routed.muffle,
                    )
                };
                // Door-routed sound radiates FROM the doorway: whatever
                // stands between the door and the listener (e.g. the
                // building itself, when you're behind it) occludes it via
                // the same straight-ray transmission rules. Additionally,
                // the bent path only carries what the straight ray does NOT:
                // with clear line of sight through the opening the straight
                // tap IS the direct sound and the routed copy vanishes — a
                // residual copy at sub-millisecond offset is a comb filter
                // (metallic), not scatter.
                let occ = if routed.extra_dist > 0.0 {
                    let door_occ = straight_path_transmission(
                        rooms,
                        doors_l,
                        routed.virt_world,
                        state.listener_world,
                    );
                    let los = straight_path_transmission(
                        rooms,
                        doors_l,
                        def.pos,
                        state.listener_world,
                    );
                    let fl = floor_of(routed.virt_world);
                    core::array::from_fn(|b| {
                        door_occ[b].max(fl[b]) * (1.0 - los[b])
                    })
                } else if rooms[state.room].outdoor {
                    // Same outdoor "room" — but buildings may stand between
                    // two open-air points (this is what occludes a thrown
                    // whistling projectile behind a wall). The bend floor
                    // keeps the whistle bending around the corner instead
                    // of vanishing.
                    let t = straight_path_transmission(
                        rooms, doors_l, def.pos, state.listener_world);
                    let fl = floor_of(def.pos);
                    core::array::from_fn(|b| t[b].max(fl[b]))
                } else {
                    [1.0; NBANDS]
                };
                // Weight amplitudes; key-space per room so blended states coexist.
                for tp in &mut pb.taps {
                    tp.key += state.room as u32 * 1024;
                    for b in 0..NBANDS {
                        tp.gains[b] *= w * occ[b];
                    }
                }
                (pb, routed.route)
            };

            let (mut pb, route) = sim_in_room(st, bs.primary_weight);
            let wp2 = bs.primary_weight * bs.primary_weight;
            let mut reverb = pb.reverb;
            for b in 0..NBANDS {
                reverb.rt60[b] *= wp2;
                reverb.level[b] *= wp2;
            }
            if let Some((ref ost, wo)) = bs.other {
                let (opb, _) = sim_in_room(ost, wo);
                let wo2 = wo * wo;
                for b in 0..NBANDS {
                    reverb.rt60[b] += opb.reverb.rt60[b] * wo2;
                    reverb.level[b] += opb.reverb.level[b] * wo2;
                }
                pb.taps.extend_from_slice(&opb.taps);
            }
            pb.reverb = reverb;

            // Straight-ray propagation: one segment source → listener, each
            // wall crossed attenuates by mass-law transmission × thickness,
            // door apertures pass freely. With clear line of sight through
            // open doors this IS the unmuffled direct sound the portal
            // model used to fake; through walls it is the bass rumble.
            if def.room != st.room {
                let trans = straight_path_transmission(
                    &self.rooms,
                    &self.doors,
                    def.pos,
                    st.listener_world,
                );
                let (dxw, dyw) = (
                    def.pos.0 - st.listener_world.0,
                    def.pos.1 - st.listener_world.1,
                );
                let d = (dxw * dxw + dyw * dyw).sqrt().max(0.3);
                let air = air_attenuation(d);
                let gains: [f32; NBANDS] =
                    core::array::from_fn(|b| trans[b] * air[b] / d);
                if gains[0] > 2e-5 {
                    let (sin, cos) = st.yaw.sin_cos();
                    let (nx, ny) = (dxw / d, dyw / d);
                    pb.taps.push(Tap {
                        key: 9000,
                        delay_s: d / SPEED_OF_SOUND,
                        dir: [cos * nx + sin * ny, -sin * nx + cos * ny, 0.0],
                        gains,
                    });
                }
            }

            // Coupled-room wet: the source room's reverberant field, heard
            // through the doorway. Weighted per blend state.
            let mut states: Vec<(&walkthrough::WalkState, f32)> = vec![(st, bs.primary_weight)];
            if let Some((ref ost, wo)) = bs.other {
                states.push((ost, wo));
            }
            let mut rem_send = [0.0f32; NBANDS];
            let mut rem_sh = [0.0f32; omg_dsp::ambi::NCH];
            let mut rem_rt60 = [0.0f32; NBANDS];
            let mut rem_n = 0usize;
            let mut rp_cache: Option<omg_core::params::ReverbParams> = None;
            for (state, w) in states.drain(..) {
                if state.room == def.room || self.rooms[def.room].outdoor {
                    continue;
                }
                // Radiators: the walk chain's entry aperture + every door
                // and WINDOW directly connecting the two rooms. Each one
                // re-radiates the source room's field omnidirectionally —
                // a window is audible off-axis, not only on the sight-line.
                let chain = route_source(&def, state.room, state.listener_world, &self.doors);
                let mut radiators: Vec<(walkthrough::Routed, bool)> = vec![(chain, true)];
                for ar in aperture_routes(&def, state.room, state.listener_world, &self.doors) {
                    let d0 = &radiators[0].0.virt_world;
                    if (ar.virt_world.0 - d0.0).abs() + (ar.virt_world.1 - d0.1).abs() > 0.1 {
                        radiators.push((ar, false));
                    }
                }
                for (ai, (routed, is_chain)) in radiators.iter().enumerate() {
                    let routed = &*routed;
                    let is_chain = *is_chain;
                let rp = *rp_cache.get_or_insert_with(|| {
                    let sr = &self.rooms[def.room];
                    let sbox = Shoebox::new(
                        Vec3::new(sr.max.0 - sr.min.0, sr.max.1 - sr.min.1, sr.height),
                        sr.walls,
                    );
                    let src_l = to_local(sr, def.pos.0, def.pos.1, walkthrough::SRC_HEIGHT);
                    let exit = routed.route[1];
                    let door_l = to_local(sr, exit.0, exit.1, walkthrough::SRC_HEIGHT);
                    self.sim_remotes[si].reverb_only(&sbox, src_l, door_l)
                });
                let mut d_after = 0.0f32;
                for seg in routed.route[1..].windows(2) {
                    d_after +=
                        ((seg[0].0 - seg[1].0).powi(2) + (seg[0].1 - seg[1].1).powi(2)).sqrt();
                }
                let w2 = w * w;
                // The wet field exits the OPEN first door unattenuated;
                // only additional doors muffle it. The ~2 m² aperture is a
                // near-field radiator: flat to ~sqrt(A), 1/d beyond.
                // The wet field exits the OPEN first door unattenuated; any
                // further doors on the chain bend it and pay knife-edge.
                let chain_bend = walkthrough::chain_bend_muffle(&routed.route[1..]);
                let aperture = 1.6 / d_after.max(1.6);
                let occ: [f32; NBANDS] = {
                    let t = straight_path_transmission(
                        &self.rooms,
                        &self.doors,
                        routed.virt_world,
                        state.listener_world,
                    );
                    let fl = floor_of(routed.virt_world);
                    core::array::from_fn(|b| t[b].max(fl[b]))
                };
                for b in 0..NBANDS {
                    // panes/panels always pay their flat transmission; open
                    // chains pay the bending of doors past the first
                    let ap_factor = if routed.glass {
                        walkthrough::GLASS_TRANSMISSION[b]
                    } else {
                        routed.wet_trans[b] * chain_bend[b]
                    };
                    rem_send[b] += w2 * rp.level[b] * ap_factor * aperture * occ[b];
                    rem_rt60[b] = rp.rt60[b];
                }
                let (dxw, dyw) = (
                    routed.virt_world.0 - state.listener_world.0,
                    routed.virt_world.1 - state.listener_world.1,
                );
                let dlen = (dxw * dxw + dyw * dyw).sqrt().max(1e-3);
                let (sin, cos) = state.yaw.sin_cos();
                let (dx, dy) = ((cos * dxw + sin * dyw) / dlen, (-sin * dxw + cos * dyw) / dlen);
                let enc = omg_dsp::ambi::encode_gains([dx, dy, 0.0]);
                for k in 0..omg_dsp::ambi::NCH {
                    rem_sh[k] += enc[k];
                }
                rem_n += 1;

                // Wall-transmitted wet: the source room's diffuse field
                // leaking through its walls (mass law) — the club "oomp"
                // from outside is reverberant bass, not dry signal.
                let sr = &self.rooms[def.room];
                let wx = state.listener_world.0.clamp(sr.min.0, sr.max.0);
                let wy = state.listener_world.1.clamp(sr.min.1, sr.max.1);
                let (tdx, tdy) = (wx - state.listener_world.0, wy - state.listener_world.1);
                let dwall = (tdx * tdx + tdy * tdy).sqrt().max(0.5);
                let wall_trans = sr.walls[0].transmission_at(sr.wall_thickness);
                for b in 0..NBANDS {
                    rem_send[b] += w2 * rp.level[b] * wall_trans[b] * (1.6 / dwall.max(1.6));
                }
                let wdl = (tdx * tdx + tdy * tdy).sqrt().max(1e-3);
                let (wdx, wdy) =
                    ((cos * tdx + sin * tdy) / wdl, (-sin * tdx + cos * tdy) / wdl);
                let wenc = omg_dsp::ambi::encode_gains([wdx, wdy, 0.0]);
                for k in 0..omg_dsp::ambi::NCH {
                    rem_sh[k] += wenc[k];
                }
                rem_n += 1;

                // Early reflections of the SOURCE's room, carried through
                // the doorway. Each image path exits the aperture at its
                // OWN crossing point — a ceiling bounce leaves higher and
                // elsewhere along the width than a grazing path. Collapsing
                // them all onto the door's center point makes a stack of
                // coherent copies at millisecond spacings from one
                // direction: the "metal box" coloration outside a lively
                // room. This is the image-source construction continued
                // through the opening.
                let sr2 = &self.rooms[def.room];
                let sbox2 = Shoebox::new(
                    Vec3::new(sr2.max.0 - sr2.min.0, sr2.max.1 - sr2.min.1, sr2.height),
                    sr2.walls,
                );
                let src_l2 = to_local(sr2, def.pos.0, def.pos.1, walkthrough::SRC_HEIGHT);
                let exit2 = routed.route[1];
                let door_l2 = to_local(sr2, exit2.0, exit2.1, walkthrough::SRC_HEIGHT);
                let mut refl = Vec::new();
                image_source_taps(&sbox2, src_l2, door_l2, 2, &mut refl);
                if let Some(di) = refl
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.delay_s.total_cmp(&b.1.delay_s))
                    .map(|(i, _)| i)
                {
                    refl.remove(di); // direct leg is already the routed path
                }
                refl.sort_by(|a, b| b.gains[1].total_cmp(&a.gains[1]));
                refl.truncate(6);
                // the aperture this radiator exits through (its true extents)
                let ap = self.doors.iter().find(|dd| {
                    let (dp, dl) = if dd.axis == 0 {
                        (exit2.0 - dd.pos.0, exit2.1 - dd.pos.1)
                    } else {
                        (exit2.1 - dd.pos.1, exit2.0 - dd.pos.0)
                    };
                    dp.abs() < 0.1 && dl.abs() <= dd.half + 0.1
                });
                let lis3 = [
                    state.listener_world.0,
                    state.listener_world.1,
                    self.rooms[state.room].floor_z + state.listener_local.z,
                ];
                let dist3 = |a: [f32; 3], b: [f32; 3]| {
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                        .sqrt()
                };
                for (ri, tref) in refl.iter().enumerate() {
                    let d_tap = tref.delay_s * omg_core::SPEED_OF_SOUND;
                    // image position (dir points from the door toward it)
                    let img = [
                        exit2.0 + tref.dir[0] * d_tap,
                        exit2.1 + tref.dir[1] * d_tap,
                        sr2.floor_z + walkthrough::SRC_HEIGHT + tref.dir[2] * d_tap,
                    ];
                    // where the image→listener line crosses the opening
                    let e3 = if let Some(dd) = ap {
                        let (plane, ia, la, ic, lc) = if dd.axis == 0 {
                            (dd.pos.0, img[0], lis3[0], img[1], lis3[1])
                        } else {
                            (dd.pos.1, img[1], lis3[1], img[0], lis3[0])
                        };
                        let denom = la - ia;
                        let t = if denom.abs() < 1e-6 {
                            0.5
                        } else {
                            ((plane - ia) / denom).clamp(0.0, 1.0)
                        };
                        let cc = if dd.axis == 0 { dd.pos.1 } else { dd.pos.0 };
                        let cl = (ic + t * (lc - ic)).clamp(cc - dd.half + 0.05, cc + dd.half - 0.05);
                        let cz = (img[2] + t * (lis3[2] - img[2])).clamp(
                            dd.zc - 0.5 * dd.height + 0.05,
                            dd.zc + 0.5 * dd.height - 0.05,
                        );
                        if dd.axis == 0 { [plane, cl, cz] } else { [cl, plane, cz] }
                    } else {
                        [exit2.0, exit2.1, walkthrough::SRC_HEIGHT]
                    };
                    let pre = dist3(img, e3).max(0.3);
                    let post = dist3(e3, lis3).max(0.3);
                    let scale = pre / (pre + post);
                    let gains: [f32; NBANDS] = core::array::from_fn(|b| {
                        tref.gains[b] * routed.muffle[b] * scale * occ[b] * w
                    });
                    // arrival direction: from the listener TOWARD the exit
                    let (ex, ey, ez) =
                        ((e3[0] - lis3[0]) / post, (e3[1] - lis3[1]) / post, (e3[2] - lis3[2]) / post);
                    pb.taps.push(Tap {
                        key: 7000 + ai as u32 * 128 + state.room as u32 * 16 + ri as u32,
                        delay_s: (pre + post) / omg_core::SPEED_OF_SOUND,
                        dir: [cos * ex + sin * ey, -sin * ex + cos * ey, ez],
                        gains,
                    });
                }

                // Dry direct through this aperture (the chain's version is
                // already produced by the listener-room simulation). Scaled
                // by the complement of the straight ray to avoid counting
                // the on-axis path twice.
                if !is_chain {
                    let los = straight_path_transmission(
                        &self.rooms,
                        &self.doors,
                        def.pos,
                        state.listener_world,
                    );
                    let total = routed.extra_dist + d_after;
                    let air = air_attenuation(total);
                    let gains: [f32; NBANDS] = core::array::from_fn(|b| {
                        routed.muffle[b] * air[b] / total.max(0.3)
                            * occ[b]
                            * (1.0 - los[b])
                            * w
                    });
                    if gains[0] > 2e-5 {
                        pb.taps.push(Tap {
                            key: 7800 + ai as u32 * 8 + state.room as u32,
                            delay_s: total / omg_core::SPEED_OF_SOUND,
                            dir: [dx, dy, 0.0],
                            gains,
                        });
                    }
                }
                }
            }
            if rem_n > 0 {
                for k in 0..omg_dsp::ambi::NCH {
                    rem_sh[k] /= rem_n as f32;
                }
                pb.remote = Some(RemoteReverb { rt60: rem_rt60, send: rem_send, sh: rem_sh });
            }

            // Per-source gain: scales dry, reflections and both wet paths.
            let g = def.gain;
            for t in &mut pb.taps {
                for b in 0..NBANDS {
                    t.gains[b] *= g;
                }
            }
            for b in 0..NBANDS {
                pb.reverb.level[b] *= g;
            }
            if let Some(r) = &mut pb.remote {
                for b in 0..NBANDS {
                    r.send[b] *= g;
                }
            }

            pb.version = self.tick_no;
            rt60_mid = pb.reverb.rt60[1];
            self.last_level[si] = pb.taps.iter().map(|t| t.gains[1]).sum::<f32>()
                + pb.remote.as_ref().map_or(0.0, |r| r.send[1]);
            self.last_blocks[si] = pb.clone();
            self.last_routes[si] = route.clone();
            blocks.push(pb);
            routes.push(route);
        }

        let info = TickInfo {
            listener: st.listener_world,
            yaw: st.yaw,
            room: st.room,
            rt60_mid,
            routes,
            env: self.environment(&bs),
        };
        (blocks, info)
    }

    /// Environment routing for a pose. The directional inflow is the
    /// ambient DOME sampled by rays against the real geometry: apertures,
    /// swinging leaves, panes and multi-room hops all emerge from ray
    /// paths, continuously in position — no horizon hand-sampling, no
    /// aperture bookkeeping, no blend anchoring. The power balance keeps
    /// what rays cannot carry: the through-shell seep; enclosure and roof
    /// exposure blend across the doorway states.
    fn environment(&mut self, bs: &BlendedState) -> Environment {
        let field = self.acoustics.outdoor_field(&self.doors);
        let mut states: Vec<(&walkthrough::WalkState, f32)> =
            vec![(&bs.primary, bs.primary_weight)];
        if let Some((ref o, w)) = bs.other {
            states.push((o, w));
        }

        let mut seep_e = [0.0f32; NBANDS];
        let (mut enclosure, mut roof) = (0.0f32, 0.0f32);
        let mut windows: Vec<EnvWindow> = Vec::new();
        for (state, w) in states {
            let w2 = w * w;
            if self.rooms[state.room].outdoor {
                continue;
            }
            let (lx, ly) = state.listener_world;
            let lz = self.rooms[state.room].floor_z + state.listener_local.z;
            for b in 0..NBANDS {
                seep_e[b] += w2 * field[state.room][b];
            }
            // Enclosure is GEOMETRIC (a shell stands between you and the
            // weather), not reverberant; blend zones make it continuous.
            enclosure += w2;
            roof += w2 * self.acoustics.roof_sky[state.room];
            // glass panes of this room: rain drops anchor ON them
            for d in self.doors.iter().filter(|d| d.glass) {
                if d.rooms.0 != state.room && d.rooms.1 != state.room {
                    continue;
                }
                if windows.len() >= MAX_ENV_WINDOWS {
                    break;
                }
                let (dx, dy, dz) = (d.pos.0 - lx, d.pos.1 - ly, d.zc - lz);
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.3);
                windows.push(EnvWindow {
                    dir: [dx / dist, dy / dist, dz / dist],
                    gain: w * 3.0 / (3.0 + dist),
                });
            }
        }

        let leaves = door_panels(&self.doors);
        let eye = Vec3::new(bs.raw.0, bs.raw.1, bs.raw.2);
        let bins = self.dome.sample(eye, &leaves);
        let mut routes: Vec<EnvRoute> = bins
            .iter()
            .enumerate()
            .filter(|(_, b)| b.energy[1] * DOME_GAIN > 1e-8)
            .map(|(k, b)| EnvRoute {
                id: DOME_ID_BASE + k as u32,
                dir: b.dir,
                gains: core::array::from_fn(|band| (b.energy[band] * DOME_GAIN).sqrt()),
                dist: 10.0,
            })
            .collect();
        routes.sort_by(|a, b| b.gains[1].total_cmp(&a.gains[1]));
        routes.truncate(MAX_ENV_ROUTES);
        Environment {
            seep: core::array::from_fn(|b| seep_e[b].sqrt()),
            enclosure,
            roof_gain: roof,
            routes,
            windows,
        }
    }
}

impl WorldSim {
    /// Best per-band factor a bent path around the geometry can deliver
    /// for an emitter at `e`, RELATIVE to its (blocked) straight leg to
    /// the listener — knife-edge losses × the extra spreading of the
    /// longer path; zero when the leg is effectively clear. Occlusion
    /// floors at this. M4: mesh-emergent — the hand-built corner list,
    /// corner-visibility matrix and multi-roof rubber band dissolved into
    /// AutoPaths over the world mesh, where a door jamb, a building
    /// corner and a roof line are the same thing: an auto-extracted edge.
    pub fn bend_factor(&mut self, e: (f32, f32), lis: (f32, f32), src_z: f32) -> [f32; NBANDS] {
        let ez = 0.5 * (src_z + walkthrough::EYE_HEIGHT);
        let a = Vec3::new(e.0, e.1, ez);
        let b = Vec3::new(lis.0, lis.1, walkthrough::EYE_HEIGHT);
        let (auto, mesh, buf) = (&mut self.auto, &self.dome.mesh, &mut self.paths_buf);
        // the occlusion floor is what keeps shadows continuous — worth a
        // deeper search than the default (only blocked exits pay it)
        let budget = PathBudget { edge_candidates: 48, pair_edges: 32, max_paths: 4 };
        auto.find(mesh, a, b, budget, buf);
        // Max over ALL paths, the mesh direct included: occlusion takes
        // max(2D straight, this), and letting the two direct estimators
        // disagree (a grazing segment reads clear in 3D, blocked in 2D)
        // once blacked out a one-meter stripe along a facade.
        let d_direct = (b - a).length().max(0.5);
        let mut best = [0.0f32; NBANDS];
        for p in buf.iter() {
            for band in 0..NBANDS {
                best[band] =
                    best[band].max(p.gains[band] * d_direct / p.length.max(d_direct));
            }
        }
        best
    }
}

impl Default for WorldSim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mid_sum_at(w: &mut WorldSim, x: f32, y: f32, src: usize) -> f32 {
        // settle the EMA a little, then read
        for _ in 0..4 {
            let _ = w.tick_at(x, y, 0.0);
        }
        let (blocks, _) = w.tick_at(x, y, 0.0);
        blocks[src].taps.iter().map(|t| t.gains[1]).sum()
    }

    /// The user-facing property: walking behind a building must never cut
    /// a source off — occlusion floors at the best knife-edge bend, so
    /// levels stay continuous across shadow boundaries (the knife edge is
    /// −5 dB at the boundary; allow a little more for the geometry step).
    #[test]
    fn shadow_boundaries_are_continuous() {
        let mut w = WorldSim::new();
        let mut prev = f32::NAN;
        // east-west walk at y = 12: crosses the Old House shadow of the
        // club's windows twice (entering and leaving)
        let mut x = 16.0;
        while x <= 40.0 {
            let cur = mid_sum_at(&mut w, x, 12.0, 2);
            assert!(cur > 1e-4, "club inaudible at ({x}, 12)");
            if prev.is_finite() {
                let ratio = (cur / prev).max(prev / cur);
                assert!(
                    ratio < 3.2,
                    "level jump {:.1} dB between x={} and x={}",
                    20.0 * ratio.log10(),
                    x - 1.0,
                    x
                );
            }
            prev = cur;
            x += 1.0;
        }
    }

    /// Closing a door turns the portal into a wood panel: markedly
    /// quieter, bass-favoring (mass law), and fully reversible.
    #[test]
    fn closing_a_door_muffles_the_portal() {
        let mut w = WorldSim::new();
        // listener mid-corridor, music playing in the living room
        let read = |w: &mut WorldSim| -> ([f32; NBANDS], f32) {
            for _ in 0..4 {
                let _ = w.tick_at(4.0, 9.0, 0.0);
            }
            let (blocks, _) = w.tick_at(4.0, 9.0, 0.0);
            let mut sum = [0.0f32; NBANDS];
            for t in &blocks[0].taps {
                for b in 0..NBANDS {
                    sum[b] += t.gains[b];
                }
            }
            (sum, sum[1])
        };
        let (_, open_mid) = read(&mut w);
        w.set_door(0, 0.0); // Living ↔ Corridor
        let (closed, closed_mid) = read(&mut w);
        assert!(
            closed_mid < 0.45 * open_mid,
            "closed door should muffle: {closed_mid} vs {open_mid}"
        );
        assert!(
            closed[0] > 2.0 * closed[2],
            "panel transmission must favor lows: {closed:?}"
        );
        w.set_door(0, 1.0);
        let (_, reopened) = read(&mut w);
        assert!(reopened > 0.75 * open_mid, "reopening should restore: {reopened} vs {open_mid}");
    }

    /// The complaint this design answers: walking from outdoors through
    /// the hall door must never step the ambient level — the outdoor
    /// field hands over from horizon routes to aperture routes + seep
    /// through the blend zone, continuously.
    #[test]
    fn ambient_field_is_continuous_through_a_doorway() {
        let mut w = WorldSim::new();
        // straight walk through the Hall ↔ Outside door at (7, 24)
        let mut prev = f32::NAN;
        let mut y = 27.0;
        while y >= 20.0 {
            let (_, info) = w.tick_at(7.0, y, 0.0);
            let e = &info.env;
            // total received ambient energy: diffuse seep + every route
            let total: f32 = e.seep[1] * e.seep[1]
                + e.routes.iter().map(|r| r.gains[1] * r.gains[1]).sum::<f32>();
            assert!(total > 1e-6, "ambient field died at y={y}");
            if prev.is_finite() {
                let ratio = (total / prev).max(prev / total);
                assert!(
                    ratio < 2.0,
                    "ambient energy jump {:.1} dB between y={} and y={}",
                    10.0 * ratio.log10(),
                    y + 0.25,
                    y
                );
            }
            prev = total;
            y -= 0.25;
        }
        // and enclosure must ramp, not snap
        let (_, out_info) = w.tick_at(7.0, 27.0, 0.0);
        let (_, in_info) = w.tick_at(7.0, 20.0, 0.0);
        let (_, mid_info) = w.tick_at(7.0, 24.0, 0.0);
        assert!(out_info.env.enclosure < 0.1);
        assert!(in_info.env.enclosure > 0.5);
        assert!(
            mid_info.env.enclosure > 0.1 && mid_info.env.enclosure < 0.9,
            "doorway enclosure should sit between: {}",
            mid_info.env.enclosure
        );
    }

    /// Outside an open door, the source room's early reflections must
    /// exit across the WHOLE opening — distinct directions (including
    /// elevation) and distinct path lengths. Collapsed onto the door's
    /// center they are a stack of coherent copies at millisecond
    /// spacings: the "metal box" coloration this test pins down.
    #[test]
    fn aperture_reflections_spread_across_the_opening() {
        let mut w = WorldSim::new();
        // voice is in the Great Hall; stand outside its door at (7, 24)
        for _ in 0..4 {
            let _ = w.tick_at(7.0, 26.0, 0.0);
        }
        let (blocks, _) = w.tick_at(7.0, 26.0, 0.0);
        let refl: Vec<&Tap> =
            blocks[1].taps.iter().filter(|t| (7000..7800).contains(&t.key)).collect();
        assert!(refl.len() >= 3, "expected doorway reflections, got {}", refl.len());
        let mut min_dot = 1.0f32;
        for i in 0..refl.len() {
            for j in i + 1..refl.len() {
                let (a, b) = (refl[i].dir, refl[j].dir);
                min_dot = min_dot.min(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
            }
        }
        assert!(min_dot < 0.995, "reflections leave from one point: min pairwise dot {min_dot}");
        assert!(
            refl.iter().any(|t| t.dir[2].abs() > 0.05),
            "no vertical spread — exit heights are not being used"
        );
        // and they must ARRIVE from the door (south of the listener),
        // not from its mirror image — an inverted direction once slipped
        // through because only the spread was pinned
        for t in &refl {
            assert!(t.dir[1] < -0.3, "reflection arrives from the wrong side: {:?}", t.dir);
        }
        let (dmin, dmax) = refl.iter().fold((f32::MAX, 0.0f32), |(lo, hi), t| {
            (lo.min(t.delay_s), hi.max(t.delay_s))
        });
        assert!(dmax - dmin > 0.002, "delays too uniform: {:.4}..{:.4}", dmin, dmax);
    }

    /// Deep shadow behind the Old House: the surviving energy must be
    /// bass-heavy — the bend floor is knife-edge shaped, not flat.
    #[test]
    fn deep_shadow_favors_lows() {
        let mut w = WorldSim::new();
        for _ in 0..4 {
            let _ = w.tick_at(27.5, 12.0, 0.0);
        }
        let (blocks, _) = w.tick_at(27.5, 12.0, 0.0);
        let mut low = 0.0f32;
        let mut high = 0.0f32;
        for t in &blocks[2].taps {
            low += t.gains[0];
            high += t.gains[2];
        }
        assert!(low > 2.0 * high, "shadow spectrum not bass-heavy: low {low} high {high}");
    }
}
