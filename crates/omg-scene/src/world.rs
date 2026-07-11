//! WorldSim: one tick of the full demo-world simulation for all sources —
//! portal routing, room blending, coupled-room reverb — producing one
//! ParamBlock per source plus viz/telemetry info. Drives both the native
//! scripted walkthrough and the interactive web build.

use crate::sim::{Facade, Sim};
use crate::walkthrough::{
    self, aperture_routes, blended_state_at, blended_state_for, route_source,
    straight_path_transmission, to_local, BlendedState, Door, RoomDef, SourceDef,
};
use omg_core::ism::image_source_taps;
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
    /// Building corners (world coords, slightly outset): diffraction nodes
    /// for around-the-corner bending outdoors.
    corners: Vec<(f32, f32)>,
    /// Corner↔corner clearness (flattened n×n), precomputed once — the
    /// middle leg of double-corner diffraction paths.
    corner_vis: Vec<bool>,
    tick_no: u64,
}

impl WorldSim {
    pub fn new() -> Self {
        let rooms = walkthrough::rooms();
        let doors = walkthrough::doors();
        let defs: Vec<SourceDef> = walkthrough::sources().into();
        let corners: Vec<(f32, f32)> = rooms
            .iter()
            .filter(|r| !r.outdoor)
            .flat_map(|r| {
                let o = 0.15;
                [
                    (r.min.0 - o, r.min.1 - o),
                    (r.max.0 + o, r.min.1 - o),
                    (r.min.0 - o, r.max.1 + o),
                    (r.max.0 + o, r.max.1 + o),
                ]
            })
            .collect();
        let corner_vis: Vec<bool> = (0..corners.len() * corners.len())
            .map(|k| {
                let (i, j) = (k / corners.len(), k % corners.len());
                i != j
                    && straight_path_transmission(&rooms, &doors, corners[i], corners[j])[1]
                        >= 0.7
            })
            .collect();
        Self {
            sims: defs
                .iter()
                .map(|_| (0..rooms.len()).map(|_| Sim::new()).collect())
                .collect(),
            sim_remotes: defs.iter().map(|_| Sim::new()).collect(),
            src_z: defs.iter().map(|_| walkthrough::SRC_HEIGHT).collect(),
            dynamic_active: [false; walkthrough::DYN_SLOTS],
            facades: {
                let out = rooms.last().unwrap();
                let (ox, oy) = out.min;
                let mut f = Vec::new();
                for r in rooms.iter().filter(|r| !r.outdoor) {
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
            corners,
            corner_vis,
            rooms,
            doors,
            defs,
            tick_no: 0,
        }
    }

    /// Open/close a door (indices into the scene door list; glass panes
    /// ignore this). Closed doors keep routing but pay panel transmission.
    pub fn set_door(&mut self, i: usize, open: bool) {
        if let Some(d) = self.doors.get_mut(i) {
            if !d.glass && d.open != open {
                d.open = open;
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
        self.defs[i].room = walkthrough::room_of(&self.rooms, x, y);
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
        let bs = blended_state_for(&self.rooms, lx, ly, yaw);
        self.tick_blended(bs)
    }

    fn tick_blended(&mut self, bs: BlendedState) -> (Vec<ParamBlock>, TickInfo) {
        self.tick_no += 1;
        let st = &bs.primary;
        let mut blocks = Vec::with_capacity(self.defs.len());
        let mut routes = Vec::with_capacity(self.defs.len());
        let mut rt60_mid = 0.0f32;

        let n_static = self.defs.len() - walkthrough::DYN_SLOTS;
        for (si, def) in self.defs.iter().enumerate() {
            if si >= n_static && !self.dynamic_active[si - n_static] {
                let mut pb = ParamBlock::default();
                pb.version = self.tick_no;
                blocks.push(pb);
                routes.push(vec![]);
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
                        pts.push(route_source(def, st.room, st.listener_world, &self.doors).virt_world);
                        for ar in
                            aperture_routes(def, st.room, st.listener_world, &self.doors)
                        {
                            pts.push(ar.virt_world);
                        }
                    }
                    for p in pts {
                        m.push((p, self.bend_factor(p, st.listener_world, src_z)));
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
                let routed = route_source(def, state.room, state.listener_world, doors_l);
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
                // tap IS the direct sound, and the routed copy shrinks to
                // door-frame scatter (avoids double-counting, which made
                // doorway sight-lines louder than being in the room).
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
                        door_occ[b].max(fl[b]) * (1.0 - 0.7 * los[b])
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
                let chain = route_source(def, state.room, state.listener_world, &self.doors);
                let mut radiators: Vec<(walkthrough::Routed, bool)> = vec![(chain, true)];
                for ar in aperture_routes(def, state.room, state.listener_world, &self.doors) {
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
                // the doorway: image sources toward the exit door, then the
                // door→listener leg (spreading, muffle, occlusion), arriving
                // from the doorway direction. This is what makes a room
                // sound like itself through its open door.
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
                for (ri, tref) in refl.iter().enumerate() {
                    let d_tap = tref.delay_s * omg_core::SPEED_OF_SOUND;
                    let scale = d_tap / (d_tap + d_after);
                    let gains: [f32; NBANDS] = core::array::from_fn(|b| {
                        tref.gains[b] * routed.muffle[b] * scale * occ[b] * w
                    });
                    pb.taps.push(Tap {
                        key: 7000 + ai as u32 * 128 + state.room as u32 * 16 + ri as u32,
                        delay_s: tref.delay_s + d_after / omg_core::SPEED_OF_SOUND,
                        dir: [dx, dy, 0.0],
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
                            * (1.0 - 0.7 * los[b])
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
            blocks.push(pb);
            routes.push(route);
        }

        let info = TickInfo {
            listener: st.listener_world,
            yaw: st.yaw,
            room: st.room,
            rt60_mid,
            routes,
        };
        (blocks, info)
    }
}

impl WorldSim {
    /// Best per-band factor a bent path around the geometry can deliver
    /// for an emitter at `e`, RELATIVE to its (blocked) straight leg to
    /// the listener: knife-edge losses × the extra spreading of the longer
    /// path. Zero when the leg is effectively clear (no bend needed) or
    /// nothing bends usefully. Occlusion floors at this — losing sight of
    /// an emitter hands its energy to the bend instead of cutting it.
    pub fn bend_factor(&self, e: (f32, f32), lis: (f32, f32), src_z: f32) -> [f32; NBANDS] {
        let straight = straight_path_transmission(&self.rooms, &self.doors, e, lis);
        if straight[1] >= 0.5 {
            return [0.0; NBANDS];
        }
        let ez = 0.5 * (src_z + walkthrough::EYE_HEIGHT);
        let e3 = [e.0, e.1, ez];
        let lst3 = [lis.0, lis.1, walkthrough::EYE_HEIGHT];
        let d_direct = {
            let (dx, dy) = (lis.0 - e.0, lis.1 - e.1);
            (dx * dx + dy * dy).sqrt().max(0.5)
        };
        let nc = self.corners.len();
        let c3 = |ci: usize| {
            let c = self.corners[ci];
            [c.0, c.1, ez]
        };
        let mut s_clear = vec![false; nc];
        let mut l_clear = vec![false; nc];
        for (ci, c) in self.corners.iter().enumerate() {
            s_clear[ci] =
                straight_path_transmission(&self.rooms, &self.doors, e, *c)[1] >= 0.7;
            l_clear[ci] =
                straight_path_transmission(&self.rooms, &self.doors, *c, lis)[1] >= 0.7;
        }
        let mut best = [0.0f32; NBANDS];
        let mut consider = |path: &[[f32; 3]]| {
            let (len, ke) = bent_path_gains(path);
            for b in 0..NBANDS {
                best[b] = best[b].max(ke[b] * d_direct / len.max(d_direct));
            }
        };
        for ci in 0..nc {
            if s_clear[ci] && l_clear[ci] {
                consider(&[e3, c3(ci), lst3]);
            }
        }
        for ci in 0..nc {
            if !s_clear[ci] {
                continue;
            }
            for cj in 0..nc {
                if cj != ci && l_clear[cj] && self.corner_vis[ci * nc + cj] {
                    consider(&[e3, c3(ci), c3(cj), lst3]);
                }
            }
        }
        for r in self.rooms.iter().filter(|r| !r.outdoor) {
            if let Some((pin, pout)) = segment_rect_crossing(e, lis, r.min, r.max) {
                let h = r.barrier_height;
                consider(&[e3, [pin.0, pin.1, h], [pout.0, pout.1, h], lst3]);
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

/// Total length and per-band knife-edge amplitude of a bent path: each
/// interior vertex diffracts with the local detour vs. the straight line
/// between its neighbors (the "rubber band" construction — standard
/// multi-edge barrier practice).
fn bent_path_gains(path: &[[f32; 3]]) -> (f32, [f32; NBANDS]) {
    let d = |a: [f32; 3], b: [f32; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let mut total = 0.0;
    for w in path.windows(2) {
        total += d(w[0], w[1]);
    }
    let mut g = [1.0f32; NBANDS];
    for i in 1..path.len() - 1 {
        let detour = d(path[i - 1], path[i]) + d(path[i], path[i + 1]) - d(path[i - 1], path[i + 1]);
        let ke = omg_core::diffraction::knife_edge_bands(detour);
        for b in 0..NBANDS {
            g[b] *= ke[b];
        }
    }
    (total, g)
}

/// Entry and exit points where segment a→b crosses rect [min, max]
/// (2D slab test). None if the segment misses or only grazes it.
fn segment_rect_crossing(
    a: (f32, f32),
    b: (f32, f32),
    min: (f32, f32),
    max: (f32, f32),
) -> Option<((f32, f32), (f32, f32))> {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, d, lo, hi) in [(a.0, dx, min.0, max.0), (a.1, dy, min.1, max.1)] {
        if d.abs() < 1e-9 {
            if p < lo || p > hi {
                return None;
            }
        } else {
            let (mut ta, mut tb) = ((lo - p) / d, (hi - p) / d);
            if ta > tb {
                core::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 >= t1 {
                return None;
            }
        }
    }
    if t1 - t0 < 1e-3 || t0 <= 1e-3 || t1 >= 1.0 - 1e-3 {
        return None; // grazing, or an endpoint inside the building
    }
    Some((
        (a.0 + t0 * dx, a.1 + t0 * dy),
        (a.0 + t1 * dx, a.1 + t1 * dy),
    ))
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
        w.set_door(0, false); // Living ↔ Corridor
        let (closed, closed_mid) = read(&mut w);
        assert!(
            closed_mid < 0.45 * open_mid,
            "closed door should muffle: {closed_mid} vs {open_mid}"
        );
        assert!(
            closed[0] > 2.0 * closed[2],
            "panel transmission must favor lows: {closed:?}"
        );
        w.set_door(0, true);
        let (_, reopened) = read(&mut w);
        assert!(reopened > 0.75 * open_mid, "reopening should restore: {reopened} vs {open_mid}");
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
