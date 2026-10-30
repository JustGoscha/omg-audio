//! The web demo's level: the walkthrough town, loaded into the engine.
//!
//! The engine knows nothing of towns — it takes triangles with
//! materials, moving boxes and sources. This is the demo putting its
//! level in, like Godot's level.gd does from its scene: the same rooms,
//! walls with their real apertures, roofs, the nave, the paved ground and
//! the furniture and the radar dome (`omg_scene::dome::build_town_mesh`, `walkthrough`). What
//! fills an aperture is a moving object: a glass pane always, a door leaf
//! for the share of the hole it covers (its openness moves it). Sources
//! are the walkthrough's, by their logical index (a speaker rig is
//! several sample-locked engine sources splitting the power).
//!
//! Native too: the demo's sound can be simulated and rendered headless
//! (`examples/gpu_town.rs`).

use omg_audio::{Geometry, Material, MaterialId, ObjectId, Simulation, SourceDesc, SourceId};
use omg_core::vec3::Vec3;
use omg_scene::walkthrough::{self, Door, RoomDef, SourceDef};

/// The static town: the world mesh and the furniture.
pub fn geometry(rooms: &[RoomDef], doors: &[Door]) -> Geometry {
    let (mesh, _glass) = omg_scene::dome::build_town_mesh(rooms, doors);
    let mut g = Geometry::new();
    let ids: Vec<MaterialId> = mesh.materials.iter().map(|m| g.material(*m)).collect();
    let positions: Vec<[f32; 3]> = mesh.positions.iter().map(|p| [p.x, p.y, p.z]).collect();
    let indices: Vec<u32> = mesh.indices.iter().flat_map(|t| *t).collect();
    let mats: Vec<MaterialId> = mesh.tri_material.iter().map(|&m| ids[m as usize]).collect();
    g.add_mesh_per_tri(&positions, &indices, &mats);
    // furniture: room-local boxes, placed on each room's floor
    for (ri, r) in rooms.iter().enumerate() {
        let o = Vec3::new(r.min.0, r.min.1, r.floor_z);
        for (a, m) in walkthrough::furniture(ri).iter().zip(walkthrough::furniture_mats(ri)) {
            let id = g.material(*m);
            let (mn, mx) = (a.min + o, a.max + o);
            g.add_box([mn.x, mn.y, mn.z], [mx.x, mx.y, mx.z], id);
        }
    }
    g
}

/// What fills door `d`'s hole now, as a box and its material (None: the
/// hole is acoustically open) — a glass pane always, a leaf for the
/// covered share (the walkthrough's `aperture_box`).
pub fn aperture(d: &Door) -> Option<([f32; 3], [f32; 3], Material)> {
    let lat_c = if d.axis == 0 { d.pos.1 } else { d.pos.0 };
    let (lat0, lat1, material) = if d.glass {
        let t = walkthrough::glass_transmission(d.heavy);
        (lat_c - d.half, lat_c + d.half, Material { transmission: t, ..Material::GLASS })
    } else {
        if d.openness >= 0.999 {
            return None;
        }
        let covered = (1.0 - d.openness) * 2.0 * d.half;
        let t = if d.heavy { walkthrough::HEAVY_PANEL_TRANSMISSION } else { walkthrough::DOOR_PANEL_TRANSMISSION };
        (lat_c - d.half, lat_c - d.half + covered, Material { transmission: t, ..Material::DOOR })
    };
    let (mn, mx) = slab(d, lat0, lat1);
    Some((mn, mx, material))
}

/// A door's whole hole, the leaf's opening (`Simulation::
/// set_object_aperture`: v6.2 prices the paths through it by how much of
/// it the leaf covers).
pub fn hole(d: &Door) -> ([f32; 3], [f32; 3]) {
    let lat_c = if d.axis == 0 { d.pos.1 } else { d.pos.0 };
    slab(d, lat_c - d.half, lat_c + d.half)
}

/// The door's 8 cm slab between `lat0` and `lat1` across its hole.
fn slab(d: &Door, lat0: f32, lat1: f32) -> ([f32; 3], [f32; 3]) {
    let plane = if d.axis == 0 { d.pos.0 } else { d.pos.1 };
    let (z0, z1) = (d.zc - 0.5 * d.height, d.zc + 0.5 * d.height);
    if d.axis == 0 {
        ([plane - 0.04, lat0, z0], [plane + 0.04, lat1, z1])
    } else {
        ([lat0, plane - 0.04, z0], [lat1, plane + 0.04, z1])
    }
}

/// Per logical source (the walkthrough's), its engine sources — one, or a
/// speaker rig's sample-locked emitters — in the order `Town::new` adds
/// them to a fresh simulation: the mixer's side knows them without one.
pub fn source_map() -> Vec<Vec<SourceId>> {
    let mut next = 0u32;
    walkthrough::sources()
        .iter()
        .map(|s| {
            (0..s.emitters.len().max(1))
                .map(|_| {
                    next += 1;
                    SourceId(next - 1)
                })
                .collect()
        })
        .collect()
}

/// The town in a simulation: its geometry, an object per aperture, the
/// sources.
pub struct Town {
    pub rooms: Vec<RoomDef>,
    pub doors: Vec<Door>,
    pub defs: Vec<SourceDef>,
    /// Per door: its object (a leaf disabled while its hole is open).
    objects: Vec<ObjectId>,
    /// Per logical source: its engine sources (one, or a speaker rig's).
    pub sources: Vec<Vec<SourceId>>,
    /// Per logical source: muted on the mixer (not traced at all), and a
    /// dynamic slot's state (position, loudness factor; None: idle).
    muted: Vec<bool>,
    dynamic: Vec<Option<([f32; 3], f32)>>,
}

impl Town {
    /// Put the town into `sim`: geometry, apertures as they stand, every
    /// walkthrough source at its authored loudness (dynamic slots — balls,
    /// cars, footsteps — start inactive).
    pub fn new(sim: &mut Simulation) -> Self {
        let rooms = walkthrough::rooms();
        let doors = walkthrough::doors();
        let defs: Vec<SourceDef> = walkthrough::sources().into();
        sim.set_geometry(geometry(&rooms, &doors));
        let objects = doors
            .iter()
            .map(|d| {
                // a leaf stays an object while its hole is open (disabled):
                // what v6.2 learnt about its doorway stays with it
                let (mn, mx) = hole(d);
                let (a, b, m) = aperture(d).unwrap_or((mn, mx, Material::DOOR));
                let id = sim.add_object(a, b, m);
                if !d.glass {
                    sim.set_object_aperture(id, mn, mx);
                    sim.set_object_enabled(id, aperture(d).is_some());
                }
                id
            })
            .collect();
        let dyn_slots = walkthrough::DYN_BASE..walkthrough::DYN_BASE + walkthrough::DYN_SLOTS;
        let sources: Vec<Vec<SourceId>> = defs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                // sample-locked emitters split the power
                let g = s.gain / (s.emitters.len().max(1) as f32).sqrt();
                let active = !dyn_slots.contains(&i);
                s.emitters
                    .iter()
                    .map(|e| sim.add_source(SourceDesc { position: [e.0, e.1, s.z], gain: g, active }))
                    .collect()
            })
            .collect();
        debug_assert_eq!(sources, source_map(), "the mixer's side counts the same ids");
        let n = defs.len();
        Self { rooms, doors, defs, objects, sources, muted: vec![false; n], dynamic: vec![None; n] }
    }

    fn base_gain(&self, i: usize) -> f32 {
        let s = &self.defs[i];
        s.gain / (s.emitters.len().max(1) as f32).sqrt()
    }

    fn is_dynamic(i: usize) -> bool {
        (walkthrough::DYN_BASE..walkthrough::DYN_BASE + walkthrough::DYN_SLOTS).contains(&i)
    }

    /// A dynamic slot (a ball, a car, the footsteps) moves, at `gain` × its
    /// authored loudness; None: idle (not traced).
    pub fn set_dynamic(&mut self, sim: &mut Simulation, slot: usize, at: Option<([f32; 3], f32)>) {
        let i = walkthrough::DYN_BASE + slot;
        if i >= self.defs.len() {
            return;
        }
        let was = self.dynamic[i].map(|d| d.1);
        self.dynamic[i] = at;
        let base = self.base_gain(i);
        let on = at.is_some() && !self.muted[i];
        for &id in &self.sources[i] {
            let p = at.map_or([self.defs[i].pos.0, self.defs[i].pos.1, self.defs[i].z], |a| a.0);
            sim.set_source(id, p, on);
            if let Some((_, g)) = at {
                if was != Some(g) {
                    sim.set_source_gain(id, base * g.max(0.0));
                }
            }
        }
    }

    /// Mute a logical source on the mixer: it is not traced at all.
    pub fn set_muted(&mut self, sim: &mut Simulation, i: usize, on: bool) {
        if i >= self.defs.len() || self.muted[i] == on {
            return;
        }
        self.muted[i] = on;
        let active = !on && (!Self::is_dynamic(i) || self.dynamic[i].is_some());
        let s = &self.defs[i];
        for (k, &id) in self.sources[i].iter().enumerate() {
            let p = match self.dynamic[i] {
                Some((p, _)) => p,
                None => {
                    let e = s.emitters.get(k).copied().unwrap_or(s.pos);
                    [e.0, e.1, s.z]
                }
            };
            sim.set_source(id, p, active);
        }
    }

    /// The logical source an engine source belongs to.
    pub fn logical(&self, id: SourceId) -> Option<usize> {
        self.sources.iter().position(|v| v.contains(&id))
    }

    /// Door `i`'s leaf openness 0 (closed) … 1 (open): its object moves
    /// (or goes, or comes back).
    pub fn set_door(&mut self, sim: &mut Simulation, i: usize, openness: f32) {
        let Some(d) = self.doors.get_mut(i) else { return };
        let o = openness.clamp(0.0, 1.0);
        if (d.openness - o).abs() < 1e-4 {
            return;
        }
        d.openness = o;
        let id = self.objects[i];
        match aperture(d) {
            Some((mn, mx, m)) => {
                sim.set_object(id, mn, mx);
                sim.set_object_material(id, m);
                sim.set_object_enabled(id, true);
            }
            None => sim.set_object_enabled(id, false),
        }
    }

    /// The room the point is in (the walkthrough's, by index).
    pub fn room_at(&self, p: [f32; 3]) -> Option<usize> {
        self.rooms.iter().position(|r| {
            !r.outdoor && p[0] >= r.min.0 && p[0] <= r.max.0 && p[1] >= r.min.1 && p[1] <= r.max.1 && p[2] >= r.floor_z && p[2] <= r.floor_z + r.height
        })
        .or_else(|| self.rooms.iter().position(|r| r.outdoor))
    }
}
