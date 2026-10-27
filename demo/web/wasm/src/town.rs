//! The web demo's level: the walkthrough town, loaded into the engine.
//!
//! The engine knows nothing of towns — it takes triangles with
//! materials, moving boxes and sources. This is the demo putting its
//! level in, like Godot's level.gd does from its scene: the same rooms,
//! walls with their real apertures, roofs, the nave, the paved ground and
//! the furniture (`omg_scene::dome::build_world_mesh`, `walkthrough`). What
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
    let (mesh, _glass) = omg_scene::dome::build_world_mesh(rooms, doors);
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
    let (lat_c, plane) = if d.axis == 0 { (d.pos.1, d.pos.0) } else { (d.pos.0, d.pos.1) };
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
    let (z0, z1) = (d.zc - 0.5 * d.height, d.zc + 0.5 * d.height);
    Some(if d.axis == 0 {
        ([plane - 0.04, lat0, z0], [plane + 0.04, lat1, z1], material)
    } else {
        ([lat0, plane - 0.04, z0], [lat1, plane + 0.04, z1], material)
    })
}

/// The town in a simulation: its geometry, an object per aperture, the
/// sources.
pub struct Town {
    pub rooms: Vec<RoomDef>,
    pub doors: Vec<Door>,
    pub defs: Vec<SourceDef>,
    /// Per door: its object (None while the hole is open).
    objects: Vec<Option<ObjectId>>,
    /// Per logical source: its engine sources (one, or a speaker rig's).
    pub sources: Vec<Vec<SourceId>>,
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
            .map(|d| aperture(d).map(|(mn, mx, m)| sim.add_object(mn, mx, m)))
            .collect();
        let dyn_slots = walkthrough::DYN_BASE..walkthrough::DYN_BASE + walkthrough::DYN_SLOTS;
        let sources = defs
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
        Self { rooms, doors, defs, objects, sources }
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
        match (aperture(d), self.objects[i]) {
            (Some((mn, mx, _)), Some(id)) => sim.set_object(id, mn, mx),
            (Some((mn, mx, m)), None) => self.objects[i] = Some(sim.add_object(mn, mx, m)),
            (None, Some(id)) => {
                sim.remove_object(id);
                self.objects[i] = None;
            }
            (None, None) => {}
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
