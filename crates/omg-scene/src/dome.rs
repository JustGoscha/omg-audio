//! The ambient dome: ambience as an audio skybox, sampled by rays.
//!
//! The outdoor ambience is a directional field on the sky dome. From the
//! listener, a fixed stratified fan of rays traces against the REAL scene
//! geometry (walls with true aperture holes, slabs, roofs, glass panes,
//! swinging door leaves): a ray that escapes to the sky delivers the
//! dome's sound from its departure direction, attenuated by whatever it
//! reflected off or passed through on the way out.
//!
//! Everything the analytic environment code used to hand-build emerges
//! here: standing outside, the whole dome arrives (dimmed behind
//! buildings, because those rays must bounce to escape); indoors, only
//! the rays that find windows, doorways and stairwells escape — the
//! openings localize the world outside, rooms with no direct opening
//! receive it through the rooms that do, and a swinging leaf sweeps the
//! inflow with its own moving geometry. No room graph, no aperture lists,
//! no blend anchoring: the field varies smoothly with position because
//! ray geometry does.
//!
//! The ray pattern is deterministic (golden spiral, specular bounces), so
//! a stationary listener gets a bit-identical estimate every tick — no
//! flicker to smooth away, and motion changes the estimate continuously.

use crate::walkthrough::{Door, RoomDef, GLASS_TRANSMISSION};
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

/// 8 azimuth sectors + zenith. Bin index is the stable route id offset.
pub const DOME_BINS: usize = 9;
/// Route ids for dome bins start here (aperture/window ids stay < 200).
pub const DOME_ID_BASE: u32 = 200;

const N_RAYS: usize = 512;
const MAX_EVENTS: usize = 6;
const HORIZON: f32 = 80.0;
/// Per-tick EMA on bin energies (~0.15 s at 20 Hz) under the engine's
/// own 0.25 s route smoothing.
const EMA: f32 = 0.35;

/// An axis-aligned rectangle in a vertical plane that rays interact with
/// but that is not part of the static mesh: glass transmits, a door leaf
/// blocks. `axis`: 0 = plane of constant x, 1 = constant y.
#[derive(Clone, Copy)]
pub struct Panel {
    pub axis: usize,
    pub plane: f32,
    pub lat: (f32, f32),
    pub z: (f32, f32),
    pub glass: bool,
}

impl Panel {
    fn hit(&self, o: Vec3, d: Vec3, t_max: f32) -> Option<f32> {
        let (po, pd) = if self.axis == 0 { (o.x, d.x) } else { (o.y, d.y) };
        if pd.abs() < 1e-8 {
            return None;
        }
        let t = (self.plane - po) / pd;
        if t < 1e-4 || t >= t_max {
            return None;
        }
        let lat = if self.axis == 0 { o.y + t * d.y } else { o.x + t * d.x };
        let z = o.z + t * d.z;
        (lat > self.lat.0 && lat < self.lat.1 && z > self.z.0 && z < self.z.1).then_some(t)
    }
}

/// One dome inflow estimate: mean departure direction and per-band energy
/// (fraction of the dome reaching the listener through this sector).
#[derive(Clone, Copy, Default)]
pub struct DomeBin {
    pub dir: [f32; 3],
    pub energy: [f32; NBANDS],
}

/// Rectangle (u0, u1, v0, v1) minus hole rectangles, as sub-rectangles:
/// vertical strips at hole u-edges, complement v-intervals per strip.
fn subrects(
    r: (f32, f32, f32, f32),
    holes: &[(f32, f32, f32, f32)],
) -> Vec<(f32, f32, f32, f32)> {
    let mut cuts = vec![r.0, r.1];
    for h in holes {
        cuts.push(h.0.clamp(r.0, r.1));
        cuts.push(h.1.clamp(r.0, r.1));
    }
    cuts.sort_by(f32::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
    let mut out = Vec::new();
    for w in cuts.windows(2) {
        let (u0, u1) = (w[0], w[1]);
        if u1 - u0 < 1e-3 {
            continue;
        }
        let um = 0.5 * (u0 + u1);
        // v-intervals of holes covering this strip, merged
        let mut vh: Vec<(f32, f32)> = holes
            .iter()
            .filter(|h| h.0 <= um && um <= h.1)
            .map(|h| (h.2.max(r.2), h.3.min(r.3)))
            .filter(|(a, b)| b > a)
            .collect();
        vh.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut v = r.2;
        for (h0, h1) in vh {
            if h0 - v > 1e-3 {
                out.push((u0, u1, v, h0));
            }
            v = v.max(h1);
        }
        if r.3 - v > 1e-3 {
            out.push((u0, u1, v, r.3));
        }
    }
    out
}

/// Build the demo world as a mesh: ground, every room's walls with true
/// holes for its apertures, ceiling/roof slabs (stairwell holes), and a
/// parapet band up to the acoustic barrier height. Glass panes come back
/// as transmissive panels (they are not mesh — rays pass them with the
/// pane's loss).
pub fn build_world_mesh(rooms: &[RoomDef], doors: &[Door]) -> (Mesh, Vec<Panel>) {
    let mut b = MeshBuilder::new();
    let mut glass = Vec::new();

    let outdoor = rooms.iter().find(|r| r.outdoor).expect("an outdoor room");
    let ground = b.material(outdoor.walls[4]);
    b.quad(
        Vec3::new(outdoor.min.0, outdoor.min.1, 0.0),
        Vec3::new(outdoor.max.0, outdoor.min.1, 0.0),
        Vec3::new(outdoor.max.0, outdoor.max.1, 0.0),
        Vec3::new(outdoor.min.0, outdoor.max.1, 0.0),
        ground,
    );

    for r in rooms.iter().filter(|r| !r.outdoor) {
        let (z0, z1) = (r.floor_z, r.floor_z + r.height);
        // a storey directly above? its floor closes the gap; otherwise the
        // parapet band continues up to the barrier line
        let above = rooms
            .iter()
            .filter(|n| !n.outdoor && n.floor_z >= z1 - 0.05 && n.floor_z <= z1 + 0.8)
            .find(|n| {
                (n.min.0 - r.min.0).abs() < 2.0 && (n.min.1 - r.min.1).abs() < 2.0
            });
        let wall_top = match above {
            Some(n) => n.floor_z,
            None => r.barrier_height.max(z1),
        };

        let sides = [
            (0usize, r.min.0, r.min.1, r.max.1, 0usize),
            (1, r.max.0, r.min.1, r.max.1, 0),
            (2, r.min.1, r.min.0, r.max.0, 1),
            (3, r.max.1, r.min.0, r.max.0, 1),
        ];
        for (wi, plane, lo, hi, axis) in sides {
            let mat = b.material(r.walls[wi]);
            // apertures in this wall plane touching this room
            let mut holes = Vec::new();
            for d in doors {
                if d.axis != axis || (d.rooms.0 != rooms_index(rooms, r) && d.rooms.1 != rooms_index(rooms, r)) {
                    continue;
                }
                let dp = if axis == 0 { d.pos.0 } else { d.pos.1 };
                let dl = if axis == 0 { d.pos.1 } else { d.pos.0 };
                if (dp - plane).abs() > 0.1 {
                    continue;
                }
                holes.push((dl - d.half, dl + d.half, d.zc - 0.5 * d.height, d.zc + 0.5 * d.height));
                if d.glass {
                    glass.push(Panel {
                        axis,
                        plane,
                        lat: (dl - d.half, dl + d.half),
                        z: (d.zc - 0.5 * d.height, d.zc + 0.5 * d.height),
                        glass: true,
                    });
                }
            }
            for (u0, u1, v0, v1) in subrects((lo, hi, z0, wall_top), &holes) {
                let p = |u: f32, v: f32| {
                    if axis == 0 { Vec3::new(plane, u, v) } else { Vec3::new(u, plane, v) }
                };
                b.quad(p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1), mat);
            }
        }

        // ceiling / roof slab at the room's top, with holes for vertical
        // portals (a stairwell: the aperture between this room and the
        // storey whose floor sits on this slab)
        let mat = b.material(r.walls[5]);
        let mut holes = Vec::new();
        for d in doors {
            let (a, bb) = d.rooms;
            let vertical = (d.zc - z1).abs() < 0.6;
            if vertical
                && (a == rooms_index(rooms, r) || bb == rooms_index(rooms, r))
                && !d.glass
            {
                holes.push((d.pos.0 - d.half, d.pos.0 + d.half, d.pos.1 - d.half, d.pos.1 + d.half));
            }
        }
        for (u0, u1, v0, v1) in subrects((r.min.0, r.max.0, r.min.1, r.max.1), &holes) {
            b.quad(
                Vec3::new(u0, v0, z1),
                Vec3::new(u1, v0, z1),
                Vec3::new(u1, v1, z1),
                Vec3::new(u0, v1, z1),
                mat,
            );
        }
    }
    (b.build(), glass)
}

/// Index of a room def within the list (defs are compared by bounds —
/// names are not unique keys across storeys sharing a footprint).
fn rooms_index(rooms: &[RoomDef], r: &RoomDef) -> usize {
    rooms
        .iter()
        .position(|n| std::ptr::eq(n, r))
        .expect("room from this list")
}

/// The dynamic door leaves as blocking panels: a leaf at openness `o`
/// covers `(1 − o)` of its aperture from one jamb (the swing modeled as
/// coverage — what matters acoustically is how much of the opening the
/// panel obstructs, continuously).
pub fn door_panels(doors: &[Door]) -> Vec<Panel> {
    doors
        .iter()
        .filter(|d| !d.glass && d.openness < 0.999)
        .map(|d| {
            let (lat_c, plane) =
                if d.axis == 0 { (d.pos.1, d.pos.0) } else { (d.pos.0, d.pos.1) };
            let covered = (1.0 - d.openness) * 2.0 * d.half;
            Panel {
                axis: d.axis,
                plane,
                lat: (lat_c - d.half, lat_c - d.half + covered),
                z: (d.zc - 0.5 * d.height, d.zc + 0.5 * d.height),
                glass: false,
            }
        })
        .collect()
}

pub struct DomeProbe {
    mesh: Mesh,
    glass: Vec<Panel>,
    dirs: Vec<Vec3>,
    /// EMA'd per-bin energies.
    ema: [[f32; NBANDS]; DOME_BINS],
    primed: bool,
}

fn bin_of(d: Vec3) -> usize {
    if d.z > 0.87 {
        return 8; // zenith
    }
    let az = d.y.atan2(d.x) + core::f32::consts::PI;
    ((az / (core::f32::consts::TAU / 8.0)) as usize).min(7)
}

impl DomeProbe {
    pub fn new(rooms: &[RoomDef], doors: &[Door]) -> Self {
        let (mesh, glass) = build_world_mesh(rooms, doors);
        // golden-spiral fan over the full sphere — fixed, stratified
        let ga = core::f32::consts::PI * (3.0 - 5.0f32.sqrt());
        let dirs = (0..N_RAYS)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f32 + 0.5) / N_RAYS as f32;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = ga * i as f32;
                Vec3::new(r * phi.cos(), r * phi.sin(), z)
            })
            .collect();
        Self { mesh, glass, dirs, ema: [[0.0; NBANDS]; DOME_BINS], primed: false }
    }

    /// Trace one ray to the sky; returns the per-band energy throughput
    /// it delivers from the dome, or None if it dies inside.
    fn trace_escape(&self, mut pos: Vec3, mut dir: Vec3, leaves: &[Panel]) -> Option<[f32; NBANDS]> {
        let mut tp = [1.0f32; NBANDS];
        for _ in 0..MAX_EVENTS {
            let mesh_hit = self.mesh.raycast(pos, dir);
            let t_mesh = mesh_hit.map_or(f32::MAX, |(t, _)| t);
            let mut t_pan = t_mesh;
            let mut pan: Option<&Panel> = None;
            for p in self.glass.iter().chain(leaves.iter()) {
                if let Some(t) = p.hit(pos, dir, t_pan) {
                    t_pan = t;
                    pan = Some(p);
                }
            }
            if let Some(p) = pan {
                if !p.glass {
                    return None; // a door leaf blocks
                }
                for b in 0..NBANDS {
                    tp[b] *= GLASS_TRANSMISSION[b] * GLASS_TRANSMISSION[b];
                }
                pos = pos + dir * (t_pan + 1e-3);
                continue;
            }
            let Some((t, tri)) = mesh_hit else {
                return Some(tp); // escaped to the sky dome
            };
            if t > HORIZON {
                return Some(tp);
            }
            // specular bounce with the surface's energy reflection
            let m = &self.mesh.materials[self.mesh.tri_material[tri as usize] as usize];
            for b in 0..NBANDS {
                tp[b] *= 1.0 - m.absorption[b];
            }
            if tp[1] < 2e-3 {
                return None;
            }
            let n = self.mesh.tri_normal(tri);
            let hit = pos + dir * t;
            dir = (dir - n * (2.0 * dir.dot(n))).normalize();
            pos = hit + dir * 1e-3;
        }
        None
    }

    /// One dome estimate from `eye`, with the current door leaves.
    pub fn sample(&mut self, eye: Vec3, leaves: &[Panel]) -> [DomeBin; DOME_BINS] {
        let mut e = [[0.0f32; NBANDS]; DOME_BINS];
        let mut dsum = [Vec3::new(0.0, 0.0, 0.0); DOME_BINS];
        for i in 0..N_RAYS {
            let d0 = self.dirs[i];
            if let Some(tp) = self.trace_escape(eye, d0, leaves) {
                let bi = bin_of(d0);
                for b in 0..NBANDS {
                    e[bi][b] += tp[b] / N_RAYS as f32;
                }
                dsum[bi] = dsum[bi] + d0 * tp[1];
            }
        }
        let alpha = if self.primed { EMA } else { 1.0 };
        self.primed = true;
        let mut out = [DomeBin::default(); DOME_BINS];
        for k in 0..DOME_BINS {
            for b in 0..NBANDS {
                self.ema[k][b] += alpha * (e[k][b] - self.ema[k][b]);
                out[k].energy[b] = self.ema[k][b];
            }
            let d = if dsum[k].length() > 1e-6 {
                dsum[k].normalize()
            } else {
                // silent bin: its center direction (harmless placeholder)
                let az = (k as f32 + 0.5) * core::f32::consts::TAU / 8.0 - core::f32::consts::PI;
                if k == 8 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(az.cos(), az.sin(), 0.0) }
            };
            out[k].dir = [d.x, d.y, d.z];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walkthrough;

    fn probe() -> DomeProbe {
        DomeProbe::new(&walkthrough::rooms(), &walkthrough::doors())
    }

    fn total_mid(bins: &[DomeBin; DOME_BINS]) -> f32 {
        bins.iter().map(|b| b.energy[1]).sum()
    }

    /// Open field: most of the dome arrives (upper hemisphere direct,
    /// some more off the ground). Deep indoors: a small directional rest.
    #[test]
    fn open_field_hears_the_dome_and_rooms_hear_slivers() {
        let mut p = probe();
        let open = total_mid(&p.sample(Vec3::new(18.0, 8.0, 1.6), &[]));
        assert!(open > 0.4 && open < 1.0, "open field dome fraction: {open}");
        let mut p = probe();
        let hall = total_mid(&p.sample(Vec3::new(7.0, 18.0, 1.6), &[]));
        assert!(hall < 0.25 * open, "indoors must be a fraction: {hall} vs {open}");
        assert!(hall > 1e-4, "but never zero — the door is open");
    }

    /// Inside the Great Hall, the inflow must point at its door (7, 24):
    /// the strongest bin faces roughly north from (7, 18).
    #[test]
    fn indoor_inflow_localizes_the_opening() {
        let mut p = probe();
        let bins = p.sample(Vec3::new(7.0, 18.0, 1.6), &[]);
        let best = bins
            .iter()
            .max_by(|a, b| a.energy[1].total_cmp(&b.energy[1]))
            .unwrap();
        assert!(
            best.dir[1] > 0.6,
            "dominant inflow should face the door (+y): {:?}",
            best.dir
        );
    }

    /// The corridor has no outside opening at all — its inflow arrives
    /// after threading TWO doorways (corridor→hall→outside). Emergent
    /// multi-hop, no room graph.
    #[test]
    fn interior_room_receives_through_two_doorways() {
        let mut p = probe();
        let bins = p.sample(Vec3::new(4.0, 10.0, 1.6), &[]);
        let t = total_mid(&bins);
        assert!(t > 1e-6, "corridor must not be dead: {t}");
        let mut ph = probe();
        let hall = total_mid(&ph.sample(Vec3::new(7.0, 18.0, 1.6), &[]));
        assert!(t < hall, "two hops must cost more than one: {t} vs {hall}");
    }

    /// Swinging the hall door shut drains the hall's inflow monotonically
    /// with coverage — moving geometry, not a switch.
    #[test]
    fn leaf_coverage_sweeps_the_inflow() {
        let mut doors = walkthrough::doors();
        let mut prev = f32::MAX;
        for step in 0..=4 {
            doors[2].openness = 1.0 - step as f32 / 4.0; // Hall ↔ Outside
            let mut p = DomeProbe::new(&walkthrough::rooms(), &doors);
            let t = total_mid(&p.sample(Vec3::new(7.0, 20.0, 1.6), &door_panels(&doors)));
            assert!(t <= prev + 1e-6, "closing must never raise inflow: {t} vs {prev}");
            prev = t;
        }
        assert!(prev < 1e-5, "a fully closed leaf seals the ray inflow: {prev}");
    }

    /// Behind a window: rays pass the pane with glass loss — brighter
    /// bands lose more (mass law), and the inflow survives.
    #[test]
    fn glass_transmits_with_its_spectrum() {
        let mut p = probe();
        // Old House upper floor: windows only (stairwell aside)
        let bins = p.sample(Vec3::new(27.5, 19.5, 4.6), &[]);
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        for b in bins.iter() {
            lo += b.energy[0];
            hi += b.energy[2];
        }
        assert!(lo > 1e-5, "upper room hears through its panes");
        assert!(lo > 2.0 * hi, "glass must favor lows: {lo} vs {hi}");
    }
}
