//! Environment acoustics: the OUTDOOR ambient field (city hum, rain) as a
//! physical quantity per room, derived from geometry alone.
//!
//! The scene's rooms form a coupling graph: shared walls, floor slabs,
//! sky-exposed roofs and exterior walls (each an area × per-band energy
//! transmission from its material and thickness), plus the live apertures
//! (doors with continuous openness, glass panes, stairwells). A steady-
//! state Sabine power balance over that graph — absorbed power = incoming
//! power, with the outdoors pinned at unit energy — yields each room's
//! outdoor-field level per band.
//!
//! This replaces the per-room hand constants the ambient bed used to
//! switch between: room transitions ride the blend zones, door swings
//! move the aperture conduits continuously, and a room with no direct
//! outside opening (the Corridor) receives its share through the rooms
//! that do — automatically, because that is what the balance says.

use crate::walkthrough::{Door, RoomDef};
use omg_core::NBANDS;

/// Coupling area below which a conduit is dropped as numerical noise.
const MIN_AREA: f32 = 0.05;

/// One static conduit between two rooms: `area` m² of surface with the
/// given per-band ENERGY transmission (min of the two faces, mass law).
struct Conduit {
    rooms: (usize, usize),
    area: f32,
    trans2: [f32; NBANDS],
}

pub struct AcousticsGraph {
    outside: usize,
    /// Per room: total absorption area per band (Σ surface × α), m² sabins.
    absorb: Vec<[f32; NBANDS]>,
    conduits: Vec<Conduit>,
    /// Per room: sky-exposed fraction of the ceiling (0 under another
    /// storey) — rain drums on this.
    pub roof_sky: Vec<f32>,
}

fn overlap(a0: f32, a1: f32, b0: f32, b1: f32) -> f32 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

fn footprint_overlap(a: &RoomDef, b: &RoomDef) -> f32 {
    overlap(a.min.0, a.max.0, b.min.0, b.max.0) * overlap(a.min.1, a.max.1, b.min.1, b.max.1)
}

impl AcousticsGraph {
    pub fn new(rooms: &[RoomDef], doors: &[Door]) -> Self {
        let outside = rooms.iter().position(|r| r.outdoor).expect("an outdoor room");
        let mut absorb = vec![[0.0f32; NBANDS]; rooms.len()];
        let mut conduits: Vec<Conduit> = Vec::new();
        let mut roof_sky = vec![0.0f32; rooms.len()];

        for (ri, r) in rooms.iter().enumerate() {
            if r.outdoor {
                continue;
            }
            let footprint = (r.max.0 - r.min.0) * (r.max.1 - r.min.1);
            let (z0, z1) = (r.floor_z, r.floor_z + r.height);

            // absorption: all six interior surfaces
            for b in 0..NBANDS {
                let mut a = footprint * (r.walls[4].absorption[b] + r.walls[5].absorption[b]);
                let spans = [r.max.1 - r.min.1, r.max.1 - r.min.1, r.max.0 - r.min.0, r.max.0 - r.min.0];
                for (wi, span) in spans.iter().enumerate() {
                    a += span * r.height * r.walls[wi].absorption[b];
                }
                absorb[ri][b] = a;
            }

            // side walls: (wall index, plane, cross range, axis, outward sign)
            let sides = [
                (0usize, r.min.0, r.min.1, r.max.1, 0usize, -1.0f32),
                (1, r.max.0, r.min.1, r.max.1, 0, 1.0),
                (2, r.min.1, r.min.0, r.max.0, 1, -1.0),
                (3, r.max.1, r.min.0, r.max.0, 1, 1.0),
            ];
            for (wi, plane, lo, hi, axis, sign) in sides {
                let t_own = r.walls[wi].transmission_at(r.wall_thickness);
                let total = (hi - lo) * r.height;
                let mut covered = 0.0f32;
                for (ni, n) in rooms.iter().enumerate() {
                    if ni == ri || n.outdoor {
                        continue;
                    }
                    let (n_lo, n_hi, n_center) = if axis == 0 {
                        (n.min.0, n.max.0, 0.5 * (n.min.0 + n.max.0))
                    } else {
                        (n.min.1, n.max.1, 0.5 * (n.min.1 + n.max.1))
                    };
                    let abuts =
                        (n_lo - plane).abs() < 0.45 || (n_hi - plane).abs() < 0.45;
                    if !abuts || (n_center - plane) * sign <= 0.0 {
                        continue;
                    }
                    let (c_lo, c_hi) = if axis == 0 { (n.min.1, n.max.1) } else { (n.min.0, n.max.0) };
                    let area = overlap(lo, hi, c_lo, c_hi)
                        * overlap(z0, z1, n.floor_z, n.floor_z + n.height);
                    if area < MIN_AREA {
                        continue;
                    }
                    covered += area;
                    if ri < ni {
                        // one wall, two faces: the more opaque decides
                        let t_n = n.walls[wi ^ 1].transmission_at(n.wall_thickness);
                        conduits.push(Conduit {
                            rooms: (ri, ni),
                            area,
                            trans2: core::array::from_fn(|b| {
                                let t = t_own[b].min(t_n[b]);
                                t * t
                            }),
                        });
                    }
                }
                // Only the ABOVE-GROUND part of a wall faces the open
                // air; below grade there is earth, which transmits
                // nothing worth modeling (an underground room couples
                // outward only through its shaft).
                let above = (z1.min(f32::MAX) - z0.max(0.0)).max(0.0) / (z1 - z0);
                let ext = (total - covered) * above;
                if ext > MIN_AREA {
                    conduits.push(Conduit {
                        rooms: (ri, outside),
                        area: ext,
                        trans2: core::array::from_fn(|b| t_own[b] * t_own[b]),
                    });
                }
            }

            // ceiling: storeys above couple through the slab; the rest is
            // roof, open to the sky
            let t_ceil = r.walls[5].transmission_at(r.wall_thickness);
            let mut covered = 0.0f32;
            for (ni, n) in rooms.iter().enumerate() {
                if ni == ri || n.outdoor {
                    continue;
                }
                if n.floor_z < z1 - 0.05 || n.floor_z > z1 + 0.8 {
                    continue;
                }
                let area = footprint_overlap(r, n);
                if area < MIN_AREA {
                    continue;
                }
                covered += area;
                let t_floor = n.walls[4].transmission_at(n.wall_thickness);
                conduits.push(Conduit {
                    rooms: (ri, ni),
                    area,
                    trans2: core::array::from_fn(|b| {
                        let t = t_ceil[b].min(t_floor[b]);
                        t * t
                    }),
                });
            }
            // a ceiling below grade is under earth, not under sky
            let sky = if z1 > 0.05 { (footprint - covered).max(0.0) } else { 0.0 };
            roof_sky[ri] = sky / footprint;
            if sky > MIN_AREA {
                conduits.push(Conduit {
                    rooms: (ri, outside),
                    area: sky,
                    trans2: core::array::from_fn(|b| t_ceil[b] * t_ceil[b]),
                });
            }
        }

        // Apertures sit IN these surfaces: carve their area out of the
        // largest conduit between the same pair (they are priced live in
        // `outdoor_field`, with door state).
        for d in doors {
            let s = 2.0 * d.half * d.height;
            if let Some(c) = conduits
                .iter_mut()
                .filter(|c| {
                    (c.rooms.0 == d.rooms.0 && c.rooms.1 == d.rooms.1)
                        || (c.rooms.0 == d.rooms.1 && c.rooms.1 == d.rooms.0)
                })
                .max_by(|a, b| a.area.total_cmp(&b.area))
            {
                c.area = (c.area - s).max(0.0);
            }
        }
        conduits.retain(|c| c.area > MIN_AREA);

        Self { outside, absorb, conduits, roof_sky }
    }

    /// Steady-state ENERGY of the outdoor field in every room, relative
    /// to the open air (pinned at 1): Sabine balance, absorbed = incoming,
    /// over static conduits plus the live apertures. Gauss–Seidel — the
    /// graph is tiny and diagonally dominant (absorption ≫ coupling).
    pub fn outdoor_field(&self, doors: &[Door]) -> Vec<[f32; NBANDS]> {
        let n = self.absorb.len();
        // incident conduits per room: (neighbor, area, energy transmission)
        let mut inc: Vec<Vec<(usize, f32, [f32; NBANDS])>> = vec![Vec::new(); n];
        for c in &self.conduits {
            inc[c.rooms.0].push((c.rooms.1, c.area, c.trans2));
            inc[c.rooms.1].push((c.rooms.0, c.area, c.trans2));
        }
        for d in doors {
            let s = 2.0 * d.half * d.height;
            let t2 = d.fill_energy();
            inc[d.rooms.0].push((d.rooms.1, s, t2));
            inc[d.rooms.1].push((d.rooms.0, s, t2));
        }
        let mut f = vec![[0.0f32; NBANDS]; n];
        f[self.outside] = [1.0; NBANDS];
        for _ in 0..16 {
            for r in 0..n {
                if r == self.outside {
                    continue;
                }
                for b in 0..NBANDS {
                    let mut num = 0.0f32;
                    let mut den = self.absorb[r][b];
                    for &(nb, s, t2) in &inc[r] {
                        num += s * t2[b] * f[nb][b];
                        den += s * t2[b];
                    }
                    f[r][b] = if den > 1e-9 { num / den } else { 0.0 };
                }
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walkthrough;

    fn field() -> (Vec<RoomDef>, Vec<Door>, Vec<[f32; NBANDS]>) {
        let rooms = walkthrough::rooms();
        let doors = walkthrough::doors();
        let g = AcousticsGraph::new(&rooms, &doors);
        let f = g.outdoor_field(&doors);
        (rooms, doors, f)
    }

    /// The balance must be physical: unit field outdoors, a genuine but
    /// partial field in every room, bass-heavier indoors (mass law), and
    /// a room with NO direct outside aperture still supplied through its
    /// neighbors.
    #[test]
    fn outdoor_field_is_physical() {
        let (rooms, _, f) = field();
        assert_eq!(f[walkthrough::OUTSIDE], [1.0; NBANDS]);
        for (ri, r) in rooms.iter().enumerate() {
            if r.outdoor {
                continue;
            }
            assert!(
                f[ri][1] > 1e-5 && f[ri][1] < 0.5,
                "{}: field {:?} out of the plausible range",
                r.name,
                f[ri]
            );
            assert!(
                f[ri][0] > f[ri][2],
                "{}: indoors must favor lows: {:?}",
                r.name,
                f[ri]
            );
        }
        // Corridor is interior-only: supplied via Living/Hall, so weaker
        // than both but alive.
        assert!(f[walkthrough::CORRIDOR][1] < f[walkthrough::HALL][1]);
        assert!(f[walkthrough::CORRIDOR][1] > 1e-5);
    }

    /// Swinging the house door closed drains the house's share of the
    /// field — continuously, monotonically.
    #[test]
    fn closing_a_door_drains_the_field_continuously() {
        let rooms = walkthrough::rooms();
        let mut doors = walkthrough::doors();
        let g = AcousticsGraph::new(&rooms, &doors);
        let mut prev = f32::NAN;
        for step in 0..=10 {
            doors[5].openness = 1.0 - step as f32 / 10.0; // Outside ↔ House
            let f = g.outdoor_field(&doors);
            let cur = f[walkthrough::HOUSE][1];
            if prev.is_finite() {
                assert!(cur < prev + 1e-9, "closing must never raise the field");
                assert!(prev / cur.max(1e-9) < 1.8, "no audible step per 10%: {prev} -> {cur}");
            }
            prev = cur;
        }
        doors[5].openness = 1.0;
        let f = g.outdoor_field(&doors);
        assert!(prev < 0.7 * f[walkthrough::HOUSE][1], "closed house is clearly quieter");
    }

    /// Stacked storeys: the ground floor of the Old House sits under the
    /// upper storey, so it has no sky-exposed roof; the upper storey does.
    #[test]
    fn roof_exposure_respects_storeys() {
        let rooms = walkthrough::rooms();
        let doors = walkthrough::doors();
        let g = AcousticsGraph::new(&rooms, &doors);
        assert!(g.roof_sky[walkthrough::HOUSE] < 0.05, "ground floor is roofed by the storey");
        assert!(g.roof_sky[walkthrough::HOUSE_UP] > 0.95, "upper storey sees the sky");
        assert!(g.roof_sky[walkthrough::CLUB] > 0.95);
    }
}
