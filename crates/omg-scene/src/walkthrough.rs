//! Scripted walkthrough scene: three connected rooms with contrasting
//! acoustics and two *fixed* sources — music in the Living Room, a narrator
//! in the Great Hall. The listener walks the waypoint path between them.
//!
//! Cross-room propagation uses portals: a source in another room is
//! rendered as a virtual source at the aperture into the listener's room,
//! delayed and attenuated by the pre-door path. Each doorway crossing is
//! priced by knife-edge diffraction at the actual jamb the path bends
//! around (`aperture_hop`) — free on the sight line through the opening,
//! increasingly bass-only as the bend deepens. Geometry decides; there are
//! no per-door muffle constants.

use omg_core::material::Material;

/// Cathedral stone: acoustically hard, but arcades, columns and rounded
/// arches break flat-wall specularity — the path tracer sees that as a
/// high scattering fraction (denser, smoother tail). The barrel vault
/// itself is real traced geometry (`Roof::Nave`); the scattering stands
/// in for the arcades, capitals and ribs the mesh does not model.
const ORNATE_STONE: Material = Material {
    absorption: [0.015, 0.03, 0.06],
    scattering: 0.35,
    transmission: [0.10, 0.014, 0.0015],
};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

pub struct RoomDef {
    pub name: &'static str,
    pub min: (f32, f32),
    pub max: (f32, f32),
    pub height: f32,
    /// Acoustic roof line for over-the-top diffraction (interior `height`
    /// plus roof slab / upper storeys — what sound must clear outdoors).
    pub barrier_height: f32,
    /// Height of this room's floor above ground (stacked storeys).
    pub floor_z: f32,
    pub walls: [Material; 6],
    pub wall_thickness: f32,
    /// Outdoor region: no walls, no reverb — direct + ground reflection only.
    pub outdoor: bool,
    /// What closes the room on top (a slab, or a vaulted nave under a
    /// gable roof). `height` is always where the walls stop being the
    /// interior boundary: the slab, or the vault's springing line.
    pub roof: Roof,
}

/// What closes a room on top.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Roof {
    /// A flat slab at `floor_z + height`; the walls run on as a parapet
    /// band up to `barrier_height`.
    Flat,
    /// A church nave. Inside: a round (semicircular) barrel vault that
    /// springs from the side walls at `floor_z + height` and spans the
    /// room across its ridge, built from `facets` flat strips — the
    /// inscribed half-polygon, which the tracer and the renderer both
    /// draw. Outside: a steep gable roof from the eaves at world z `eave`
    /// up to the ridge at `barrier_height`. The side walls rise to the
    /// eaves, the end walls to the gable apex; the space between vault
    /// and roof is a sealed attic. `ridge_axis`: 0 = the ridge runs
    /// along x (gables on the x-walls), 1 = along y (gables on the
    /// y-walls).
    Nave { ridge_axis: usize, eave: f32, facets: usize },
}

impl RoomDef {
    /// Extent across the ridge (lo, hi), then along it (lo, hi).
    pub fn nave_spans(&self, ridge_axis: usize) -> ((f32, f32), (f32, f32)) {
        if ridge_axis == 1 {
            ((self.min.0, self.max.0), (self.min.1, self.max.1))
        } else {
            ((self.min.1, self.max.1), (self.min.0, self.max.0))
        }
    }

    /// The vault's cross-section, springing line to springing line, as
    /// `(across, world z)` corners of the inscribed half-polygon. Empty
    /// for a flat roof.
    pub fn vault_profile(&self) -> Vec<(f32, f32)> {
        let Roof::Nave { ridge_axis, facets, .. } = self.roof else { return Vec::new() };
        let ((lo, hi), _) = self.nave_spans(ridge_axis);
        let (mid, r) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let zs = self.floor_z + self.height;
        let n = facets.max(2);
        (0..=n)
            .map(|k| match k {
                // springing lines exact (they sit on the wall planes)
                0 => (lo, zs),
                k if k == n => (hi, zs),
                _ => {
                    let th = core::f32::consts::PI * (1.0 - k as f32 / n as f32);
                    (mid + r * th.cos(), zs + r * th.sin())
                }
            })
            .collect()
    }

    /// Cross-section area of the vault above its springing line (m²).
    fn vault_section(&self) -> f32 {
        let p = self.vault_profile();
        let zs = self.floor_z + self.height;
        // shoelace, closed back along the springing line
        let mut a = 0.0f32;
        for k in 0..p.len() {
            let (x0, z0) = p[k];
            let (x1, z1) = p[(k + 1) % p.len()];
            a += x0 * (z1 - zs) - x1 * (z0 - zs);
        }
        0.5 * a.abs()
    }

    /// World z of the outside roof line above plan point (x, y): what a
    /// sound passing over the building there must clear. A flat roof:
    /// the parapet top everywhere. A nave: the eaves over the side walls,
    /// rising linearly to the ridge (on the end walls, the gable outline).
    pub fn roof_line_at(&self, x: f32, y: f32) -> f32 {
        let flat = self.barrier_height.max(self.floor_z + self.height);
        let Roof::Nave { ridge_axis, eave, .. } = self.roof else { return flat };
        let ((lo, hi), _) = self.nave_spans(ridge_axis);
        let a = if ridge_axis == 1 { x } else { y };
        let half = 0.5 * (hi - lo);
        let f = (1.0 - (a - 0.5 * (lo + hi)).abs() / half.max(1e-3)).clamp(0.0, 1.0);
        eave + (self.barrier_height - eave) * f
    }

    /// World z of the highest interior point: the slab, or the vault crown.
    pub fn interior_top(&self) -> f32 {
        self.vault_profile().iter().fold(self.floor_z + self.height, |m, p| m.max(p.1))
    }

    /// Interior air volume (m³): the box up to the springing line plus
    /// the vault above it.
    pub fn volume(&self) -> f32 {
        let mut v = (self.max.0 - self.min.0) * (self.max.1 - self.min.1) * self.height;
        if let Roof::Nave { ridge_axis, .. } = self.roof {
            let (_, (a0, a1)) = self.nave_spans(ridge_axis);
            v += self.vault_section() * (a1 - a0);
        }
        v
    }

    /// Interior ceiling area (m²): the slab, or the vault's developed
    /// surface (the half-polygon's perimeter × the nave's length).
    pub fn ceiling_area(&self) -> f32 {
        let slab = (self.max.0 - self.min.0) * (self.max.1 - self.min.1);
        let Roof::Nave { ridge_axis, .. } = self.roof else { return slab };
        let p = self.vault_profile();
        let arc: f32 = p
            .windows(2)
            .map(|s| ((s[1].0 - s[0].0).powi(2) + (s[1].1 - s[0].1).powi(2)).sqrt())
            .sum();
        let (_, (a0, a1)) = self.nave_spans(ridge_axis);
        arc * (a1 - a0)
    }

    /// Interior area of side wall `wi` (walls[] order: x-min, x-max,
    /// y-min, y-max): up to the springing line, plus the vault's lunette
    /// on a nave's end walls.
    pub fn wall_area(&self, wi: usize) -> f32 {
        let span = if wi < 2 { self.max.1 - self.min.1 } else { self.max.0 - self.min.0 };
        let mut a = span * self.height;
        if let Roof::Nave { ridge_axis, .. } = self.roof {
            // walls 0/1 stand in x-planes; the end walls are the planes
            // across the ridge
            if (if wi < 2 { 0 } else { 1 }) == ridge_axis {
                a += self.vault_section();
            }
        }
        a
    }

    /// Height of the box with this room's footprint and volume: the
    /// legacy shoebox engines' stand-in for a vaulted room (equal volume,
    /// so their reverb tracks the real room's size). `height` for a slab.
    pub fn box_height(&self) -> f32 {
        self.volume() / ((self.max.0 - self.min.0) * (self.max.1 - self.min.1)).max(1e-6)
    }

    /// The legacy engines' shoebox for this room (see `box_height`).
    pub fn shoebox(&self) -> Shoebox {
        Shoebox::new(
            Vec3::new(self.max.0 - self.min.0, self.max.1 - self.min.1, self.box_height()),
            self.walls,
        )
    }
}

pub fn rooms() -> Vec<RoomDef> {
    vec![
        RoomDef {
            name: "Living Room",
            min: (0.0, 0.0),
            max: (8.0, 6.0),
            height: 2.7,
            barrier_height: 3.1,
            floor_z: 0.0,
            walls: [
                Material::BRICK,
                Material::BRICK,
                Material::BRICK,
                Material::BRICK,
                Material::CARPET,
                Material::ACOUSTIC_TILE,
            ],
            wall_thickness: 0.24,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Corridor",
            min: (3.2, 6.0),
            max: (4.8, 14.0),
            height: 2.4,
            barrier_height: 3.1,
            floor_z: 0.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.20,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Great Hall",
            min: (0.0, 14.0),
            max: (14.0, 24.0),
            height: 7.0,
            barrier_height: 7.4,
            floor_z: 0.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.30,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Entrance",
            min: (20.0, 28.0),
            max: (22.0, 34.0),
            height: 2.6,
            barrier_height: 3.0,
            floor_z: 0.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.20,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Club",
            min: (22.0, 26.0),
            max: (32.0, 38.0),
            height: 4.5,
            barrier_height: 4.9,
            floor_z: 0.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.35,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Old House",
            min: (24.0, 16.0),
            max: (31.0, 23.0),
            height: 2.8, // interior ground floor; barrier_height covers the storeys
            barrier_height: 5.9,
            floor_z: 0.0,
            walls: [Material::BRICK; 6],
            wall_thickness: 0.25,
            outdoor: false,
            roof: Roof::Flat,
        },
        // Solid street furniture on the square: modeled as thin
        // non-enterable rooms so they occlude, reflect (facades) and
        // diffract (corners) with zero extra machinery.
        RoomDef {
            name: "Colonnade",
            min: (16.0, 15.0),
            max: (16.5, 21.0),
            height: 2.5,
            barrier_height: 2.9,
            floor_z: 0.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.25,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Kiosk",
            min: (14.5, 34.0),
            max: (16.5, 36.0),
            height: 2.7,
            barrier_height: 3.0,
            floor_z: 0.0,
            walls: [Material::WOOD_PANEL; 6],
            wall_thickness: 0.1,
            outdoor: false,
            roof: Roof::Flat,
        },
        RoomDef {
            name: "Old House Upper",
            min: (24.0, 16.0),
            max: (31.0, 23.0),
            height: 2.6,
            barrier_height: 5.9,
            floor_z: 3.0, // above the ground floor + slab
            walls: [Material::BRICK; 6],
            wall_thickness: 0.25,
            outdoor: false,
            roof: Roof::Flat,
        },
        // A stone cathedral, far north across the field: a vast nave
        // whose reverb should read as its size long before you reach it.
        // Romanesque section: 15 m walls carry a round barrel vault
        // (crown 23 m) under a steep 50° gable roof (eaves 19 m, ridge
        // 28.5 m) running north from the portal gable. The vault is real
        // geometry — +41% air volume over the old flat 15 m ceiling, so
        // the traced tail rings longer, as a church should.
        RoomDef {
            name: "Cathedral",
            min: (0.0, 52.0),
            max: (16.0, 74.0),
            height: 15.0, // the vault's springing line
            barrier_height: 28.5, // the ridge
            floor_z: 0.0,
            walls: [ORNATE_STONE; 6],
            wall_thickness: 0.5,
            outdoor: false,
            roof: Roof::Nave { ridge_axis: 1, eave: 19.0, facets: 8 },
        },
        // An underground bunker beneath the east field, reached by a
        // stair shaft. Earth, not air, is what surrounds it — the
        // acoustics graph couples only above-ground surfaces outward.
        RoomDef {
            name: "Bunker",
            min: (34.0, 6.0),
            max: (40.0, 14.0),
            height: 2.2,
            barrier_height: 0.0,
            floor_z: -3.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.4,
            outdoor: false,
            roof: Roof::Flat,
        },
        // A brick bell tower ~300 m north across the field. This is its
        // SHAFT: solid, doorless brick up to 24 m. The bells hang above it
        // in the open belfry stage (arches on all four sides, spire on
        // top), so they ring out over the town; the shaft only occludes,
        // reflects and shadows what would have to pass through it.
        RoomDef {
            name: "Belfry",
            min: (7.0, 318.0),
            max: (13.0, 324.0),
            height: 24.0,
            barrier_height: 24.5,
            floor_z: 0.0,
            walls: [Material::BRICK; 6],
            wall_thickness: 0.4,
            outdoor: false,
            roof: Roof::Flat,
        },
        // The bunker's way in: a concrete blockhouse over a stair shaft.
        // One room from the bunker floor (-3 m) to the blockhouse roof
        // (+2.4 m): the blast door opens into it from the street, the
        // shaft drops through a hole in the ground (cut by the mesh for
        // the below-grade hatch door), and that open hatch leads into the
        // bunker. Before, the bunker was a sealed box: its door sat on no
        // wall and nothing connected the street to it.
        RoomDef {
            name: "Bunker Stair",
            min: (31.4, 6.0),
            max: (34.0, 8.0),
            height: 5.4,
            barrier_height: 2.6,
            floor_z: -3.0,
            walls: [Material::CONCRETE; 6],
            wall_thickness: 0.3,
            outdoor: false,
            roof: Roof::Flat,
        },
        // Outside must come LAST: enclosed rooms overlap its rectangle and
        // room_of matches in order. It surrounds all buildings.
        RoomDef {
            name: "Outside",
            min: (-28.0, -1900.0),
            max: (48.0, 2000.0),
            height: 30.0,
            barrier_height: 30.0,
            floor_z: 0.0,
            walls: [Material::GRASS; 6],
            wall_thickness: 0.15,
            outdoor: true,
            roof: Roof::Flat,
        },
    ]
}

pub const LIVING: usize = 0;
pub const CORRIDOR: usize = 1;
pub const HALL: usize = 2;
pub const ENTRANCE: usize = 3;
pub const CLUB: usize = 4;
pub const HOUSE: usize = 5;
pub const HOUSE_UP: usize = 8;
pub const CATHEDRAL: usize = 9;
pub const BUNKER: usize = 10;
pub const BELFRY: usize = 11;
pub const BUNKER_STAIR: usize = 12;
pub const OUTSIDE: usize = 13;

/// A doorway between two rooms. `axis`: 0 = opening in an x=const wall,
/// 1 = opening in a y=const wall. Arbitrary graph topology (BFS routing).
#[derive(Clone, Copy)]
pub struct Door {
    pub rooms: (usize, usize),
    pub pos: (f32, f32),
    pub axis: usize,
    pub half: f32,
    /// Vertical extent of the opening (m) — aperture area for the power
    /// balance is `2·half × height`.
    pub height: f32,
    /// WORLD z of the aperture center (doors ~1 m, windows sill-high,
    /// upper-storey windows above the slab) — routes point at this.
    pub zc: f32,
    /// Glass pane: not walkable, not a routing edge — but an acoustic
    /// radiator like any aperture, with glass transmission.
    pub glass: bool,
    /// Panel openness 0 (closed) … 1 (fully open). The swinging leaf IS
    /// the filter: transmission is the area-weighted energy mix of the
    /// open slit and the panel, so opening a door sweeps the sound
    /// continuously instead of snapping it (see `fill_energy`).
    pub openness: f32,
    /// Heavy steel panel (blast door) instead of wood.
    pub heavy: bool,
}

impl Door {
    /// Per-band ENERGY transmission of whatever fills this aperture:
    /// glass panes transmit as glass; a door passes the open fraction of
    /// its area freely and the rest through the panel (incoherent
    /// sub-apertures — powers add).
    pub fn fill_energy(&self) -> [f32; NBANDS] {
        if self.glass {
            let g = glass_transmission(self.heavy);
            core::array::from_fn(|b| g[b] * g[b])
        } else {
            let o = self.openness.clamp(0.0, 1.0);
            let panel =
                if self.heavy { HEAVY_PANEL_TRANSMISSION } else { DOOR_PANEL_TRANSMISSION };
            core::array::from_fn(|b| o + (1.0 - o) * panel[b] * panel[b])
        }
    }

    /// Amplitude form of `fill_energy`.
    pub fn fill_amplitude(&self) -> [f32; NBANDS] {
        let e = self.fill_energy();
        core::array::from_fn(|b| e[b].sqrt())
    }
}

pub fn doors() -> Vec<Door> {
    vec![
        Door { rooms: (LIVING, CORRIDOR), pos: (4.0, 6.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0, heavy: false },
        Door { rooms: (CORRIDOR, HALL), pos: (4.0, 14.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0, heavy: false },
        Door { rooms: (HALL, OUTSIDE), pos: (7.0, 24.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0, heavy: false },
        // the club's sound-lock: both vestibule doors are heavy
        // acoustic doors — closed, they seal like the bunker's
        Door { rooms: (OUTSIDE, ENTRANCE), pos: (20.0, 31.0), axis: 0, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0, heavy: true },
        Door { rooms: (ENTRANCE, CLUB), pos: (22.0, 31.0), axis: 0, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 0.0, heavy: true },
        Door { rooms: (OUTSIDE, HOUSE), pos: (26.5, 23.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0, heavy: false },
        // cathedral portal: a tall bronze-clad double door (E-toggleable,
        // index 6) — closed, it transmits like the blast door, not wood
        Door { rooms: (OUTSIDE, CATHEDRAL), pos: (8.0, 52.0), axis: 1, half: 1.2, height: 3.5, zc: 1.75, glass: false, openness: 0.0, heavy: true },
        // bunker blockhouse: a heavy steel door at the stair head (index 7)
        Door { rooms: (OUTSIDE, BUNKER_STAIR), pos: (31.4, 7.0), axis: 0, half: 0.7, height: 2.0, zc: 1.0, glass: false, openness: 1.0, heavy: true },
        // windows
        Door { rooms: (LIVING, OUTSIDE), pos: (3.0, 0.0), axis: 1, half: 1.3, height: 1.4, zc: 1.5, glass: true, openness: 1.0, heavy: false },
        // club windows are laminated glazing: the boom passes, the hiss dies
        Door { rooms: (CLUB, OUTSIDE), pos: (32.0, 32.0), axis: 0, half: 1.8, height: 1.4, zc: 1.5, glass: true, openness: 1.0, heavy: true },
        Door { rooms: (CLUB, OUTSIDE), pos: (26.0, 38.0), axis: 1, half: 1.8, height: 1.4, zc: 1.5, glass: true, openness: 1.0, heavy: true },
        // house windows look out onto the square, club and hall
        Door { rooms: (HOUSE, OUTSIDE), pos: (29.3, 23.0), axis: 1, half: 1.4, height: 1.4, zc: 1.5, glass: true, openness: 1.0, heavy: false },
        Door { rooms: (HOUSE, OUTSIDE), pos: (24.0, 19.5), axis: 0, half: 1.4, height: 1.4, zc: 1.5, glass: true, openness: 1.0, heavy: false },
        // upper-storey windows onto the square and toward the club
        Door { rooms: (HOUSE_UP, OUTSIDE), pos: (29.3, 23.0), axis: 1, half: 1.4, height: 1.4, zc: 4.5, glass: true, openness: 1.0, heavy: false },
        Door { rooms: (HOUSE_UP, OUTSIDE), pos: (24.0, 19.5), axis: 0, half: 1.4, height: 1.4, zc: 4.5, glass: true, openness: 1.0, heavy: false },
        // stairwell: the vertical portal between the two storeys — always
        // open, never toggled (indices ≥ 8 are outside the E-key range)
        Door { rooms: (HOUSE, HOUSE_UP), pos: (24.8, 21.5), axis: 0, half: 0.7, height: 2.2, zc: 3.0, glass: false, openness: 1.0, heavy: false },
        // the open hatch at the foot of the bunker stair, below grade (the
        // mesh cuts the shaft's ground hole above it)
        Door { rooms: (BUNKER_STAIR, BUNKER), pos: (34.0, 7.0), axis: 0, half: 0.8, height: 2.2, zc: -1.9, glass: false, openness: 1.0, heavy: false },
    ]
}

pub const DOOR_HALF_WIDTH: f32 = 0.55;

/// Single-pane glass, per band (much leakier than any wall).
pub const GLASS_TRANSMISSION: [f32; NBANDS] = [0.50, 0.32, 0.20];

/// Laminated/double glazing (club windows): mass law with an air gap —
/// bass thumps through, mids and highs die. This is what makes a club
/// BOOM from outside instead of hissing.
pub const HEAVY_GLASS_TRANSMISSION: [f32; NBANDS] = [0.18, 0.05, 0.012];

/// The pane transmission a glass aperture actually has.
pub fn glass_transmission(heavy: bool) -> [f32; NBANDS] {
    if heavy { HEAVY_GLASS_TRANSMISSION } else { GLASS_TRANSMISSION }
}

/// A closed door: ~4 cm wood panel filling the aperture (mass law).
pub const DOOR_PANEL_TRANSMISSION: [f32; NBANDS] = [0.28, 0.12, 0.05];

/// A heavy steel blast door (the bunker): nearly a wall when shut.
pub const HEAVY_PANEL_TRANSMISSION: [f32; NBANDS] = [0.10, 0.03, 0.01];

/// Half-width of the portal blend zone: within this distance of a doorway
/// the acoustics of both connected rooms are simulated and crossfaded.
pub const BLEND_RADIUS: f32 = 1.5;

/// Effective bend point and per-band knife-edge loss for one aperture hop.
/// Where the straight line p→q meets the door plane INSIDE the opening,
/// sound passes with at most edge-proximity loss (illuminated side of the
/// Kurze–Anderson kernel); outside it, the path bends around the nearer
/// jamb and pays the shadow-zone loss of its detour. This replaces the old
/// fixed per-door muffle constants — geometry decides now.
pub fn aperture_hop(
    p: (f32, f32),
    q: (f32, f32),
    d: &Door,
) -> ((f32, f32), [f32; NBANDS]) {
    let dist =
        |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    // door plane coordinate and opening span in the cross coordinate
    let (plane, lo, hi) = if d.axis == 0 {
        (d.pos.0, d.pos.1 - d.half, d.pos.1 + d.half)
    } else {
        (d.pos.1, d.pos.0 - d.half, d.pos.0 + d.half)
    };
    let (pa, qa) = if d.axis == 0 { (p.0, q.0) } else { (p.1, q.1) };
    let (pc, qc) = if d.axis == 0 { (p.1, q.1) } else { (p.0, q.0) };
    // The hop only passes THROUGH the opening if the segment actually
    // transits the door plane between its endpoints. A hairpin (both
    // neighbors on the same side — e.g. a listener beside the wall the
    // door exits from) must be priced as a bend around the jamb; the
    // clamped crossing would otherwise flip sides on a hair's width and
    // snap the whole chain between free and shadowed.
    let denom = qa - pa;
    let (c, transits) = if denom.abs() < 1e-6 {
        (0.5 * (pc + qc), false)
    } else {
        let t_raw = (plane - pa) / denom;
        (pc + t_raw.clamp(0.0, 1.0) * (qc - pc), (0.0..=1.0).contains(&t_raw))
    };
    let inside = transits && c > lo && c < hi;
    // bend/reference point: the crossing clamped into the opening (jamb
    // margin keeps virtual sources out of the wall itself)
    let cc = c.clamp(lo + 0.02, hi - 0.02);
    let v = if d.axis == 0 { (plane, cc) } else { (cc, plane) };
    // Open-path factor: free on the sight line through the opening (the
    // coherent-point lit-side edge ripple of the knife-edge kernel is not
    // applied — sources here are extended/reverberant fields and it
    // averages away across the source and the two jambs); in the shadow
    // zone the path bends around the nearer jamb and pays the knife-edge
    // loss of its detour.
    let ke = if inside {
        [1.0; NBANDS]
    } else {
        let jamb = if (c - lo).abs() < (c - hi).abs() { lo } else { hi };
        let e = if d.axis == 0 { (plane, jamb) } else { (jamb, plane) };
        let detour = (dist(p, e) + dist(e, q) - dist(p, q)).max(0.0);
        omg_core::diffraction::knife_edge_bands(detour)
    };
    if d.glass {
        return (v, ke); // callers price the pane's flat transmission
    }
    // A swinging panel: the open fraction of the aperture carries the
    // bent/free field, the panel fraction mass-law transmission — energy
    // area mix, continuous in openness (the moving leaf IS the filter).
    let o = d.openness.clamp(0.0, 1.0);
    let mixed: [f32; NBANDS] = core::array::from_fn(|b| {
        (o * ke[b] * ke[b]
            + (1.0 - o) * DOOR_PANEL_TRANSMISSION[b] * DOOR_PANEL_TRANSMISSION[b])
            .sqrt()
    });
    (v, mixed)
}

/// Per-band knife-edge product over the interior vertices of a 2D
/// polyline (the "rubber band" multi-edge construction) — used for wet
/// energy following a multi-door chain.
pub fn chain_bend_muffle(points: &[(f32, f32)]) -> [f32; NBANDS] {
    let dist =
        |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let mut g = [1.0f32; NBANDS];
    for i in 1..points.len().saturating_sub(1) {
        let detour =
            dist(points[i - 1], points[i]) + dist(points[i], points[i + 1])
                - dist(points[i - 1], points[i + 1]);
        let ke = omg_core::diffraction::knife_edge_bands(detour);
        for b in 0..NBANDS {
            g[b] *= ke[b];
        }
    }
    g
}

#[derive(Clone, Copy)]
pub struct SourceDef {
    pub name: &'static str,
    /// Centroid: used for routing, transmission and cross-room rendering.
    pub pos: (f32, f32),
    pub room: usize,
    /// Linear gain applied to everything this source produces.
    pub gain: f32,
    /// WORLD z of the emitter (a rooftop owl sits at 24 m; a room
    /// source at floor_z + SRC_HEIGHT). Dynamic slots override per tick.
    pub z: f32,
    /// Same-room emitter positions playing the identical, sample-locked
    /// signal (a speaker rig). Power is split across them.
    pub emitters: &'static [(f32, f32)],
}

/// Club PA: four corner speakers.
pub const CLUB_SPEAKERS: [(f32, f32); 4] =
    [(23.5, 27.5), (30.5, 27.5), (23.5, 36.5), (30.5, 36.5)];

pub fn sources() -> [SourceDef; 14] {
    [
        SourceDef {
            name: "music",
            z: 1.5,
            pos: (2.0, 3.0),
            room: LIVING,
            gain: 1.0,
            emitters: &[(2.0, 3.0)],
        },
        SourceDef {
            name: "voice",
            z: 1.5,
            pos: (10.5, 20.5),
            room: HALL,
            gain: 0.5,
            emitters: &[(10.5, 20.5)],
        },
        SourceDef {
            name: "club",
            z: 1.5,
            pos: (27.0, 32.0),
            room: CLUB,
            // realistic PA level: far above speech/piano; the ear-adaptation
            // stage (acoustic reflex) is what keeps it listenable inside.
            gain: 5.0,
            emitters: &CLUB_SPEAKERS,
        },
        SourceDef {
            name: "flute",
            z: 1.5,
            pos: (8.0, 66.0),
            room: CATHEDRAL,
            gain: 1.0,
            emitters: &[(8.0, 66.0)],
        },
        SourceDef {
            name: "radio",
            z: -1.5, // bunker floor −3 m + source height
            pos: (37.5, 12.5),
            room: BUNKER,
            // a little transistor radio, not a hi-fi
            gain: 0.5,
            emitters: &[(37.5, 12.5)],
        },
        // Wildlife & carillon (fx one-shots on placed sources — like
        // the feet, they inherit the world's real acoustics for free).
        // The owls PERCH ABOVE the rooms they sit on: room = OUTSIDE,
        // z on the roof line — elevation is native to the traced engine.
        SourceDef {
            name: "owl·cathedral",
            pos: (4.0, 53.0),
            room: OUTSIDE,
            gain: 1.6,
            // on the west roof pitch a meter behind the portal gable,
            // half a meter above the slates (roof line 23.75 m at x = 4)
            z: 24.25,
            emitters: &[(4.0, 53.0)],
        },
        SourceDef {
            name: "owl·house",
            pos: (27.5, 19.5),
            room: OUTSIDE,
            gain: 1.4,
            z: 6.3,
            emitters: &[(27.5, 19.5)],
        },
        SourceDef {
            name: "bells",
            pos: (10.0, 321.0),
            room: OUTSIDE,
            // a real bell peals ~100 dB at the tower; 250 m of air and
            // spreading make it the distant Sunday sound, bass carrying.
            // Hung mid-stage in the open arches, above the 24 m shaft
            // (stage 24–27 m): open air on every side, not inside brick.
            gain: 30.0,
            z: 25.5,
            emitters: &[(10.0, 321.0)],
        },
        // Dynamic slots: thrown projectiles + passing cars (positions
        // set per tick by the frontend).
        SourceDef { name: "ball0", pos: (0.0, 0.0), room: OUTSIDE, gain: 0.9, z: 1.5, emitters: &[(0.0, 0.0)] },
        SourceDef { name: "ball1", pos: (0.0, 0.0), room: OUTSIDE, gain: 0.9, z: 1.5, emitters: &[(0.0, 0.0)] },
        SourceDef { name: "ball2", pos: (0.0, 0.0), room: OUTSIDE, gain: 0.9, z: 1.5, emitters: &[(0.0, 0.0)] },
        SourceDef { name: "car0", pos: (-18.0, 0.0), room: OUTSIDE, gain: 2.2, z: 1.5, emitters: &[(-18.0, 0.0)] },
        SourceDef { name: "car1", pos: (-18.0, 0.0), room: OUTSIDE, gain: 2.2, z: 1.5, emitters: &[(-18.0, 0.0)] },
        // The listener's own feet: footstep one-shots placed at ground
        // level under the walker, so steps pick up each room's acoustics
        // (parquet knock + hall reverb) with zero special-casing.
        SourceDef { name: "feet", pos: (0.0, 0.0), room: OUTSIDE, gain: 1.0, z: 0.3, emitters: &[(0.0, 0.0)] },
    ]
}

pub const DYN_SLOTS: usize = 6;

/// Furniture (PT-early C5): solid boxes inside rooms, ROOM-LOCAL
/// coordinates. The traced early engine shadows sources behind them
/// (segment-validated chains + knife-edge bends); the ISM backend
/// cannot see them — that difference IS the demo. Keep the demo-side
/// mirror (web/main.js FURNITURE) in sync.
pub fn furniture(room: usize) -> &'static [omg_core::pt::Aabb] {
    use omg_core::pt::Aabb;
    use omg_core::vec3::Vec3;
    const fn boxy(
        x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32, t: [f32; 3],
    ) -> Aabb {
        Aabb { min: Vec3 { x: x0, y: y0, z: z0 }, max: Vec3 { x: x1, y: y1, z: z1 }, transmission: t }
    }
    // per-band amplitude THROUGH a piece: soft things pass muffled
    // bass, stone passes nothing
    const STONE: [f32; 3] = [0.02, 0.002, 0.0];
    const SOFT: [f32; 3] = [0.5, 0.25, 0.1];   // sofa, bed, pillows
    const WOOD: [f32; 3] = [0.5, 0.3, 0.15];   // thin tables, chairs
    const CABINET: [f32; 3] = [0.3, 0.1, 0.03]; // solid casework, bar
    const BOOKS: [f32; 3] = [0.15, 0.04, 0.01]; // packed bookshelf
    const PANEL: [f32; 3] = [0.4, 0.2, 0.08];  // whiteboard, TV
    // Cathedral nave (16 x 22 x 15 local): two colonnades of three
    // stone pillars; the flute stands mid-nave at local (8, 14).
    static CATHEDRAL_F: [omg_core::pt::Aabb; 6] = [
        boxy(3.75, 4.75, 0.0, 5.25, 6.25, 9.0, STONE),
        boxy(3.75, 10.25, 0.0, 5.25, 11.75, 9.0, STONE),
        boxy(3.75, 15.75, 0.0, 5.25, 17.25, 9.0, STONE),
        boxy(10.75, 4.75, 0.0, 12.25, 6.25, 9.0, STONE),
        boxy(10.75, 10.25, 0.0, 12.25, 11.75, 9.0, STONE),
        boxy(10.75, 15.75, 0.0, 12.25, 17.25, 9.0, STONE),
    ];
    // Club (10 x 12 local): the bar counter along the west wall.
    static CLUB_F: [omg_core::pt::Aabb; 1] = [boxy(1.0, 3.0, 0.0, 2.0, 7.5, 1.1, CABINET)];
    // Living room (8 x 6 x 2.7 local), piano at (2, 3): a real living
    // room — sofa + coffee table + armchair around the piano, sideboard
    // on the south wall, and a ceiling-high bookshelf dividing the east
    // end: stand behind it and the piano goes into shadow.
    static LIVING_F: [omg_core::pt::Aabb; 5] = [
        boxy(2.0, 4.4, 0.0, 4.2, 5.3, 0.75, SOFT),    // sofa
        boxy(2.5, 3.3, 0.0, 3.7, 3.9, 0.45, WOOD),    // coffee table
        boxy(4.9, 4.3, 0.0, 5.7, 5.1, 0.75, SOFT),    // armchair
        boxy(0.3, 0.2, 0.0, 2.3, 0.7, 0.9, CABINET),  // sideboard
        boxy(5.4, 0.4, 0.0, 5.8, 3.4, 2.2, BOOKS),    // bookshelf divider
    ];
    // Great Hall (14 x 10 x 7 local) as a seminar room. The narrator
    // stands at local (10.5, 6.5) behind a lectern; two columns of
    // three tables with a chair each face her, and a tall whiteboard
    // stands by the east wall — step behind it and she shadows.
    static HALL_F: [omg_core::pt::Aabb; 14] = [
        boxy(1.8, 1.6, 0.0, 3.6, 2.3, 0.74, WOOD),
        boxy(1.8, 4.4, 0.0, 3.6, 5.1, 0.74, WOOD),
        boxy(1.8, 7.2, 0.0, 3.6, 7.9, 0.74, WOOD),
        boxy(4.6, 1.6, 0.0, 6.4, 2.3, 0.74, WOOD),
        boxy(4.6, 4.4, 0.0, 6.4, 5.1, 0.74, WOOD),
        boxy(4.6, 7.2, 0.0, 6.4, 7.9, 0.74, WOOD),
        boxy(1.2, 1.7, 0.0, 1.65, 2.15, 0.45, WOOD), // chairs
        boxy(1.2, 4.5, 0.0, 1.65, 4.95, 0.45, WOOD),
        boxy(1.2, 7.3, 0.0, 1.65, 7.75, 0.45, WOOD),
        boxy(4.0, 1.7, 0.0, 4.45, 2.15, 0.45, WOOD),
        boxy(4.0, 4.5, 0.0, 4.45, 4.95, 0.45, WOOD),
        boxy(4.0, 7.3, 0.0, 4.45, 7.75, 0.45, WOOD),
        boxy(10.2, 5.9, 0.0, 10.8, 6.3, 1.2, CABINET),  // lectern
        boxy(12.0, 2.5, 0.0, 12.3, 5.5, 2.2, PANEL),  // whiteboard
    ];
    // Old House ground floor (7 x 7 local): couch + TV, an open
    // kitchen along the east wall, pillows on the floor.
    static HOUSE_F: [omg_core::pt::Aabb; 7] = [
        boxy(1.0, 2.0, 0.0, 3.2, 2.9, 0.75, SOFT),    // couch
        boxy(1.6, 3.4, 0.0, 2.6, 4.0, 0.4, WOOD),     // coffee table
        boxy(0.3, 4.4, 0.4, 0.55, 5.9, 1.4, PANEL),   // TV on the west wall
        boxy(4.4, 0.3, 0.0, 6.7, 0.95, 0.95, CABINET), // kitchen counter
        boxy(6.0, 1.3, 0.0, 6.7, 2.0, 1.9, CABINET),  // fridge
        boxy(2.2, 4.8, 0.0, 2.8, 5.4, 0.25, SOFT),    // floor pillow
        boxy(3.4, 5.2, 0.0, 4.0, 5.8, 0.25, SOFT),    // floor pillow
    ];
    // Old House upper floor (local z above its own floor): bed + a
    // big wall TV + dresser + a pillow nest.
    static HOUSE_UP_F: [omg_core::pt::Aabb; 4] = [
        boxy(0.8, 0.8, 0.0, 2.6, 2.8, 0.6, SOFT),     // bed
        boxy(0.25, 3.6, 0.8, 0.45, 5.6, 2.0, PANEL),  // big TV, west wall
        boxy(5.8, 0.4, 0.0, 6.6, 2.0, 1.0, CABINET),  // dresser
        boxy(4.6, 4.6, 0.0, 5.2, 5.2, 0.25, SOFT),    // pillow nest
    ];
    match room {
        CATHEDRAL => &CATHEDRAL_F,
        CLUB => &CLUB_F,
        LIVING => &LIVING_F,
        HALL => &HALL_F,
        HOUSE => &HOUSE_F,
        HOUSE_UP => &HOUSE_UP_F,
        _ => &[],
    }
}

/// Full acoustic identity of each furniture piece, PARALLEL to
/// `furniture(room)` row for row (a gate asserts the lengths match).
/// Transmission repeats the boxy tables; absorption/scattering are what
/// the late tracer bounces with and what the early solver reflects
/// with — a sofa eats the tail, a bookshelf diffuses it, a table top
/// slaps back.
/// Pieces below this volume (m³) don't get specular reflector faces or
/// late-field panels — a chair shadows the early field (extras) but
/// doesn't bend a decay curve or slap back audibly, and every reflector
/// face costs chain-cache pressure. One constant, used by the early
/// overlay, the late panels, and both discovery kernels' box lists.
pub const FURN_REFLECTOR_MIN_VOL: f32 = 0.25;

pub fn furniture_mats(room: usize) -> &'static [omg_core::material::Material] {
    use omg_core::material::Material;
    const STONE_M: Material = Material {
        absorption: [0.02, 0.03, 0.04],
        scattering: 0.30,
        transmission: [0.02, 0.002, 0.0],
    };
    const SOFT_M: Material = Material {
        absorption: [0.35, 0.60, 0.70],
        scattering: 0.65,
        transmission: [0.5, 0.25, 0.1],
    };
    const WOOD_M: Material = Material {
        absorption: [0.10, 0.10, 0.12],
        scattering: 0.40,
        transmission: [0.5, 0.3, 0.15],
    };
    const CABINET_M: Material = Material {
        absorption: [0.12, 0.14, 0.18],
        scattering: 0.35,
        transmission: [0.3, 0.1, 0.03],
    };
    const BOOKS_M: Material = Material {
        absorption: [0.25, 0.42, 0.55],
        scattering: 0.80,
        transmission: [0.15, 0.04, 0.01],
    };
    const PANEL_M: Material = Material {
        absorption: [0.18, 0.10, 0.06],
        scattering: 0.15,
        transmission: [0.4, 0.2, 0.08],
    };
    static CATHEDRAL_M: [Material; 6] = [STONE_M; 6];
    static CLUB_M: [Material; 1] = [CABINET_M];
    static LIVING_M: [Material; 5] = [SOFT_M, WOOD_M, SOFT_M, CABINET_M, BOOKS_M];
    static HALL_M: [Material; 14] = [
        WOOD_M, WOOD_M, WOOD_M, WOOD_M, WOOD_M, WOOD_M, // tables
        WOOD_M, WOOD_M, WOOD_M, WOOD_M, WOOD_M, WOOD_M, // chairs
        CABINET_M, // lectern
        PANEL_M,   // whiteboard
    ];
    static HOUSE_M: [Material; 7] =
        [SOFT_M, WOOD_M, PANEL_M, CABINET_M, CABINET_M, SOFT_M, SOFT_M];
    static HOUSE_UP_M: [Material; 4] = [SOFT_M, PANEL_M, CABINET_M, SOFT_M];
    match room {
        CATHEDRAL => &CATHEDRAL_M,
        CLUB => &CLUB_M,
        LIVING => &LIVING_M,
        HALL => &HALL_M,
        HOUSE => &HOUSE_M,
        HOUSE_UP => &HOUSE_UP_M,
        _ => &[],
    }
}

/// (time s, x, y) — piecewise-linear listener path.
/// Lingers near the music, walks the corridor, pauses at the narrator,
/// then leaves through the exterior door into open air.
const WAYPOINTS: [(f32, f32, f32); 20] = [
    (0.0, 3.0, 3.0),
    (6.0, 6.0, 2.5),
    (11.0, 5.5, 4.8),
    (14.0, 4.0, 5.4),
    (16.0, 4.0, 7.0),
    (24.0, 4.0, 13.2),
    (27.0, 4.0, 15.5),
    (33.0, 7.0, 18.5),
    (39.0, 10.0, 20.0),
    (43.0, 10.3, 20.4),
    (48.0, 8.5, 22.0),
    (52.0, 7.0, 23.5),
    (58.0, 7.0, 28.5),
    (63.0, 8.5, 32.5),
    (68.0, 12.0, 35.0),
    (76.0, 18.5, 31.5),
    (81.0, 20.8, 31.0),
    (85.0, 22.8, 31.0),
    (92.0, 28.0, 32.0),
    (98.0, 28.5, 32.5),
];

pub const DURATION_S: f32 = 98.0;
pub const EYE_HEIGHT: f32 = 1.6;
pub const SRC_HEIGHT: f32 = 1.5;
const WALL_MARGIN: f32 = 0.3;

fn path_at(t: f32) -> (f32, f32) {
    let t = t.clamp(0.0, WAYPOINTS[WAYPOINTS.len() - 1].0);
    for w in WAYPOINTS.windows(2) {
        let (t0, x0, y0) = w[0];
        let (t1, x1, y1) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let f = f * f * (3.0 - 2.0 * f);
            return (x0 + f * (x1 - x0), y0 + f * (y1 - y0));
        }
    }
    let last = WAYPOINTS[WAYPOINTS.len() - 1];
    (last.1, last.2)
}

pub fn room_of(rooms: &[RoomDef], x: f32, y: f32) -> usize {
    room_of_z(rooms, x, y, EYE_HEIGHT)
}

/// Room containing (x, y) whose vertical extent holds eye height `z` —
/// stacked storeys select by z, ground-based rooms as before.
pub fn room_of_z(rooms: &[RoomDef], x: f32, y: f32, z: f32) -> usize {
    for (i, r) in rooms.iter().enumerate() {
        if x >= r.min.0
            && x <= r.max.0
            && y >= r.min.1
            && y <= r.max.1
            && z >= r.floor_z - 0.3
            && z <= r.interior_top() + 0.9
        {
            return i;
        }
    }
    // no storey matched: retry ignoring z (e.g. a projectile above a roof
    // still belongs to the outdoor volume below it)
    for (i, r) in rooms.iter().enumerate() {
        if x >= r.min.0 && x <= r.max.0 && y >= r.min.1 && y <= r.max.1 && r.outdoor {
            return i;
        }
    }
    for (i, r) in rooms.iter().enumerate() {
        if x >= r.min.0 && x <= r.max.0 && y >= r.min.1 && y <= r.max.1 {
            return i;
        }
    }
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (i, r) in rooms.iter().enumerate() {
        let cx = 0.5 * (r.min.0 + r.max.0);
        let cy = 0.5 * (r.min.1 + r.max.1);
        let d = (cx - x).powi(2) + (cy - y).powi(2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn clamp_into(r: &RoomDef, x: f32, y: f32) -> (f32, f32) {
    (
        x.clamp(r.min.0 + WALL_MARGIN, r.max.0 - WALL_MARGIN),
        y.clamp(r.min.1 + WALL_MARGIN, r.max.1 - WALL_MARGIN),
    )
}

/// World point → room-local simulation coordinates.
pub fn to_local(r: &RoomDef, x: f32, y: f32, z: f32) -> Vec3 {
    to_local_margin(r, x, y, z, WALL_MARGIN)
}

/// Like `to_local` with an explicit wall margin. Portal virtual sources use
/// a tiny margin so they sit in the wall plane: their first-order wall
/// image then nearly coincides with them (natural half-space doubling)
/// instead of creating a ~2 ms comb.
pub fn to_local_margin(r: &RoomDef, x: f32, y: f32, z: f32, margin: f32) -> Vec3 {
    let cx = x.clamp(r.min.0 + margin, r.max.0 - margin);
    let cy = y.clamp(r.min.1 + margin, r.max.1 - margin);
    Vec3::new(cx - r.min.0, cy - r.min.1, z)
}

pub struct WalkState {
    pub listener_world: (f32, f32),
    pub yaw: f32,
    pub room: usize,
    pub shoebox: Shoebox,
    pub listener_local: Vec3,
}

fn state_in_room(rooms: &[RoomDef], room: usize, lx: f32, ly: f32, lz: f32, yaw: f32) -> WalkState {
    let r = &rooms[room];
    let (lx, ly) = clamp_into(r, lx, ly);
    let local_z = (lz - r.floor_z).clamp(0.2, r.box_height() - 0.2);
    WalkState {
        listener_world: (lx, ly),
        yaw,
        room,
        shoebox: r.shoebox(),
        listener_local: Vec3::new(lx - r.min.0, ly - r.min.1, local_z),
    }
}

pub fn state_at(rooms: &[RoomDef], t: f32) -> WalkState {
    let (lx, ly) = path_at(t);
    let (ax, ay) = path_at(t + 0.4);
    let yaw = if (ax - lx).abs() + (ay - ly).abs() > 1e-4 {
        (ay - ly).atan2(ax - lx)
    } else {
        let (bx, by) = path_at(t - 0.4);
        (ly - by).atan2(lx - bx)
    };
    state_in_room(rooms, room_of(rooms, lx, ly), lx, ly, EYE_HEIGHT, yaw)
}

/// Primary room state + (near a doorway) the state of the room across it,
/// with equal-power crossfade weights. Continuous through the crossing:
/// at the door plane both weights are ≈ 0.707 regardless of which side
/// `room_of` currently reports.
pub struct BlendedState {
    pub primary: WalkState,
    pub primary_weight: f32,
    pub other: Option<(WalkState, f32)>,
    /// The UNCLAMPED pose (x, y, eye z): ray probes want the true body
    /// position, not one snapped into a room's rectangle.
    pub raw: (f32, f32, f32),
}

pub fn blended_state_at(rooms: &[RoomDef], t: f32) -> BlendedState {
    let st = state_at(rooms, t);
    let (lx, ly) = st.listener_world;
    blended_state_for(rooms, &doors(), lx, ly, EYE_HEIGHT, st.yaw)
}

/// Blend state for an arbitrary listener pose (interactive/web input).
/// `lz` is the EYE height in world coordinates (feet + 1.6). `all_doors`
/// is the scene's door list (the blend anchors on its non-glass doors).
pub fn blended_state_for(
    rooms: &[RoomDef],
    all_doors: &[Door],
    lx: f32,
    ly: f32,
    lz: f32,
    yaw: f32,
) -> BlendedState {
    let raw = (lx, ly, lz);
    let primary = state_in_room(rooms, room_of_z(rooms, lx, ly, lz), lx, ly, lz, yaw);
    let (lx, ly) = primary.listener_world;

    let mut nearest: Option<(usize, f32)> = None;
    for (i, d) in all_doors.iter().enumerate() {
        if d.glass {
            continue;
        }
        let dist = ((lx - d.pos.0).powi(2) + (ly - d.pos.1).powi(2)).sqrt();
        if nearest.map_or(true, |(_, nd)| dist < nd) {
            nearest = Some((i, dist));
        }
    }

    if let Some((door, dist)) = nearest {
        if dist < BLEND_RADIUS {
            let d = &all_doors[door];
            let other_room = if primary.room == d.rooms.0 {
                Some(d.rooms.1)
            } else if primary.room == d.rooms.1 {
                Some(d.rooms.0)
            } else {
                None
            };
            if let Some(o) = other_room {
                let theta = (1.0 - dist / BLEND_RADIUS) * core::f32::consts::FRAC_PI_4;
                let other = state_in_room(rooms, o, lx, ly, lz, primary.yaw);
                return BlendedState {
                    primary,
                    primary_weight: theta.cos(),
                    other: Some((other, theta.sin())),
                    raw,
                };
            }
        }
    }
    BlendedState { primary, primary_weight: 1.0, other: None, raw }
}

/// How a fixed source reaches the listener's room.
pub struct Routed {
    /// Apparent source position for simulation in the listener's room:
    /// the source itself, or the aperture it is heard through.
    pub virt_world: (f32, f32),
    /// Radiating through glass (window) rather than an opening.
    pub glass: bool,
    /// Path length before the virtual source (source → doors), meters.
    pub extra_dist: f32,
    /// Per-band amplitude factor for all doorway crossings.
    pub muffle: [f32; NBANDS],
    /// Flat (pane/panel) transmission along the route — glass and closed
    /// doors; 1.0 through open apertures. The wet path pays this even
    /// where it pays no bending.
    pub wet_trans: [f32; NBANDS],
    /// Wall axis of the exit aperture (`Door::axis`: 0 = x-const wall,
    /// 1 = y-const wall), None when the source radiates unrouted. Lets
    /// the diffuse re-radiation pay Lambert obliquity toward the target.
    pub exit_axis: Option<usize>,
    /// Full apparent path (source → doors → listener), for visualization.
    pub route: Vec<(f32, f32)>,
}

impl Routed {
    /// Lambert obliquity of the exit aperture toward `to`: a diffuse
    /// radiator is loudest on-axis and near-silent seen edge-on; the
    /// floor is the energy that edge diffraction wraps around anyway.
    pub fn lambert_toward(&self, to: (f32, f32)) -> f32 {
        const FLOOR: f32 = 0.15;
        match self.exit_axis {
            Some(axis) => {
                let (dx, dy) = (to.0 - self.virt_world.0, to.1 - self.virt_world.1);
                let len = (dx * dx + dy * dy).sqrt().max(1e-3);
                let cos = if axis == 0 { dx.abs() } else { dy.abs() } / len;
                cos.max(FLOOR)
            }
            None => 1.0,
        }
    }
}

/// Room-index space of the door graph: every room any door names, plus
/// both endpoints — the BFS tables are sized from the graph it walks,
/// not from the demo's room list, so any scene's rooms/doors route.
fn graph_size(a: usize, b: usize, all_doors: &[Door]) -> usize {
    all_doors
        .iter()
        .map(|d| d.rooms.0.max(d.rooms.1))
        .fold(a.max(b), usize::max)
        + 1
}

pub fn route_source(
    src: &SourceDef,
    lis_room: usize,
    lis: (f32, f32),
    all_doors: &[Door],
) -> Routed {
    if src.room == lis_room {
        return Routed {
            virt_world: src.pos,
            glass: false,
            extra_dist: 0.0,
            muffle: [1.0; NBANDS],
            wet_trans: [1.0; NBANDS],
            exit_axis: None,
            route: vec![src.pos, lis],
        };
    }

    // BFS over the door graph from the source's room to the listener's.
    let n_rooms = graph_size(src.room, lis_room, all_doors);
    let mut prev: Vec<Option<usize>> = vec![None; n_rooms]; // door index used
    let mut visited = vec![false; n_rooms];
    let mut queue = std::collections::VecDeque::new();
    visited[src.room] = true;
    queue.push_back(src.room);
    while let Some(r) = queue.pop_front() {
        if r == lis_room {
            break;
        }
        for (di, d) in all_doors.iter().enumerate() {
            if d.glass {
                continue;
            }
            let next = if d.rooms.0 == r {
                d.rooms.1
            } else if d.rooms.1 == r {
                d.rooms.0
            } else {
                continue;
            };
            if !visited[next] {
                visited[next] = true;
                prev[next] = Some(di);
                queue.push_back(next);
            }
        }
    }

    // Reconstruct door chain listener → source, then reverse.
    let mut chain: Vec<usize> = Vec::new();
    let mut r = lis_room;
    while r != src.room {
        let di = prev[r].expect("rooms are connected");
        chain.push(di);
        r = if all_doors[di].rooms.0 == r { all_doors[di].rooms.1 } else { all_doors[di].rooms.0 };
    }
    chain.reverse();

    price_chain(src.pos, &chain, lis, all_doors)
}

/// The door indices a source's sound crosses to reach the listener's room
/// (BFS shortest hop count). Topology only — independent of the exact
/// target point, so one chain can be re-priced toward many targets.
pub fn door_chain(src_room: usize, lis_room: usize, all_doors: &[Door]) -> Vec<usize> {
    if src_room == lis_room {
        return Vec::new();
    }
    let n_rooms = graph_size(src_room, lis_room, all_doors);
    let mut prev: Vec<Option<usize>> = vec![None; n_rooms];
    let mut visited = vec![false; n_rooms];
    let mut queue = std::collections::VecDeque::new();
    visited[src_room] = true;
    queue.push_back(src_room);
    while let Some(r) = queue.pop_front() {
        if r == lis_room {
            break;
        }
        for (di, d) in all_doors.iter().enumerate() {
            if d.glass {
                continue;
            }
            let next = if d.rooms.0 == r {
                d.rooms.1
            } else if d.rooms.1 == r {
                d.rooms.0
            } else {
                continue;
            };
            if !visited[next] {
                visited[next] = true;
                prev[next] = Some(di);
                queue.push_back(next);
            }
        }
    }
    let mut chain: Vec<usize> = Vec::new();
    let mut r = lis_room;
    while r != src_room {
        let di = prev[r].expect("rooms are connected");
        chain.push(di);
        r = if all_doors[di].rooms.0 == r { all_doors[di].rooms.1 } else { all_doors[di].rooms.0 };
    }
    chain.reverse();
    chain
}

/// Price a door chain from `src_pos` toward `target`: effective bend
/// points (each hop's straight line clamped into its aperture) and the
/// knife-edge loss of every bend the chain forces. One forward pass —
/// earlier vertices already refined, later ones still at door centers;
/// doors are far apart relative to their widths.
pub fn price_chain(
    src_pos: (f32, f32),
    chain: &[usize],
    target: (f32, f32),
    all_doors: &[Door],
) -> Routed {
    let dist =
        |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    if chain.is_empty() {
        return Routed {
            virt_world: src_pos,
            glass: false,
            extra_dist: 0.0,
            muffle: [1.0; NBANDS],
            wet_trans: [1.0; NBANDS],
            exit_axis: None,
            route: vec![src_pos, target],
        };
    }
    let mut pts: Vec<(f32, f32)> = vec![src_pos];
    pts.extend(chain.iter().map(|&di| all_doors[di].pos));
    pts.push(target);
    let mut muffle = [1.0f32; NBANDS];
    let mut wet_trans = [1.0f32; NBANDS];
    for (k, &di) in chain.iter().enumerate() {
        let d = &all_doors[di];
        let (v, ke) = aperture_hop(pts[k], pts[k + 2], d);
        pts[k + 1] = v;
        let fill = d.fill_amplitude();
        for b in 0..NBANDS {
            // aperture_hop already mixes the panel into the bent path;
            // the wet field pays the filler alone (it doesn't bend).
            muffle[b] *= ke[b];
            wet_trans[b] *= fill[b];
        }
    }

    let n = chain.len();
    let mut extra = 0.0;
    for w in pts[..=n].windows(2) {
        extra += dist(w[0], w[1]);
    }

    Routed {
        virt_world: pts[n],
        glass: false,
        extra_dist: extra,
        muffle,
        wet_trans,
        exit_axis: chain.last().map(|&di| all_doors[di].axis),
        route: pts,
    }
}

/// All apertures (doors + windows) directly connecting two rooms, as
/// radiator routes: the source room's field exits each one and re-radiates
/// omnidirectionally (this is what makes a window audible off-axis).
pub fn aperture_routes(
    src: &SourceDef,
    lis_room: usize,
    lis: (f32, f32),
    all_doors: &[Door],
) -> Vec<Routed> {
    let dist =
        |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    all_doors
        .iter()
        .filter(|d| {
            (d.rooms.0 == src.room && d.rooms.1 == lis_room)
                || (d.rooms.1 == src.room && d.rooms.0 == lis_room)
        })
        .map(|d| {
            let (v, ke) = aperture_hop(src.pos, lis, d);
            Routed {
                virt_world: v,
                glass: d.glass,
                // a pane transmits (flat loss) rather than diffracting; an
                // open aperture is priced by the bend it forces
                extra_dist: dist(src.pos, v),
                muffle: if d.glass { glass_transmission(d.heavy) } else { ke },
                wet_trans: d.fill_amplitude(),
                exit_axis: Some(d.axis),
                route: vec![src.pos, v, lis],
            }
        })
        .collect()
}

/// Straight-segment propagation source → listener: every room-boundary
/// crossing attenuates per that wall's transmission (mass law × thickness),
/// EXCEPT crossings through a door aperture, which pass freely. Same-t
/// crossings (shared walls between adjacent rooms) are merged as one
/// physical wall. Returns per-band amplitude factor of the whole path
/// (1.0 = unobstructed line of sight).
pub fn straight_path_transmission(
    rooms: &[RoomDef],
    all_doors: &[Door],
    a: (f32, f32),
    b: (f32, f32),
) -> [f32; NBANDS] {
    path_transmission(rooms, all_doors, a, b, None, false)
}

/// Height-aware `straight_path_transmission`: `a` and `b` carry world z,
/// and a wall is only paid where the straight segment actually passes
/// through it — a ray crossing a wall plane above the building's top
/// (or below an upper storey's floor) goes over (or under) it for free.
/// This is what lets a bell hung ABOVE a solid tower shaft ring out
/// over it, while a source standing behind a house is still shadowed.
///
/// `open_end`: skip a crossing at the very end of the segment. Use it
/// when `b` is a point ON an aperture (a doorway the route continues
/// through) — that aperture is priced by the route itself.
pub fn straight_path_transmission_z(
    rooms: &[RoomDef],
    all_doors: &[Door],
    a: (f32, f32, f32),
    b: (f32, f32, f32),
    open_end: bool,
) -> [f32; NBANDS] {
    path_transmission(rooms, all_doors, (a.0, a.1), (b.0, b.1), Some((a.2, b.2)), open_end)
}

fn path_transmission(
    rooms: &[RoomDef],
    all_doors: &[Door],
    a: (f32, f32),
    b: (f32, f32),
    z: Option<(f32, f32)>,
    open_end: bool,
) -> [f32; NBANDS] {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    // (t, per-band amplitude) of each boundary crossing
    let mut crossings: Vec<(f32, [f32; NBANDS])> = Vec::new();

    // buried rooms (the bunker) cannot occlude above-ground paths
    for r in rooms.iter().filter(|r| !r.outdoor && r.floor_z + r.height > 0.2) {
        // Edge order matches walls[]: x-min, x-max, y-min, y-max.
        let edges = [
            (0usize, r.min.0, r.min.1, r.max.1, 0usize),
            (1, r.max.0, r.min.1, r.max.1, 0),
            (2, r.min.1, r.min.0, r.max.0, 1),
            (3, r.max.1, r.min.0, r.max.0, 1),
        ];
        for (wall_idx, plane, lo, hi, axis) in edges {
            let (pa, pb) = if axis == 0 { (a.0, b.0) } else { (a.1, b.1) };
            let dp = pb - pa;
            if dp.abs() < 1e-9 {
                continue;
            }
            let t = (plane - pa) / dp;
            // exclude only the source-end self-hit (door-mounted virtual
            // sources sit ON a plane); a listener hugging a wall must still
            // be attenuated by it.
            if !(1e-4..=1.0).contains(&t) {
                continue;
            }
            // the aperture `b` sits on is the route's to price, not ours
            if open_end && (1.0 - t) * len < 0.05 {
                continue;
            }
            let along = if axis == 0 { a.1 + t * dy } else { a.0 + t * dx };
            if along < lo || along > hi {
                continue;
            }
            // over the roof line or under the floor: this wall isn't hit
            // (the line where it crosses: eaves on a nave's side walls,
            // the gable outline on its end walls)
            if let Some((za, zb)) = z {
                let zt = za + t * (zb - za);
                let (px, py) = if axis == 0 { (plane, along) } else { (along, plane) };
                if zt > r.roof_line_at(px, py) || zt < r.floor_z {
                    continue;
                }
            }
            // Through a door aperture on this plane? Free passage.
            let hit_aperture = all_doors.iter().find(|d| {
                d.axis == axis
                    && (if axis == 0 { d.pos.0 - plane } else { d.pos.1 - plane }).abs() < 0.08
                    && (along - if axis == 0 { d.pos.1 } else { d.pos.0 }).abs() < d.half
            });
            let amp = match hit_aperture {
                // openness-continuous filler (open slit + panel / pane)
                Some(d) => d.fill_amplitude(),
                None => r.walls[wall_idx].transmission_at(r.wall_thickness),
            };
            crossings.push((t, amp));
        }
    }

    // Merge same-t crossings (shared boundary of two rooms = one wall):
    // take the more opaque side.
    crossings.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut total = [1.0f32; NBANDS];
    let mut i = 0;
    while i < crossings.len() {
        let mut amp = crossings[i].1;
        let mut j = i + 1;
        while j < crossings.len() && crossings[j].0 - crossings[i].0 < 1e-3 {
            for bnd in 0..NBANDS {
                amp[bnd] = amp[bnd].min(crossings[j].1[bnd]);
            }
            j += 1;
        }
        for bnd in 0..NBANDS {
            total[bnd] *= amp[bnd];
        }
        i = j;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On the sight line through an open door, the portal costs (almost)
    /// nothing; hiding beside the jamb costs a lot, and costs the highs
    /// far more than the lows. Geometry prices doors now, not constants.
    fn test_door(openness: f32) -> Door {
        Door {
            rooms: (0, 1),
            pos: (4.0, 6.0),
            axis: 1,
            half: 0.55,
            height: 2.0,
            zc: 1.0,
            glass: false,
            openness,
            heavy: false,
        }
    }

    #[test]
    fn aperture_is_free_on_axis_and_muffled_off_axis() {
        let d = test_door(1.0);
        // straight through the middle of the opening
        let (_, on) = aperture_hop((4.0, 3.0), (4.0, 9.0), &d);
        assert!(on[2] > 0.85, "on-axis high band should pass: {:?}", on);
        // listener tucked around the corner, well off the opening
        let (_, off) = aperture_hop((4.0, 3.0), (7.5, 6.4), &d);
        assert!(off[2] < 0.3, "deep off-axis highs must shadow: {:?}", off);
        assert!(off[0] > 2.0 * off[2], "off-axis must favor lows: {:?}", off);
    }

    /// The effective virtual source sits inside the opening, near the jamb
    /// the path actually bends around — not at the door center.
    #[test]
    fn aperture_vertex_tracks_the_bend() {
        let d = test_door(1.0);
        let (v, _) = aperture_hop((3.6, 3.0), (7.5, 6.4), &d);
        assert!((v.1 - 6.0).abs() < 1e-4, "vertex on the door plane");
        assert!(v.0 > 4.0, "vertex pulled toward the listener-side jamb: {v:?}");
        assert!(v.0 <= 4.0 + d.half, "vertex stays inside the opening: {v:?}");
    }

    /// A swinging leaf sweeps the transmission continuously: monotone in
    /// openness, panel-flat at 0, free at 1, and LINEAR in transmitted
    /// power (the open area is what admits energy). This is what makes
    /// opening a door sound like moving geometry, not toggling a filter.
    #[test]
    fn door_openness_prices_continuously() {
        let probe = |o: f32| -> [f32; NBANDS] {
            aperture_hop((4.0, 3.0), (4.0, 9.0), &test_door(o)).1
        };
        let mut prev: Option<[f32; NBANDS]> = None;
        let mut o = 0.0f32;
        while o <= 1.0 + 1e-6 {
            let t = probe(o);
            if let Some(p) = prev {
                for b in 0..NBANDS {
                    assert!(t[b] >= p[b] - 1e-6, "opening must never reduce band {b}");
                    let de = t[b] * t[b] - p[b] * p[b];
                    assert!(de <= 0.055, "energy step must stay ∝ area: {de} in band {b} at {o}");
                }
            }
            prev = Some(t);
            o += 0.05;
        }
        let (closed, open) = (probe(0.0), probe(1.0));
        for b in 0..NBANDS {
            assert!((closed[b] - DOOR_PANEL_TRANSMISSION[b]).abs() < 1e-6);
            assert!((open[b] - 1.0).abs() < 1e-6);
        }
        let half = probe(0.5);
        assert!(half[2] > 2.0 * closed[2] && half[2] < 0.95, "half-open sits between: {half:?}");
    }
}

/// The street's asphalt strip (x range): cars pass on it, and it is the
/// one outdoor ground that is not grass (web/main.js draws it the same).
pub const STREET_X: (f32, f32) = (-20.2, -15.8);

/// Where the ambience recording was "made": ear height on open grass,
/// south of the town. The shaped emitters are calibrated so their free
/// field here equals the open dome the bed used to play everywhere.
pub const AMBIENT_REFERENCE: (f32, f32, f32) = (10.0, -150.0, 1.6);

/// Height of the cricket layer above the ground, m.
pub const CRICKET_Z: f32 = 0.1;

/// The demo's ambience bed as shaped emitters: crickets in the grass —
/// the outdoor ground (GRASS) minus every building footprint standing on
/// it and the asphalt street. Fly up and they come from below; walk off
/// and they collapse toward the meadow (`ambient.rs`).
pub fn ambient_emitters(rooms: &[RoomDef]) -> Vec<crate::ambient::AmbientEmitter> {
    use crate::ambient::{AmbientEmitter, AmbientShape, Rect};
    let out = rooms.iter().find(|r| r.outdoor).expect("an outdoor room");
    // holes must be disjoint (each carves its own area): add every
    // footprint as the pieces not already covered
    let mut holes: Vec<Rect> = Vec::new();
    let add = |r: (f32, f32, f32, f32), holes: &mut Vec<Rect>| {
        let have: Vec<(f32, f32, f32, f32)> =
            holes.iter().map(|&((a, b), (c, d))| (a, c, b, d)).collect();
        for (u0, u1, v0, v1) in crate::dome::subrects(r, &have) {
            holes.push(((u0, v0), (u1, v1)));
        }
    };
    for r in rooms.iter().filter(|r| !r.outdoor && r.floor_z < 0.5 && r.floor_z + r.height > 0.2) {
        add((r.min.0, r.max.0, r.min.1, r.max.1), &mut holes);
    }
    add((STREET_X.0, STREET_X.1, out.min.1, out.max.1), &mut holes);
    vec![AmbientEmitter {
        shape: AmbientShape::GroundArea { min: out.min, max: out.max, z: CRICKET_Z, holes },
        level: 1.0,
    }]
}
