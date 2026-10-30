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
        // The echo canyon, south of the town: two sheer stone cliffs,
        // 40 m apart face to face, 50 m long, 24 m high — solid like the
        // kiosk. A clap between them comes back as a train of discrete
        // echoes (one per 2·40 m / c ≈ 233 ms, ping-ponging when you stand
        // off-centre). See `CANYON`.
        cliff("West Cliff", CANYON.west),
        cliff("East Cliff", CANYON.east),
        // Outside must come LAST: enclosed rooms overlap its rectangle and
        // room_of matches in order. It surrounds all buildings.
        RoomDef {
            name: "Outside",
            // west to the radar dome across the road (`RADOME`), as far as
            // the Godot level's ground tile reaches
            min: (-60.0, -1900.0),
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

/// The echo canyon's two cliffs, each (min, max) in plan: the inner
/// faces stand at x = -6 and x = 34, from y = -120 to -70; the canyon
/// floor and the track from the street are `GROUND` patches.
pub struct Canyon {
    pub west: ((f32, f32), (f32, f32)),
    pub east: ((f32, f32), (f32, f32)),
    pub height: f32,
}

pub const CANYON: Canyon = Canyon {
    west: ((-12.0, -120.0), (-6.0, -70.0)),
    east: ((34.0, -120.0), (40.0, -70.0)),
    height: 24.0,
};

fn cliff(name: &'static str, (min, max): ((f32, f32), (f32, f32))) -> RoomDef {
    RoomDef {
        name,
        min,
        max,
        height: CANYON.height,
        barrier_height: CANYON.height + 0.5,
        floor_z: 0.0,
        walls: [Material::ROCK; 6],
        wall_thickness: 0.5,
        outdoor: false,
        roof: Roof::Flat,
    }
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
pub const WEST_CLIFF: usize = 13;
pub const EAST_CLIFF: usize = 14;
pub const OUTSIDE: usize = 15;

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
pub const GLASS_TRANSMISSION: [f32; NBANDS] = omg_core::material::Material::GLASS.transmission;

/// Laminated/double glazing (club windows): mass law with an air gap —
/// bass thumps through, mids and highs die. This is what makes a club
/// BOOM from outside instead of hissing.
pub const HEAVY_GLASS_TRANSMISSION: [f32; NBANDS] = omg_core::material::Material::HEAVY_GLASS.transmission;

/// The pane transmission a glass aperture actually has.
pub fn glass_transmission(heavy: bool) -> [f32; NBANDS] {
    if heavy { HEAVY_GLASS_TRANSMISSION } else { GLASS_TRANSMISSION }
}

/// A closed door: ~4 cm wood panel filling the aperture (mass law).
pub const DOOR_PANEL_TRANSMISSION: [f32; NBANDS] = omg_core::material::Material::DOOR.transmission;

/// A heavy steel blast door (the bunker): nearly a wall when shut.
pub const HEAVY_PANEL_TRANSMISSION: [f32; NBANDS] = omg_core::material::Material::HEAVY_DOOR.transmission;

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

pub fn sources() -> [SourceDef; 16] {
    [
        // the living room's grand piano: the sound rises off its
        // soundboard, half a meter above the case
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
            // on the bunker's table: floor −3 m + table 0.75 m + the set
            z: -2.1,
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
        // Static sources added after the dynamic slots (appended, so every
        // earlier index stays as the frontends know it).
        // A small garden fountain beside the belfry path, halfway to the
        // tower: the jet's splash a little above the basin rim.
        SourceDef {
            name: "fountain",
            pos: FOUNTAIN,
            room: OUTSIDE,
            gain: 0.2,
            z: 0.8,
            emitters: &[FOUNTAIN],
        },
        // A voice on the stage at the centre of the radar dome (`RADOME`):
        // a speaker standing on it, projecting.
        SourceDef {
            name: "stage",
            pos: RADOME.center,
            room: OUTSIDE,
            gain: 1.0,
            z: RADOME.stage_h + 1.6,
            emitters: &[RADOME.center],
        },
    ]
}

/// The garden fountain beside the belfry path (basin centre).
pub const FOUNTAIN: (f32, f32) = (14.2, 199.0);

/// A radar dome across the west road, like Teufelsberg's: a sphere of
/// fabric panels standing on the ground (its centre `center_z` up, cut by
/// the grade), a doorway facing the street (east), and a round stage at
/// its centre. A sphere focuses: what leaves the centre comes back to it
/// together, `2 · radius / c` later, over and over.
#[derive(Clone, Copy)]
pub struct Radome {
    pub center: (f32, f32),
    pub radius: f32,
    pub center_z: f32,
    /// The doorway: width and height (m), facing +x.
    pub door_w: f32,
    pub door_h: f32,
    /// The stage: radius and height (m).
    pub stage_r: f32,
    pub stage_h: f32,
    /// Icosphere subdivisions (4: 5120 facets, ~0.8 m at r 12 m).
    pub subdiv: u32,
}

pub const RADOME: Radome = Radome {
    center: (-42.0, 30.0),
    radius: 12.0,
    center_z: 4.0,
    door_w: 3.0,
    door_h: 3.2,
    stage_r: 2.5,
    stage_h: 0.8,
    subdiv: 4,
};

/// The dome's skin: coated fibreglass membrane panels — smooth (a mirror
/// that focuses), light (the bass passes). Godot tags it "radome"
/// (`omg-godot` backend::materials, the same values).
pub const RADOME_FABRIC: Material = Material::new([0.08, 0.05, 0.04], 0.05, [0.30, 0.10, 0.03]);

/// Sources `DYN_BASE .. DYN_BASE + DYN_SLOTS` are the dynamic slots
/// (positions set per tick by the frontend); every other source is static.
pub const DYN_BASE: usize = 8;
pub const DYN_SLOTS: usize = 6;

/// The dynamic slot of source `si`, if it is one.
pub fn dyn_slot(si: usize) -> Option<usize> {
    (DYN_BASE..DYN_BASE + DYN_SLOTS).contains(&si).then(|| si - DYN_BASE)
}

// ------------------------------------------------------------ interiors
//
// The interior layout is authored HERE, once: one table per room of
// `Piece`s (kind, centre, size, yaw, material) in `layout`. Everything
// else derives from it — the tracer's boxes (`furniture` /
// `furniture_mats`), the exported `assets/demo/layout.json` the web demo
// and the Godot demo build their visuals and colliders from
// (`layout_json`; regenerate with `cargo run -p omg-scene --example
// export_layout` — the `demo_layout` test fails while the committed
// file is stale), and the layout gate (`tests/demo_layout.rs`: no
// overlaps, wall rules, door clearance, walkable paths from every door
// to every source).

/// What a furniture piece IS: its look in the frontends, its wall rule
/// in the layout gate, and its JSON `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Sofa,
    Armchair,
    CoffeeTable,
    SideTable,
    Rug,
    FloorLamp,
    /// An upright piano (back against a wall, keyboard in front).
    Piano,
    /// A grand piano, free-standing; its front is the keyboard.
    GrandPiano,
    PianoBench,
    Bookshelf,
    /// A free-standing double-sided bookcase dividing a room; one end
    /// against a wall.
    Divider,
    Sideboard,
    Desk,
    Chair,
    Table,
    Tv,
    TvStand,
    Bed,
    Nightstand,
    Wardrobe,
    Dresser,
    BunkBed,
    KitchenCounter,
    Fridge,
    /// A row of joined seats (seminar seating), one box.
    SeatRow,
    Lectern,
    Whiteboard,
    Bar,
    BackBar,
    Stool,
    /// A high-backed booth bench.
    Booth,
    Stage,
    DjBooth,
    /// A PA stack's sub cabinet; the emitting top box sits above it.
    Speaker,
    Pew,
    Altar,
    Pillar,
    Crate,
    Shelf,
    Locker,
    Bench,
    CoatRack,
    /// A small lamp standing on a desk or nightstand (decor).
    TableLamp,
    /// A floor-standing church candlestick (decor).
    CandleStand,
    /// A stone baptismal font.
    Font,
    /// A water barrel / drum.
    Barrel,
}

/// How a piece must stand relative to the room's walls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WallRule {
    Free,
    /// Its back (the side opposite its facing) stands against a wall:
    /// beds (headboard), bookshelves, sideboards, wardrobes, wall TVs.
    Back,
    /// Some face touches a wall (a divider's end, a booth, a stage).
    Any,
}

impl Kind {
    pub const fn name(self) -> &'static str {
        use Kind::*;
        match self {
            Sofa => "sofa",
            Armchair => "armchair",
            CoffeeTable => "coffee_table",
            SideTable => "side_table",
            Rug => "rug",
            FloorLamp => "floor_lamp",
            Piano => "piano",
            GrandPiano => "grand_piano",
            PianoBench => "piano_bench",
            Bookshelf => "bookshelf",
            Divider => "divider",
            Sideboard => "sideboard",
            Desk => "desk",
            Chair => "chair",
            Table => "table",
            Tv => "tv",
            TvStand => "tv_stand",
            Bed => "bed",
            Nightstand => "nightstand",
            Wardrobe => "wardrobe",
            Dresser => "dresser",
            BunkBed => "bunk_bed",
            KitchenCounter => "kitchen_counter",
            Fridge => "fridge",
            SeatRow => "seat_row",
            Lectern => "lectern",
            Whiteboard => "whiteboard",
            Bar => "bar",
            BackBar => "back_bar",
            Stool => "stool",
            Booth => "booth",
            Stage => "stage",
            DjBooth => "dj_booth",
            Speaker => "speaker",
            Pew => "pew",
            Altar => "altar",
            Pillar => "pillar",
            Crate => "crate",
            Shelf => "shelf",
            Locker => "locker",
            Bench => "bench",
            CoatRack => "coat_rack",
            TableLamp => "table_lamp",
            CandleStand => "candle_stand",
            Font => "font",
            Barrel => "barrel",
        }
    }

    pub const fn wall_rule(self) -> WallRule {
        use Kind::*;
        match self {
            // (desks may stand free: the seminar presenter's faces the room)
            Piano | Bookshelf | Sideboard | Tv | TvStand | Bed | Nightstand | Wardrobe | Dresser
            | KitchenCounter | Fridge | Whiteboard | BackBar | Shelf | Locker | CoatRack => {
                WallRule::Back
            }
            Divider | BunkBed | Booth | Stage => WallRule::Any,
            _ => WallRule::Free,
        }
    }

    /// Lies flat on the floor (rugs): walkable, may lie under anything.
    pub const fn flat(self) -> bool {
        matches!(self, Kind::Rug)
    }

    /// Takes part in the acoustics (`furniture`). Rugs, lamps and
    /// candlesticks are decor: a 1 cm rug box would only tax every floor
    /// bounce.
    pub const fn acoustic(self) -> bool {
        !matches!(self, Kind::Rug | Kind::FloorLamp | Kind::TableLamp | Kind::CandleStand)
    }

    /// Base colour for the frontends (0xRRGGBB).
    pub const fn color(self) -> u32 {
        use Kind::*;
        match self {
            Sofa | Armchair => 0x5a4636,
            Booth => 0x6a2236,
            Rug => 0x6a3a32,
            CoffeeTable | SideTable | Table | Desk | Nightstand | PianoBench | Chair | Stool
            | Bench => 0x4a3626,
            FloorLamp | TableLamp => 0xd8c8a0,
            CandleStand => 0x8a7040,
            Font => 0x8a8a90,
            Barrel => 0x3e5a3a,
            Piano | GrandPiano => 0x15130f,
            Bookshelf | Divider | Shelf => 0x5a4530,
            Sideboard | Dresser | Wardrobe | TvStand => 0x4a3626,
            Tv => 0x14161c,
            Bed | BunkBed => 0x707a8a,
            KitchenCounter => 0x4a4038,
            Fridge => 0xb8bcc0,
            SeatRow => 0x35404f,
            Lectern => 0x5a4530,
            Whiteboard => 0xd8d8d0,
            Bar | BackBar => 0x3a2a1e,
            Stage => 0x1e1a22,
            DjBooth => 0x2a2630,
            Speaker => 0x111114,
            Pew => 0x3e2c1e,
            Altar => 0x3a3a44,
            Pillar => 0x4c4c58,
            Crate => 0x5a5236,
            Locker => 0x48523e,
            CoatRack => 0x3a2a1e,
        }
    }
}

/// Acoustic material class of a piece.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FurnMat {
    Stone,
    /// Upholstery, mattresses, cushions.
    Soft,
    /// Thin tables, chairs, benches.
    Wood,
    /// Solid casework: cabinets, counters, the bar, the piano case.
    Cabinet,
    /// A packed bookshelf.
    Books,
    /// Whiteboards, TVs.
    Panel,
}

impl FurnMat {
    pub const fn name(self) -> &'static str {
        match self {
            FurnMat::Stone => "stone",
            FurnMat::Soft => "soft",
            FurnMat::Wood => "wood",
            FurnMat::Cabinet => "cabinet",
            FurnMat::Books => "books",
            FurnMat::Panel => "panel",
        }
    }

    /// Full acoustic identity: transmission is the per-band amplitude
    /// THROUGH the piece (soft things pass muffled bass, stone passes
    /// nothing); absorption/scattering are what the late tracer bounces
    /// with and the early solver reflects with — a sofa eats the tail,
    /// a bookshelf diffuses it, a table top slaps back.
    pub const fn material(self) -> Material {
        match self {
            FurnMat::Stone => Material {
                absorption: [0.02, 0.03, 0.04],
                scattering: 0.30,
                transmission: [0.02, 0.002, 0.0],
            },
            FurnMat::Soft => Material {
                absorption: [0.35, 0.60, 0.70],
                scattering: 0.65,
                transmission: [0.5, 0.25, 0.1],
            },
            FurnMat::Wood => Material {
                absorption: [0.10, 0.10, 0.12],
                scattering: 0.40,
                transmission: [0.5, 0.3, 0.15],
            },
            FurnMat::Cabinet => Material {
                absorption: [0.12, 0.14, 0.18],
                scattering: 0.35,
                transmission: [0.3, 0.1, 0.03],
            },
            FurnMat::Books => Material {
                absorption: [0.25, 0.42, 0.55],
                scattering: 0.80,
                transmission: [0.15, 0.04, 0.01],
            },
            FurnMat::Panel => Material {
                absorption: [0.18, 0.10, 0.06],
                scattering: 0.15,
                transmission: [0.4, 0.2, 0.08],
            },
        }
    }
}

/// One piece of furniture, ROOM-LOCAL.
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub kind: Kind,
    /// Footprint centre in room-local plan coordinates (m).
    pub c: (f32, f32),
    /// Own-frame size (m): width (side to side), depth (back to front),
    /// height.
    pub size: (f32, f32, f32),
    /// Which way the piece faces (its front: where you sit, the screen,
    /// the keyboard, a bed's foot), degrees counter-clockwise from +y:
    /// 0 = +y (north), 90 = −x (west), 180 = −y (south), 270 = +x
    /// (east). Multiples of 90 only — the tracer's boxes are axis-aligned.
    pub yaw: i32,
    /// Bottom above the room's floor (wall-mounted panels, the DJ deck
    /// on the stage); 0 for anything standing on the floor.
    pub z0: f32,
    pub mat: FurnMat,
}

impl Piece {
    /// Unit plan vector the piece faces.
    pub fn facing(&self) -> (f32, f32) {
        match self.yaw.rem_euclid(360) {
            0 => (0.0, 1.0),
            90 => (-1.0, 0.0),
            180 => (0.0, -1.0),
            _ => (1.0, 0.0),
        }
    }

    /// Half extents along x and y.
    pub fn half_xy(&self) -> (f32, f32) {
        if self.yaw.rem_euclid(180) == 0 {
            (0.5 * self.size.0, 0.5 * self.size.1)
        } else {
            (0.5 * self.size.1, 0.5 * self.size.0)
        }
    }

    /// Room-local box (min, max), z from the room's floor.
    pub fn local_box(&self) -> ([f32; 3], [f32; 3]) {
        let (hx, hy) = self.half_xy();
        (
            [self.c.0 - hx, self.c.1 - hy, self.z0],
            [self.c.0 + hx, self.c.1 + hy, self.z0 + self.size.2],
        )
    }

    /// The tracer's box (room-local) with the material's transmission.
    pub fn aabb(&self) -> omg_core::pt::Aabb {
        let (a, b) = self.local_box();
        omg_core::pt::Aabb {
            min: Vec3::new(a[0], a[1], a[2]),
            max: Vec3::new(b[0], b[1], b[2]),
            transmission: self.mat.material().transmission,
        }
    }
}

/// A floor-standing piece: kind, centre (x, y), size (w, d, h), yaw, material.
#[allow(clippy::too_many_arguments)]
const fn pc(kind: Kind, x: f32, y: f32, w: f32, d: f32, h: f32, yaw: i32, mat: FurnMat) -> Piece {
    Piece { kind, c: (x, y), size: (w, d, h), yaw, z0: 0.0, mat }
}

/// A raised piece (wall-mounted, or standing on something): as `pc`,
/// its bottom `z0` above the floor.
#[allow(clippy::too_many_arguments)]
const fn up(
    kind: Kind, x: f32, y: f32, w: f32, d: f32, h: f32, yaw: i32, z0: f32, mat: FurnMat,
) -> Piece {
    Piece { kind, c: (x, y), size: (w, d, h), yaw, z0, mat }
}

// Facings for `Piece::yaw`.
const FACE_N: i32 = 0;
const FACE_W: i32 = 90;
const FACE_S: i32 = 180;
const FACE_E: i32 = 270;

/// Number of entries in `rooms()`.
pub const ROOM_COUNT: usize = OUTSIDE + 1;

/// The interior layout of `room`, ROOM-LOCAL — one table per room.
/// Wall-hugging pieces keep 8 cm off the wall plane (the frontends draw
/// 15 cm walls centred on it).
pub fn layout(room: usize) -> &'static [Piece] {
    use FurnMat::*;
    use Kind::*;
    const N: i32 = FACE_N;
    const W: i32 = FACE_W;
    const S: i32 = FACE_S;
    const E: i32 = FACE_E;
    // Living Room (8 × 6 × 2.7): a music room in two zones split by a
    // ceiling-high double-sided bookcase standing out from the south
    // wall. WEST: the grand piano free-standing in the middle of its
    // half (the music source rises from its soundboard), keyboard and
    // bench to the south so the pianist faces north with the open
    // curved side toward the listeners; the music library against the
    // west wall behind it; a conversation group facing the piano from
    // the east — sofa backed onto the bookcase, coffee table on a rug,
    // an armchair closing the group from the south with a reading lamp;
    // a sideboard against the north wall beside the door. EAST: a
    // reading nook behind the bookcase (armchair facing the books, side
    // table, lamp) and a desk against the east wall. Stand in the nook
    // and the piano goes into the bookcase's shadow.
    static LIVING_L: &[Piece] = &[
        pc(GrandPiano, 2.0, 3.0, 1.5, 1.9, 1.0, S, Cabinet),
        pc(PianoBench, 2.0, 1.6, 0.8, 0.36, 0.5, N, Wood),
        pc(Bookshelf, 0.255, 3.0, 2.0, 0.35, 2.0, E, Books),
        pc(Sofa, 4.9, 3.0, 2.1, 0.9, 0.85, W, Soft),
        pc(CoffeeTable, 3.85, 3.0, 1.0, 0.5, 0.42, W, Wood),
        pc(Rug, 3.9, 3.0, 2.6, 2.0, 0.01, W, Soft),
        pc(Armchair, 3.85, 1.45, 0.85, 0.85, 0.9, N, Soft),
        pc(FloorLamp, 4.75, 1.0, 0.3, 0.3, 1.6, N, Wood),
        pc(Sideboard, 1.6, 5.695, 1.8, 0.45, 0.8, S, Cabinet),
        pc(Divider, 5.6, 2.09, 4.02, 0.4, 2.2, W, Books),
        pc(Armchair, 7.475, 1.55, 0.85, 0.85, 0.9, W, Soft),
        pc(SideTable, 7.5, 0.7, 0.45, 0.45, 0.55, W, Wood),
        pc(FloorLamp, 7.6, 2.35, 0.3, 0.3, 1.6, W, Wood),
        pc(Desk, 7.61, 4.95, 1.4, 0.62, 0.75, W, Wood),
        pc(Chair, 6.95, 4.95, 0.45, 0.45, 0.9, E, Wood),
        up(TableLamp, 7.65, 5.3, 0.2, 0.2, 0.4, W, 0.75, Wood),
    ];
    // Great Hall (14 × 10 × 7) as a seminar room. The front is the east
    // wall: the whiteboard on it with a flip chart beside it, the
    // presenter's desk free-standing in front of the wall (she sits
    // behind it facing the room), and the narrator behind a lectern at
    // local (10.5, 6.5) facing west. The audience sits in three rows of
    // table + seat row (front row fixed, then 2.0 and 1.95 m pitch — the
    // back row's seats end at x = 3.0: 5 cm further west put a 15 dB
    // step into the walk past the north door outside) facing her, in two
    // blocks either side of a central aisle on her axis (y = 6.5); a
    // refreshment sideboard by the south wall; both doors open onto
    // free floor.
    static HALL_L: &[Piece] = &[
        pc(Table, 7.8, 5.0, 1.6, 0.6, 0.74, E, Wood),
        pc(Table, 7.8, 8.0, 1.6, 0.6, 0.74, E, Wood),
        pc(Table, 5.8, 5.0, 1.6, 0.6, 0.74, E, Wood),
        pc(Table, 5.8, 8.0, 1.6, 0.6, 0.74, E, Wood),
        pc(Table, 3.85, 5.0, 1.6, 0.6, 0.74, E, Wood),
        pc(Table, 3.85, 8.0, 1.6, 0.6, 0.74, E, Wood),
        pc(SeatRow, 7.175, 5.0, 1.6, 0.45, 0.85, E, Wood),
        pc(SeatRow, 7.175, 8.0, 1.6, 0.45, 0.85, E, Wood),
        pc(SeatRow, 5.175, 5.0, 1.6, 0.45, 0.85, E, Wood),
        pc(SeatRow, 5.175, 8.0, 1.6, 0.45, 0.85, E, Wood),
        pc(SeatRow, 3.225, 5.0, 1.6, 0.45, 0.85, E, Wood),
        pc(SeatRow, 3.225, 8.0, 1.6, 0.45, 0.85, E, Wood),
        pc(Lectern, 9.75, 6.5, 0.6, 0.5, 1.15, W, Cabinet),
        up(Whiteboard, 13.86, 6.5, 3.0, 0.12, 1.2, W, 0.9, Panel),
        pc(Whiteboard, 13.72, 9.2, 0.7, 0.4, 1.7, W, Panel),
        pc(Desk, 12.9, 3.0, 1.6, 0.72, 0.75, E, Wood),
        pc(Chair, 13.5, 3.0, 0.45, 0.45, 0.9, W, Wood),
        pc(Sideboard, 9.5, 0.33, 1.8, 0.5, 0.9, N, Cabinet),
    ];
    // Entrance (2 × 6): the club's sound-lock — a bench at the south
    // end, coats on a wall rack at the north end, the door axis free
    // but for a doormat between the two doors.
    static ENTRANCE_L: &[Piece] = &[
        pc(Bench, 1.0, 0.29, 1.4, 0.42, 0.45, N, Wood),
        pc(CoatRack, 1.0, 5.77, 1.0, 0.3, 1.8, S, Wood),
        pc(Rug, 1.0, 3.0, 1.0, 0.6, 0.01, W, Soft),
    ];
    // Club (10 × 12 × 4.5). The stage along the south wall with the DJ
    // deck on it, flanked by the two main PA stacks; two fill stacks in
    // the north corners (the club source's four emitters stand on these
    // sub cabinets). The bar runs along the west wall north of the door
    // (back bar against the wall, stools facing it); two booths against
    // the east wall either side of the window; a high table by the
    // south-west speaker; the dance floor in the middle stays clear.
    static CLUB_L: &[Piece] = &[
        pc(Stage, 5.0, 1.19, 5.2, 2.22, 0.5, N, Cabinet),
        up(DjBooth, 5.0, 1.25, 1.6, 0.6, 1.0, N, 0.5, Cabinet),
        pc(Speaker, 1.5, 1.5, 0.7, 0.7, 1.2, N, Cabinet),
        pc(Speaker, 8.5, 1.5, 0.7, 0.7, 1.2, N, Cabinet),
        pc(Speaker, 1.5, 10.5, 0.7, 0.7, 1.2, S, Cabinet),
        pc(Speaker, 8.5, 10.5, 0.7, 0.7, 1.2, S, Cabinet),
        pc(BackBar, 0.29, 8.0, 2.8, 0.42, 2.0, E, Cabinet),
        pc(Bar, 1.5, 8.0, 2.8, 0.6, 1.1, E, Cabinet),
        pc(Stool, 2.2, 7.0, 0.4, 0.4, 0.75, W, Wood),
        pc(Stool, 2.2, 7.7, 0.4, 0.4, 0.75, W, Wood),
        pc(Stool, 2.2, 8.4, 0.4, 0.4, 0.75, W, Wood),
        pc(Stool, 2.2, 9.1, 0.4, 0.4, 0.75, W, Wood),
        pc(Table, 1.3, 3.2, 0.6, 0.6, 1.05, N, Wood),
        pc(Booth, 9.26, 2.65, 1.32, 0.5, 1.1, N, Soft),
        pc(Table, 9.36, 3.25, 1.12, 0.6, 0.75, N, Wood),
        pc(Booth, 9.26, 3.85, 1.32, 0.5, 1.1, S, Soft),
        pc(Booth, 9.26, 8.25, 1.32, 0.5, 1.1, N, Soft),
        pc(Table, 9.36, 8.85, 1.12, 0.6, 0.75, N, Wood),
        pc(Booth, 9.26, 9.45, 1.32, 0.5, 1.1, S, Soft),
    ];
    // Old House ground floor (7 × 7 × 2.8), open plan. The stairs run up
    // the west wall (kept clear). Kitchen along the south wall (counter,
    // fridge in the corner), the dining table in front of it turned
    // east–west, two chairs either side; the living corner in the north-east: sofa facing a TV on
    // the east wall across a coffee table and rug.
    static HOUSE_L: &[Piece] = &[
        pc(KitchenCounter, 4.8, 0.38, 2.8, 0.6, 0.9, N, Cabinet),
        pc(Fridge, 6.57, 0.42, 0.7, 0.68, 1.9, N, Cabinet),
        pc(Table, 5.1, 2.4, 1.4, 0.8, 0.75, W, Wood),
        pc(Chair, 4.3, 2.05, 0.42, 0.42, 0.9, E, Wood),
        pc(Chair, 4.3, 2.75, 0.42, 0.42, 0.9, E, Wood),
        pc(Chair, 5.9, 2.05, 0.42, 0.42, 0.9, W, Wood),
        pc(Chair, 5.9, 2.75, 0.42, 0.42, 0.9, W, Wood),
        pc(Sofa, 3.85, 5.1, 2.1, 0.9, 0.85, E, Soft),
        pc(CoffeeTable, 5.0, 5.1, 1.0, 0.55, 0.42, E, Wood),
        pc(Rug, 5.0, 5.1, 2.4, 2.4, 0.01, E, Soft),
        pc(TvStand, 6.695, 5.1, 1.6, 0.45, 0.5, W, Cabinet),
        up(Tv, 6.86, 5.1, 1.3, 0.1, 0.75, W, 0.95, Panel),
        pc(FloorLamp, 3.6, 6.45, 0.3, 0.3, 1.6, E, Wood),
    ];
    // Old House upper floor (7 × 7 × 2.6; the stairwell opens along the
    // west wall, kept clear): a bedroom. The bed's headboard against the
    // south wall with a nightstand (and lamp) each side, a dresser beside
    // them, the
    // wardrobe on the east wall, a writing desk under the north window
    // with its chair, a rug at the foot of the bed.
    static HOUSE_UP_L: &[Piece] = &[
        pc(Bed, 4.4, 1.13, 1.6, 2.1, 0.9, N, Soft),
        pc(Nightstand, 3.3, 0.28, 0.45, 0.4, 0.55, N, Wood),
        pc(Nightstand, 5.5, 0.28, 0.45, 0.4, 0.55, N, Wood),
        up(TableLamp, 3.3, 0.28, 0.2, 0.2, 0.35, N, 0.55, Wood),
        up(TableLamp, 5.5, 0.28, 0.2, 0.2, 0.35, N, 0.55, Wood),
        pc(Dresser, 2.425, 0.33, 0.9, 0.5, 0.9, N, Cabinet),
        pc(Wardrobe, 6.62, 3.6, 1.8, 0.6, 2.1, W, Cabinet),
        pc(Desk, 5.3, 6.62, 1.3, 0.6, 0.75, S, Wood),
        pc(Chair, 5.3, 5.95, 0.45, 0.45, 0.9, N, Wood),
        pc(Rug, 4.4, 2.8, 2.0, 1.0, 0.01, N, Soft),
    ];
    // Cathedral nave (16 × 22 × 15). Two arcades of six stone columns
    // (x = 4 and 12); pews in two blocks either side of the central
    // aisle on the portal's axis, all facing the altar in the apse at the
    // north end between two candle stands; a font inside the portal; the
    // flautist plays in the open chancel between the front pews and the
    // altar.
    static CATHEDRAL_L: &[Piece] = &[
        pc(Pillar, 4.0, 3.0, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 4.0, 6.2, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 4.0, 9.4, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 4.0, 12.6, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 4.0, 15.8, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 4.0, 19.0, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 12.0, 3.0, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 12.0, 6.2, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 12.0, 9.4, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 12.0, 12.6, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 12.0, 15.8, 0.9, 0.9, 8.5, N, Stone),
        pc(Pillar, 12.0, 19.0, 0.9, 0.9, 8.5, N, Stone),
        pc(Pew, 6.05, 5.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 9.95, 5.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 6.05, 6.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 9.95, 6.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 6.05, 7.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 9.95, 7.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 6.05, 8.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 9.95, 8.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 6.05, 9.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Pew, 9.95, 9.5, 2.3, 0.6, 0.95, N, Wood),
        pc(Altar, 8.0, 19.0, 2.4, 1.2, 1.1, S, Stone),
        pc(CandleStand, 6.2, 18.2, 0.3, 0.3, 1.2, S, Wood),
        pc(CandleStand, 9.8, 18.2, 0.3, 0.3, 1.2, S, Wood),
        pc(Font, 6.3, 2.5, 0.6, 0.6, 0.9, N, Stone),
    ];
    // Bunker (6 × 8 × 2.2, below grade): a shelter. The radio stands on
    // a table near the north end with a chair either side; two bunks
    // along the east wall, supply shelving on the north wall with a water
    // barrel beside it, two crates stacked in the north-east corner, a
    // locker by the west wall; the hatch from
    // the stair (south-west) opens onto free floor.
    static BUNKER_L: &[Piece] = &[
        pc(Table, 3.5, 6.5, 1.2, 0.8, 0.75, N, Wood),
        pc(Chair, 2.6, 6.5, 0.42, 0.42, 0.9, E, Wood),
        pc(Chair, 4.4, 6.5, 0.42, 0.42, 0.9, W, Wood),
        pc(BunkBed, 5.5, 2.6, 2.0, 0.84, 1.6, W, Soft),
        pc(BunkBed, 5.5, 4.7, 2.0, 0.84, 1.6, W, Soft),
        pc(Shelf, 1.4, 7.67, 1.8, 0.5, 1.9, S, Cabinet),
        pc(Crate, 5.4, 7.45, 0.9, 0.9, 0.8, S, Wood),
        up(Crate, 5.4, 7.45, 0.7, 0.7, 0.6, S, 0.8, Wood),
        pc(Barrel, 2.6, 7.55, 0.5, 0.5, 0.9, N, Cabinet),
        pc(Locker, 0.33, 3.6, 1.2, 0.5, 1.9, E, Cabinet),
    ];
    match room {
        LIVING => LIVING_L,
        HALL => HALL_L,
        ENTRANCE => ENTRANCE_L,
        CLUB => CLUB_L,
        HOUSE => HOUSE_L,
        HOUSE_UP => HOUSE_UP_L,
        CATHEDRAL => CATHEDRAL_L,
        BUNKER => BUNKER_L,
        _ => &[],
    }
}

/// A floor region of a room that furniture must leave free
/// (ROOM-LOCAL): stairs and stairwells, the dance floor, the chancel.
#[derive(Clone, Copy, Debug)]
pub struct ClearZone {
    pub name: &'static str,
    pub min: (f32, f32),
    pub max: (f32, f32),
}

pub fn keep_clear(room: usize) -> &'static [ClearZone] {
    const fn z(name: &'static str, x0: f32, y0: f32, x1: f32, y1: f32) -> ClearZone {
        ClearZone { name, min: (x0, y0), max: (x1, y1) }
    }
    static LIVING_Z: &[ClearZone] = &[z("pianist", 1.5, 0.9, 2.5, 1.4)];
    static HOUSE_Z: &[ClearZone] = &[z("stairs", 0.0, 0.0, 1.75, 7.0)];
    static HOUSE_UP_Z: &[ClearZone] = &[z("stairwell", 0.0, 0.0, 1.75, 7.0)];
    static CLUB_Z: &[ClearZone] = &[z("dance floor", 2.8, 3.0, 7.6, 9.0)];
    static CATHEDRAL_Z: &[ClearZone] = &[z("chancel", 5.0, 11.0, 11.0, 15.5)];
    match room {
        LIVING => LIVING_Z,
        HOUSE => HOUSE_Z,
        HOUSE_UP => HOUSE_UP_Z,
        CLUB => CLUB_Z,
        CATHEDRAL => CATHEDRAL_Z,
        _ => &[],
    }
}

/// Named floor spots (ROOM-LOCAL) a walker must be able to reach from
/// every door: the room's open centre ("hub") and room-specific ones
/// (the foot of the stairs, the bar). Handy as spawn / camera points.
pub fn spots(room: usize) -> &'static [(&'static str, (f32, f32))] {
    match room {
        LIVING => &[("hub", (3.2, 4.7)), ("nook", (6.4, 1.6))],
        CORRIDOR => &[("hub", (0.8, 4.0))],
        HALL => &[("hub", (9.0, 3.0)), ("aisle", (5.0, 6.5))],
        ENTRANCE => &[("hub", (1.0, 3.0))],
        CLUB => &[("hub", (5.0, 6.0)), ("bar", (3.0, 8.0))],
        HOUSE => &[("hub", (2.6, 3.5)), ("stair foot", (1.9, 1.0)), ("kitchen", (4.8, 1.15))],
        HOUSE_UP => &[("hub", (3.5, 4.2)), ("landing", (1.2, 5.6))],
        CATHEDRAL => &[("hub", (8.0, 12.0))],
        BUNKER => &[("hub", (2.5, 4.0))],
        _ => &[],
    }
}

/// The tracer's furniture boxes (PT-early C5), ROOM-LOCAL, derived from
/// `layout` (acoustic pieces only). The traced early engine shadows
/// sources behind them (segment-validated chains + knife-edge bends);
/// the ISM backend cannot see them — that difference IS the demo.
pub fn furniture(room: usize) -> &'static [omg_core::pt::Aabb] {
    static T: std::sync::OnceLock<Vec<Vec<omg_core::pt::Aabb>>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        (0..ROOM_COUNT)
            .map(|r| layout(r).iter().filter(|p| p.kind.acoustic()).map(Piece::aabb).collect())
            .collect()
    })
    .get(room)
    .map_or(&[], |v| v.as_slice())
}

/// Pieces below this volume (m³) don't get specular reflector faces or
/// late-field panels — a chair shadows the early field (extras) but
/// doesn't bend a decay curve or slap back audibly, and every reflector
/// face costs chain-cache pressure. One constant, used by the early
/// overlay, the late panels, and both discovery kernels' box lists.
pub const FURN_REFLECTOR_MIN_VOL: f32 = 0.25;

/// Full acoustic identity of each furniture box, PARALLEL to
/// `furniture(room)` row for row.
pub fn furniture_mats(room: usize) -> &'static [Material] {
    static T: std::sync::OnceLock<Vec<Vec<Material>>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        (0..ROOM_COUNT)
            .map(|r| {
                layout(r).iter().filter(|p| p.kind.acoustic()).map(|p| p.mat.material()).collect()
            })
            .collect()
    })
    .get(room)
    .map_or(&[], |v| v.as_slice())
}

/// The interior layout as JSON (`assets/demo/layout.json`): plain
/// arrays/objects, meters, WORLD coordinates, z up (as in this file),
/// yaw in degrees. Shape:
///
/// ```text
/// { "version", "units", "up", "yaw_convention",
///   "rooms": [ { "index", "name", "min":[x,y], "max":[x,y], "floor_z",
///                "height", "keep_clear":[{ "name", "min", "max" }],
///                "spots":[{ "name", "pos":[x,y] }],
///                "furniture":[ { "kind", "pos":[cx,cy,z_bottom],
///                    "size":[w,d,h], "yaw", "facing":[fx,fy],
///                    "min":[x,y,z], "max":[x,y,z], "material",
///                    "acoustic", "color":"#rrggbb" } ] } ],
///   "sources": [ { "index", "name", "room", "pos":[x,y,z],
///                  "emitters":[[x,y], …] } ] }
/// ```
///
/// Rooms without furniture, keep-clear zones or spots are left out.
pub fn layout_json() -> String {
    fn n(v: f32) -> String {
        let mut s = format!("{:.3}", v);
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        if s == "-0" {
            s = "0".into();
        }
        s
    }
    fn arr(v: &[f32]) -> String {
        format!("[{}]", v.iter().map(|&x| n(x)).collect::<Vec<_>>().join(", "))
    }
    let rooms = rooms();
    let mut o = String::new();
    o.push_str("{\n");
    o.push_str("  \"version\": 1,\n");
    o.push_str("  \"generated_by\": \"omg-scene walkthrough::layout_json (cargo run -p omg-scene --example export_layout)\",\n");
    o.push_str("  \"units\": \"meters\",\n");
    o.push_str("  \"up\": \"z\",\n");
    o.push_str("  \"yaw_convention\": \"degrees counter-clockwise from +y seen from above: 0 = the piece's front faces +y, 90 = -x, 180 = -y, 270 = +x. size = [width, depth, height] in the piece's own frame; pos = [centre x, centre y, bottom z]; min/max = the world AABB\",\n");
    o.push_str("  \"rooms\": [\n");
    let listed: Vec<usize> = (0..rooms.len())
        .filter(|&i| !layout(i).is_empty() || !keep_clear(i).is_empty() || !spots(i).is_empty())
        .collect();
    for (k, &ri) in listed.iter().enumerate() {
        let r = &rooms[ri];
        let (ox, oy, oz) = (r.min.0, r.min.1, r.floor_z);
        o.push_str("    {\n");
        o.push_str(&format!("      \"index\": {ri},\n"));
        o.push_str(&format!("      \"name\": \"{}\",\n", r.name));
        o.push_str(&format!("      \"min\": {},\n", arr(&[r.min.0, r.min.1])));
        o.push_str(&format!("      \"max\": {},\n", arr(&[r.max.0, r.max.1])));
        o.push_str(&format!("      \"floor_z\": {},\n", n(r.floor_z)));
        o.push_str(&format!("      \"height\": {},\n", n(r.height)));
        let zones: Vec<String> = keep_clear(ri)
            .iter()
            .map(|z| {
                format!(
                    "{{ \"name\": \"{}\", \"min\": {}, \"max\": {} }}",
                    z.name,
                    arr(&[z.min.0 + ox, z.min.1 + oy]),
                    arr(&[z.max.0 + ox, z.max.1 + oy])
                )
            })
            .collect();
        o.push_str(&format!("      \"keep_clear\": [{}],\n", zones.join(", ")));
        let sp: Vec<String> = spots(ri)
            .iter()
            .map(|(name, p)| {
                format!("{{ \"name\": \"{}\", \"pos\": {} }}", name, arr(&[p.0 + ox, p.1 + oy]))
            })
            .collect();
        o.push_str(&format!("      \"spots\": [{}],\n", sp.join(", ")));
        o.push_str("      \"furniture\": [");
        let pieces = layout(ri);
        for (j, p) in pieces.iter().enumerate() {
            let (a, b) = p.local_box();
            let f = p.facing();
            o.push_str(if j == 0 { "\n" } else { ",\n" });
            o.push_str(&format!(
                "        {{ \"kind\": \"{}\", \"pos\": {}, \"size\": {}, \"yaw\": {}, \"facing\": {}, \"min\": {}, \"max\": {}, \"material\": \"{}\", \"acoustic\": {}, \"color\": \"#{:06x}\" }}",
                p.kind.name(),
                arr(&[p.c.0 + ox, p.c.1 + oy, p.z0 + oz]),
                arr(&[p.size.0, p.size.1, p.size.2]),
                p.yaw.rem_euclid(360),
                arr(&[f.0, f.1]),
                arr(&[a[0] + ox, a[1] + oy, a[2] + oz]),
                arr(&[b[0] + ox, b[1] + oy, b[2] + oz]),
                p.mat.name(),
                p.kind.acoustic(),
                p.kind.color(),
            ));
        }
        o.push_str(if pieces.is_empty() { "]\n" } else { "\n      ]\n" });
        o.push_str(if k + 1 == listed.len() { "    }\n" } else { "    },\n" });
    }
    o.push_str("  ],\n");
    o.push_str(&format!(
        "  \"canyon\": {{ \"height\": {}, \"material\": \"rock\", \"cliffs\": [{}] }},\n",
        n(CANYON.height),
        [CANYON.west, CANYON.east]
            .iter()
            .map(|(a, b)| format!("{{ \"min\": {}, \"max\": {} }}", arr(&[a.0, a.1]), arr(&[b.0, b.1])))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    o.push_str(&format!(
        "  \"radome\": {{ \"center\": {}, \"radius\": {}, \"center_z\": {}, \"door_w\": {}, \"door_h\": {}, \"stage_r\": {}, \"stage_h\": {}, \"subdiv\": {}, \"material\": \"radome\" }},\n",
        arr(&[RADOME.center.0, RADOME.center.1]),
        n(RADOME.radius),
        n(RADOME.center_z),
        n(RADOME.door_w),
        n(RADOME.door_h),
        n(RADOME.stage_r),
        n(RADOME.stage_h),
        RADOME.subdiv
    ));
    o.push_str("  \"ground\": [\n");
    let out = rooms.iter().find(|r| r.outdoor).expect("an outdoor room");
    for (i, g) in GROUND.iter().enumerate() {
        o.push_str(&format!(
            "    {{ \"name\": \"{}\", \"surface\": \"{}\", \"min\": {}, \"max\": {} }}{}\n",
            g.name,
            g.surface,
            arr(&[g.min.0, g.min.1.max(out.min.1)]),
            arr(&[g.max.0, g.max.1.min(out.max.1)]),
            if i + 1 == GROUND.len() { "" } else { "," }
        ));
    }
    o.push_str("  ],\n");
    o.push_str("  \"sources\": [\n");
    let srcs = sources();
    let statics: Vec<usize> = (0..srcs.len()).filter(|&i| dyn_slot(i).is_none()).collect();
    for (k, &i) in statics.iter().enumerate() {
        let s = &srcs[i];
        let em: Vec<String> = s.emitters.iter().map(|e| arr(&[e.0, e.1])).collect();
        o.push_str(&format!(
            "    {{ \"index\": {i}, \"name\": \"{}\", \"room\": {}, \"pos\": {}, \"emitters\": [{}] }}{}\n",
            s.name,
            s.room,
            arr(&[s.pos.0, s.pos.1, s.z]),
            em.join(", "),
            if k + 1 == statics.len() { "" } else { "," }
        ));
    }
    o.push_str("  ]\n}\n");
    o
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
    (33.0, 9.0, 17.5),
    (39.0, 9.2, 19.6),
    (43.0, 9.3, 20.0),
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

/// Is there a non-glass door path between rooms `a` and `b`?
fn connected(a: usize, b: usize, all_doors: &[Door]) -> bool {
    let n = graph_size(a, b, all_doors);
    let mut seen = vec![false; n];
    let mut stack = vec![a];
    seen[a] = true;
    while let Some(r) = stack.pop() {
        if r == b {
            return true;
        }
        for d in all_doors.iter().filter(|d| !d.glass) {
            let next = if d.rooms.0 == r { d.rooms.1 } else if d.rooms.1 == r { d.rooms.0 } else { continue };
            if !seen[next] {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

pub fn route_source(
    src: &SourceDef,
    lis_room: usize,
    lis: (f32, f32),
    all_doors: &[Door],
) -> Routed {
    // Same room — or no door path at all (a listener inside a doorless
    // solid structure: the colonnade, the kiosk, the belfry stage; the
    // Godot demo lets you walk in): the straight line, whose walls the
    // straight-path pricing charges. Never a panic on the audio path.
    if src.room == lis_room || !connected(src.room, lis_room, all_doors) {
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

/// The street's asphalt strip (x range): cars pass on it (its ground
/// patch is in `GROUND`).
pub const STREET_X: (f32, f32) = (-20.2, -15.8);

/// Outdoor ground that is not grass: the paved square and paths, the
/// street and its kerbs. Each patch is its own ground surface with its
/// own material (`dome::build_world_mesh`), keeps the crickets out
/// (`ambient_emitters`), picks the footstep sound, and is drawn by both
/// demos from layout.json. Patches never overlap each other; the outdoor
/// rect clips them, and a building standing on one simply covers it.
pub struct GroundPatch {
    pub name: &'static str,
    /// Footstep / material name: "paving", "asphalt", "stone" or
    /// "concrete".
    pub surface: &'static str,
    pub min: (f32, f32),
    pub max: (f32, f32),
}

impl GroundPatch {
    pub fn material(&self) -> Material {
        match self.surface {
            "asphalt" => Material::ASPHALT,
            "paving" => Material::PAVING,
            "stone" => Material::ROCK,
            _ => Material::CONCRETE,
        }
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.min.0 && x <= self.max.0 && y >= self.min.1 && y <= self.max.1
    }
}

const fn patch(name: &'static str, surface: &'static str, x0: f32, y0: f32, x1: f32, y1: f32) -> GroundPatch {
    GroundPatch { name, surface, min: (x0, y0), max: (x1, y1) }
}

/// The street and kerbs run the whole outdoor length (clipped to it).
pub const GROUND: &[GroundPatch] = &[
    patch("square", "paving", 5.5, 24.2, 21.5, 44.0),
    patch("lane to the old house", "paving", 21.5, 23.2, 34.5, 26.0),
    patch("lane, west end", "paving", 14.0, 23.2, 21.5, 24.2),
    patch("path to the cathedral", "paving", 6.0, 44.0, 10.0, 51.8),
    patch("path to the street", "paving", -15.0, 26.0, 5.5, 29.0),
    patch("bunker path", "paving", 24.0, 6.0, 31.4, 8.0),
    patch("bunker lane", "paving", 34.5, 14.0, 37.0, 23.2),
    patch("belfry path", "paving", 8.6, 74.4, 11.4, 316.0),
    patch("canyon floor", "stone", -6.0, -120.0, 34.0, -62.0),
    patch("canyon track", "stone", -15.0, -66.0, -6.0, -62.0),
    patch("street", "asphalt", STREET_X.0, -1e4, STREET_X.1, 1e4),
    patch("west kerb", "concrete", -21.0, -1e4, STREET_X.0, 1e4),
    patch("east kerb", "concrete", STREET_X.1, -1e4, -15.0, 1e4),
];

/// The outdoor ground's surface at (x, y): a patch's, else "grass".
pub fn ground_surface(x: f32, y: f32) -> &'static str {
    GROUND.iter().find(|g| g.contains(x, y)).map_or("grass", |g| g.surface)
}

/// Where the ambience recording was "made": ear height on open grass,
/// south of the town. The shaped emitters are calibrated so their free
/// field here equals the open dome the bed used to play everywhere.
pub const AMBIENT_REFERENCE: (f32, f32, f32) = (10.0, -150.0, 1.6);

/// Height of the cricket layer above the ground, m.
pub const CRICKET_Z: f32 = 0.1;

/// The demo's ambience bed as shaped emitters: crickets in the grass —
/// the outdoor ground (GRASS) minus every building footprint standing on
/// it and every paved patch (`GROUND`). Fly up and they come from below; walk off
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
    for g in GROUND {
        let (y0, y1) = (g.min.1.max(out.min.1), g.max.1.min(out.max.1));
        add((g.min.0, g.max.0, y0, y1), &mut holes);
    }
    vec![AmbientEmitter {
        shape: AmbientShape::GroundArea { min: out.min, max: out.max, z: CRICKET_Z, holes },
        level: 1.0,
    }]
}
