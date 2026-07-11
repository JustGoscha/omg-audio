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
        },
        // Outside must come LAST: enclosed rooms overlap its rectangle and
        // room_of matches in order. It surrounds all buildings.
        RoomDef {
            name: "Outside",
            min: (-8.0, -8.0),
            max: (42.0, 46.0),
            height: 30.0,
            barrier_height: 30.0,
            floor_z: 0.0,
            walls: [Material::GRASS; 6],
            wall_thickness: 0.15,
            outdoor: true,
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
pub const OUTSIDE: usize = 9;

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
    /// open slit and the wood panel, so opening a door sweeps the sound
    /// continuously instead of snapping it (see `fill_energy`).
    pub openness: f32,
}

impl Door {
    /// Per-band ENERGY transmission of whatever fills this aperture:
    /// glass panes transmit as glass; a door passes the open fraction of
    /// its area freely and the rest through the panel (incoherent
    /// sub-apertures — powers add).
    pub fn fill_energy(&self) -> [f32; NBANDS] {
        if self.glass {
            core::array::from_fn(|b| GLASS_TRANSMISSION[b] * GLASS_TRANSMISSION[b])
        } else {
            let o = self.openness.clamp(0.0, 1.0);
            core::array::from_fn(|b| {
                o + (1.0 - o) * DOOR_PANEL_TRANSMISSION[b] * DOOR_PANEL_TRANSMISSION[b]
            })
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
        Door { rooms: (LIVING, CORRIDOR), pos: (4.0, 6.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0 },
        Door { rooms: (CORRIDOR, HALL), pos: (4.0, 14.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0 },
        Door { rooms: (HALL, OUTSIDE), pos: (7.0, 24.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0 },
        Door { rooms: (OUTSIDE, ENTRANCE), pos: (20.0, 31.0), axis: 0, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0 },
        Door { rooms: (ENTRANCE, CLUB), pos: (22.0, 31.0), axis: 0, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0 },
        Door { rooms: (OUTSIDE, HOUSE), pos: (26.5, 23.0), axis: 1, half: 0.55, height: 2.0, zc: 1.0, glass: false, openness: 1.0 },
        // windows
        Door { rooms: (LIVING, OUTSIDE), pos: (3.0, 0.0), axis: 1, half: 1.3, height: 1.4, zc: 1.5, glass: true, openness: 1.0 },
        Door { rooms: (CLUB, OUTSIDE), pos: (32.0, 32.0), axis: 0, half: 1.8, height: 1.4, zc: 1.5, glass: true, openness: 1.0 },
        Door { rooms: (CLUB, OUTSIDE), pos: (26.0, 38.0), axis: 1, half: 1.8, height: 1.4, zc: 1.5, glass: true, openness: 1.0 },
        // house windows look out onto the square, club and hall
        Door { rooms: (HOUSE, OUTSIDE), pos: (29.3, 23.0), axis: 1, half: 1.4, height: 1.4, zc: 1.5, glass: true, openness: 1.0 },
        Door { rooms: (HOUSE, OUTSIDE), pos: (24.0, 19.5), axis: 0, half: 1.4, height: 1.4, zc: 1.5, glass: true, openness: 1.0 },
        // upper-storey windows onto the square and toward the club
        Door { rooms: (HOUSE_UP, OUTSIDE), pos: (29.3, 23.0), axis: 1, half: 1.4, height: 1.4, zc: 4.5, glass: true, openness: 1.0 },
        Door { rooms: (HOUSE_UP, OUTSIDE), pos: (24.0, 19.5), axis: 0, half: 1.4, height: 1.4, zc: 4.5, glass: true, openness: 1.0 },
        // stairwell: the vertical portal between the two storeys — always
        // open, never toggled (indices ≥ 6 are outside the E-key range)
        Door { rooms: (HOUSE, HOUSE_UP), pos: (24.8, 21.5), axis: 0, half: 0.7, height: 2.2, zc: 3.0, glass: false, openness: 1.0 },
    ]
}

pub const DOOR_HALF_WIDTH: f32 = 0.55;

/// Single-pane glass, per band (much leakier than any wall).
pub const GLASS_TRANSMISSION: [f32; NBANDS] = [0.50, 0.32, 0.20];

/// A closed door: ~4 cm wood panel filling the aperture (mass law).
pub const DOOR_PANEL_TRANSMISSION: [f32; NBANDS] = [0.28, 0.12, 0.05];

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

pub struct SourceDef {
    pub name: &'static str,
    /// Centroid: used for routing, transmission and cross-room rendering.
    pub pos: (f32, f32),
    pub room: usize,
    /// Linear gain applied to everything this source produces.
    pub gain: f32,
    /// Same-room emitter positions playing the identical, sample-locked
    /// signal (a speaker rig). Power is split across them.
    pub emitters: &'static [(f32, f32)],
}

/// Club PA: four corner speakers.
pub const CLUB_SPEAKERS: [(f32, f32); 4] =
    [(23.5, 27.5), (30.5, 27.5), (23.5, 36.5), (30.5, 36.5)];

pub fn sources() -> [SourceDef; 6] {
    [
        SourceDef {
            name: "music",
            pos: (2.0, 3.0),
            room: LIVING,
            gain: 1.0,
            emitters: &[(2.0, 3.0)],
        },
        SourceDef {
            name: "voice",
            pos: (10.5, 20.5),
            room: HALL,
            gain: 0.5,
            emitters: &[(10.5, 20.5)],
        },
        SourceDef {
            name: "club",
            pos: (27.0, 32.0),
            room: CLUB,
            // realistic PA level: far above speech/piano; the ear-adaptation
            // stage (acoustic reflex) is what keeps it listenable inside.
            gain: 5.0,
            emitters: &CLUB_SPEAKERS,
        },
        // Dynamic slots: thrown projectiles (positions set per tick).
        SourceDef { name: "ball0", pos: (0.0, 0.0), room: OUTSIDE, gain: 0.9, emitters: &[(0.0, 0.0)] },
        SourceDef { name: "ball1", pos: (0.0, 0.0), room: OUTSIDE, gain: 0.9, emitters: &[(0.0, 0.0)] },
        SourceDef { name: "ball2", pos: (0.0, 0.0), room: OUTSIDE, gain: 0.9, emitters: &[(0.0, 0.0)] },
    ]
}

pub const DYN_SLOTS: usize = 3;

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
            && z <= r.floor_z + r.height + 0.9
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
    let size = Vec3::new(r.max.0 - r.min.0, r.max.1 - r.min.1, r.height);
    let local_z = (lz - r.floor_z).clamp(0.2, r.height - 0.2);
    WalkState {
        listener_world: (lx, ly),
        yaw,
        room,
        shoebox: Shoebox::new(size, r.walls),
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
    /// The doorway this blend crosses (scene door index), when blending.
    pub door: Option<usize>,
}

pub fn blended_state_at(rooms: &[RoomDef], t: f32) -> BlendedState {
    let st = state_at(rooms, t);
    let (lx, ly) = st.listener_world;
    blended_state_for(rooms, lx, ly, EYE_HEIGHT, st.yaw)
}

/// Blend state for an arbitrary listener pose (interactive/web input).
/// `lz` is the EYE height in world coordinates (feet + 1.6).
pub fn blended_state_for(rooms: &[RoomDef], lx: f32, ly: f32, lz: f32, yaw: f32) -> BlendedState {
    let primary = state_in_room(rooms, room_of_z(rooms, lx, ly, lz), lx, ly, lz, yaw);
    let (lx, ly) = primary.listener_world;

    let all_doors = doors();
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
                    door: Some(door),
                };
            }
        }
    }
    BlendedState { primary, primary_weight: 1.0, other: None, door: None }
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
    /// Full apparent path (source → doors → listener), for visualization.
    pub route: Vec<(f32, f32)>,
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
            route: vec![src.pos, lis],
        };
    }

    // BFS over the door graph from the source's room to the listener's.
    let n_rooms = rooms().len();
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
    let n_rooms = rooms().len();
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
                muffle: if d.glass { GLASS_TRANSMISSION } else { ke },
                wet_trans: d.fill_amplitude(),
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
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    // (t, per-band amplitude) of each boundary crossing
    let mut crossings: Vec<(f32, [f32; NBANDS])> = Vec::new();

    for r in rooms.iter().filter(|r| !r.outdoor) {
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
            let along = if axis == 0 { a.1 + t * dy } else { a.0 + t * dx };
            if along < lo || along > hi {
                continue;
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
