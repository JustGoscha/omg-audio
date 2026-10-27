//! Layout gate for the demo town's interiors (`walkthrough::layout`).
//!
//! The furniture is authored once, in Rust; the tracer, the web demo and
//! the Godot demo all build from it. This gate keeps it physically and
//! functionally sane:
//! - `assets/demo/layout.json` is exactly what `layout_json()` produces
//!   (regenerate: `cargo run -p omg-scene --example export_layout`, or
//!   run this test with `OMG_WRITE_LAYOUT=1`);
//! - every piece stands inside its room, clear of the (15 cm, centred)
//!   wall slabs, below the ceiling, axis-aligned;
//! - no two pieces intersect (rugs may lie under anything);
//! - pieces that belong against a wall (beds by their headboard,
//!   bookshelves, sideboards, wardrobes, wall TVs …) touch one, with
//!   their back;
//! - doorways keep 0.9 m of free floor in front of them, windows are not
//!   blocked by anything taller than the sill, keep-clear zones (stairs,
//!   the dance floor, the chancel) stay empty;
//! - no sound source sits inside a piece;
//! - a walker 0.8 m wide (flood fill on a 0.25 m grid) gets from every
//!   door to the room's spots and to every source: within 1.3 m of it,
//!   or right up to the piece it stands on (the grand piano, a table).

use omg_scene::walkthrough::{self, Kind, Piece, WallRule};

const WALL_HALF: f32 = 0.075;
const WALL_TOUCH: f32 = 0.15;
const DOOR_DEPTH: f32 = 0.9;
const DOOR_SIDE: f32 = 0.2;
const CELL: f32 = 0.25;
/// Half the free width a walking path needs (0.8 m paths).
const WALK_R: f32 = 0.4;
const SOURCE_REACH: f32 = 1.3;
const SPOT_SNAP: f32 = 0.3;
const EPS: f32 = 1e-3;

type Rect = ((f32, f32), (f32, f32));

fn rect_of(p: &Piece) -> Rect {
    let (a, b) = p.local_box();
    ((a[0], a[1]), (b[0], b[1]))
}

fn rects_overlap(a: Rect, b: Rect) -> bool {
    a.0 .0 < b.1 .0 - EPS && b.0 .0 < a.1 .0 - EPS && a.0 .1 < b.1 .1 - EPS && b.0 .1 < a.1 .1 - EPS
}

fn dist_to_rect(p: (f32, f32), r: Rect) -> f32 {
    let dx = (r.0 .0 - p.0).max(0.0).max(p.0 - r.1 .0);
    let dy = (r.0 .1 - p.1).max(0.0).max(p.1 - r.1 .1);
    (dx * dx + dy * dy).sqrt()
}

fn furnished_rooms() -> Vec<usize> {
    (0..walkthrough::ROOM_COUNT).filter(|&r| !walkthrough::layout(r).is_empty()).collect()
}

#[test]
fn layout_json_is_fresh() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/demo/layout.json");
    let want = walkthrough::layout_json();
    if std::env::var_os("OMG_WRITE_LAYOUT").is_some() {
        std::fs::write(&path, &want).expect("write layout.json");
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        have == want,
        "demo/assets/demo/layout.json is stale — run `cargo run -p omg-scene --example export_layout`"
    );
}

/// The paved ground patches tile without overlapping (each is one
/// ground surface; overlaps would be coplanar double ground) and lie on
/// the outdoor ground.
#[test]
fn ground_patches_are_disjoint_and_outdoors() {
    let rooms = walkthrough::rooms();
    let out = rooms.iter().find(|r| r.outdoor).unwrap();
    let g = walkthrough::GROUND;
    for (i, a) in g.iter().enumerate() {
        assert!(
            a.min.0 >= out.min.0 && a.max.0 <= out.max.0 && a.min.0 < a.max.0 && a.min.1 < a.max.1,
            "{} lies off the outdoor ground",
            a.name
        );
        for b in &g[i + 1..] {
            let overlap = a.min.0 < b.max.0 - 1e-3
                && b.min.0 < a.max.0 - 1e-3
                && a.min.1 < b.max.1 - 1e-3
                && b.min.1 < a.max.1 - 1e-3;
            assert!(!overlap, "ground patches {} and {} overlap", a.name, b.name);
        }
    }
    assert_eq!(walkthrough::ground_surface(-18.0, 20.0), "asphalt");
    assert_eq!(walkthrough::ground_surface(12.0, 30.0), "paving");
    assert_eq!(walkthrough::ground_surface(-25.0, 20.0), "grass");
}

#[test]
fn acoustic_tables_derive_from_the_layout() {
    for room in 0..walkthrough::ROOM_COUNT {
        let pieces: Vec<&Piece> =
            walkthrough::layout(room).iter().filter(|p| p.kind.acoustic()).collect();
        let boxes = walkthrough::furniture(room);
        let mats = walkthrough::furniture_mats(room);
        assert_eq!(boxes.len(), pieces.len(), "room {room}");
        assert_eq!(mats.len(), pieces.len(), "room {room}");
        for ((b, p), m) in boxes.iter().zip(&pieces).zip(mats) {
            let (lo, hi) = p.local_box();
            assert_eq!([b.min.x, b.min.y, b.min.z], lo);
            assert_eq!([b.max.x, b.max.y, b.max.z], hi);
            assert_eq!(b.transmission, m.transmission);
        }
    }
}

#[test]
fn pieces_stand_inside_their_rooms_and_never_intersect() {
    let rooms = walkthrough::rooms();
    let mut errs = Vec::new();
    for ri in furnished_rooms() {
        let r = &rooms[ri];
        let (w, d) = (r.max.0 - r.min.0, r.max.1 - r.min.1);
        let pieces = walkthrough::layout(ri);
        for (i, p) in pieces.iter().enumerate() {
            let tag = format!("{} #{i} {}", r.name, p.kind.name());
            if p.yaw.rem_euclid(90) != 0 {
                errs.push(format!("{tag}: yaw {} is not a multiple of 90°", p.yaw));
            }
            if !(p.size.0 > 0.0 && p.size.1 > 0.0 && p.size.2 > 0.0 && p.z0 >= 0.0) {
                errs.push(format!("{tag}: degenerate size {:?} / z0 {}", p.size, p.z0));
            }
            let (a, b) = p.local_box();
            if a[0] < WALL_HALF - EPS
                || a[1] < WALL_HALF - EPS
                || b[0] > w - WALL_HALF + EPS
                || b[1] > d - WALL_HALF + EPS
            {
                errs.push(format!("{tag}: {a:?}..{b:?} cuts into a wall of the {w}×{d} room"));
            }
            if b[2] > r.height + EPS {
                errs.push(format!("{tag}: top {} above the {} m ceiling", b[2], r.height));
            }
            for (j, q) in pieces.iter().enumerate().skip(i + 1) {
                if p.kind.flat() != q.kind.flat() {
                    continue; // a rug under anything is fine
                }
                let (c, e) = q.local_box();
                let z_overlap = a[2] < e[2] - EPS && c[2] < b[2] - EPS;
                if z_overlap && rects_overlap(rect_of(p), rect_of(q)) {
                    errs.push(format!("{tag} intersects #{j} {}", q.kind.name()));
                }
            }
            for z in walkthrough::keep_clear(ri) {
                if rects_overlap(rect_of(p), (z.min, z.max)) {
                    errs.push(format!("{tag} stands in the keep-clear zone '{}'", z.name));
                }
            }
        }
    }
    assert!(errs.is_empty(), "layout errors:\n  {}", errs.join("\n  "));
}

#[test]
fn wall_pieces_stand_against_a_wall() {
    let rooms = walkthrough::rooms();
    let mut errs = Vec::new();
    for ri in furnished_rooms() {
        let r = &rooms[ri];
        let (w, d) = (r.max.0 - r.min.0, r.max.1 - r.min.1);
        for (i, p) in walkthrough::layout(ri).iter().enumerate() {
            let (a, b) = p.local_box();
            // gap to each wall: x-min, x-max, y-min, y-max
            let gaps = [a[0], w - b[0], a[1], d - b[1]];
            let ok = match p.kind.wall_rule() {
                WallRule::Free => true,
                WallRule::Any => gaps.iter().any(|&g| g <= WALL_TOUCH),
                WallRule::Back => {
                    let (fx, fy) = p.facing();
                    // the back is opposite the facing
                    let g = if fx > 0.5 {
                        gaps[0]
                    } else if fx < -0.5 {
                        gaps[1]
                    } else if fy > 0.5 {
                        gaps[2]
                    } else {
                        gaps[3]
                    };
                    g <= WALL_TOUCH
                }
            };
            if !ok {
                errs.push(format!(
                    "{} #{i} {} (facing {:?}) must stand {} a wall; gaps {gaps:?}",
                    r.name,
                    p.kind.name(),
                    p.facing(),
                    if p.kind.wall_rule() == WallRule::Back { "with its back to" } else { "against" },
                ));
            }
        }
    }
    assert!(errs.is_empty(), "wall rule errors:\n  {}", errs.join("\n  "));
}

/// Doors of `ri` a walker uses at this floor (not the stairwell portal
/// seen from below), with the inward normal when the door sits on a wall.
fn floor_doors(ri: usize) -> Vec<(walkthrough::Door, (f32, f32), Option<(f32, f32)>)> {
    let rooms = walkthrough::rooms();
    let r = &rooms[ri];
    walkthrough::doors()
        .into_iter()
        .filter(|dr| dr.rooms.0 == ri || dr.rooms.1 == ri)
        .filter(|dr| (dr.zc - (r.floor_z + 1.0)).abs() < 1.5 || dr.glass)
        .map(|dr| {
            let p = (dr.pos.0 - r.min.0, dr.pos.1 - r.min.1);
            let (w, d) = (r.max.0 - r.min.0, r.max.1 - r.min.1);
            let n = if dr.axis == 0 {
                if p.0.abs() < 0.05 {
                    Some((1.0, 0.0))
                } else if (p.0 - w).abs() < 0.05 {
                    Some((-1.0, 0.0))
                } else {
                    None
                }
            } else if p.1.abs() < 0.05 {
                Some((0.0, 1.0))
            } else if (p.1 - d).abs() < 0.05 {
                Some((0.0, -1.0))
            } else {
                None
            };
            (dr, p, n)
        })
        .collect()
}

/// The floor strip `depth` deep in front of an opening on a wall.
fn front_strip(p: (f32, f32), n: (f32, f32), half: f32, depth: f32, axis: usize) -> Rect {
    let (u0, u1) = if axis == 0 { (p.1 - half, p.1 + half) } else { (p.0 - half, p.0 + half) };
    if axis == 0 {
        let (x0, x1) = if n.0 > 0.0 { (p.0, p.0 + depth) } else { (p.0 - depth, p.0) };
        ((x0, u0), (x1, u1))
    } else {
        let (y0, y1) = if n.1 > 0.0 { (p.1, p.1 + depth) } else { (p.1 - depth, p.1) };
        ((u0, y0), (u1, y1))
    }
}

#[test]
fn doorways_and_windows_stay_clear() {
    let rooms = walkthrough::rooms();
    let mut errs = Vec::new();
    for ri in furnished_rooms() {
        let r = &rooms[ri];
        for (dr, p, n) in floor_doors(ri) {
            let Some(n) = n else { continue };
            if dr.glass {
                // nothing taller than the sill right in front of a window
                let sill = dr.zc - 0.5 * dr.height - r.floor_z;
                let strip = front_strip(p, n, dr.half, 0.5, dr.axis);
                for (i, q) in walkthrough::layout(ri).iter().enumerate() {
                    let top = q.z0 + q.size.2;
                    if top > sill + 0.02 && rects_overlap(rect_of(q), strip) {
                        errs.push(format!(
                            "{} #{i} {} ({top} m tall) blocks the window at {:?} (sill {sill} m)",
                            r.name,
                            q.kind.name(),
                            dr.pos
                        ));
                    }
                }
            } else {
                let strip = front_strip(p, n, dr.half + DOOR_SIDE, DOOR_DEPTH, dr.axis);
                for (i, q) in walkthrough::layout(ri).iter().enumerate() {
                    if !q.kind.flat() && rects_overlap(rect_of(q), strip) {
                        errs.push(format!(
                            "{} #{i} {} blocks the doorway at {:?}",
                            r.name,
                            q.kind.name(),
                            dr.pos
                        ));
                    }
                }
            }
        }
    }
    assert!(errs.is_empty(), "doorway errors:\n  {}", errs.join("\n  "));
}

#[test]
fn no_source_sits_inside_furniture() {
    let rooms = walkthrough::rooms();
    let mut errs = Vec::new();
    let srcs = walkthrough::sources();
    for s in (0..srcs.len()).filter(|&i| walkthrough::dyn_slot(i).is_none()).map(|i| &srcs[i]) {
        for ri in 0..walkthrough::ROOM_COUNT {
            let r = &rooms[ri];
            for e in s.emitters {
                let l = [e.0 - r.min.0, e.1 - r.min.1, s.z - r.floor_z];
                for (i, p) in walkthrough::layout(ri).iter().enumerate() {
                    let (a, b) = p.local_box();
                    if (0..3).all(|k| l[k] > a[k] - 0.05 && l[k] < b[k] + 0.05) {
                        errs.push(format!(
                            "source {} emitter {e:?} is inside {} #{i} {}",
                            s.name,
                            r.name,
                            p.kind.name()
                        ));
                    }
                }
            }
        }
    }
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

/// Free-floor grid of a room for a walker needing `WALK_R` either side.
struct Grid {
    nx: usize,
    ny: usize,
    free: Vec<bool>,
}

impl Grid {
    fn new(ri: usize) -> Self {
        let rooms = walkthrough::rooms();
        let r = &rooms[ri];
        let (w, d) = (r.max.0 - r.min.0, r.max.1 - r.min.1);
        let (nx, ny) = ((w / CELL).floor() as usize, (d / CELL).floor() as usize);
        // what a walker bumps into: anything not flat between the ankles
        // and the shoulders
        let blockers: Vec<Rect> = walkthrough::layout(ri)
            .iter()
            .filter(|p| !p.kind.flat() && p.z0 < 1.2 && p.z0 + p.size.2 > 0.2)
            .map(rect_of)
            .collect();
        let mut free = vec![false; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let c = Self::centre(i, j);
                let in_room = c.0 >= WALK_R - EPS
                    && c.1 >= WALK_R - EPS
                    && c.0 <= w - WALK_R + EPS
                    && c.1 <= d - WALK_R + EPS;
                free[j * nx + i] = in_room && blockers.iter().all(|&b| dist_to_rect(c, b) >= WALK_R);
            }
        }
        Grid { nx, ny, free }
    }

    fn centre(i: usize, j: usize) -> (f32, f32) {
        ((i as f32 + 0.5) * CELL, (j as f32 + 0.5) * CELL)
    }

    /// The free cell nearest `p` within `snap`.
    fn snap(&self, p: (f32, f32), snap: f32) -> Option<usize> {
        let mut best = None;
        let mut bd = snap;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let k = j * self.nx + i;
                let c = Self::centre(i, j);
                let dd = ((c.0 - p.0).powi(2) + (c.1 - p.1).powi(2)).sqrt();
                if self.free[k] && dd <= bd {
                    bd = dd;
                    best = Some(k);
                }
            }
        }
        best
    }

    fn flood(&self, start: usize) -> Vec<bool> {
        let mut seen = vec![false; self.free.len()];
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(k) = stack.pop() {
            let (i, j) = ((k % self.nx) as i64, (k / self.nx) as i64);
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (a, b) = (i + di, j + dj);
                if a < 0 || b < 0 || a >= self.nx as i64 || b >= self.ny as i64 {
                    continue;
                }
                let n = b as usize * self.nx + a as usize;
                if self.free[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        seen
    }
}

#[test]
fn every_door_reaches_every_spot_and_source() {
    let rooms = walkthrough::rooms();
    let srcs = walkthrough::sources();
    let mut errs = Vec::new();
    for ri in furnished_rooms() {
        let r = &rooms[ri];
        let g = Grid::new(ri);
        let starts: Vec<_> = floor_doors(ri).into_iter().filter(|(dr, ..)| !dr.glass).collect();
        assert!(!starts.is_empty(), "{}: no door to start from", r.name);
        for (dr, p, n) in starts {
            let s = match n {
                Some(n) => (p.0 + n.0 * 0.45, p.1 + n.1 * 0.45),
                None => p,
            };
            let Some(k0) = g.snap(s, SPOT_SNAP) else {
                errs.push(format!("{}: no free floor inside the door at {:?}", r.name, dr.pos));
                continue;
            };
            let seen = g.flood(k0);
            for (name, sp) in walkthrough::spots(ri) {
                let reached = g.snap(*sp, SPOT_SNAP).is_some_and(|k| seen[k]);
                if !reached {
                    errs.push(format!(
                        "{}: spot '{name}' {sp:?} unreachable from the door at {:?}",
                        r.name, dr.pos
                    ));
                }
            }
            for s in srcs.iter().filter(|s| s.room == ri) {
                for e in s.emitters {
                    let l = (e.0 - r.min.0, e.1 - r.min.1);
                    // what the source stands on or rises from (a piano,
                    // a table, a speaker cabinet): walking up to it counts
                    let carriers: Vec<Rect> = walkthrough::layout(ri)
                        .iter()
                        .filter(|p| !p.kind.flat())
                        .map(rect_of)
                        .filter(|&q| dist_to_rect(l, q) == 0.0)
                        .collect();
                    let near = (0..seen.len()).any(|k| {
                        let c = Grid::centre(k % g.nx, k / g.nx);
                        seen[k]
                            && (((c.0 - l.0).powi(2) + (c.1 - l.1).powi(2)).sqrt() <= SOURCE_REACH
                                || carriers.iter().any(|&q| dist_to_rect(c, q) <= WALK_R + 0.25))
                    });
                    if !near {
                        errs.push(format!(
                            "{}: source {} emitter {e:?} unreachable from the door at {:?}",
                            r.name, s.name, dr.pos
                        ));
                    }
                }
            }
        }
    }
    assert!(errs.is_empty(), "walkability errors:\n  {}", errs.join("\n  "));
}

#[test]
fn every_kind_is_used_sensibly() {
    // seats face something: no sofa or armchair with its front a hand's
    // width from a wall
    let rooms = walkthrough::rooms();
    for ri in furnished_rooms() {
        let r = &rooms[ri];
        let (w, d) = (r.max.0 - r.min.0, r.max.1 - r.min.1);
        for p in walkthrough::layout(ri) {
            if !matches!(p.kind, Kind::Sofa | Kind::Armchair | Kind::Chair | Kind::Pew | Kind::SeatRow)
            {
                continue;
            }
            let (a, b) = p.local_box();
            let (fx, fy) = p.facing();
            let front_gap = if fx > 0.5 {
                w - b[0]
            } else if fx < -0.5 {
                a[0]
            } else if fy > 0.5 {
                d - b[1]
            } else {
                a[1]
            };
            assert!(front_gap > 0.4, "{}: a {} faces the wall {front_gap} m away", r.name, p.kind.name());
        }
    }
}
