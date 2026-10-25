//! Effect: the OVER-THE-SILHOUETTE bend (`AutoPaths::find`, key 90000) —
//! the taut string from source to listener over everything standing in
//! their vertical plane, priced as a chain of knife edges.
//!
//! Claims guaranteed:
//! - the string clears what it bends over, thin features included: a
//!   parapet band with no top face (a downward probe slips past it to
//!   the roof slab) is cleared at its top edge, and no leg passes
//!   through any surface;
//! - a string never reaches an enclosed end for free: a listener inside
//!   a closed room pays the roof it drops through, and a listener below
//!   grade pays the earth.
//!
//! Combined in the demo: `crates/omg-scene/tests/bells_open_vs_indoor.rs`
//! (bells louder indoors than in the open — this path used to drop
//! through roofs, parapets and earth for free) and `wildlife.rs`.

mod common;

use common::{db, flat_opaque, flat_transmissive, v};
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::paths::{AutoPaths, FoundPath, PathBudget};
use omg_core::vec3::Vec3;

const BUDGET: PathBudget =
    PathBudget { edge_candidates: 48, pair_edges: 32, max_paths: 8, taps: false };

fn silhouette(mesh: &Mesh, src: Vec3, lis: Vec3) -> Option<FoundPath> {
    let mut auto = AutoPaths::new(mesh, 512);
    let mut buf = Vec::new();
    auto.find(mesh, src, lis, BUDGET, &mut buf);
    buf.into_iter().find(|p| p.key == 90_000)
}

/// A wide ground slab, so downward probes always land somewhere.
fn ground(mb: &mut MeshBuilder) {
    let m = mb.material(flat_opaque(0.1, 0.1));
    mb.abox(v(-200.0, -200.0, -2.0), v(200.0, 200.0, 0.0), m);
}

/// Building 10 m deep across the line, roof slab top at 6 m, with a
/// parapet band on its near (source-side) edge rising to 7.5 m. The
/// parapet is a single vertical quad: no top face to probe.
fn building_with_parapet() -> Mesh {
    let mut mb = MeshBuilder::new();
    ground(&mut mb);
    let wall = mb.material(flat_transmissive(0.05, 0.1, [0.03, 0.01, 0.003]));
    mb.abox(v(40.0, -20.0, 0.0), v(50.0, 20.0, 6.0), wall);
    mb.quad(
        v(40.0, -20.0, 6.0),
        v(40.0, 20.0, 6.0),
        v(40.0, 20.0, 7.5),
        v(40.0, -20.0, 7.5),
        wall,
    );
    mb.build()
}

#[test]
fn string_clears_a_thin_parapet() {
    let mesh = building_with_parapet();
    let (src, lis) = (v(0.0, 0.0, 9.0), v(60.0, 0.0, 1.6));
    let p = silhouette(&mesh, src, lis).expect("a string over the building");
    let peak = p.points.iter().map(|q| q.z).fold(f32::MIN, f32::max);
    assert!(
        peak >= 7.5,
        "the string must clear the parapet top (7.5 m), peaks at {peak:.2}: {:?}",
        p.points
    );
    let mut auto = AutoPaths::new(&mesh, 512);
    for w in p.points.windows(2) {
        let tr = auto.transmission(&mesh, w[0], w[1]);
        assert!(
            tr.iter().all(|&t| t > 0.999),
            "leg {:?} → {:?} passes through geometry: {tr:?}",
            w[0],
            w[1]
        );
    }
    // a pure knife-edge chain: bass bends more readily than treble
    assert!(p.gains[0] > p.gains[1] && p.gains[1] > p.gains[2], "{:?}", p.gains);
}

#[test]
fn listener_in_a_closed_room_pays_the_roof() {
    // a closed hut behind a wall: the string goes over the wall, but the
    // last leg must come down THROUGH the hut's roof
    let mut mb = MeshBuilder::new();
    ground(&mut mb);
    let wall = mb.material(flat_transmissive(0.05, 0.1, [0.03, 0.01, 0.003]));
    mb.abox(v(30.0, -20.0, 0.0), v(31.0, 20.0, 8.0), wall);
    let roof = [0.05, 0.02, 0.005];
    let hut = mb.material(flat_transmissive(0.05, 0.1, roof));
    mb.abox(v(50.0, -4.0, 0.0), v(58.0, 4.0, 3.0), hut);
    let mesh = mb.build();
    let (src, lis) = (v(0.0, 0.0, 2.0), v(54.0, 0.0, 1.6));
    if let Some(p) = silhouette(&mesh, src, lis) {
        for b in 0..3 {
            assert!(
                p.gains[b] <= roof[b] * 1.001,
                "band {b}: string reached the closed hut for less than its roof: {} dB vs roof {} dB",
                db(p.gains[b]),
                db(roof[b])
            );
        }
    }
}

#[test]
fn buried_listener_pays_the_earth() {
    // a listener 1.5 m below grade: the ground itself pokes above the
    // sight line, and the string used to drop through it for free
    let mut mb = MeshBuilder::new();
    let earth = [1e-3, 1e-4, 1e-5];
    let m = mb.material(flat_transmissive(0.1, 0.1, earth));
    mb.abox(v(-200.0, -200.0, -6.0), v(200.0, 200.0, 0.0), m);
    let mesh = mb.build();
    let (src, lis) = (v(0.0, 0.0, 20.0), v(80.0, 0.0, -1.5));
    if let Some(p) = silhouette(&mesh, src, lis) {
        for b in 0..3 {
            assert!(
                p.gains[b] <= earth[b] * 1.001,
                "band {b}: string reached a buried listener for {} dB, earth costs {} dB",
                db(p.gains[b]),
                db(earth[b])
            );
        }
    }
}
