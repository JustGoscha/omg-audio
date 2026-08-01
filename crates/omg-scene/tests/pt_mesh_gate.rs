//! C6b gates. (1) Box equivalence: in a mesh-built empty box the
//! world-mesh solver must agree with the analytic box solver — every
//! mesh record matches an analytic record (delay ≤ 0.5 ms, level
//! ≤ 1 dB), the direct path is exact, and ray discovery finds the
//! bulk of the ≤2-order set. (2) The portal-free doorway: on the REAL
//! world mesh, a source in the living room reaches a listener in the
//! corridor through the door hole at full strength, and a listener
//! behind masonry only via mass-law transmission — no portal code,
//! no room graph, no routing involved anywhere.

use omg_core::material::Material;
use omg_core::mesh::MeshBuilder;
use omg_core::pt::{pt_discover, PathRecord};
use omg_core::pt_mesh::{mesh_chains, mesh_record, MChain, SurfaceTable};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_scene::dome::build_world_mesh;
use omg_scene::walkthrough;

fn db(a: f32) -> f32 {
    20.0 * a.max(1e-9).log10()
}

#[test]
fn mesh_solver_matches_analytic_in_a_box() {
    // the golden live-room, built as a mesh with one surface per wall
    let size = Vec3::new(6.0, 4.0, 3.0);
    let walls = [
        Material::CONCRETE,
        Material::CONCRETE,
        Material::CONCRETE,
        Material::CONCRETE,
        Material::WOOD_PANEL,
        Material::CONCRETE,
    ];
    let mut b = MeshBuilder::new();
    let v = Vec3::new;
    let quads: [([Vec3; 4], usize); 6] = [
        ([v(0., 0., 0.), v(0., 4., 0.), v(0., 4., 3.), v(0., 0., 3.)], 0),
        ([v(6., 0., 0.), v(6., 0., 3.), v(6., 4., 3.), v(6., 4., 0.)], 1),
        ([v(0., 0., 0.), v(0., 0., 3.), v(6., 0., 3.), v(6., 0., 0.)], 2),
        ([v(0., 4., 0.), v(6., 4., 0.), v(6., 4., 3.), v(0., 4., 3.)], 3),
        ([v(0., 0., 0.), v(6., 0., 0.), v(6., 4., 0.), v(0., 4., 0.)], 4),
        ([v(0., 0., 3.), v(0., 4., 3.), v(6., 4., 3.), v(6., 0., 3.)], 5),
    ];
    for (q, wi) in quads {
        let m = b.material(walls[wi]);
        b.begin_surface();
        b.quad(q[0], q[1], q[2], q[3], m);
    }
    let mesh = b.build();
    let table = SurfaceTable::build(&mesh);

    let room = Shoebox::new(size, walls);
    let src = Vec3::new(1.5, 1.0, 1.5);
    let lis = Vec3::new(4.5, 3.0, 1.6);

    // analytic reference (rays + seeds, converged)
    let mut refs: Vec<PathRecord> = Vec::new();
    let mut tick = Vec::new();
    for rot in 0..8 {
        pt_discover(&room, &[src], lis, 4096, rot, &mut tick);
        for r in &tick {
            if !refs.iter().any(|q| q.key() == r.key()) {
                refs.push(*r);
            }
        }
    }

    // mesh discovery + solve
    let mut chains: Vec<MChain> = Vec::new();
    for rot in 0..8 {
        mesh_chains(&mesh, lis, 4096, rot, &mut chains);
    }
    chains.sort();
    chains.dedup();
    let mut buf = Vec::new();
    let mut recs = Vec::new();
    // direct path is always a candidate
    if let Some(r) = mesh_record(&mesh, &table, &[], 0, src, lis, &[], &mut buf) {
        recs.push(r);
    }
    for (chain, order) in &chains {
        if let Some(r) =
            mesh_record(&mesh, &table, &chain[..*order as usize], 0, src, lis, &[], &mut buf)
        {
            recs.push(r);
        }
    }

    // soundness: every mesh record matches an analytic one
    for r in &recs {
        let hit = refs.iter().any(|q| {
            (q.delay_s - r.delay_s).abs() * 1000.0 <= 0.5
                && (0..3).all(|b| (db(q.gains[b]) - db(r.gains[b])).abs() <= 1.0)
        });
        assert!(
            hit,
            "mesh path unmatched: order {} delay {:.2} ms",
            r.order,
            r.delay_s * 1000.0
        );
    }
    // coverage: the bulk of the analytic ≤2 set is found by mesh rays
    let low: Vec<&PathRecord> = refs.iter().filter(|r| r.order <= 2).collect();
    let found = low
        .iter()
        .filter(|q| {
            recs.iter().any(|r| {
                (q.delay_s - r.delay_s).abs() * 1000.0 <= 0.5
                    && (0..3).all(|b| (db(q.gains[b]) - db(r.gains[b])).abs() <= 1.0)
            })
        })
        .count();
    assert!(
        found * 10 >= low.len() * 8,
        "mesh discovery found {found}/{} of the ≤2-order set",
        low.len()
    );
    // direct exact
    let d = recs.iter().find(|r| r.order == 0).expect("direct");
    let true_delay = (src - lis).length() / 343.0;
    assert!((d.delay_s - true_delay).abs() < 1e-5);
    println!("box equivalence: {}/{} low-order, {} records total", found, low.len(), recs.len());
}

#[test]
fn doorways_thread_with_zero_portal_code() {
    let rooms = walkthrough::rooms();
    let doors = walkthrough::doors();
    let (mesh, _) = build_world_mesh(&rooms, &doors);
    let table = SurfaceTable::build(&mesh);
    let mut buf = Vec::new();

    // piano in the living room; the living-room door to the corridor
    // is at (4, 6). A listener in the corridor with a sight line
    // through the opening:
    let src = Vec3::new(2.0, 3.0, 1.5);
    let clear = Vec3::new(4.4, 7.5, 1.6);
    let r = mesh_record(&mesh, &table, &[], 0, src, clear, &[], &mut buf)
        .expect("direct through the door hole");
    let free_space = 1.0 / (src - clear).length();
    assert!(
        (db(r.gains[1]) - db(free_space)).abs() < 3.0,
        "through the open door should be ~free-space: {:.1} vs {:.1} dB",
        db(r.gains[1]),
        db(free_space)
    );

    // a listener OFF the sight line, behind the wall: only mass-law
    // transmission survives (heavily attenuated but nonzero in bass,
    // treble gone) — or the path is dropped as inaudible. Either way,
    // FAR below the doorway path.
    let blocked = Vec3::new(1.0, 7.5, 1.6);
    let r2 = mesh_record(&mesh, &table, &[], 0, src, blocked, &[], &mut buf);
    if let Some(r2) = r2 {
        assert!(
            db(r2.gains[1]) < db(r.gains[1]) - 20.0,
            "masonry should cost ≥20 dB mid-band: {:.1} vs {:.1}",
            db(r2.gains[1]),
            db(r.gains[1])
        );
        assert!(
            db(r2.gains[0]) - db(r2.gains[2]) > 6.0,
            "mass law: bass must outlive treble through the wall"
        );
    }
    println!(
        "doorway: clear {:.1} dB mid, behind wall {}",
        db(r.gains[1]),
        r2.map_or("inaudible (dropped)".into(), |r| format!("{:.1} dB mid", db(r.gains[1])))
    );
}
