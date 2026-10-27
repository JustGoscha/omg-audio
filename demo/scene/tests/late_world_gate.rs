//! C6d gates. (1) Box parity: the stochastic tracer over a mesh-built
//! golden box must report the same RT60/level as over the analytic
//! shoebox — same tracer, two geometry backends. (2) Measured coupling:
//! with `early = traced`, a listener in the corridor hears the living
//! room's wet field through the open door — level nonzero, the tail
//! ANISOTROPIC toward the doorway — and closing the leaf (a panel
//! overlay on the world geometry, no portal code) drops it and turns
//! reversible.

use omg_core::material::Material;
use omg_core::mesh::MeshBuilder;
use omg_core::rng::Rng;
use omg_core::scene::Shoebox;
use omg_core::tracer::{estimate_reverb, trace, Echogram};
use omg_core::vec3::Vec3;
use omg_scene::quality;
use omg_scene::world::WorldSim;

#[test]
fn mesh_late_field_matches_shoebox() {
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
        b.quad(q[0], q[1], q[2], q[3], m);
    }
    let mesh = b.build();
    let room = Shoebox::new(size, walls);

    let src = Vec3::new(1.5, 1.0, 1.5);
    let lis = Vec3::new(4.5, 3.0, 1.6);

    // average a few seeds — both estimates are stochastic
    let avg = |f: &mut dyn FnMut(u64) -> ([f32; 3], [f32; 3])| {
        let (mut rt, mut lv) = ([0.0f32; 3], [0.0f32; 3]);
        for seed in 0..4u64 {
            let (r, l) = f(seed);
            for b in 0..3 {
                rt[b] += r[b] / 4.0;
                lv[b] += l[b] / 4.0;
            }
        }
        (rt, lv)
    };
    let (rt_box, lv_box) = avg(&mut |seed| {
        let mut rng = Rng::new(100 + seed);
        let mut e = Echogram::new();
        trace(&room, src, lis, 16_384, [1.0; 3], &mut rng, &mut e);
        let p = estimate_reverb(&e);
        (p.rt60, p.level)
    });
    let (rt_mesh, lv_mesh) = avg(&mut |seed| {
        let mut rng = Rng::new(100 + seed);
        let mut e = Echogram::new();
        trace(&mesh, src, lis, 16_384, [1.0; 3], &mut rng, &mut e);
        let p = estimate_reverb(&e);
        (p.rt60, p.level)
    });
    for b in 0..3 {
        let r = rt_mesh[b] / rt_box[b];
        assert!(
            (0.8..1.25).contains(&r),
            "band {b} rt60 mesh {} vs box {}",
            rt_mesh[b],
            rt_box[b]
        );
        let l = lv_mesh[b] / lv_box[b].max(1e-6);
        assert!(
            (0.75..1.33).contains(&l),
            "band {b} level mesh {} vs box {}",
            lv_mesh[b],
            lv_box[b]
        );
    }
    println!("box parity: rt60 {rt_mesh:?} vs {rt_box:?}, level {lv_mesh:?} vs {lv_box:?}");
}

#[test]
fn doorway_wet_is_measured_directional_and_door_sensitive() {
    quality::set_early(1);
    let mut w = WorldSim::new();
    const PIANO: usize = 0; // living room
    // corridor listener with the living door at (4, 6) to the south
    let read = |w: &mut WorldSim| {
        // enough ticks for the round-robin budget to reach this source
        // several times and the EMA to settle
        for _ in 0..60 {
            let _ = w.tick_at(4.4, 7.5, 0.0);
        }
        let (blocks, _) = w.tick_at(4.4, 7.5, 0.0);
        let pb = &blocks[PIANO];
        let diffuse = pb.reverb.level[1];
        let send = pb.remote.as_ref().map_or(0.0, |r| r.send[1]);
        (diffuse + send, send, pb.reverb.rt60[1])
    };
    let (open_wet, open_send, rt60) = read(&mut w);
    eprintln!("open: wet {open_wet:.5} send {open_send:.5} rt60 {rt60:.2}");
    assert!(open_wet > 1e-4, "wet field through the open door: {open_wet}");
    // ~29% measured: the tail partly RE-DIFFUSES in the listener's own
    // corridor (correct physics) — pin that a noticeable directional
    // component exists, not a precise split.
    assert!(
        open_send > 0.15 * open_wet,
        "through-a-door wet should carry a directional component: send {open_send} of {open_wet}"
    );
    assert!(
        rt60 > 0.15 && rt60 < 3.0,
        "coupled rt60 should be a sane room decay: {rt60}"
    );

    w.set_door(0, 0.0); // close Living ↔ Corridor
    let (closed_wet, _, _) = read(&mut w);
    assert!(
        closed_wet < 0.6 * open_wet,
        "closing the leaf must drop the measured wet: {closed_wet} vs {open_wet}"
    );

    w.set_door(0, 1.0);
    let (reopened, _, _) = read(&mut w);
    assert!(
        reopened > 0.6 * open_wet,
        "reopening must restore the wet: {reopened} vs {open_wet}"
    );
    println!(
        "doorway wet: open {open_wet:.5} (send {open_send:.5}, rt60 {rt60:.2}s), closed {closed_wet:.5}, reopened {reopened:.5}"
    );
}
