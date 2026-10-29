//! A host's level dumped and replayed headless (`Geometry::record`,
//! `save`, `load`): the same triangles, materials, thicknesses and ground
//! — the same field traced from it.

use omg_audio::{Compute, Config, Geometry, Material, Quality, Simulation, SourceDesc};

fn level(record: bool) -> Geometry {
    let mut g = Geometry::new();
    if record {
        g.record();
    }
    let brick = g.material(Material::BRICK);
    let glass = g.material(Material::GLASS);
    g.set_thickness(glass, 0.012);
    g.add_box([0.0, 0.0, 0.0], [8.0, 0.3, 3.0], brick);
    g.add_box([0.0, 6.0, 0.0], [8.0, 6.3, 3.0], brick);
    g.add_mesh(&[[2.0, 3.0, 0.0], [4.0, 3.0, 0.0], [4.0, 3.0, 2.5], [2.0, 3.0, 2.5]], &[0, 1, 2, 0, 2, 3], glass);
    g.ground_plane(0.0, brick);
    g
}

fn heard(g: Geometry) -> (f32, f32) {
    let mut sim = Simulation::new(Config { quality: Quality::Level(3), compute: Compute::Cpu, v4: false, v5: false, v6: true, ..Config::default() });
    sim.set_geometry(g);
    let id = sim.add_source(SourceDesc::at([3.0, 5.0, 1.2]));
    sim.set_listener([3.0, 1.5, 1.6], 0.0, 0.0, 0.0);
    let mut f = sim.tick();
    for _ in 0..5 {
        f = sim.tick();
    }
    let b = f.block(id).expect("traced");
    (b.taps.iter().map(|t| t.gains[1] * t.gains[1]).sum(), b.reverb.level[1])
}

#[test]
fn a_dumped_level_replays_the_same() {
    let dir = std::env::temp_dir().join(format!("omg-geometry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("level.geo");
    let path = path.to_str().unwrap();
    let g = level(true);
    let tris = g.triangle_count();
    g.save(path).unwrap();
    let back = Geometry::load(path).unwrap();
    assert_eq!(back.triangle_count(), tris);
    // traced from either, bit for bit (the tracer repeats exactly)
    assert_eq!(heard(level(false)), heard(back));
    std::fs::remove_dir_all(&dir).ok();
}
