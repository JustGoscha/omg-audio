//! The ray walk spread over threads (`Tracer::threads`) gives exactly the
//! field one thread gives: each ray draws its own random numbers, and
//! what the rays leave is replayed in ray order.

use omg_core::material::Material;
use omg_core::mesh::MeshBuilder;
use omg_core::v3::{Budget, Source, Tracer, World};
use omg_core::vec3::Vec3;

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

#[test]
fn the_walk_is_the_same_on_any_number_of_threads() {
    // a hall with a pillar and a door to a second room: reflections,
    // edges, a partition
    let mut mb = MeshBuilder::new();
    let conc = mb.material(Material::CONCRETE);
    let wood = mb.material(Material::WOOD);
    let door = mb.material(Material::DOOR);
    // a 12 × 8 × 4 m hall of 20 cm slabs, a pillar, a door leaf in the
    // west wall's opening
    let t = 0.2;
    mb.abox(v(-t, -t, -t), v(12.0 + t, 8.0 + t, 0.0), conc);
    mb.abox(v(-t, -t, 4.0), v(12.0 + t, 8.0 + t, 4.0 + t), conc);
    mb.abox(v(-t, -t, 0.0), v(12.0 + t, 0.0, 4.0), conc);
    mb.abox(v(-t, 8.0, 0.0), v(12.0 + t, 8.0 + t, 4.0), conc);
    mb.abox(v(12.0, 0.0, 0.0), v(12.0 + t, 8.0, 4.0), conc);
    mb.abox(v(-t, 0.0, 0.0), v(0.0, 3.0, 4.0), conc);
    mb.abox(v(-t, 4.0, 0.0), v(0.0, 8.0, 4.0), conc);
    mb.abox(v(-t, 3.0, 2.2), v(0.0, 4.0, 4.0), conc);
    mb.abox(v(-0.05, 3.0, 0.0), v(0.0, 4.0, 2.2), door);
    mb.abox(v(5.0, 3.0, 0.0), v(6.0, 4.0, 4.0), wood);
    let mesh = mb.build();
    let world = World::new(&mesh, &[]);
    let lis = v(2.0, 2.0, 1.6);
    let srcs = [Source { id: 0, pos: v(9.0, 6.0, 1.2), weight: 1.0 }, Source { id: 1, pos: v(5.5, 6.5, 1.0), weight: 0.5 }];
    let b = Budget { rays: 300, bounces: 4, late_rays: 70, ..Budget::default() };
    let run = |threads: usize| {
        let mut t = Tracer::new(7);
        t.threads = threads;
        let mut out = Vec::new();
        for k in 0..4 {
            out = t.trace(&world, lis + v(0.1 * k as f32, 0.0, 0.0), &srcs, b);
        }
        out
    };
    let (one, three) = (run(1), run(3));
    for (a, b) in one.iter().zip(&three) {
        assert_eq!(a.taps.len(), b.taps.len());
        for (x, y) in a.taps.iter().zip(&b.taps) {
            assert_eq!((x.key, x.delay_s.to_bits(), x.gains.map(f32::to_bits)), (y.key, y.delay_s.to_bits(), y.gains.map(f32::to_bits)));
        }
        assert_eq!(a.reverb.rt60.map(f32::to_bits), b.reverb.rt60.map(f32::to_bits));
        assert_eq!(a.reverb.level.map(f32::to_bits), b.reverb.level.map(f32::to_bits));
        assert_eq!(a.reflected.map(f32::to_bits), b.reflected.map(f32::to_bits));
    }
}
