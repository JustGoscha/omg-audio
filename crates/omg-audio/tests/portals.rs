//! Portals (`Config::portals`): what reaches the ears through an opening
//! from an enclosed source, estimated from the source's side, sums to
//! what the walk alone estimates where the walk is exact.

use omg_audio::{Config, Geometry, Material, Quality, Simulation, SourceDesc};

/// A 10 × 8 × 4 m room (absorption 0.1, `scatter`), the source and the
/// ears in it, an opening — an aperture with no leaf, blocking nothing —
/// across it between them; per band, what the walk deposits a tick (dB,
/// mean over `ticks`).
fn walked(scatter: f32, portals: bool, k: f32, ticks: usize) -> [f64; 3] {
    let mut g = Geometry::new();
    let m = g.material(Material::new([0.1; 3], scatter, [0.0; 3]));
    let (lx, ly, lz, t) = (10.0f32, 8.0f32, 4.0f32, 0.3f32);
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
    g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
    g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
    let mut b = Quality::High.budget(false);
    b.diffraction = false;
    b.k = k;
    let cfg = Config { quality: Quality::Custom(b), v4: false, v5: false, v6: true, v61: true, v62: true, portals, seed: 1, kernel_thread: false, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    sim.set_geometry(g);
    let o = sim.add_object([4.96, 0.0, 0.0], [5.04, ly, lz], Material::DOOR);
    sim.set_object_aperture(o, [4.96, 0.0, 0.0], [5.04, ly, lz]);
    sim.set_object_enabled(o, false);
    let id = sim.add_source(SourceDesc::at([3.0, 4.0, 1.5]));
    sim.set_listener([7.0, 4.5, 1.6], 0.0, 0.0, 0.0);
    let mut acc = [0.0f64; 3];
    for _ in 0..ticks {
        sim.tick();
        let e = sim.walked_energy(id).expect("traced");
        for b in 0..3 {
            acc[b] += e[b] as f64;
        }
    }
    acc.map(|v| 10.0 * (v / ticks as f64).log10())
}

/// Every reflection diffuse (the walk exact: next-event connections
/// only): portals and the walk agree within 0.15 dB per band (the high
/// band's air over the joined path to 0.2).
#[test]
fn portals_and_the_walk_agree_in_a_diffuse_room() {
    let (w, p) = (walked(1.0, false, 0.06, 120), walked(1.0, true, 0.06, 120));
    for b in 0..3 {
        let tol = if b == 2 { 0.2 } else { 0.15 };
        assert!((p[b] - w[b]).abs() < tol, "band {b}: portals {:.2} dB, the walk {:.2} dB", p[b], w[b]);
    }
}

/// Half the reflections mirrors: the walk catches a mirror path in a
/// sphere around the source, its radius k per metre of path, and reads
/// such paths low (−0.4 dB here at the rungs' k = 0.06); as k → 0 it
/// converges to what portals read at any k (the record is traced from
/// the source: no sphere). At k = 0.015 the two agree within 0.2 dB.
#[test]
fn portals_read_mirror_paths_as_the_walk_does_in_its_limit() {
    let (w, p) = (walked(0.5, false, 0.015, 160), walked(0.5, true, 0.06, 120));
    for b in 0..3 {
        assert!((p[b] - w[b]).abs() < 0.2, "band {b}: portals {:.2} dB, the walk (k 0.015) {:.2} dB", p[b], w[b]);
    }
}
