//! Diffraction in motion: the sound bent round a corner is there the tick
//! the straight line goes behind it, round a thick end as round a thin one.
use omg_audio::{Config, Geometry, Material, Quality, Simulation, SourceDesc};

fn db(x: f32) -> f32 {
    10.0 * x.max(1e-30).log10()
}

/// The ears walk east along y = 6 (1.4 m/s, 20 Hz ticks) past an obstacle
/// south of them with a source behind it, until the straight line to it
/// runs behind the obstacle; twenty sources behind sheds out east keep the
/// edge searches busy (one source a tick searches at the lowest level).
/// The exact paths' (direct, bends) largest step a tick, and how far they
/// fall into the shadow.
fn walk_past(obstacle: ([f32; 3], [f32; 3]), source: [f32; 3], until: f32, v64: bool) -> (f32, f32) {
    let mut g = Geometry::new();
    let wall = g.material(Material::CONCRETE);
    let ground = g.material(Material::GRASS);
    g.ground_plane(0.0, ground);
    g.add_box(obstacle.0, obstacle.1, wall);
    for k in 0..20 {
        let x = 30.0 + 6.0 * (k % 5) as f32;
        let y = -20.0 + 8.0 * (k / 5) as f32;
        g.add_box([x, y, 0.0], [x + 2.0, y + 0.25, 3.0], wall);
    }
    let cfg = Config { quality: Quality::Level(0), v6: true, v61: true, v62: true, v63: true, v64, kernel_thread: false, ..Config::default() };
    let mut sim = Simulation::new(cfg);
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at(source));
    for k in 0..20 {
        let x = 31.0 + 6.0 * (k % 5) as f32;
        let y = -21.5 + 8.0 * (k / 5) as f32;
        sim.add_source(SourceDesc::at([x, y, 1.5]));
    }
    let mut level = |sim: &mut Simulation| -> f32 {
        let f = sim.tick();
        db(f.block(src).map_or(0.0, |pb| pb.taps.iter().filter(|t| omg_core::v3::tap_kind(t.key) != 3).map(|t| t.gains[1] * t.gains[1]).sum()))
    };
    let mut x = -12.0f32;
    for _ in 0..40 {
        sim.set_listener([x, 6.0, 1.6], 0.0, 0.0, 0.0);
        level(&mut sim);
    }
    let mut lv = Vec::new();
    while x < until {
        x += 1.4 * 0.05;
        sim.set_listener([x, 6.0, 1.6], 0.0, 0.0, 0.0);
        lv.push(level(&mut sim));
    }
    let step = lv.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    (step, lv[0] - lv[lv.len() - 1])
}

/// Past a wall's end — thin (2 cm), thick (25 cm: a door's jamb, a wall
/// as built) — and a building's corner: the bend round it carries the
/// source the tick the straight line goes behind it (v6.4: looked ahead,
/// Schissler, Mückl & Calamia 2021 §5.2.2), and round a thick end as
/// round a thin one (v6.4: an end's two edges one corner). v6.3 stepped
/// 36 dB in a tick past the thick end and fell 51 dB into its shadow
/// against the thin one's 10: its single bend ran a leg through the end.
#[test]
fn passing_a_corner_the_bend_is_there_at_once_v64() {
    let thin = ([0.0, 0.0, 0.0], [10.0, 0.02, 3.0]);
    let thick = ([0.0, 0.0, 0.0], [10.0, 0.25, 3.0]);
    let building = ([0.0, -10.0, 0.0], [10.0, 0.0, 6.0]);
    let cases = [("thin wall's end", thin, [6.0, -6.0, 1.5], 4.0), ("thick wall's end", thick, [6.0, -6.0, 1.5], 4.0), ("building's corner", building, [-6.0, -5.0, 1.5], 8.0)];
    let mut falls = Vec::new();
    for (name, ob, src, until) in cases {
        let (s63, f63) = walk_past(ob, src, until, false);
        let (s64, f64) = walk_past(ob, src, until, true);
        eprintln!("past a {name}: largest step a tick v6.3 {s63:.1} dB, v6.4 {s64:.1} dB; into the shadow {f63:.1} / {f64:.1} dB down");
        assert!(s64 < 6.0, "v6.4, past a {name}: the level stepped {s64:.1} dB in a tick");
        falls.push(f64);
    }
    assert!((falls[1] - falls[0]).abs() < 3.0, "v6.4: round a thick end {:.1} dB down, round a thin one {:.1}", falls[1], falls[0]);
}
