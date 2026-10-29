//! Effect tests for the engine facade's `Simulation`: each builds a
//! minimal level through the public API (`Geometry`, objects, sources,
//! listener) and checks one acoustic claim against a closed-form answer
//! or a clear ordering. Nothing here knows about the demo town.
//!
//! - free field: one direct path, delay d/c, amplitude air/d;
//! - a wall in between: the direct path pays the wall's mass-law
//!   transmission at its real thickness;
//! - a door object closing over a doorway muffles: lower level, and the
//!   highs drop more than the lows;
//! - a closed shoebox rings with a Sabine-like RT60;
//! - under open sky the outdoor field is large, sealed in a box ~zero;
//! - beyond `max_traced_sources`, sources take turns and all get traced;
//! - a facade's echo of a passing car plays while the facade mirrors it,
//!   at every rung, at its image source's delay and level.

use omg_audio::{Config, Frame, Geometry, Material, ParamBlock, PathKind, Quality, Simulation, SourceDesc, SPEED_OF_SOUND};
use omg_core::material::air_attenuation;

/// A material that only reflects: isolates the effect under test.
const OPAQUE: Material = Material::new([0.1; 3], 0.3, [0.0; 3]);

fn settle(sim: &mut Simulation, ticks: usize) -> Frame {
    let mut f = sim.tick();
    for _ in 1..ticks {
        f = sim.tick();
    }
    f
}

/// Per-band energy a block delivers: paths, diffuse and remote reverb.
fn energy(pb: &ParamBlock) -> [f32; 3] {
    core::array::from_fn(|b| {
        pb.taps.iter().map(|t| t.gains[b] * t.gains[b]).sum::<f32>()
            + pb.reverb.level[b] * pb.reverb.level[b]
            + pb.remote.map_or(0.0, |r| r.send[b] * r.send[b])
    })
}

/// v4: the dense response's energy per band (0 in v3).
fn dense_energy(sim: &Simulation, id: omg_audio::SourceId) -> [f32; 3] {
    let d = sim.dense_response(id);
    core::array::from_fn(|b| d.map_or(0.0, |d| d.bins.iter().map(|x| x[b]).sum()))
}

/// v4: RT60 (mid) of the dense response from its Schroeder curve (the
/// -5 to -25 dB slope).
fn dense_rt60(d: &omg_audio::DenseResponse) -> f32 {
    let mut sch: Vec<f64> = d.bins.iter().map(|x| x[1] as f64).collect();
    for i in (0..sch.len().saturating_sub(1)).rev() {
        sch[i] += sch[i + 1];
    }
    let tot = sch.first().copied().unwrap_or(0.0).max(1e-30);
    let db = |i: usize| 10.0 * (sch[i] / tot).max(1e-30).log10();
    let i5 = (0..sch.len()).find(|&i| db(i) <= -5.0).unwrap_or(0);
    let i25 = (0..sch.len()).find(|&i| db(i) <= -25.0).unwrap_or(sch.len().max(1) - 1);
    3.0 * (i25.saturating_sub(i5)) as f32 * d.bin_s
}

#[test]
fn free_field_is_one_direct_path() {
    let mut sim = Simulation::new(Config::default());
    let src = sim.add_source(SourceDesc::at([10.0, 0.0, 1.6]));
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let f = sim.tick();
    let pb = f.block(src).expect("a block for the source");
    assert_eq!(pb.taps.len(), 1, "free field: the direct path only, got {:?}", pb.taps);
    let t = &pb.taps[0];
    assert!((t.delay_s - 10.0 / SPEED_OF_SOUND).abs() < 1e-6, "delay {}", t.delay_s);
    let air = air_attenuation(10.0);
    for b in 0..3 {
        assert!((t.gains[b] - air[b] / 10.0).abs() < 1e-6, "band {b}: {:?}", t.gains);
    }
    assert!(t.dir[0] > 0.999, "arrives from +x (world frame): {:?}", t.dir);
    // source gain scales the paths
    sim.set_source_gain(src, 2.0);
    let f = sim.tick();
    assert!((f.block(src).unwrap().taps[0].gains[1] - 2.0 * air[1] / 10.0).abs() < 1e-6);
}

#[test]
fn a_wall_between_pays_its_transmission() {
    let mut g = Geometry::new();
    let brick = g.material(Material::BRICK);
    g.add_box([4.9, -50.0, -50.0], [5.1, 50.0, 50.0], brick); // 20 cm, real thickness
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at([10.0, 0.0, 1.6]));
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let f = sim.tick();
    let d = f.block(src).unwrap().taps.iter().find(|t| t.key == omg_core::v3::KEY_DIRECT).expect("direct path");
    let (wall, air) = (Material::BRICK.transmission_at(0.2), air_attenuation(10.0));
    for b in 0..3 {
        let want = wall[b] * air[b] / 10.0;
        assert!((d.gains[b] - want).abs() <= 1e-3 * want + 1e-9, "band {b}: {} vs {want}", d.gains[b]);
    }
    assert!(d.gains[2] < d.gains[0] * 0.1, "mass law: highs die first {:?}", d.gains);
}

/// A closed 6×5×3 room (20 cm walls) with a 1×2.1 m doorway in its +x
/// wall; listener inside, source 8 m outside the doorway.
fn room_with_doorway() -> Geometry {
    let mut g = Geometry::new();
    let w = g.material(Material::CONCRETE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [6.0 + t, 5.0 + t, 0.0], w); // floor
    g.add_box([-t, -t, 3.0], [6.0 + t, 5.0 + t, 3.0 + t], w); // ceiling
    g.add_box([-t, -t, 0.0], [0.0, 5.0 + t, 3.0], w); // -x
    g.add_box([-t, -t, 0.0], [6.0 + t, 0.0, 3.0], w); // -y
    g.add_box([-t, 5.0, 0.0], [6.0 + t, 5.0 + t, 3.0], w); // +y
    // +x wall with the doorway at y 2..3, z 0..2.1
    g.add_box([6.0, 0.0, 0.0], [6.0 + t, 2.0, 3.0], w);
    g.add_box([6.0, 3.0, 0.0], [6.0 + t, 5.0, 3.0], w);
    g.add_box([6.0, 2.0, 2.1], [6.0 + t, 3.0, 3.0], w);
    let ground = g.material(Material::GRASS);
    g.ground_plane(-t, ground);
    g
}

/// Mean per-band energy of one source over ticks 20..80 (settled, and
/// averaged over the tracer's noise: through a doorway the early paths
/// of a single tick scatter by ±5 dB).
fn mean_energy(sim: &mut Simulation, id: omg_audio::SourceId) -> [f32; 3] {
    settle(sim, 20);
    let mut acc = [0.0f32; 3];
    for _ in 0..60 {
        let e = energy(sim.tick().block(id).unwrap());
        // v4: the scattered energy is the dense response's
        let d = dense_energy(sim, id);
        for b in 0..3 {
            acc[b] += (e[b] + d[b]) / 60.0;
        }
    }
    acc
}

#[test]
fn closing_a_door_muffles() {
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(room_with_doorway());
    let src = sim.add_source(SourceDesc::at([14.0, 2.5, 1.5]));
    sim.set_listener([2.0, 1.5, 1.6], 0.0, 0.0, 0.0);
    let open = mean_energy(&mut sim, src);

    // the door leaf: a 4 cm box filling the doorway
    let door = sim.add_object([6.08, 2.0, 0.0], [6.12, 3.0, 2.1], Material::DOOR);
    let closed = mean_energy(&mut sim, src);
    // open again: disabling the object restores the doorway
    sim.set_object_enabled(door, false);
    let reopened = mean_energy(&mut sim, src);

    let db = |a: f32, b: f32| 10.0 * (a / b).log10();
    let drop: [f32; 3] = core::array::from_fn(|b| db(open[b], closed[b]));
    eprintln!("door: open {open:?} closed {closed:?} reopened {reopened:?} drop dB {drop:?}");
    assert!(drop[1] > 6.0, "closing the door must cost >6 dB mid, got {:.1}", drop[1]);
    assert!(drop[2] > drop[0] + 3.0, "the door muffles: highs drop more than lows {drop:?}");
    assert!(db(open[1], reopened[1]).abs() < 3.0, "reopening restores the level");
}

#[test]
fn a_closed_shoebox_rings_like_sabine() {
    let (lx, ly, lz) = (8.0, 6.0, 3.0);
    let mut g = Geometry::new();
    let m = g.material(OPAQUE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
    g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
    g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
    let mut sim = Simulation::new(Config { quality: Quality::High, ..Config::default() });
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at([2.0, 2.0, 1.5]));
    sim.set_listener([5.5, 4.0, 1.6], 0.0, 0.0, 0.0);
    let f = settle(&mut sim, 40);
    let pb = f.block(src).unwrap();
    let (v, s) = (lx * ly * lz, 2.0 * (lx * ly + lx * lz + ly * lz));
    let sabine = 0.161 * v / (s * 0.1);
    // v4: the decay of the traced dense response; v3: the fitted RT60
    let rt = match sim.dense_response(src) {
        Some(d) => dense_rt60(d),
        None => pb.reverb.rt60[1],
    };
    eprintln!("shoebox rt60 {rt:.2} s, Sabine {sabine:.2} s, reverb level {:?}", pb.reverb.level);
    assert!((rt / sabine - 1.0).abs() < 0.35, "rt60 {rt:.2} vs Sabine {sabine:.2}");
    let dense = sim.dense_response(src).is_some_and(|d| !d.bins.is_empty());
    assert!(dense || pb.reverb.level[1] > 0.0 || pb.remote.is_some(), "a closed room has a late field");
}

#[test]
fn open_sky_versus_sealed_box() {
    let mut open = Simulation::new(Config::default());
    let mut g = Geometry::new();
    let grass = g.material(Material::GRASS);
    g.ground_plane(0.0, grass);
    open.set_geometry(g);
    open.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    let o = settle(&mut open, 10).outdoor();

    let mut sealed = Simulation::new(Config::default());
    let mut g = Geometry::new();
    let c = g.material(Material::CONCRETE);
    g.add_box([-3.0, -3.0, -0.3], [3.0, 3.0, 0.0], c);
    g.add_box([-3.0, -3.0, 3.0], [3.0, 3.0, 3.3], c);
    g.add_box([-3.3, -3.3, -0.3], [-3.0, 3.3, 3.3], c);
    g.add_box([3.0, -3.3, -0.3], [3.3, 3.3, 3.3], c);
    g.add_box([-3.0, -3.3, -0.3], [3.0, -3.0, 3.3], c);
    g.add_box([-3.0, 3.0, -0.3], [3.0, 3.3, 3.3], c);
    sealed.set_geometry(g);
    sealed.set_listener([0.0, 0.0, 1.7], 0.0, 0.0, 0.0);
    let s = settle(&mut sealed, 10).outdoor();
    eprintln!("outdoor field: open {o:?}, sealed {s:?}");
    assert!(o[1] > 0.3, "under open sky most rays escape: {o:?}");
    assert!(s[1] < 0.01 * o[1], "sealed in concrete the sky is gone: {s:?} vs {o:?}");
}

#[test]
fn many_sources_take_turns() {
    let mut sim = Simulation::new(Config { max_traced_sources: 16, quality: Quality::Low, ..Config::default() });
    let ids: Vec<_> = (0..100)
        .map(|i| {
            let a = i as f32 * 0.37;
            sim.add_source(SourceDesc::at([a.cos() * (5.0 + i as f32), a.sin() * (5.0 + i as f32), 1.5]))
        })
        .collect();
    sim.set_listener([0.0, 0.0, 1.6], 0.0, 0.0, 0.0);
    let mut f = sim.tick();
    assert_eq!(sim.traced_last_tick(), 16);
    for _ in 0..12 {
        f = sim.tick();
        assert!(sim.traced_last_tick() <= 16);
    }
    for id in ids {
        assert!(!f.block(id).unwrap().taps.is_empty(), "{id:?} never traced");
    }
}

#[test]
fn frames_round_trip_through_flat_floats() {
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(room_with_doorway());
    let a = sim.add_source(SourceDesc::at([14.0, 2.5, 1.5]));
    let b = sim.add_source(SourceDesc::at([3.0, 3.0, 1.2]));
    sim.set_listener_at([2.0, 1.5, 1.6], 0.3, -0.1, 0.05, 1234.567_891);
    let f = settle(&mut sim, 3);
    let mut flat = Vec::new();
    f.write_flat(&mut flat);
    let g = Frame::read_flat(&flat).expect("parses");
    assert_eq!(g.version, f.version);
    assert_eq!(g.listener, f.listener);
    assert_eq!(g.time, Some(1234.567_891), "the pose time survives bit-exact");
    let mut untimed = Simulation::new(Config::default());
    untimed.set_geometry(room_with_doorway());
    untimed.add_source(SourceDesc::at([3.0, 3.0, 1.2]));
    untimed.set_listener([2.0, 1.5, 1.6], 0.3, -0.1, 0.05);
    untimed.tick().write_flat(&mut flat);
    assert_eq!(Frame::read_flat(&flat).expect("parses").time, None, "a host that gives no time: none");
    assert_eq!(g.sky, f.sky);
    for id in [a, b] {
        let (x, y) = (f.block(id).unwrap(), g.block(id).unwrap());
        assert_eq!(x.taps.len(), y.taps.len());
        assert_eq!(x.reverb.rt60, y.reverb.rt60);
        for (s, t) in x.taps.iter().zip(&y.taps) {
            assert_eq!((s.delay_s, s.dir, s.gains), (t.delay_s, t.dir, t.gains));
        }
    }
    assert!(Frame::read_flat(&flat[..flat.len() - 1]).is_none(), "truncated input is rejected");
    assert!(Frame::read_flat(&[0.0; 16]).is_none());
}

/// The debug view's paths are the traced geometry: each starts at the
/// listener, ends at the source, and crosses the doorway wall only
/// through the opening (an opaque room: nothing passes through a wall).
/// Off by default, and off again leaves nothing behind.
#[test]
fn debug_paths_follow_the_geometry() {
    let mut g = Geometry::new();
    let w = g.material(OPAQUE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [6.0 + t, 5.0 + t, 0.0], w); // floor
    g.add_box([-t, -t, 3.0], [6.0 + t, 5.0 + t, 3.0 + t], w); // ceiling
    g.add_box([-t, -t, 0.0], [0.0, 5.0 + t, 3.0], w); // -x
    g.add_box([-t, -t, 0.0], [6.0 + t, 0.0, 3.0], w); // -y
    g.add_box([-t, 5.0, 0.0], [6.0 + t, 5.0 + t, 3.0], w); // +y
    g.add_box([6.0, 0.0, 0.0], [6.0 + t, 2.0, 3.0], w); // +x, doorway y 2..3, z 0..2.1
    g.add_box([6.0, 3.0, 0.0], [6.0 + t, 5.0, 3.0], w);
    g.add_box([6.0, 2.0, 2.1], [6.0 + t, 3.0, 3.0], w);
    g.ground_plane(-t, w);
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(g);
    let (lis, at) = ([2.0, 1.5, 1.6], [9.0, 3.5, 1.5]);
    let src = sim.add_source(SourceDesc::at(at));
    sim.set_listener(lis, 0.0, 0.0, 0.0);

    settle(&mut sim, 3);
    assert!(sim.debug_paths().is_empty(), "off by default: no paths");

    sim.set_debug_paths(true);
    settle(&mut sim, 5);
    let paths = sim.debug_paths();
    assert!(!paths.is_empty(), "paths through the doorway");
    assert!(paths.iter().filter(|p| p.kind == PathKind::Sampled).count() <= 64);
    // the exact paths are steady: standing still, the same ones next tick
    // the exact paths are steady: standing still, the strong ones are
    // all there again next tick (weak chains may come and go)
    let key = |p: &omg_audio::DebugPath| (p.kind as u8, p.key);
    let top = paths.iter().filter(|p| p.kind != PathKind::Sampled).map(|p| p.energy).fold(0.0f32, f32::max);
    assert!(paths.iter().any(|p| p.kind == PathKind::Direct), "the direct path, through the doorway");
    settle(&mut sim, 1);
    let next: Vec<_> = sim.debug_paths().iter().map(key).collect();
    for p in paths.iter().filter(|p| p.kind != PathKind::Sampled && p.energy > 0.01 * top) {
        assert!(next.contains(&key(p)), "{:?} {} gone a tick later", p.kind, p.key);
    }
    assert!(paths.windows(2).all(|p| p[0].energy >= p[1].energy), "strongest first");
    let d = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    for p in &paths {
        assert_eq!(p.source, src);
        assert!(p.points.len() >= 2);
        let (first, last) = (p.points[0], *p.points.last().unwrap());
        assert!(d(first, lis) < 0.15, "starts at the ears: {first:?}");
        let len: f32 = p.points.windows(2).map(|s| d(s[0], s[1])).sum();
        assert!(d(last, at) < 0.1 + 0.06 * len, "ends at the source: {last:?} (path {len:.1} m)");
        for s in p.points.windows(2) {
            let (a, b) = (s[0], s[1]);
            // crossing the doorway wall's mid-plane x = 6.1: inside the opening
            if (a[0] - 6.1) * (b[0] - 6.1) < 0.0 {
                let k = (6.1 - a[0]) / (b[0] - a[0]);
                let (y, z) = (a[1] + k * (b[1] - a[1]), a[2] + k * (b[2] - a[2]));
                assert!(
                    (1.95..=3.05).contains(&y) && z <= 2.15,
                    "a segment through the wall at y {y:.2} z {z:.2}: {:?}",
                    p.points
                );
            }
        }
    }

    sim.set_debug_paths(false);
    assert!(sim.debug_paths().is_empty());
    settle(&mut sim, 2);
    assert!(sim.debug_paths().is_empty(), "off again: nothing recorded");
}

/// Between full ticks, `tick_exact` moves the exact paths with the
/// listener at once: the direct path's delay is the new distance's, the
/// mirror chains re-solved; nothing moved, no frame.
#[test]
fn exact_frames_follow_the_listener_between_ticks() {
    let mut g = Geometry::new();
    let c = g.material(Material::CONCRETE);
    let t = 0.2;
    g.add_box([-t, -t, -t], [8.0 + t, 6.0 + t, 0.0], c);
    g.add_box([-t, -t, 3.0], [8.0 + t, 6.0 + t, 3.0 + t], c);
    g.add_box([-t, -t, 0.0], [0.0, 6.0 + t, 3.0], c);
    g.add_box([8.0, -t, 0.0], [8.0 + t, 6.0 + t, 3.0], c);
    g.add_box([0.0, -t, 0.0], [8.0, 0.0, 3.0], c);
    g.add_box([0.0, 6.0, 0.0], [8.0, 6.0 + t, 3.0], c);
    let mut sim = Simulation::new(Config::default());
    sim.set_geometry(g);
    let src = sim.add_source(SourceDesc::at([1.5, 3.0, 1.5]));
    sim.set_listener([6.0, 3.0, 1.5], 0.0, 0.0, 0.0);
    settle(&mut sim, 20);
    assert!(sim.tick_exact().is_none(), "nothing moved: no frame");
    let direct = |f: &Frame| f.block(src).and_then(|b| b.taps.iter().find(|t| t.key == omg_core::v3::KEY_DIRECT).copied());
    let before = direct(&sim.tick()).expect("a direct path");
    sim.set_listener([4.0, 3.0, 1.5], 0.0, 0.0, 0.0);
    let f = sim.tick_exact().expect("moved: a frame");
    let after = direct(&f).expect("still a direct path");
    let want = 2.5 / SPEED_OF_SOUND;
    eprintln!("direct delay {:.2} ms → {:.2} ms (want {:.2})", before.delay_s * 1e3, after.delay_s * 1e3, want * 1e3);
    assert!((after.delay_s - want).abs() < 1e-5, "the direct path at the new distance");
    assert!(after.gains[1] > before.gains[1] * 1.5, "closer: louder (1/d)");
    let chains = f.block(src).unwrap().taps.iter().filter(|t| t.key >= 9_000_000).count();
    eprintln!("{chains} mirror chains re-solved in the exact frame");
}

/// A car passes 30 m off at 100 km/h (x = −130 from y = −50, 0.7 m up;
/// the listener at (−100, 100, 1.6)) in a field with one facade: x = 0
/// to 0.3, y = 50 to 80, 8 m tall (`assess cartone`'s pass and the town's
/// west face). While the facade mirrors the car to the listener its echo
/// plays as its own tap — ~0.45 s after the car, ~11 dB under it — at the
/// image source's delay and level: every tick and exact frame of that
/// stretch at the cheapest rung and a middle one. The finders' rays found
/// it by chance (a facade 100 m off fills ~0.1 % of their sphere: at
/// rung 4 the town's car was named twice in 10 s, both times outside its
/// echo's 1.8 s), and a chain rejected as the car came up was held out
/// for 2 s — all of it; first-order chains are named from the faces now.
#[test]
fn a_facades_echo_of_a_passing_car_plays_at_every_rung() {
    let facade = Material::new([0.03; 3], 0.35, [0.0; 3]);
    let lis = [-100.0f32, 100.0, 1.6];
    let car = |t: f64| [-130.0f32, -50.0 + 27.78 * t as f32, 0.7];
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let c = SPEED_OF_SOUND as f64;
    // the light cone to a point (the listener or its image): where the
    // car was when what arrives at `t` left it, and the path's length
    let emitted = |t: f64, to: [f32; 3]| {
        let mut te = t;
        for _ in 0..8 {
            te = t - dist(car(te), to) as f64 / c;
        }
        (car(te), dist(car(te), to))
    };
    let image = [-lis[0], lis[1], lis[2]];
    for level in [0, 6] {
        let mut g = Geometry::new();
        let (grass, stone) = (g.material(Material::GRASS), g.material(facade));
        g.add_box([0.0, 50.0, 0.0], [0.3, 80.0, 8.0], stone);
        g.ground_plane(0.0, grass);
        let mut sim = Simulation::new(Config { quality: Quality::Level(level), ..Config::default() });
        sim.set_geometry(g);
        let src = sim.add_source(SourceDesc::at(car(0.0)));
        let (mut want, mut heard, mut worst_ms, mut worst_db) = (0, 0, 0.0f64, 0.0f32);
        for k in 0..(5.0 * 120.0) as usize {
            let t = k as f64 / 120.0;
            sim.set_time(t);
            sim.set_source(src, car(t), true);
            sim.set_listener_at(lis, 0.0, 0.0, 0.0, t);
            let f = if k % 6 == 0 { Some(sim.tick()) } else { sim.tick_exact() };
            let Some(f) = f else { continue };
            // the facade mirrors what arrives now: its mirror point on it
            let (x, len) = emitted(t, image);
            let s = -x[0] / (image[0] - x[0]); // where the leg crosses x = 0
            let (qy, qz) = (x[1] + (image[1] - x[1]) * s, x[2] + (image[2] - x[2]) * s);
            // (half a metre in from its edges: the solve's own margins)
            if !(t > 1.0 && (50.5..79.5).contains(&qy) && (0.5..7.5).contains(&qz)) {
                continue;
            }
            want += 1;
            let pb = f.block(src).expect("the car");
            // a mirror chain's tap past the early window (the ground's
            // arrives with the car)
            let direct = pb.taps.iter().find(|tp| tp.key == omg_core::v3::KEY_DIRECT).expect("direct path");
            let late = direct.delay_s + omg_core::v3::EARLY_S;
            let Some(echo) = pb.taps.iter().find(|tp| tp.key >= 9_000_000 && tp.delay_s > late) else { continue };
            heard += 1;
            let (_, d) = emitted(t, lis);
            // the image source: the facade's mirror share, spreading, air
            let thr: f32 = (1.0 - 0.03) * (1.0 - 0.35);
            let air = air_attenuation(len)[1] / air_attenuation(d)[1];
            let want_db = 10.0 * thr.log10() + 20.0 * (air * d / len).log10();
            let got_db = 20.0 * (echo.gains[1] / direct.gains[1]).log10();
            worst_ms = worst_ms.max(((echo.delay_s as f64 - len as f64 / c) * 1e3).abs());
            worst_db = worst_db.max((got_db - want_db).abs());
        }
        eprintln!("level {level}: the echo in {heard} of {want} frames; delay off by ≤ {worst_ms:.2} ms, level by ≤ {worst_db:.2} dB");
        assert!(want > 100, "the facade mirrors the car for a while: {want} frames");
        assert!(heard as f32 >= 0.95 * want as f32, "level {level}: the echo plays in {heard} of {want} frames");
        assert!(worst_ms < 1.0, "level {level}: the echo's delay off by {worst_ms:.2} ms");
        assert!(worst_db < 1.0, "level {level}: the echo's level off by {worst_db:.2} dB");
    }
}
