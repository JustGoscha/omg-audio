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
//!   at every rung, at its image source's delay and level;
//! - a large hall rings as long as Sabine and sounds as loud as diffuse-
//!   field theory says (v6.1, v5).

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
            + pb.early.level[b] * pb.early.level[b]
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
/// stretch at the cheapest rung and a middle one — and its twin off the
/// ground (the facade, then the ground on the car's side: the ground's
/// mirror share under it, 0.03 ms later), as the direct sound has its own. The finders' rays found
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
        let (mut twins, mut twin_db) = (0, 0.0f32);
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
            // its twin off the ground on the car's side (facade, then
            // ground: the ground's mirror share under it, a few hundredths
            // of a millisecond later)
            if let Some(tw) = pb.taps.iter().find(|tp| tp.key >= 9_000_000 && tp.key != echo.key && (tp.delay_s - echo.delay_s).abs() < 1e-3) {
                twins += 1;
                let gm = Material::GRASS;
                let share = (1.0 - gm.absorption[1] - gm.transmission[1] * gm.transmission[1]) * (1.0 - gm.scattering);
                twin_db = twin_db.max((20.0 * (tw.gains[1] / echo.gains[1]).log10() - 10.0 * share.log10()).abs());
            }
        }
        eprintln!("level {level}: the echo in {heard} of {want} frames; delay off by ≤ {worst_ms:.2} ms, level by ≤ {worst_db:.2} dB; its ground twin in {twins}, off the ground's share by ≤ {twin_db:.2} dB");
        assert!(want > 100, "the facade mirrors the car for a while: {want} frames");
        assert!(heard as f32 >= 0.95 * want as f32, "level {level}: the echo plays in {heard} of {want} frames");
        assert!(worst_ms < 1.0, "level {level}: the echo's delay off by {worst_ms:.2} ms");
        assert!(worst_db < 1.0, "level {level}: the echo's level off by {worst_db:.2} dB");
        assert!(twins as f32 >= 0.95 * want as f32, "level {level}: its ground twin in {twins} of {want} frames");
        assert!(twin_db < 1.0, "level {level}: the twin off the ground's share by {twin_db:.2} dB");
    }
}

/// An inside corner of two facades 100 m from the listener (8 m tall: x = 0
/// from y = 0 to 80, and y = 80 from x = −80 to 0) and a car passing at 100
/// km/h on x = −110: its echo off both walls — off the first (x = 0), then
/// the second — plays as its own tap at the double image's delay and
/// level, every tick and exact frame while both mirror points lie on the
/// walls, at the cheapest rung and a middle one — and its twins off the
/// grass (third order: the ground before, between or after the walls), a
/// few ms later at the ground's mirror share. Second order between two
/// faces was left to the finders' rays, which come across two far walls in
/// turn rarely (by the town's west road such an echo, −15 dB at 122 ms,
/// was missed: `examples/echo_ref2`); pairs of large faces name it now
/// (`Tracer::name_mirrors`), with its ground twins.
#[test]
fn an_inside_corners_echo_of_a_passing_car_plays() {
    let facade = Material::new([0.03; 3], 0.35, [0.0; 3]);
    let lis = [-100.0f32, 20.0, 1.6];
    let car = |t: f64| [-110.0f32, -10.0 + 27.78 * t as f32, 0.7];
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let c = SPEED_OF_SOUND as f64;
    let emitted = |t: f64, to: [f32; 3]| {
        let mut te = t;
        for _ in 0..8 {
            te = t - dist(car(te), to) as f64 / c;
        }
        (car(te), dist(car(te), to))
    };
    // the listener across x = 0, then across y = 80
    let (im1, image) = ([-lis[0], lis[1], lis[2]], [-lis[0], 160.0 - lis[1], lis[2]]);
    for level in [0, 6] {
        let mut g = Geometry::new();
        let (stone, grass) = (g.material(facade), g.material(Material::GRASS));
        g.add_box([0.0, 0.0, 0.0], [0.3, 80.3, 8.0], stone);
        g.add_box([-80.0, 80.0, 0.0], [0.3, 80.3, 8.0], stone);
        g.ground_plane(0.0, grass);
        let mut sim = Simulation::new(Config { quality: Quality::Level(level), ..Config::default() });
        sim.set_geometry(g);
        let src = sim.add_source(SourceDesc::at(car(0.0)));
        let (mut want, mut heard, mut worst_ms, mut worst_db) = (0, 0, 0.0f64, 0.0f32);
        let (mut twins, mut twin_db) = (0, 0.0f32);
        for k in 0..(3.5 * 120.0) as usize {
            let t = k as f64 / 120.0;
            sim.set_time(t);
            sim.set_source(src, car(t), true);
            sim.set_listener_at(lis, 0.0, 0.0, 0.0, t);
            let f = if k % 6 == 0 { Some(sim.tick()) } else { sim.tick_exact() };
            let Some(f) = f else { continue };
            let (x, len) = emitted(t, image);
            // the mirror points: x → image crosses y = 80, then on to the
            // first image it crosses x = 0
            let s2 = (80.0 - x[1]) / (image[1] - x[1]);
            let q2 = [x[0] + (image[0] - x[0]) * s2, 80.0, x[2] + (image[2] - x[2]) * s2];
            let s1 = -q2[0] / (im1[0] - q2[0]);
            let q1 = [0.0, q2[1] + (im1[1] - q2[1]) * s1, q2[2] + (im1[2] - q2[2]) * s1];
            // (half a metre in from the walls' edges)
            let on = (-79.5..-0.5).contains(&q2[0]) && (0.5..79.5).contains(&q1[1]) && (0.5..7.5).contains(&q1[2]) && (0.5..7.5).contains(&q2[2]);
            if !(t > 0.5 && on) {
                continue;
            }
            want += 1;
            let pb = f.block(src).expect("the car");
            let direct = pb.taps.iter().find(|tp| tp.key == omg_core::v3::KEY_DIRECT).expect("direct path");
            let at = len as f64 / c;
            let Some(echo) = pb.taps.iter().find(|tp| tp.key >= 9_000_000 && (tp.delay_s as f64 - at).abs() < 2e-3) else { continue };
            heard += 1;
            let (_, d) = emitted(t, lis);
            let share: f32 = (1.0 - 0.03) * (1.0 - 0.35);
            let air = air_attenuation(len)[1] / air_attenuation(d)[1];
            let want_db = 20.0 * share.log10() + 20.0 * (air * d / len).log10();
            let got_db = 20.0 * (echo.gains[1] / direct.gains[1]).log10();
            worst_ms = worst_ms.max(((echo.delay_s as f64 - at) * 1e3).abs());
            worst_db = worst_db.max((got_db - want_db).abs());
            // its twins off the ground (third order: before, between or
            // after the walls), a few ms later, the grass's share under it
            let gm = Material::GRASS;
            let gshare = (1.0 - gm.absorption[1] - gm.transmission[1] * gm.transmission[1]) * (1.0 - gm.scattering);
            let tw: Vec<&omg_audio::Tap> = pb.taps.iter().filter(|tp| tp.key >= 9_000_000 && tp.key != echo.key && tp.delay_s > echo.delay_s && tp.delay_s - echo.delay_s < 0.01).collect();
            if !tw.is_empty() {
                twins += 1;
                let e: f32 = tw.iter().map(|t| t.gains[1] * t.gains[1]).sum::<f32>() / tw.len() as f32;
                twin_db = twin_db.max((10.0 * (e / (echo.gains[1] * echo.gains[1])).log10() - 10.0 * gshare.log10()).abs());
            }
        }
        eprintln!("level {level}: the corner's echo in {heard} of {want} frames; delay off by ≤ {worst_ms:.2} ms, level by ≤ {worst_db:.2} dB; its ground twins in {twins}, their mean off the ground's share by ≤ {twin_db:.2} dB");
        assert!(want > 100, "the corner mirrors the car for a while: {want} frames");
        assert!(heard as f32 >= 0.95 * want as f32, "level {level}: the corner's echo plays in {heard} of {want} frames");
        assert!(worst_ms < 1.0, "level {level}: its delay off by {worst_ms:.2} ms");
        assert!(worst_db < 1.0, "level {level}: its level off by {worst_db:.2} dB");
        assert!(twins as f32 >= 0.95 * want as f32, "level {level}: its ground twins in {twins} of {want} frames");
        assert!(twin_db < 1.0, "level {level}: the twins off the ground's share by {twin_db:.2} dB");
    }
}

/// A street between two facades 30 m apart (x = 0 and x = 30, 200 m long,
/// 10 m tall) and a car driving down it at 100 km/h, 50–150 m off: its
/// flutter of third order, A → B → A and B → A → B (the same length, from
/// either side), plays at the triple image's delay, from each side with
/// its twin off the grass — 0.05 ms later at the ground's mirror share —
/// every frame while it arrives after the early window, at the cheapest
/// rung and a middle one, in v5 (two taps a side) and v6.2 (one: chains
/// within `ECHO_MERGE_S` are one arrival). The finders' rays name such
/// chains (`Tracer::find_chains`); their twins they came across rarely,
/// and unnamed a late chain plays nothing: before, each side played 1.4
/// dB under the flutter and its twin in every late frame. Each chain that
/// solves names its twin now — and v6.2's exact frames between full ticks
/// (`Tracer::exact_taps`) play a merged tap whole: re-solved alone, its
/// lead dropped the twin in 220 of the 264 frames.
#[test]
fn a_streets_flutter_off_a_passing_car_plays_with_its_ground_twins() {
    let facade = Material::new([0.03; 3], 0.35, [0.0; 3]);
    let lis = [5.0f32, 10.0, 1.6];
    let car = |t: f64| [25.0f32, 60.0 + 27.78 * t as f32, 0.7];
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let c = SPEED_OF_SOUND as f64;
    let emitted = |t: f64, to: [f32; 3]| {
        let mut te = t;
        for _ in 0..8 {
            te = t - dist(car(te), to) as f64 / c;
        }
        dist(car(te), to)
    };
    // the listener across x = 0, x = 30, x = 0 (B → A → B: across 30, 0,
    // 30, as far on the other side); its twin across the ground too
    let image = [-60.0 - lis[0], lis[1], lis[2]];
    let twin_image = [image[0], image[1], -lis[2]];
    let share: f32 = ((1.0 - 0.03) * (1.0 - 0.35f32)).powi(3);
    let gm = Material::GRASS;
    let gshare = (1.0 - gm.absorption[1] - gm.transmission[1] * gm.transmission[1]) * (1.0 - gm.scattering);
    for (mode, level) in [("v5", 0), ("v5", 6), ("v6.2", 0), ("v6.2", 6)] {
        let mut g = Geometry::new();
        let (stone, grass) = (g.material(facade), g.material(Material::GRASS));
        g.add_box([-0.3, 0.0, 0.0], [0.0, 200.0, 10.0], stone);
        g.add_box([30.0, 0.0, 0.0], [30.3, 200.0, 10.0], stone);
        g.ground_plane(0.0, grass);
        let v6 = mode == "v6.2";
        let mut sim = Simulation::new(Config { quality: Quality::Level(level), v4: !v6, v5: !v6, v6, v61: v6, v62: v6, ..Config::default() });
        sim.set_geometry(g);
        let src = sim.add_source(SourceDesc::at(car(0.0)));
        let (mut want, mut heard, mut worst_ms, mut worst_db, mut alone_db) = (0, 0, 0.0f64, 0.0f32, f32::MAX);
        for k in 0..(3.5 * 120.0) as usize {
            let t = k as f64 / 120.0;
            sim.set_time(t);
            sim.set_source(src, car(t), true);
            sim.set_listener_at(lis, 0.0, 0.0, 0.0, t);
            let f = if k % 6 == 0 { Some(sim.tick()) } else { sim.tick_exact() };
            let Some(f) = f else { continue };
            let (len, len_g, d) = (emitted(t, image), emitted(t, twin_image), emitted(t, lis));
            // after the early window, the solver's alone
            if !(t > 0.5 && (len - d) as f64 / c >= omg_core::v3::EARLY_S as f64) {
                continue;
            }
            want += 1;
            let pb = f.block(src).expect("the car");
            let direct = pb.taps.iter().find(|tp| tp.key == omg_core::v3::KEY_DIRECT).expect("direct path");
            let at = len as f64 / c;
            // the flutter and its twin, under the direct sound (energy)
            let rel = |l: f32| (air_attenuation(l)[1] / air_attenuation(d)[1] * d / l).powi(2);
            let (e_chain, e_twin) = (share * rel(len), share * gshare * rel(len_g));
            // from either side: A → B → A arrives off x = 0, B → A → B off
            // x = 30 — each the chain taps within 1.5 ms
            let mut sides = 0;
            for side in [-1.0f32, 1.0] {
                let near: Vec<&omg_audio::Tap> = pb.taps.iter().filter(|tp| tp.key >= 9_000_000 && tp.dir[0] * side > 0.0 && (tp.delay_s as f64 - at).abs() < 1.5e-3).collect();
                let Some(first) = near.iter().min_by(|a, b| a.delay_s.total_cmp(&b.delay_s)) else { continue };
                let e: f32 = near.iter().map(|tp| (tp.gains[1] / direct.gains[1]).powi(2)).sum();
                worst_ms = worst_ms.max(((first.delay_s as f64 - at) * 1e3).abs());
                worst_db = worst_db.max((10.0 * (e / (e_chain + e_twin)).log10()).abs());
                alone_db = alone_db.min((10.0 * (e / e_chain).log10()).abs());
                sides += 1;
            }
            if sides == 2 {
                heard += 1;
            }
        }
        eprintln!("{mode} level {level}: the flutter from both sides in {heard} of {want} late frames; delay off by ≤ {worst_ms:.3} ms, level (with its twin) by ≤ {worst_db:.2} dB, ≥ {alone_db:.2} dB off the chain alone");
        assert!(want > 200, "the flutter arrives late for a while: {want} frames");
        assert!(heard as f32 >= 0.95 * want as f32, "{mode} level {level}: the flutter plays in {heard} of {want} frames");
        assert!(worst_ms < 0.01, "{mode} level {level}: its delay off by {worst_ms:.3} ms");
        assert!(worst_db < 0.3, "{mode} level {level}: the flutter with its twin off by {worst_db:.2} dB");
    }
}

/// A large closed hall (30×20×15 m, walls absorbing 6 %): the
/// reverberation time and the reverberant level each engine plays,
/// against Sabine's time and diffuse-field theory's level — both with
/// the air's absorption over the long paths (ISO 9613-1 at the mid band:
/// 4mV in the absorption area).
fn hall_against_theory(mode: &str) -> (f32, f32, f32, f32) {
    let (lx, ly, lz) = (30.0f32, 20.0, 15.0);
    let alpha = 0.06f32;
    let mut g = Geometry::new();
    let m = g.material(Material::new([alpha; 3], 0.3, [0.0; 3]));
    let t = 0.3;
    g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
    g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
    g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
    g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
    g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
    g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
    let cfg = Config {
        quality: Quality::High,
        v4: mode == "v5",
        v5: mode == "v5",
        v6: mode.starts_with("v6"),
        v61: mode == "v6.1" || mode == "v6.2",
        v62: mode == "v6.2",
        ..Config::default()
    };
    let mut sim = Simulation::new(cfg);
    sim.set_geometry(g);
    let (sp, lp) = ([8.0f32, 7.0, 1.5], [21.0f32, 12.0, 1.6]);
    let src = sim.add_source(SourceDesc::at(sp));
    sim.set_listener(lp, 0.0, 0.0, 0.0);
    let f = settle(&mut sim, 60);
    let pb = f.block(src).unwrap();
    let r = ((sp[0] - lp[0]).powi(2) + (sp[1] - lp[1]).powi(2) + (sp[2] - lp[2]).powi(2)).sqrt();
    let (v, s) = (lx * ly * lz, 2.0 * (lx * ly + lx * lz + ly * lz));
    // the air's energy attenuation per metre at the mid band
    let m_air = -(air_attenuation(1.0)[1].powi(2)).ln();
    let a = s * alpha + 4.0 * m_air * v;
    let sabine = 0.161 * v / a;
    let direct = pb.taps.iter().find(|tp| tp.key == omg_core::v3::KEY_DIRECT).map_or(0.0, |tp| tp.gains[1] * tp.gains[1]);
    // everything reflected: the early arrivals and the late field
    let reflected: f32 = pb.taps.iter().filter(|tp| tp.key != omg_core::v3::KEY_DIRECT).map(|tp| tp.gains[1] * tp.gains[1]).sum::<f32>()
        + match sim.dense_response(src) {
            Some(d) => d.bins.iter().map(|x| x[1]).sum::<f32>(),
            None => pb.reverb.level[1].powi(2) + pb.remote.as_ref().map_or(0.0, |rr| rr.send[1].powi(2)) + pb.early.level[1].powi(2),
        };
    let rt = match sim.dense_response(src) {
        Some(d) => dense_rt60(d),
        None => pb.reverb.rt60[1],
    };
    let ratio_db = 10.0 * (reflected / direct).log10();
    let theory_db = 10.0 * (16.0 * std::f32::consts::PI * r * r / a).log10();
    eprintln!("{mode}: rt60 {rt:.2} s (Sabine {sabine:.2} s); reflected over direct {ratio_db:+.1} dB (diffuse theory {theory_db:+.1} dB)");
    (rt, sabine, ratio_db, theory_db)
}

/// v6.1, v6.2 and v5 play a hall as long and as loud as theory says:
/// within 15 % of Sabine's time and 3 dB of diffuse-field theory's
/// reflected level (measured: 6.50 / 6.32 s against 7.25; +14.7 / +15.1
/// dB against +16.9 for v6.1 / v5 — the trace's field a little less
/// diffuse than theory's; v6.2, its dense early field played, nearer).
/// v6 played it 5.00 s (clamped) and +6.5 dB: its late level in the
/// echogram's legacy units, 12 dB under (ENGINE_V4.md, "Level").
#[test]
fn a_hall_rings_and_sounds_as_long_and_loud_as_theory_says() {
    hall_against_theory("v6");
    for mode in ["v6.1", "v6.2", "v5"] {
        let (rt, sabine, ratio, theory) = hall_against_theory(mode);
        assert!((rt / sabine - 1.0).abs() < 0.15, "{mode}: rt60 {rt:.2} s against Sabine's {sabine:.2} s");
        assert!((ratio - theory).abs() < 3.0, "{mode}: reflected {ratio:+.1} dB over direct, theory {theory:+.1} dB");
    }
}

/// `room_with_doorway` with a leaf in its doorway covering `1 − open` of
/// it, and the doorway as the leaf's aperture (v6.2): the room's source
/// heard outside by `ears` — through the doorway, off its straight line:
/// what the estimate (early taps, reverb) carries.
fn doorway_scene(mode: &str, ears: [f32; 3], open: f32) -> (Simulation, omg_audio::SourceId, omg_audio::ObjectId) {
    let mut sim = Simulation::new(Config { quality: Quality::High, v6: true, v61: true, v62: mode == "v6.2", kernel_thread: false, ..Config::default() });
    sim.set_geometry(room_with_doorway());
    let src = sim.add_source(SourceDesc::at([2.0, 1.5, 1.5]));
    sim.set_listener(ears, 0.0, 0.0, 0.0);
    let door = sim.add_object([6.08, 2.0, 0.0], [6.12, 3.0, 2.1], Material::DOOR);
    sim.set_object_aperture(door, [6.08, 2.0, 0.0], [6.12, 3.0, 2.1]);
    set_leaf(&mut sim, door, open);
    (sim, src, door)
}

fn set_leaf(sim: &mut Simulation, door: omg_audio::ObjectId, open: f32) {
    sim.set_object(door, [6.08, 2.0, 0.0], [6.12, 2.0 + (1.0 - open).max(1e-3), 2.1]);
    sim.set_object_enabled(door, open < 0.999);
}

/// A door swinging shut (openness e^(−7 t), as `assess doorswing`'s),
/// heard against the door standing at each openness: the swing's level at
/// every frame (ticks at 20 Hz, exact frames at 120 Hz between) against
/// the standing door's at that frame's openness (averaged, interpolated
/// in log openness). Returns the rms and the mean difference (dB).
fn swing_against_standing(mode: &str, ears: [f32; 3]) -> (f32, f32) {
    let db = |e: [f32; 3]| 10.0 * e[1].max(1e-20).log10();
    let opens = [1.0f32, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0];
    let truth: Vec<f32> = opens
        .iter()
        .map(|&o| {
            let (mut sim, src, _) = doorway_scene(mode, ears, o);
            settle(&mut sim, 30);
            let mut acc = 0.0f32;
            for _ in 0..80 {
                acc += energy(sim.tick().block(src).unwrap())[1] / 80.0;
            }
            10.0 * acc.max(1e-20).log10()
        })
        .collect();
    let standing = |o: f32| -> f32 {
        let i = opens.iter().position(|&x| x <= o).unwrap_or(opens.len() - 1).max(1);
        let (a, b) = (opens[i - 1], opens[i]);
        let f = if b > 0.0 { (o.ln() - a.ln()) / (b.ln() - a.ln()) } else { (a - o) / a };
        truth[i - 1] + f.clamp(0.0, 1.0) * (truth[i] - truth[i - 1])
    };
    let (mut sim, src, door) = doorway_scene(mode, ears, 1.0);
    let frame = 1.0 / 120.0;
    let mut t = 0.0f64;
    for k in 0..240 {
        sim.set_time(t);
        if k % 6 == 0 {
            sim.tick();
        }
        t += frame;
    }
    let (mut sq, mut sum, mut n) = (0.0f32, 0.0f32, 0);
    let mut level = 0.0f32;
    for k in 0..150 {
        let s = k as f32 * frame as f32;
        let mut open = (-7.0 * s).exp();
        if open < 0.001 {
            open = 0.0;
        }
        set_leaf(&mut sim, door, open);
        sim.set_time(t);
        let f = if k % 6 == 0 { Some(sim.tick()) } else { sim.tick_exact() };
        if let Some(f) = f {
            level = db(energy(f.block(src).unwrap()));
        }
        let d = level - standing(open);
        sq += d * d;
        sum += d;
        n += 1;
        t += frame;
    }
    ((sq / n as f32).sqrt(), sum / n as f32)
}

#[test]
fn a_door_swinging_shut_is_heard_where_it_is() {
    // in front of the doorway, and beside it (seeing it at a slant)
    let ears = [[8.0, 6.0, 1.6], [7.5, 2.8, 1.6], [7.2, 1.2, 1.6], [7.0, 3.8, 1.6]];
    let (mut e61, mut e62) = (0.0f32, 0.0f32);
    for e in ears {
        let (rms61, mean61) = swing_against_standing("v6.1", e);
        let (rms62, mean62) = swing_against_standing("v6.2", e);
        eprintln!("ears {e:?}: the swing against the door standing: v6.1 rms {rms61:.1} dB (mean {mean61:+.1}), v6.2 rms {rms62:.1} dB (mean {mean62:+.1})");
        e61 += rms61 / ears.len() as f32;
        e62 += rms62 / ears.len() as f32;
    }
    eprintln!("mean rms: v6.1 {e61:.1} dB, v6.2 {e62:.1} dB");
    assert!(e62 < e61 - 0.5, "v6.2 follows the door closer than v6.1: {e62:.1} vs {e61:.1} dB");
    assert!(e62 < 3.5, "v6.2 within 3.5 dB rms of the door standing where it is: {e62:.1} dB");
}

/// A loud room heard through its walls alone (8 × 6 × 3 m, concrete, no
/// door): what comes through is reached through the walls — at the
/// draw's floor, one ray in fifty, carrying × 50: a heavy-tailed few (the
/// same room's level up to 4 dB apart between seeds over 20–30 s
/// standing, its one-second means swinging over 5–7 dB). v6.2 also walks
/// on through a wall a sounding source is enclosed behind, a quarter of
/// the time (× 4): three seeds within 1.5 dB, steadier second to second
/// (here the worst seed's sd 1.77 → 0.75 dB).
#[test]
fn a_room_heard_through_its_walls_settles() {
    let run = |v62: bool, seed: u64| -> (f32, f32) {
        let mut g = Geometry::new();
        let w = g.material(Material::CONCRETE);
        let t = 0.2;
        let (lx, ly, lz) = (8.0, 6.0, 3.0);
        g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], w);
        g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], w);
        g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], w);
        g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], w);
        g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], w);
        g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], w);
        let grass = g.material(Material::GRASS);
        g.ground_plane(-t, grass);
        let mut sim = Simulation::new(Config { quality: Quality::Level(3), v6: true, v61: true, v62, seed, kernel_thread: false, ..Config::default() });
        sim.set_geometry(g);
        let src = sim.add_source(SourceDesc::at([3.0, 3.0, 1.5]));
        sim.set_listener([13.0, 3.0, 1.6], 0.0, 0.0, 0.0);
        let mut late = Vec::new();
        for _ in 0..400 {
            let f = sim.tick();
            let pb = f.block(src).unwrap();
            late.push(pb.reverb.level[1] * pb.reverb.level[1] + pb.remote.map_or(0.0, |r| r.send[1] * r.send[1]));
        }
        let db = |x: f32| 10.0 * x.max(1e-30).log10();
        // one-second means past the first two, their spread; the last 10 s
        let secs: Vec<f32> = late.chunks(20).skip(2).map(|c| db(c.iter().sum::<f32>() / 20.0)).collect();
        let m = secs.iter().sum::<f32>() / secs.len() as f32;
        let sd = (secs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (secs.len() - 1) as f32).sqrt();
        (db(late[200..].iter().sum::<f32>() / 200.0), sd)
    };
    let span = |v62: bool| {
        let r: Vec<(f32, f32)> = (1..=3).map(|s| run(v62, s)).collect();
        let lv: Vec<f32> = r.iter().map(|x| x.0).collect();
        let span = lv.iter().cloned().fold(f32::MIN, f32::max) - lv.iter().cloned().fold(f32::MAX, f32::min);
        (span, r.iter().map(|x| x.1).fold(0.0f32, f32::max), lv)
    };
    let (s61, sd61, l61) = span(false);
    let (s62, sd62, l62) = span(true);
    eprintln!("through the walls, three seeds: v6.1 levels {l61:.1?} (span {s61:.1} dB, 1-s sd ≤ {sd61:.2}), v6.2 {l62:.1?} (span {s62:.1} dB, 1-s sd ≤ {sd62:.2})");
    assert!(s62 < 1.5, "v6.2: the seeds agree within 1.5 dB ({s62:.1})");
    assert!(sd62 < 1.0, "v6.2: steady second to second ({sd62:.2} dB)");
    assert!(sd62 < sd61, "v6.2 steadier than v6.1: {sd62:.2} vs {sd61:.2} dB");
}

/// A 10 × 7 × 4 m shoebox (α 0.1 / 0.2 / 0.3) against its image sources
/// (Allen & Berkley, `omg_core::ism`, to order 40: the exact sum of
/// arrivals for mirror walls): the energy of the direct sound, the first
/// 80 ms and the rest, per band. v6.2 within 1.5 dB early and 2.5 dB late
/// (measured −0.6 … −1.3 / −0.6 … −1.8 dB, `examples/shoebox_ref`);
/// v6.1, which played only the strongest 24 early bins, 5–8 dB short of
/// the early energy (the dense early field, ENGINE_V6.md).
#[test]
fn a_shoebox_plays_its_image_sources_early_and_late() {
    use omg_core::vec3::Vec3;
    let (size, alpha, sp, lp) = ([10.0f32, 7.0, 4.0], [0.1f32, 0.2, 0.3], [2.5f32, 2.0, 1.5], [7.0f32, 4.5, 1.6]);
    let mat = Material::new(alpha, 0.0, [0.0; 3]);
    let t_direct = ((sp[0] - lp[0]).powi(2) + (sp[1] - lp[1]).powi(2) + (sp[2] - lp[2]).powi(2)).sqrt() / omg_core::SPEED_OF_SOUND;
    let room = omg_core::scene::Shoebox::new(Vec3::new(size[0], size[1], size[2]), [mat; 6]);
    let mut taps = Vec::new();
    omg_core::ism::image_source_taps(&room, Vec3::new(sp[0], sp[1], sp[2]), Vec3::new(lp[0], lp[1], lp[2]), 40, &mut taps);
    let parts = |taps: &[omg_core::params::Tap], onset: f32| -> [[f64; 3]; 3] {
        let mut p = [[0.0f64; 3]; 3];
        for t in taps {
            let k = if (t.delay_s - onset).abs() < 1e-5 { 0 } else if t.delay_s - onset < 0.08 { 1 } else { 2 };
            for b in 0..3 {
                p[k][b] += (t.gains[b] as f64).powi(2);
            }
        }
        p
    };
    let ism = parts(&taps, t_direct);
    let engine = |mode: &str| -> [[f64; 3]; 3] {
        let mut g = Geometry::new();
        let m = g.material(mat);
        let t = 0.3;
        let (lx, ly, lz) = (size[0], size[1], size[2]);
        g.add_box([-t, -t, -t], [lx + t, ly + t, 0.0], m);
        g.add_box([-t, -t, lz], [lx + t, ly + t, lz + t], m);
        g.add_box([-t, -t, 0.0], [0.0, ly + t, lz], m);
        g.add_box([lx, -t, 0.0], [lx + t, ly + t, lz], m);
        g.add_box([0.0, -t, 0.0], [lx, 0.0, lz], m);
        g.add_box([0.0, ly, 0.0], [lx, ly + t, lz], m);
        let cfg = Config { quality: Quality::High, v6: true, v61: true, v62: mode == "v6.2", kernel_thread: false, ..Config::default() };
        let mut sim = Simulation::new(cfg);
        sim.set_geometry(g);
        let src = sim.add_source(SourceDesc::at(sp));
        sim.set_listener(lp, 0.0, 0.0, 0.0);
        let f = settle(&mut sim, 60);
        let pb = f.block(src).unwrap();
        let mut p = [[0.0f64; 3]; 3];
        for tp in &pb.taps {
            let k = if tp.key == omg_core::v3::KEY_DIRECT { 0 } else if tp.delay_s - pb.onset_s < 0.08 { 1 } else { 2 };
            for b in 0..3 {
                p[k][b] += (tp.gains[b] as f64).powi(2);
            }
        }
        for b in 0..3 {
            p[1][b] += (pb.early.level[b] as f64).powi(2);
            p[2][b] += (pb.reverb.level[b] as f64).powi(2) + pb.remote.map_or(0.0, |r| (r.send[b] as f64).powi(2));
        }
        p
    };
    let off = |p: &[[f64; 3]; 3], k: usize| -> [f64; 3] { core::array::from_fn(|b| 10.0 * (p[k][b] / ism[k][b]).log10()) };
    let (e62, e61) = (engine("v6.2"), engine("v6.1"));
    eprintln!("shoebox against its image sources, dB: v6.2 direct {:?} early {:?} late {:?}; v6.1 early {:?}", off(&e62, 0), off(&e62, 1), off(&e62, 2), off(&e61, 1));
    for b in 0..3 {
        assert!(off(&e62, 0)[b].abs() < 0.1, "the direct sound is exact");
        assert!(off(&e62, 1)[b].abs() < 1.5, "v6.2 early {:?}", off(&e62, 1));
        assert!(off(&e62, 2)[b].abs() < 2.5, "v6.2 late {:?}", off(&e62, 2));
        assert!(off(&e61, 1)[b] < -4.0, "v6.1 played the early part whole? {:?}", off(&e61, 1));
    }
}
