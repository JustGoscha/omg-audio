//! The bell heard south of the cathedral (the town test's points): its
//! exact paths (direct and edge bends) per tick, and the edges they bend
//! over (DEBUG paths). TICKS=n (40)  AT=x,y,z
use omg_audio::{Config, PathKind, Quality, Simulation};
use omg_web::town::Town;

fn main() {
    let ticks: usize = std::env::var("TICKS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let at: Vec<f32> = std::env::var("AT").unwrap_or_else(|_| "12,42,1.6".into()).split(',').filter_map(|x| x.parse().ok()).collect();
    let v5 = std::env::var("V5").is_ok();
    let mut sim = Simulation::new(if v5 { Config { quality: Quality::High, kernel_thread: false, kernel_step_samples: 1 << 24, ..Config::default() } } else { Config { quality: Quality::High, v4: false, v5: false, v6: true, v61: true, v62: true, kernel_thread: false, ..Config::default() } });
    let mut town = Town::new(&mut sim);
    let bells = town.defs.iter().position(|d| d.name == "bells").expect("bells");
    for i in 0..town.defs.len() {
        town.set_muted(&mut sim, i, i != bells);
    }
    let id = town.sources[bells][0];
    println!("bell at {:?} z {}", town.defs[bells].pos, town.defs[bells].z);
    sim.set_debug_paths(true);
    sim.set_listener([at[0], at[1], at[2]], 0.0, 0.0, 0.0);
    for k in 0..ticks {
        let f = sim.tick();
        let pb = f.block(id).expect("traced");
        let e: f32 = pb.taps.iter().filter(|t| t.key < 9_000_000).map(|t| t.gains[1] * t.gains[1]).sum();
        if k % 5 == 4 || k == ticks - 1 {
            println!("tick {k}: exact {:.1} dB, {} taps", 10.0 * e.max(1e-30).log10(), pb.taps.len());
        }
    }
    for p in sim.debug_paths().iter().filter(|p| matches!(p.kind, PathKind::Bend)).take(6) {
        let pts: Vec<String> = p.points.iter().map(|v| format!("({:.1},{:.1},{:.1})", v[0], v[1], v[2])).collect();
        println!("  edge path {:.1} dB: {}", 10.0 * p.energy.max(1e-30).log10(), pts.join(" → "));
    }
}
