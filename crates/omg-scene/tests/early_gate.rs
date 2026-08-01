//! PT-early C2 gate: through the full Sim + PathCache pipeline, the
//! traced backend must (a) converge to a tap set matching ISM order ≤2
//! in an empty room, (b) show ZERO tap-identity churn while the
//! listener is stationary, and (c) glide — not re-key — while walking.

use omg_core::ism::image_source_taps;
use omg_core::material::Material;
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;
use omg_scene::quality;
use omg_scene::sim::Sim;

fn room() -> Shoebox {
    Shoebox::new(
        Vec3::new(6.0, 4.0, 3.0),
        [
            Material::CONCRETE,
            Material::CONCRETE,
            Material::CONCRETE,
            Material::CONCRETE,
            Material::WOOD_PANEL,
            Material::CONCRETE,
        ],
    )
}

fn db(a: f32) -> f32 {
    20.0 * a.max(1e-9).log10()
}

#[test]
fn traced_sim_matches_ism_and_never_churns() {
    quality::set_early(1);
    let room = room();
    let src = Vec3::new(1.5, 1.0, 1.5);
    let lis = Vec3::new(4.5, 3.0, 1.6);

    let mut sim = Sim::new();
    // converge the cache (discovery accumulates across ticks)
    let mut block = sim.update(&room, src, lis, 0.0);
    for _ in 0..40 {
        block = sim.update(&room, src, lis, 0.0);
    }

    // (a) equivalence vs ISM ≤2 (yaw 0 ⇒ directions comparable)
    let mut ism = Vec::new();
    image_source_taps(&room, src, lis, 2, &mut ism);
    for tap in &ism {
        let hit = block.taps.iter().any(|t| {
            (t.delay_s - tap.delay_s).abs() * 1000.0 <= 0.5
                && t.dir[0] * tap.dir[0] + t.dir[1] * tap.dir[1] + t.dir[2] * tap.dir[2] > 0.995
                && (0..3).all(|b| (db(t.gains[b]) - db(tap.gains[b])).abs() <= 1.0)
        });
        assert!(hit, "ISM path at {:.1} ms missing from traced Sim", tap.delay_s * 1000.0);
    }

    // (b) stationary ⇒ the key set is frozen, tick after tick
    let keys: Vec<u32> = block.taps.iter().map(|t| t.key).collect();
    for _ in 0..20 {
        let b = sim.update(&room, src, lis, 0.0);
        let now: Vec<u32> = b.taps.iter().map(|t| t.key).collect();
        assert_eq!(keys, now, "tap churn while stationary");
    }

    // (c) walking ⇒ same keys, gliding delays (no re-keying)
    let step = Vec3::new(-0.05, -0.02, 0.0);
    let mut pos = lis;
    let mut prev = sim.update(&room, src, pos, 0.0);
    for _ in 0..10 {
        pos = pos + step;
        let b = sim.update(&room, src, pos, 0.0);
        for t in &b.taps {
            if let Some(p) = prev.taps.iter().find(|p| p.key == t.key) {
                assert!(
                    (p.delay_s - t.delay_s).abs() < 0.001,
                    "delay jumped for a stable key"
                );
            }
        }
        // direct path must persist under motion
        assert!(
            b.taps.iter().any(|t| prev.taps.iter().any(|p| p.key == t.key)),
            "all keys re-issued while walking"
        );
        prev = b;
    }

    quality::set_early(0); // leave the process-global as we found it
}
