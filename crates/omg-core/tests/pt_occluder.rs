//! PT-early C5 gate: things IN the room. A pillar stands between the
//! source and a listener walking through its shadow — the direct tap
//! must hand off to the knife-edge bend with NO level step (adjacent
//! samples within 3 dB), keep its identity (same tap key throughout),
//! attenuate meaningfully at shadow center, and recover on the far
//! side. Blocked reflections must simply cease to exist.

use omg_core::material::Material;
use omg_core::pt::{record_for_occ, solve_chain_occ, Aabb, OPAQUE};
use omg_core::scene::Shoebox;
use omg_core::vec3::Vec3;

fn db(a: f32) -> f32 {
    20.0 * a.max(1e-9).log10()
}

fn setup() -> (Shoebox, Vec3, Aabb) {
    let room = Shoebox::new(
        Vec3::new(8.0, 6.0, 3.0),
        [Material::CONCRETE; 6],
    );
    let src = Vec3::new(1.5, 3.0, 1.5);
    // a chest-high pillar mid-room, wide enough to cast a real shadow
    let pillar = Aabb {
        min: Vec3::new(3.6, 2.2, 0.0),
        max: Vec3::new(4.4, 3.8, 2.2),
        transmission: OPAQUE,
    };
    (room, src, pillar)
}

#[test]
fn walking_through_the_shadow_never_steps() {
    let (room, src, pillar) = setup();
    let occ = [pillar];

    let mut prev_db: Option<f32> = None;
    let mut min_db = f32::MAX;
    let mut clear_db = f32::MIN;
    let mut key0 = None;
    // walk across the shadow at x = 6.5 (the pillar is between)
    let n = 120;
    for i in 0..=n {
        let y = 0.5 + 5.0 * i as f32 / n as f32;
        let lis = Vec3::new(6.5, y, 1.6);
        let r = record_for_occ(&room, &[], 0, src, lis, &occ)
            .expect("direct path must always exist (bent or straight)");
        match key0 {
            None => key0 = Some(r.key()),
            Some(k) => assert_eq!(k, r.key(), "direct tap re-keyed mid-walk"),
        }
        let level = db(r.gains[1]);
        if let Some(p) = prev_db {
            assert!(
                (level - p).abs() < 3.0,
                "level step at y={y:.2}: {p:.1} → {level:.1} dB"
            );
        }
        prev_db = Some(level);
        min_db = min_db.min(level);
        clear_db = clear_db.max(level);
    }
    // the shadow must actually shadow (mid band, chest-high pillar)
    assert!(
        clear_db - min_db > 4.0,
        "no audible shadow: clear {clear_db:.1} dB vs min {min_db:.1} dB"
    );
}

#[test]
fn blocked_reflections_cease_and_clear_ones_survive() {
    let (room, src, pillar) = setup();
    let occ = [pillar];
    let lis = Vec3::new(6.5, 3.0, 1.6); // dead center of the shadow

    // straight line blocked ⇒ solve refuses the direct chain
    assert!(solve_chain_occ(&room, &[], src, lis, &occ).is_none());
    // ...but record_for_occ bends it instead
    let bent = record_for_occ(&room, &[], 0, src, lis, &occ).unwrap();
    let clear = record_for_occ(&room, &[], 0, src, lis, &[]).unwrap();
    assert!(bent.delay_s > clear.delay_s, "bent path must be longer");
    assert!(
        db(bent.gains[2]) < db(clear.gains[2]) - 6.0,
        "treble must shadow hard"
    );
    assert!(
        db(bent.gains[0]) - db(bent.gains[2]) > 3.0,
        "bass must wrap better than treble (got low {:.1} dB, high {:.1} dB)",
        db(bent.gains[0]),
        db(bent.gains[2])
    );

    // the ceiling bounce clears the pillar and must survive untouched
    let ceil = record_for_occ(&room, &[5], 0, src, lis, &occ).expect("ceiling blocked");
    let ceil_free = record_for_occ(&room, &[5], 0, src, lis, &[]).unwrap();
    assert!((db(ceil.gains[1]) - db(ceil_free.gains[1])).abs() < 0.5);
    // a floor bounce midway hits the pillar: only the stone's through-
    // transmission seeps (−54 dB mid for OPAQUE) — audibly gone
    assert!(solve_chain_occ(&room, &[4], src, lis, &occ).is_none());
    let floor = record_for_occ(&room, &[4], 0, src, lis, &occ);
    let floor_free = record_for_occ(&room, &[4], 0, src, lis, &[]).unwrap();
    if let Some(f) = floor {
        assert!(
            db(f.gains[1]) < db(floor_free.gains[1]) - 30.0,
            "blocked floor bounce barely attenuated"
        );
    }
}
