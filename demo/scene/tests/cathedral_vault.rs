//! The cathedral is a church now: 15 m walls carry a round barrel vault
//! (crown 23 m) under a steep gable roof (eaves 19 m, ridge 28.5 m) —
//! `Roof::Nave`, real mesh the tracers see, not a scattering stand-in.
//!
//! Claims:
//! - the world mesh has that section: straight up from the nave floor a
//!   ray meets the vault at its crown; from the sky, the ridge;
//! - the room's legacy numbers follow the shape (eaves over the side
//!   walls, the ridge mid-span, ~41% more air than the flat-ceiled box);
//! - A/B against the same nave under its old flat 15 m ceiling, the
//!   traced tail rings LONGER — what the extra volume should do. Mid
//!   band sits at the engine's 5 s RT60 ceiling in both, so the claim
//!   is checked where the decay is resolved (the treble), and the mid
//!   band must not get shorter.

use omg_core::v3::{Budget, Source, Tracer, World};
use omg_core::vec3::Vec3;
use omg_scene::dome::build_world_mesh;
use omg_scene::walkthrough::{self, Roof, CATHEDRAL};

fn rooms(vaulted: bool) -> Vec<walkthrough::RoomDef> {
    let mut rooms = walkthrough::rooms();
    if !vaulted {
        // the old cathedral: flat slab at 15 m, parapet to 16.5 m
        rooms[CATHEDRAL].roof = Roof::Flat;
        rooms[CATHEDRAL].barrier_height = 16.5;
    }
    rooms
}

#[test]
fn the_mesh_has_a_vault_and_a_gable_roof() {
    let (mesh, _) = build_world_mesh(&rooms(true), &walkthrough::doors());
    // up from the nave's center line: the vault crown (8 facets: the
    // crown is a facet corner at 15 + 8 = 23 m)
    let (t, _) = mesh.raycast(Vec3::new(8.0, 63.0, 1.6), Vec3::new(0.0, 0.0, 1.0)).expect("a vault overhead");
    assert!((1.6 + t - 23.0).abs() < 0.05, "vault crown at {} m", 1.6 + t);
    // halfway out, the vault is lower than the crown, above the springing
    let (t, _) = mesh.raycast(Vec3::new(3.0, 63.0, 1.6), Vec3::new(0.0, 0.0, 1.0)).unwrap();
    assert!((19.0..23.0).contains(&(1.6 + t)), "vault at x = 3: {} m", 1.6 + t);
    // down from the sky: the roof ridge, then the eaves side
    let (t, _) = mesh.raycast(Vec3::new(8.0, 63.0, 40.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
    assert!((40.0 - t - 28.5).abs() < 0.05, "ridge at {} m", 40.0 - t);
    let (t, _) = mesh.raycast(Vec3::new(1.0, 63.0, 40.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
    let want = 19.0 + 9.5 / 8.0;
    assert!((40.0 - t - want).abs() < 0.05, "roof at x = 1: {} m, want {want}", 40.0 - t);
    // the portal gable closes the attic: level through it at 25 m the
    // segment from the forecourt into the roof space hits the wall
    let (t, _) = mesh.raycast(Vec3::new(8.0, 40.0, 25.0), Vec3::new(0.0, 1.0, 0.0)).expect("gable wall");
    assert!((40.0 + t - 52.0).abs() < 0.05, "gable plane at y = {}", 40.0 + t);
}

#[test]
fn legacy_numbers_follow_the_shape() {
    let (vault, flat) = (rooms(true), rooms(false));
    let (c, f) = (&vault[CATHEDRAL], &flat[CATHEDRAL]);
    assert_eq!(f.volume(), 16.0 * 22.0 * 15.0);
    let gain = c.volume() / f.volume();
    eprintln!("cathedral volume {:.0} → {:.0} m³ (×{gain:.2})", f.volume(), c.volume());
    assert!((1.35..1.45).contains(&gain), "vault adds ~41% air: ×{gain}");
    assert!((c.interior_top() - 23.0).abs() < 1e-3);
    // what sound passing over must clear: eaves over the side walls,
    // the ridge mid-span, the gable outline between
    assert!((c.roof_line_at(0.0, 60.0) - 19.0).abs() < 1e-4);
    assert!((c.roof_line_at(8.0, 60.0) - 28.5).abs() < 1e-4);
    assert!((c.roof_line_at(4.0, 52.0) - 23.75).abs() < 1e-4);
    // the flat room is unchanged everywhere
    assert_eq!(f.roof_line_at(4.0, 52.0), 16.5);
    assert_eq!(f.box_height(), 15.0);
    // the shoebox engines get an equal-volume box
    assert!((c.box_height() * 16.0 * 22.0 - c.volume()).abs() < 1.0);
    // a listener up in the vault still belongs to the cathedral
    assert_eq!(walkthrough::room_of_z(&vault, 8.0, 63.0, 21.0), CATHEDRAL);
}

#[test]
fn the_vault_rings_longer_than_the_flat_ceiling() {
    let doors = walkthrough::doors();
    let (src, lis) = (Vec3::new(8.0, 66.0, 1.5), Vec3::new(8.0, 60.0, 1.6));
    let rt = |vaulted: bool| {
        let (mesh, _) = build_world_mesh(&rooms(vaulted), &doors);
        let world = World { grade: Some(0.0), ..World::new(&mesh, &[]) };
        let mut t = Tracer::new(7);
        let budget = Budget {
            rays: 1024,
            bounces: 4,
            k: 0.06,
            ema: 0.5,
            nee: 0,
            late_rays: 2048,
            late_ema: 0.5,
            late_nee: 0,
            diffraction: false,
        };
        let mut f = None;
        for _ in 0..8 {
            f = Some(t.trace(&world, lis, &[Source { id: 0, pos: src, weight: 1.0 }], budget).remove(0));
        }
        f.unwrap().reverb.rt60
    };
    let (flat, vault) = (rt(false), rt(true));
    eprintln!("traced RT60 flat ceiling {flat:?} → vaulted {vault:?}");
    assert!(vault[2] > 1.1 * flat[2], "treble tail must lengthen: {flat:?} → {vault:?}");
    assert!(vault[1] >= flat[1] - 0.05, "mid tail must not shorten: {flat:?} → {vault:?}");
}
