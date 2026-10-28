//! The engine's material presets (`omg_core::material::Material::*`)
//! carry the demo's numbers exactly where the demo already had them — a
//! level moved from the demo onto the engine must sound the same.

use omg_core::material::Material;
use omg_scene::walkthrough as wt;

#[test]
fn presets_match_the_demo_materials() {
    assert_eq!(Material::GLASS.transmission, wt::GLASS_TRANSMISSION);
    assert_eq!(Material::HEAVY_GLASS.transmission, wt::HEAVY_GLASS_TRANSMISSION);
    assert_eq!(Material::DOOR.transmission, wt::DOOR_PANEL_TRANSMISSION);
    assert_eq!(Material::HEAVY_DOOR.transmission, wt::HEAVY_PANEL_TRANSMISSION);
    // the demo's panes are concrete-hard; its door leaves are wood panel
    assert_eq!(Material::GLASS.absorption, Material::CONCRETE.absorption);
    assert_eq!(Material::GLASS.scattering, Material::CONCRETE.scattering);
    assert_eq!(Material::DOOR.absorption, Material::WOOD_PANEL.absorption);
    assert_eq!(Material::WOOD.absorption, Material::WOOD_PANEL.absorption);
}

#[test]
fn presets_are_physical() {
    let all = [
        Material::STONE,
        Material::BRICK,
        Material::CONCRETE,
        Material::WOOD,
        Material::GLASS,
        Material::HEAVY_GLASS,
        Material::CARPET,
        Material::GRASS,
        Material::EARTH,
        Material::METAL,
        Material::CURTAIN,
        Material::DRYWALL,
        Material::DOOR,
        Material::HEAVY_DOOR,
    ];
    for m in all {
        assert!(m.absorption.iter().all(|a| (0.0..=1.0).contains(a)));
        assert!((0.0..=1.0).contains(&m.scattering));
        // mass law: the bass passes most; above it a thin plate's
        // coincidence dip (glass near 3 kHz) may lift the highs a little
        // over the mids, never far
        let t = m.transmission;
        assert!(t[0] >= t[1] && t[0] >= t[2] && t[2] <= 1.2 * t[1], "{m:?}");
    }
    let s = Material::simple(0.3, 0.2);
    assert_eq!(s.absorption, [0.3; 3]);
    assert_eq!(Material::new([0.1; 3], 0.5, [0.0; 3]).with_transmission([0.2; 3]).transmission, [0.2; 3]);
}
