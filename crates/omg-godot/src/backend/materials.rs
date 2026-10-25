//! Acoustic material names a level tags its geometry with: omg-audio's
//! presets plus a few the demo town needs (furniture, marble, acoustic
//! ceiling tiles). Transmission is the whole element's (omg-audio pairs a
//! wall's front and back faces to price its thickness).

use omg_audio::Material;

const fn m(absorption: [f32; 3], scattering: f32, transmission: [f32; 3]) -> Material {
    Material::new(absorption, scattering, transmission)
}

pub const PRESETS: &[(&str, Material)] = &[
    // omg-audio presets
    ("stone", Material::STONE),
    ("brick", Material::BRICK),
    ("concrete", Material::CONCRETE),
    ("wood", Material::WOOD),
    ("glass", Material::GLASS),
    ("heavy_glass", Material::HEAVY_GLASS),
    ("door", Material::DOOR),
    ("heavy_door", Material::HEAVY_DOOR),
    ("carpet", Material::CARPET),
    ("curtain", Material::CURTAIN),
    ("drywall", Material::DRYWALL),
    ("metal", Material::METAL),
    ("grass", Material::GRASS),
    ("earth", Material::EARTH),
    ("acoustic_tile", Material::ACOUSTIC_TILE),
    // aliases and extras
    ("plaster", Material::BRICK),
    ("wood_door", Material::DOOR),
    ("marble", m([0.01, 0.01, 0.02], 0.08, [0.08, 0.01, 0.001])),
    ("asphalt", m([0.02, 0.04, 0.06], 0.10, [0.05, 0.01, 0.001])),
    // cobbles / setts: hard, the joints scatter
    ("paving", m([0.02, 0.04, 0.06], 0.25, [0.08, 0.012, 0.0015])),
    // furniture (the walkthrough scene's furniture materials)
    ("soft", m([0.35, 0.60, 0.70], 0.65, [0.5, 0.25, 0.1])),
    ("furniture_wood", m([0.10, 0.10, 0.12], 0.40, [0.5, 0.3, 0.15])),
    ("cabinet", m([0.12, 0.14, 0.18], 0.35, [0.3, 0.1, 0.03])),
    ("books", m([0.25, 0.42, 0.55], 0.80, [0.15, 0.04, 0.01])),
    ("panel", m([0.18, 0.10, 0.06], 0.15, [0.4, 0.2, 0.08])),
];

pub fn names() -> impl Iterator<Item = &'static str> {
    PRESETS.iter().map(|(n, _)| *n)
}

pub fn get(name: &str) -> Option<Material> {
    PRESETS.iter().find(|(n, _)| *n == name).map(|(_, m)| *m)
}

#[cfg(test)]
mod tests {
    #[test]
    fn presets_resolve() {
        assert!(super::get("brick").is_some() && super::get("heavy_door").is_some());
        assert!(super::get("nope").is_none());
        assert_eq!(super::names().count(), super::PRESETS.len());
    }
}
