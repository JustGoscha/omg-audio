//! omg-core: propagation simulation. No I/O, no allocation on hot paths,
//! compiles unchanged to native and wasm. Produces `ParamBlock`s that the
//! DSP layer (omg-dsp) renders — the two sides only speak through params.

pub mod ism;
pub mod material;
pub mod params;
pub mod rng;
pub mod scene;
pub mod tracer;
pub mod vec3;

pub const SPEED_OF_SOUND: f32 = 343.0;

/// Frequency bands used for all acoustic quantities.
/// Band edges: low < 250 Hz, mid 250–2500 Hz, high > 2500 Hz.
pub const NBANDS: usize = 3;

#[cfg(test)]
mod tests {
    use crate::ism::image_source_taps;
    use crate::material::Material;
    use crate::rng::Rng;
    use crate::scene::Shoebox;
    use crate::tracer::{estimate_reverb, trace, Echogram};
    use crate::vec3::Vec3;
    use crate::SPEED_OF_SOUND;

    fn room() -> Shoebox {
        Shoebox::new(Vec3::new(8.0, 6.0, 3.0), [Material::DRYWALL; 6])
    }

    #[test]
    fn ism_order3_shoebox_has_63_images() {
        let mut taps = Vec::new();
        image_source_taps(&room(), Vec3::new(2.0, 2.0, 1.5), Vec3::new(5.0, 4.0, 1.5), 3, &mut taps);
        // Known closed-form count for a shoebox: 1 + 6 + 18 + 38.
        assert_eq!(taps.len(), 63);
    }

    #[test]
    fn ism_direct_path_delay_matches_distance() {
        let src = Vec3::new(2.0, 2.0, 1.5);
        let lis = Vec3::new(5.0, 4.0, 1.5);
        let mut taps = Vec::new();
        image_source_taps(&room(), src, lis, 0, &mut taps);
        assert_eq!(taps.len(), 1);
        let expect = (src - lis).length() / SPEED_OF_SOUND;
        assert!((taps[0].delay_s - expect).abs() < 1e-6);
    }

    #[test]
    fn rt60_tracks_absorption() {
        // Same room, dead vs. live materials → RT60 must order correctly.
        let mut rng = Rng::new(7);
        let src = Vec3::new(2.0, 2.0, 1.5);
        let lis = Vec3::new(5.0, 4.0, 1.5);

        let mut rt = |mat: Material| {
            let r = Shoebox::new(Vec3::new(8.0, 6.0, 3.0), [mat; 6]);
            let mut e = Echogram::new();
            trace(&r, src, lis, 20_000, [1.0; 3], &mut rng, &mut e);
            estimate_reverb(&e).rt60[1]
        };

        let live = rt(Material::CONCRETE);
        let dead = rt(Material::ACOUSTIC_TILE);
        assert!(live > 2.0 * dead, "concrete {live} vs tile {dead}");

        // Sabine for all-drywall (α_mid=0.08): T = 0.161·V/(α·S) ≈ 1.6 s.
        let drywall = rt(Material::DRYWALL);
        assert!(drywall > 0.8 && drywall < 3.0, "drywall rt60 {drywall}");
    }

    #[test]
    fn late_level_is_distance_independent() {
        // The diffuse field does not get louder as you approach the source;
        // the direct path does. (Regression: reverb used to scale ~1/r².)
        let mut rng = Rng::new(11);
        let r = Shoebox::new(Vec3::new(10.0, 8.0, 4.0), [Material::DRYWALL; 6]);
        let src = Vec3::new(2.0, 4.0, 1.5);

        let mut level_at = |lis: Vec3| {
            let mut e = Echogram::new();
            trace(&r, src, lis, 40_000, [1.0; 3], &mut rng, &mut e);
            estimate_reverb(&e).level[1]
        };

        let near = level_at(Vec3::new(3.0, 4.0, 1.5)); // 1 m
        let far = level_at(Vec3::new(8.0, 4.0, 1.5)); // 6 m
        let ratio = near / far.max(1e-6);
        assert!(
            (0.5..2.0).contains(&ratio),
            "late level should be ~distance-independent, near/far = {ratio}"
        );
    }
}
