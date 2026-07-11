use crate::NBANDS;

/// Per-band acoustic surface properties. Absorption is the fraction of
/// *energy* lost per reflection; scattering is the fraction of reflected
/// energy that leaves diffusely (Lambertian) instead of specularly.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub absorption: [f32; NBANDS],
    pub scattering: f32,
    /// Amplitude transmitted THROUGH a 15 cm wall of this material, per
    /// band (mass law: lows pass, highs die — the "club rumble" physics).
    /// Other thicknesses: `transmission_at(thickness)`.
    pub transmission: [f32; NBANDS],
}

pub const REF_THICKNESS_M: f32 = 0.15;

impl Material {
    pub const CONCRETE: Material = Material {
        absorption: [0.01, 0.02, 0.04],
        scattering: 0.10,
        transmission: [0.10, 0.014, 0.0015],
    };
    pub const DRYWALL: Material = Material {
        absorption: [0.20, 0.08, 0.05],
        scattering: 0.10,
        transmission: [0.20, 0.045, 0.0150],
    };
    pub const CARPET: Material = Material {
        absorption: [0.05, 0.30, 0.60],
        scattering: 0.40,
        transmission: [0.30, 0.100, 0.0300],
    };
    pub const WOOD_PANEL: Material = Material {
        absorption: [0.20, 0.10, 0.08],
        scattering: 0.15,
        transmission: [0.22, 0.060, 0.0200],
    };
    pub const ACOUSTIC_TILE: Material = Material {
        absorption: [0.30, 0.70, 0.80],
        scattering: 0.30,
        transmission: [0.30, 0.100, 0.0300],
    };
    /// Plastered single-leaf brick — the European interior/exterior wall.
    pub const BRICK: Material = Material {
        absorption: [0.02, 0.03, 0.04],
        scattering: 0.12,
        transmission: [0.045, 0.008, 0.0012],
    };
    pub const GRASS: Material = Material {
        absorption: [0.11, 0.26, 0.60],
        scattering: 0.50,
        transmission: [0.10, 0.020, 0.0050],
    };

    /// Mass-law thickness scaling: transmission loss grows ~6 dB per
    /// doubling of mass, i.e. amplitude ∝ ref/thickness (NOT compounding —
    /// that would double the dB loss per doubling).
    pub fn transmission_at(&self, thickness_m: f32) -> [f32; NBANDS] {
        let k = (REF_THICKNESS_M / thickness_m.max(0.02)).min(2.0);
        core::array::from_fn(|b| (self.transmission[b] * k).min(0.9))
    }

    /// Per-band amplitude (not energy) reflection factor.
    pub fn reflection_amplitude(&self) -> [f32; NBANDS] {
        let mut r = [0.0; NBANDS];
        for b in 0..NBANDS {
            r[b] = (1.0 - self.absorption[b]).max(0.0).sqrt();
        }
        r
    }
}

/// Frequency-dependent air absorption, amplitude decay per meter
/// (crude fit of ISO 9613-1 at 20 °C / 50 % RH for the three bands).
pub const AIR_ABSORPTION_PER_M: [f32; NBANDS] = [1.0e-4, 5.0e-4, 3.0e-3];

pub fn air_attenuation(dist: f32) -> [f32; NBANDS] {
    let mut a = [0.0; NBANDS];
    for b in 0..NBANDS {
        a[b] = (-AIR_ABSORPTION_PER_M[b] * dist).exp();
    }
    a
}
