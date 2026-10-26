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

    // ---- Presets for level authors -------------------------------------
    // Bands: low < 250 Hz, mid 250–2500 Hz, high > 2500 Hz. Where the
    // demo already used a material, the preset carries IDENTICAL numbers
    // (so a level moved onto the engine sounds the same); new ones come
    // from published octave-band tables, mapped low ≈ 125 Hz, mid ≈ mean
    // of 250 Hz–2 kHz, high ≈ 4 kHz.

    /// Dressed stone masonry (the demo's cathedral stone): hard, with the
    /// relief of real stonework (arcades, capitals, rough ashlar) as a
    /// high scattering fraction.
    pub const STONE: Material = Material {
        absorption: [0.015, 0.03, 0.06],
        scattering: 0.35,
        transmission: [0.10, 0.014, 0.0015],
    };
    /// Wood panelling / timber boards (= `WOOD_PANEL`).
    pub const WOOD: Material = Material::WOOD_PANEL;
    /// Single-pane window glass: reflects like a hard surface, transmits
    /// far more than any wall (the demo's pane: concrete-hard faces with
    /// `GLASS` transmission). Meant as a thin element: as an engine object
    /// (box) or a mesh material with thickness `REF_THICKNESS_M`, the
    /// transmission applies as given.
    pub const GLASS: Material = Material {
        absorption: [0.01, 0.02, 0.04],
        scattering: 0.10,
        transmission: [0.50, 0.32, 0.20],
    };
    /// Laminated / double glazing (the demo's club windows): bass thumps
    /// through, mids and highs die.
    pub const HEAVY_GLASS: Material = Material {
        absorption: [0.01, 0.02, 0.04],
        scattering: 0.10,
        transmission: [0.18, 0.05, 0.012],
    };
    /// A closed interior door leaf: ~4 cm wood panel (the demo's door).
    pub const DOOR: Material = Material {
        absorption: [0.20, 0.10, 0.08],
        scattering: 0.15,
        transmission: [0.28, 0.12, 0.05],
    };
    /// A heavy steel door: metal faces, the transmission of the demo's
    /// blast door (nearly a wall when shut).
    pub const HEAVY_DOOR: Material = Material {
        absorption: [0.05, 0.09, 0.02],
        scattering: 0.10,
        transmission: [0.10, 0.03, 0.01],
    };
    /// Rough, bare soil — the ground under a level. Absorption: "soil,
    /// rough" 0.15/0.25/0.40/0.55/0.60/0.60 (125 Hz–4 kHz; C. M. Harris,
    /// Handbook of Acoustical Measurements and Noise Control, 3rd ed.,
    /// absorption-coefficient table). Nothing audible passes through it.
    pub const EARTH: Material = Material {
        absorption: [0.15, 0.45, 0.60],
        scattering: 0.40,
        transmission: [0.010, 0.002, 0.0005],
    };
    /// Sheet steel / painted metal panels. Absorption: steel
    /// 0.05/0.10/0.10/0.10/0.07/0.02 (125 Hz–4 kHz; F. A. Everest &
    /// K. C. Pohlmann, Master Handbook of Acoustics, absorption table);
    /// transmission: the demo's heavy steel door.
    pub const METAL: Material = Material {
        absorption: [0.05, 0.09, 0.02],
        scattering: 0.10,
        transmission: [0.10, 0.03, 0.01],
    };
    /// Asphalt road surface: dense and smooth, a near-mirror ground
    /// (absorption ≈ 0.02/0.03/0.03/0.05/0.05/0.07, 125 Hz–4 kHz, the
    /// usual "asphalt, smooth" table row). Transmission: a road slab on
    /// earth passes nothing audible.
    pub const ASPHALT: Material = Material {
        absorption: [0.02, 0.04, 0.06],
        scattering: 0.10,
        transmission: [0.05, 0.01, 0.001],
    };
    /// Cobbles / setts / paving slabs: as hard as asphalt, but the joints
    /// and the relief scatter a quarter of the energy.
    pub const PAVING: Material = Material {
        absorption: [0.02, 0.04, 0.06],
        scattering: 0.25,
        transmission: [0.08, 0.012, 0.0015],
    };
    /// Smooth quarried rock — a cliff face, a quarry wall: as hard as
    /// stone, without masonry's carved relief, so it returns sound
    /// specularly (echoes stay discrete). Nothing audible passes through.
    pub const ROCK: Material = Material {
        absorption: [0.02, 0.03, 0.05],
        scattering: 0.10,
        transmission: [0.010, 0.002, 0.0005],
    };
    /// Heavy draped curtain (velour, ~0.6 kg/m², draped to half area):
    /// 0.14/0.35/0.55/0.72/0.70/0.65 (125 Hz–4 kHz; L. L. Beranek, Noise
    /// and Vibration Control, absorption table). Fabric leaks sound
    /// freely; use it as an object or a thin mesh material.
    pub const CURTAIN: Material = Material {
        absorption: [0.14, 0.58, 0.65],
        scattering: 0.50,
        transmission: [0.70, 0.50, 0.30],
    };

    /// A material from its parts: per-band energy absorption (0..1),
    /// scattering fraction (0..1) and per-band amplitude transmission
    /// through a `REF_THICKNESS_M` (15 cm) wall of it.
    pub const fn new(absorption: [f32; NBANDS], scattering: f32, transmission: [f32; NBANDS]) -> Self {
        Self { absorption, scattering, transmission }
    }

    /// The quick way: one absorption for every band, a scattering
    /// fraction, and the transmission of a solid brick wall (sound gets
    /// through muffled, as through most walls). Adjust with
    /// `with_transmission`.
    pub const fn simple(absorption: f32, scattering: f32) -> Self {
        Self {
            absorption: [absorption; NBANDS],
            scattering,
            transmission: Material::BRICK.transmission,
        }
    }

    /// This material with another per-band transmission.
    pub const fn with_transmission(mut self, transmission: [f32; NBANDS]) -> Self {
        self.transmission = transmission;
        self
    }

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
