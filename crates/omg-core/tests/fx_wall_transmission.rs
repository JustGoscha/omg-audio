//! Effect: sound transmission THROUGH walls — the stochastic tracer's
//! mass-law ray branch (`tracer::trace`, "club boom through walls") and
//! the material model behind it (`Material::transmission`,
//! `Material::transmission_at`).
//!
//! Physical claim guaranteed: with two closed rooms sharing one wall,
//! the late field of the neighbour room is fed ONLY through that wall;
//! the leaked energy scales with the wall's energy transmission t²,
//! comes out bass-heavy (lows pass, highs die — relative to the source
//! room's own spectrum), costs the source room energy, and vanishes
//! exactly when the wall is opaque. The material model follows the mass
//! law: higher frequency ⇒ less transmission, heavier wall ⇒ less, and
//! each doubling of thickness costs 6 dB of amplitude (not compounding).
//!
//! Combined in the demo: `omg_scene::world::WorldSim` late field of the
//! club / neighbour rooms (walkthrough "boom next door" checks), and the
//! early-path wall crossings in `omg_scene` pt_mesh_gate.

mod common;

use common::{band_energy, db_e, flat_opaque, flat_transmissive, two_rooms, v};
use omg_core::material::{Material, REF_THICKNESS_M};
use omg_core::mesh::Mesh;
use omg_core::rng::Rng;
use omg_core::tracer::{trace, Echogram};
use omg_core::vec3::Vec3;
use omg_core::NBANDS;

const SRC: Vec3 = Vec3::new(2.0, 2.5, 1.5);
const LIS_A: Vec3 = Vec3::new(4.0, 3.2, 1.6); // same room as the source
const LIS_B: Vec3 = Vec3::new(9.0, 2.5, 1.6); // next door

/// Outer shell: flat 10 % absorption, perfectly opaque — any energy in
/// room B can only have come through the party wall.
const OUTER: Material = flat_opaque(0.10, 0.3);

fn party(t: [f32; NBANDS]) -> Material {
    flat_transmissive(0.10, 0.3, t)
}

fn energy(mesh: &Mesh, lis: Vec3, seed: u64) -> [f32; NBANDS] {
    let mut rng = Rng::new(seed);
    let mut e = Echogram::new();
    trace(mesh, SRC, lis, 20_000, [1.0; 3], &mut rng, &mut e);
    band_energy(&e)
}

#[test]
fn opaque_party_wall_leaks_nothing() {
    let mesh = two_rooms(6.0, 5.0, 3.0, OUTER, party([0.0; NBANDS]), None);
    let b = energy(&mesh, LIS_B, 1);
    assert_eq!(b, [0.0; NBANDS], "energy reached room B through an opaque wall: {b:?}");
    // sanity: the source room itself is alive
    let a = energy(&mesh, LIS_A, 1);
    assert!(a.iter().all(|&x| x > 1e-3), "source room silent: {a:?}");
}

#[test]
fn leak_through_wall_is_bass_heavy() {
    // mass-law-shaped wall: lows pass, highs die (energy t² = 9e-2, 1e-2, 9e-4)
    let t = [0.3, 0.1, 0.03];
    let mesh = two_rooms(6.0, 5.0, 3.0, OUTER, party(t), None);
    let a = energy(&mesh, LIS_A, 2);
    let b = energy(&mesh, LIS_B, 3);
    assert!(b.iter().all(|&x| x > 0.0), "no energy next door: {b:?}");
    // spectrum next door RELATIVE to the source room's own spectrum:
    // walls are frequency-flat, so the only tilt is the wall's t²
    let rel: [f32; NBANDS] = core::array::from_fn(|k| db_e(b[k] / a[k]));
    assert!(rel[0] > rel[1] && rel[1] > rel[2], "not bass-heavy: rel {rel:?} dB");
    // low-vs-high tilt ≈ 10·log10(0.09 / 0.0009) = 20 dB
    let tilt = rel[0] - rel[2];
    assert!((tilt - 20.0).abs() < 3.0, "low-high tilt {tilt:.1} dB, want ≈ 20 dB (rel {rel:?})");
    let tilt_lm = rel[0] - rel[1];
    assert!((tilt_lm - 9.54).abs() < 2.5, "low-mid tilt {tilt_lm:.1} dB, want ≈ 9.5 dB");
    // and the leak is a real attenuation, not a copy of the source room:
    // low band ≥ 6 dB down, high band ≥ 20 dB down
    assert!(rel[0] < -6.0 && rel[2] < -20.0, "leak too loud: rel {rel:?} dB");
}

#[test]
fn leaked_energy_scales_with_t_squared() {
    // doubling the amplitude transmission ⇒ ≈ 4× (6 dB) more energy
    // next door (the branch weight is t²/p — unbiased for any p)
    let lo = two_rooms(6.0, 5.0, 3.0, OUTER, party([0.1; NBANDS]), None);
    let hi = two_rooms(6.0, 5.0, 3.0, OUTER, party([0.2; NBANDS]), None);
    let e_lo = energy(&lo, LIS_B, 4);
    let e_hi = energy(&hi, LIS_B, 5);
    for k in 0..NBANDS {
        let step = db_e(e_hi[k] / e_lo[k]);
        assert!((step - 6.02).abs() < 1.5, "band {k}: t×2 gave {step:.2} dB, want ≈ +6 dB");
    }
}

#[test]
fn transmission_drains_the_source_room() {
    // energy that leaves through the wall is gone from room A
    let opaque = two_rooms(6.0, 5.0, 3.0, OUTER, party([0.0; NBANDS]), None);
    let leaky = two_rooms(6.0, 5.0, 3.0, OUTER, party([0.6; NBANDS]), None);
    let a0 = energy(&opaque, LIS_A, 6);
    let a1 = energy(&leaky, LIS_A, 6);
    for k in 0..NBANDS {
        let d = db_e(a1[k] / a0[k]);
        assert!(d < -0.5, "band {k}: leaky wall only changed room A by {d:.2} dB");
    }
}

// ------------------------------------------------ material mass law

const BUILTIN: [(&str, Material); 7] = [
    ("concrete", Material::CONCRETE),
    ("drywall", Material::DRYWALL),
    ("carpet", Material::CARPET),
    ("wood", Material::WOOD_PANEL),
    ("tile", Material::ACOUSTIC_TILE),
    ("brick", Material::BRICK),
    ("grass", Material::GRASS),
];

#[test]
fn higher_frequency_transmits_less() {
    for (name, m) in BUILTIN {
        let t = m.transmission;
        assert!(t[0] > t[1] && t[1] > t[2], "{name}: transmission {t:?} not low > mid > high");
        for th in [0.05f32, 0.15, 0.3] {
            let ta = m.transmission_at(th);
            assert!(ta[0] >= ta[1] && ta[1] >= ta[2], "{name} @ {th} m: {ta:?}");
        }
        // energy conservation per hit: absorbed + transmitted ≤ 1
        for b in 0..NBANDS {
            assert!(m.absorption[b] + t[b] * t[b] <= 1.0, "{name} band {b}");
        }
    }
}

#[test]
fn heavier_walls_transmit_less() {
    // concrete and brick (heavy) vs drywall and wood panel (light)
    for heavy in [Material::CONCRETE, Material::BRICK] {
        for light in [Material::DRYWALL, Material::WOOD_PANEL] {
            for b in 0..NBANDS {
                assert!(
                    heavy.transmission[b] < light.transmission[b],
                    "band {b}: heavy {:?} vs light {:?}",
                    heavy.transmission,
                    light.transmission
                );
            }
        }
    }
}

#[test]
fn transmission_at_follows_mass_law_thickness() {
    let m = Material::BRICK;
    // reference thickness reproduces the authored table
    let r = m.transmission_at(REF_THICKNESS_M);
    for b in 0..NBANDS {
        assert!((r[b] - m.transmission[b]).abs() < 1e-7);
    }
    // −6 dB per doubling of thickness (amplitude halves, not squares)
    let t1 = m.transmission_at(0.15);
    let t2 = m.transmission_at(0.30);
    let t4 = m.transmission_at(0.60);
    for b in 0..NBANDS {
        assert!((t2[b] / t1[b] - 0.5).abs() < 1e-5, "band {b}: 0.15→0.30 ratio {}", t2[b] / t1[b]);
        assert!((t4[b] / t2[b] - 0.5).abs() < 1e-5, "band {b}: 0.30→0.60 ratio {}", t4[b] / t2[b]);
    }
    // monotonic non-increasing over a thickness sweep, and bounded
    for (_, m) in BUILTIN {
        let mut prev = m.transmission_at(0.001);
        for i in 1..100 {
            let th = 0.01 * i as f32;
            let t = m.transmission_at(th);
            for b in 0..NBANDS {
                assert!(t[b] <= prev[b] + 1e-7, "thicker transmitted more at {th} m");
                assert!(t[b] >= 0.0 && t[b] <= 0.9, "out of range {}", t[b]);
            }
            prev = t;
        }
    }
    // thin-wall clamp: never more than 2× the reference (≤ +6 dB)
    let thin = m.transmission_at(0.0);
    for b in 0..NBANDS {
        assert!(thin[b] <= 2.0 * m.transmission[b] + 1e-7);
    }
}

#[test]
fn shared_wall_geometry_is_closed() {
    // guard for the fixture itself: a ray fired from room A in any
    // direction always hits something (no holes in the shells)
    let mesh = two_rooms(6.0, 5.0, 3.0, OUTER, party([0.0; NBANDS]), None);
    let mut rng = Rng::new(9);
    for _ in 0..2000 {
        let d = rng.unit_sphere();
        assert!(mesh.raycast(SRC, d).is_some(), "ray escaped along {d:?}");
        assert!(mesh.raycast(v(9.0, 2.5, 1.5), d).is_some());
    }
}

#[test]
fn authored_thickness_overrides_the_geometry() {
    // A single-plane wall (no back face to pair with) is priced at the
    // nominal 0.1 m — unless its material carries a thickness override,
    // which then wins over whatever the geometry says (paired or not).
    use omg_core::mesh::MeshBuilder;
    use omg_core::paths::segment_transmission;
    let m = Material::BRICK;
    let build = |thick: f32, solid: bool| {
        let mut mb = MeshBuilder::new();
        let id = mb.material(m);
        if solid {
            mb.abox(v(4.9, -50.0, -50.0), v(5.1, 50.0, 50.0), id); // 0.2 m slab
        } else {
            mb.quad(v(5.0, -50.0, -50.0), v(5.0, 50.0, -50.0), v(5.0, 50.0, 50.0), v(5.0, -50.0, 50.0), id);
        }
        mb.set_thickness(id, thick);
        mb.build()
    };
    let mut hits = Vec::new();
    let (a, b) = (v(0.0, 0.0, 1.0), v(10.0, 0.0, 1.0));
    let plane = segment_transmission(&build(0.0, false), a, b, &mut hits);
    let slab = segment_transmission(&build(0.0, true), a, b, &mut hits);
    let plane_ovr = segment_transmission(&build(0.3, false), a, b, &mut hits);
    let slab_ovr = segment_transmission(&build(0.3, true), a, b, &mut hits);
    for band in 0..NBANDS {
        assert!((plane[band] - m.transmission_at(0.1)[band]).abs() < 1e-7);
        assert!((slab[band] - m.transmission_at(0.2)[band]).abs() < 1e-6);
        assert!((plane_ovr[band] - m.transmission_at(0.3)[band]).abs() < 1e-7);
        assert!((slab_ovr[band] - m.transmission_at(0.3)[band]).abs() < 1e-7, "a paired slab is still ONE wall");
    }
    assert_eq!(build(0.3, false).material_thickness(0), Some(0.3));
    assert_eq!(build(0.0, false).material_thickness(0), None);
}

// ------------------------------------------------ porous layers

/// A layer that reflects nothing (a drape, a mineral-wool panel) is no
/// mass-law partition: a 10 cm absorber box in a mesh costs its own
/// transmission, one pass — not the mass law's × 1.5 for a wall thinner
/// than 15 cm — and in front of a concrete wall the two passes give the
/// panel's installed absorption, 0.84 / 0.99 / 0.97.
#[test]
fn absorber_panel_is_priced_per_pass() {
    use omg_core::mesh::MeshBuilder;
    use omg_core::paths::segment_transmission;
    assert!(Material::ABSORBER.reflects_nothing() && Material::DRAPE.reflects_nothing());
    assert!(!Material::CONCRETE.reflects_nothing() && !Material::CURTAIN.reflects_nothing());
    let mut b = MeshBuilder::new();
    let m = b.material(Material::ABSORBER);
    // a 10 cm panel across x = 0 … 0.1
    let p = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    for x in [0.0f32, 0.1] {
        b.begin_surface();
        b.quad(p(x, -2.0, -2.0), p(x, 2.0, -2.0), p(x, 2.0, 2.0), p(x, -2.0, 2.0), m);
    }
    let mesh = b.build();
    let mut hits = Vec::new();
    let t = segment_transmission(&mesh, p(-1.0, 0.0, 0.0), p(1.0, 0.0, 0.0), &mut hits);
    let installed = [0.84f32, 0.99, 0.97];
    for k in 0..NBANDS {
        assert!((t[k] - Material::ABSORBER.transmission[k]).abs() < 1e-6, "band {k}: {t:?}");
        let e = t[k] * t[k];
        let alpha = 1.0 - e * 0.98 * e;
        assert!((alpha - installed[k]).abs() < 0.005, "band {k}: installed α {alpha}");
    }
}
