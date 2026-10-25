//! Level geometry for the acoustic simulation: triangles with materials.
//!
//! Hand the engine the same triangles you render (or a simplified
//! collision mesh — acoustics needs walls, floors and roofs, not
//! door-handles), tagged with a material each. Things that move or open —
//! doors, windows, crates — are *objects* on the `Simulation`, not
//! geometry: geometry is static and rebuilding it costs a BVH build.
//!
//! Coordinates: meters, right-handed, **+z up** (`from_y_up` converts
//! a y-up engine's points: three.js, Godot, Unity after a handedness flip).

use omg_core::material::Material;
use omg_core::mesh::{Mesh, MeshBuilder};
use omg_core::vec3::Vec3;

/// A material registered with a `Geometry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

/// Half-size of the ground plane (m): far beyond anything audible.
const GROUND_HALF: f32 = 2000.0;
/// Margin around the level where the ground keeps fine tessellation.
const GROUND_FINE_MARGIN: f32 = 50.0;
/// Tessellation of the far ground (m): rays rarely travel there.
const GROUND_COARSE_EDGE: f32 = 64.0;

/// A level's static acoustic geometry. Build it once (or whenever the
/// level changes) and pass it to `Simulation::set_geometry`.
pub struct Geometry {
    mb: MeshBuilder,
    n_materials: u32,
    tris: usize,
    lo: Vec3,
    hi: Vec3,
    ground: Option<(f32, MaterialId)>,
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a point from a y-up right-handed engine (three.js, Godot) to
/// the engine's z-up frame: (x, y, z) → (x, −z, y).
pub fn from_y_up(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[2], p[1]]
}

fn v3(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[1], p[2])
}

impl Geometry {
    pub fn new() -> Self {
        let big = f32::INFINITY;
        Self {
            mb: MeshBuilder::new(),
            n_materials: 0,
            tris: 0,
            lo: Vec3::new(big, big, big),
            hi: Vec3::new(-big, -big, -big),
            ground: None,
        }
    }

    /// Register a material (a preset like `Material::BRICK`, or your own
    /// via `Material::new` / `Material::simple`).
    pub fn material(&mut self, m: Material) -> MaterialId {
        assert!(self.n_materials < u16::MAX as u32, "at most 65535 materials");
        self.n_materials += 1;
        MaterialId(self.mb.material(m) as u32)
    }

    /// Price every wall of this material as `thickness_m` thick (mass law)
    /// instead of inferring it from the geometry. The inference pairs a
    /// wall's front and back faces; single-plane walls (no back face)
    /// default to 0.1 m. 0 restores the inference.
    pub fn set_thickness(&mut self, material: MaterialId, thickness_m: f32) {
        self.mb.set_thickness(material.0 as u16, thickness_m);
    }

    fn check(&self, m: MaterialId) {
        assert!(m.0 < self.n_materials, "unknown material {m:?}");
    }

    fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, m: MaterialId) {
        self.lo = self.lo.min(a).min(b).min(c);
        self.hi = self.hi.max(a).max(b).max(c);
        self.mb.tri(a, b, c, m.0 as u16);
        self.tris += 1;
    }

    /// An indexed triangle mesh, one material for all of it. `indices`
    /// holds three vertex indices per triangle (winding does not matter).
    pub fn add_mesh(&mut self, positions: &[[f32; 3]], indices: &[u32], material: MaterialId) {
        self.check(material);
        for t in indices.chunks_exact(3) {
            let p = |i: u32| v3(positions[i as usize]);
            self.tri(p(t[0]), p(t[1]), p(t[2]), material);
        }
    }

    /// An indexed triangle mesh with one material per triangle
    /// (`materials.len() == indices.len() / 3`).
    pub fn add_mesh_per_tri(&mut self, positions: &[[f32; 3]], indices: &[u32], materials: &[MaterialId]) {
        assert_eq!(materials.len(), indices.len() / 3, "one material per triangle");
        for (t, &m) in indices.chunks_exact(3).zip(materials) {
            self.check(m);
            let p = |i: u32| v3(positions[i as usize]);
            self.tri(p(t[0]), p(t[1]), p(t[2]), m);
        }
    }

    /// A solid axis-aligned box (a wall slab, a pillar, a building block).
    pub fn add_box(&mut self, min: [f32; 3], max: [f32; 3], material: MaterialId) {
        self.check(material);
        let (a, b) = (v3(min), v3(max));
        let v = Vec3::new;
        let quads = [
            [v(a.x, a.y, a.z), v(a.x, b.y, a.z), v(a.x, b.y, b.z), v(a.x, a.y, b.z)],
            [v(b.x, a.y, a.z), v(b.x, a.y, b.z), v(b.x, b.y, b.z), v(b.x, b.y, a.z)],
            [v(a.x, a.y, a.z), v(a.x, a.y, b.z), v(b.x, a.y, b.z), v(b.x, a.y, a.z)],
            [v(a.x, b.y, a.z), v(b.x, b.y, a.z), v(b.x, b.y, b.z), v(a.x, b.y, b.z)],
            [v(a.x, a.y, a.z), v(b.x, a.y, a.z), v(b.x, b.y, a.z), v(a.x, b.y, a.z)],
            [v(a.x, a.y, b.z), v(a.x, b.y, b.z), v(b.x, b.y, b.z), v(b.x, a.y, b.z)],
        ];
        for q in quads {
            self.tri(q[0], q[1], q[2], material);
            self.tri(q[0], q[2], q[3], material);
        }
    }

    /// Open ground at height `z` reaching to the horizon, and solid earth
    /// below it (sound does not pass under the ground). Optional: a level
    /// whose own floor geometry covers the playable area needs none, but
    /// outdoors the ground reflection is half of what makes a place sound
    /// like outside. Use `Material::GRASS`, `Material::EARTH`, … .
    pub fn ground_plane(&mut self, z: f32, material: MaterialId) {
        self.check(material);
        self.ground = Some((z, material));
    }

    /// Triangles added so far (the ground plane excluded).
    pub fn triangle_count(&self) -> usize {
        self.tris
    }

    /// Build the mesh (and the grade, when there is a ground plane).
    pub(crate) fn build(mut self) -> (Mesh, Option<f32>) {
        let grade = self.ground.map(|g| g.0);
        if let Some((z, m)) = self.ground {
            let (lo, hi) = if self.tris > 0 {
                (self.lo, self.hi)
            } else {
                (Vec3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, z))
            };
            let f = GROUND_FINE_MARGIN;
            let (x0, x1) = ((lo.x - f).max(-GROUND_HALF), (hi.x + f).min(GROUND_HALF));
            let (y0, y1) = ((lo.y - f).max(-GROUND_HALF), (hi.y + f).min(GROUND_HALF));
            let g = GROUND_HALF;
            let m = m.0 as u16;
            let quad = |mb: &mut MeshBuilder, u0: f32, u1: f32, v0: f32, v1: f32| {
                if u1 > u0 && v1 > v0 {
                    mb.quad(Vec3::new(u0, v0, z), Vec3::new(u1, v0, z), Vec3::new(u1, v1, z), Vec3::new(u0, v1, z), m);
                }
            };
            self.mb.begin_surface();
            // the level and its surroundings: wall-grade patches
            quad(&mut self.mb, x0, x1, y0, y1);
            // the far field to the horizon, coarse
            self.mb.coarse(GROUND_COARSE_EDGE);
            quad(&mut self.mb, -g, g, -g, y0);
            quad(&mut self.mb, -g, g, y1, g);
            quad(&mut self.mb, -g, x0, y0, y1);
            quad(&mut self.mb, x1, g, y0, y1);
            self.mb.coarse(0.0);
        }
        (self.mb.build(), grade)
    }
}
