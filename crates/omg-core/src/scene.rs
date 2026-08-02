use crate::material::Material;
use crate::vec3::Vec3;

/// A surface hit inside acoustic geometry.
pub struct GeomHit {
    pub t: f32,
    /// Unit normal at the hit, oriented AGAINST the incoming ray
    /// (n·d < 0), so reflection code needs no winding convention.
    pub normal: Vec3,
    pub material: Material,
}

/// Anything the stochastic tracer can bounce rays around in. Implemented
/// by the analytic `Shoebox` and by arbitrary triangle `Mesh`es — the
/// tracer is identical over both.
pub trait AcousticGeometry {
    /// Nearest surface hit from `p` along normalized `d`.
    fn raycast_hit(&self, p: Vec3, d: Vec3) -> Option<GeomHit>;
}

/// Wall indices: 0=x·min 1=x·max 2=y·min 3=y·max 4=z·min (floor) 5=z·max (ceiling).
/// Room occupies [0,size.x] × [0,size.y] × [0,size.z].
/// Axes: +x forward, +y left, +z up (listener faces +x for now).
#[derive(Clone, Debug)]
pub struct Shoebox {
    pub size: Vec3,
    pub walls: [Material; 6],
}

impl Shoebox {
    pub fn new(size: Vec3, walls: [Material; 6]) -> Self {
        Self { size, walls }
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.x > 0.0 && p.x < self.size.x && p.y > 0.0 && p.y < self.size.y && p.z > 0.0 && p.z < self.size.z
    }

    /// Nearest wall hit for a ray starting inside the box.
    /// Returns (t, wall_index). Direction must be normalized.
    pub fn raycast(&self, p: Vec3, d: Vec3) -> (f32, usize) {
        let mut best_t = f32::MAX;
        let mut best_w = 0;
        for axis in 0..3 {
            let di = d.get(axis);
            if di > 1e-9 {
                let t = (self.size.get(axis) - p.get(axis)) / di;
                if t < best_t {
                    best_t = t;
                    best_w = 2 * axis + 1;
                }
            } else if di < -1e-9 {
                let t = -p.get(axis) / di;
                if t < best_t {
                    best_t = t;
                    best_w = 2 * axis;
                }
            }
        }
        (best_t, best_w)
    }
}

impl AcousticGeometry for Shoebox {
    fn raycast_hit(&self, p: Vec3, d: Vec3) -> Option<GeomHit> {
        // a ray that TRANSMITTED through a wall has left the box's
        // world: outside is void (the mesh geometry has a real outside;
        // the legacy box does not — without this, CPU and GPU invented
        // different phantom walls out there)
        if !self.contains(p) {
            return None;
        }
        let (t, wall) = self.raycast(p, d);
        if !t.is_finite() || t <= 0.0 {
            return None;
        }
        let mut normal = Vec3::new(0.0, 0.0, 0.0);
        // inward-facing = against a ray leaving the interior
        normal.set(wall / 2, if wall % 2 == 0 { 1.0 } else { -1.0 });
        Some(GeomHit { t, normal, material: self.walls[wall] })
    }
}

/// A base geometry plus transient axis-aligned box blockers with their
/// own materials — door leaves and glass panes. What fills a doorway is
/// world STATE, not authored mesh, so the tracer sees it as an overlay:
/// rays hit the nearest of base or box, boxes reflect/absorb like any
/// surface. (C6d: this is how closing a door reshapes the late field
/// with zero portal code.)
pub struct WithPanels<'a, G> {
    pub base: &'a G,
    /// (min, max, material) per box.
    pub panels: &'a [(Vec3, Vec3, Material)],
}

impl<G: AcousticGeometry> AcousticGeometry for WithPanels<'_, G> {
    fn raycast_hit(&self, p: Vec3, d: Vec3) -> Option<GeomHit> {
        let mut best = self.base.raycast_hit(p, d);
        for (mn, mx, m) in self.panels {
            // slab entry test; track the axis we enter through
            let (mut t0, mut t1) = (0.0f32, f32::MAX);
            let mut axis = 3usize;
            for a in 0..3 {
                let (da, pa, lo, hi) = (d.get(a), p.get(a), mn.get(a), mx.get(a));
                if da.abs() < 1e-9 {
                    if pa < lo || pa > hi {
                        t0 = f32::MAX;
                        break;
                    }
                } else {
                    let (mut ta, mut tb) = ((lo - pa) / da, (hi - pa) / da);
                    if ta > tb {
                        core::mem::swap(&mut ta, &mut tb);
                    }
                    if ta > t0 {
                        t0 = ta;
                        axis = a;
                    }
                    t1 = t1.min(tb);
                    if t0 > t1 {
                        t0 = f32::MAX;
                        break;
                    }
                }
            }
            if t0 > 1e-4
                && t0 < f32::MAX
                && axis < 3
                && best.as_ref().map_or(true, |h| t0 < h.t)
            {
                let mut normal = Vec3::new(0.0, 0.0, 0.0);
                normal.set(axis, if d.get(axis) > 0.0 { -1.0 } else { 1.0 });
                best = Some(GeomHit { t: t0, normal, material: *m });
            }
        }
        best
    }
}
