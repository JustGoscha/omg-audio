use crate::material::Material;
use crate::vec3::Vec3;

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
