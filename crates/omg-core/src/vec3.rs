#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let l = self.length();
        if l > 1e-12 {
            self * (1.0 / l)
        } else {
            Self::new(1.0, 0.0, 0.0)
        }
    }

    pub fn get(self, axis: usize) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    pub fn set(&mut self, axis: usize, v: f32) {
        match axis {
            0 => self.x = v,
            1 => self.y = v,
            _ => self.z = v,
        }
    }
}

impl core::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl core::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl core::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
