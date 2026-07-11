/// xorshift64* — tiny deterministic RNG, no dependencies, wasm-friendly.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform direction on the unit sphere.
    pub fn unit_sphere(&mut self) -> crate::vec3::Vec3 {
        let z = 1.0 - 2.0 * self.next_f32();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = core::f32::consts::TAU * self.next_f32();
        crate::vec3::Vec3::new(r * phi.cos(), r * phi.sin(), z)
    }
}
