
pub struct Schlick {
    t: f32,
}

impl Schlick {
    pub fn new(cos_theta: f32) -> Self {
        Schlick {
            t: (1.0 - cos_theta).powf(5.0)
        }
    }

    /// Spectral operation
    pub fn eval(&self, f0: f32, f90) -> f32 {
        f0 + (f90 - f0) * t
    }
}
