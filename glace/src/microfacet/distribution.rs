#[derive(Debug, Copy, Clone)]
pub struct Ggx {
    /// Isotropic, "width"/"roughness"
    pub alpha: f32,
}

impl Ggx {
    pub fn pdf(&self, n_dot_m: f32) -> f32 {
        // Based upon [Karis2013] Listening 2
        let r2 = self.alpha * self.alpha;
        let f = (n_dot_m * r2 - n_dot_m) * n_dot_m + 1.0;
        r2 / (f * f * f32::PI)
    }
}
