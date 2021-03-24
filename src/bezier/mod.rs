#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Line1d {
    pub p0: f32,
    pub p1: f32,
}

impl Line1d {
    #[inline]
    pub fn eval(&self, t: f32) -> f32 {
        self.p0 + (self.p1 - self.p0) * t
    }

    #[inline]
    pub fn raycast(&self, p: f32) -> f32 {
        (p - self.p0) / (self.p1 - self.p0)
    }
}

pub struct QuadMonotonic1d {
    pub p0: f32,
    pub p1: f32,
    pub p2: f32,
}

impl QuadMonotonic1d {
    pub fn eval(&self, t: f32) -> f32 {
        (1.0 - t) * (1.0 - t) * self.p0 + 2.0 * t * (1.0 - t) * self.p1 + t * t * self.p2
    }

    pub fn raycast(&self, p: f32) -> f32 {
        let a = self.p0 - 2.0 * self.p1 + self.p2;

        if a.abs() < 0.0001 {
            Line1d { p0: self.p0, p1: self.p2 }.raycast(p)
        } else {
            let b = self.p0 - self.p1;
            let c = self.p0 - p;
            let dscr_sq = b * b - a * c;
            let sign = (self.p2 > p) as i32 - (self.p0 > p) as i32;

            (b + sign as f32 * dscr_sq.sqrt()) / a
        }
    }
}