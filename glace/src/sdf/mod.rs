use crate::{f32x2, Gl};

/// Signed distance from a point.
pub trait Sdf2d {
    /// Circle.
    fn circle(&self, radius: f32) -> f32;

    /// Line segment.
    fn line(&self, a: f32x2, b: f32x2) -> f32;
}

impl Sdf2d for f32x2 {
    fn circle(&self, radius: f32) -> f32 {
        self.length() - radius
    }

    fn line(&self, a: f32x2, b: f32x2) -> f32 {
        let a2p = *self - a;
        let a2b = b - a;

        // project a-p onto a-b
        let t = a2p.dot(a2b) / a2b.dot(a2b);
        (a2p - a2b * Gl::clamp(t, 0.0, 1.0)).length()
    }
}
