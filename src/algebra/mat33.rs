use super::{Vec3};
use core::ops::Mul;

#[derive(Debug, Copy, Clone)]
pub struct Mat33<T> {
    pub x: Vec3<T>,
    pub y: Vec3<T>,
    pub z: Vec3<T>,
}

impl Mul<Mat33<f32>> for Vec3<f32> {
    type Output = Self;
    fn mul(self, rhs: Mat33<f32>) -> Self::Output {
        Vec3 {
            x: self.dot(rhs.x),
            y: self.dot(rhs.y),
            z: self.dot(rhs.z),
        }
    }
}