use super::Vec4;
#[cfg(not(target_arch = "spirv"))]
use super::{vec3, vec4, Vec3};
use core::ops::Mul;

#[derive(Debug, Copy, Clone)]
pub struct Mat44<T> {
    pub x: Vec4<T>,
    pub y: Vec4<T>,
    pub z: Vec4<T>,
    pub w: Vec4<T>,
}

impl Mat44<f32> {
    // Right handed coordinate system
    #[cfg(not(target_arch = "spirv"))]
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let scale = 1.0 / (0.5 * fov).tan();
        Mat44 {
            x: vec4(scale / aspect, 0.0, 0.0, 0.0),
            y: vec4(0.0, scale, 0.0, 0.0),
            z: vec4(0.0, 0.0, -far / (far - near), -(far * near) / (far - near)),
            w: vec4(0.0, 0.0, -1.0, 0.0),
        }
    }

    // Right handed coordinate system
    #[cfg(not(target_arch = "spirv"))]
    pub fn perspective_inv(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let inv_scale = (0.5 * fov).tan();
        Mat44 {
            x: vec4(inv_scale * aspect, 0.0, 0.0, 0.0),
            y: vec4(0.0, inv_scale, 0.0, 0.0),
            z: vec4(0.0, 0.0, 0.0, -1.0),
            w: vec4(0.0, 0.0, (near - far) / (far * near), 1.0 / near),
        }
    }

    // Right handed coordinate system
    #[cfg(not(target_arch = "spirv"))]
    pub fn look_at(eye: Vec3<f32>, dir: Vec3<f32>) -> Self {
        let up = vec3(0.0, 1.0, 0.0);
        let fwd = dir.normalize();
        let right = up.cross(fwd).normalize();
        let up = fwd.cross(right);

        Mat44 {
            x: vec4(right.x, up.x, fwd.x, 0.0),
            y: vec4(right.y, up.y, fwd.y, 0.0),
            z: vec4(right.z, up.z, fwd.z, 0.0),
            w: vec4(eye.x, eye.y, eye.z, 1.0),
        }
    }

    // Right handed coordinate system
    #[cfg(not(target_arch = "spirv"))]
    pub fn look_at_inv(eye: Vec3<f32>, dir: Vec3<f32>) -> Self {
        let up = vec3(0.0, 1.0, 0.0);
        let fwd = dir.normalize();
        let right = up.cross(fwd).normalize();
        let up = fwd.cross(right);

        Mat44 {
            x: vec4(right.x, right.y, right.z, -right.dot(eye)),
            y: vec4(up.x, up.y, up.z, -up.dot(eye)),
            z: vec4(fwd.x, fwd.y, fwd.z, -fwd.dot(eye)),
            w: vec4(0.0, 0.0, 0.0, 1.0),
        }
    }
}

impl Mul<Mat44<f32>> for Vec4<f32> {
    type Output = Self;
    fn mul(self, rhs: Mat44<f32>) -> Self::Output {
        Vec4 {
            x: self.dot(rhs.x),
            y: self.dot(rhs.y),
            z: self.dot(rhs.z),
            w: self.dot(rhs.w),
        }
    }
}
