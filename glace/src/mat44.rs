use super::Vec4;
#[cfg(not(target_arch = "spirv"))]
use super::{vec3, vec4, Vec3};
use crate::std::scalar::Scalar;
use core::ops::Mul;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(target_arch = "spirv", spirv(matrix))]
pub struct Mat44<T> {
    pub c0: Vec4<T>,
    pub c1: Vec4<T>,
    pub c2: Vec4<T>,
    pub c3: Vec4<T>,
}

impl<T: Scalar> Default for Mat44<T> {
    fn default() -> Self {
        Self {
            c0: Default::default(),
            c1: Default::default(),
            c2: Default::default(),
            c3: Default::default(),
        }
    }
}

impl Mat44<f32> {
    // Right handed coordinate system
    #[cfg(not(target_arch = "spirv"))]
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let focal = 1.0 / (0.5 * fov).tan();
        Mat44 {
            c0: vec4(focal / aspect, 0.0, 0.0, 0.0),
            c1: vec4(0.0, focal, 0.0, 0.0),
            c2: vec4(0.0, 0.0, -far / (far - near), -(far * near) / (far - near)),
            c3: vec4(0.0, 0.0, -1.0, 0.0),
        }
    }

    // Right handed coordinate system
    #[cfg(not(target_arch = "spirv"))]
    pub fn perspective_inv(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let inv_focal = (0.5 * fov).tan();
        Mat44 {
            c0: vec4(inv_focal * aspect, 0.0, 0.0, 0.0),
            c1: vec4(0.0, inv_focal, 0.0, 0.0),
            c2: vec4(0.0, 0.0, 0.0, -1.0),
            c3: vec4(0.0, 0.0, (near - far) / (far * near), 1.0 / near),
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
            c0: vec4(right.x, up.x, fwd.x, eye.x),
            c1: vec4(right.y, up.y, fwd.y, eye.y),
            c2: vec4(right.z, up.z, fwd.z, eye.z),
            c3: vec4(0.0, 0.0, 0.0, 1.0),
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
            c0: vec4(right.x, right.y, right.z, -right.dot(eye)),
            c1: vec4(up.x, up.y, up.z, -up.dot(eye)),
            c2: vec4(fwd.x, fwd.y, fwd.z, -fwd.dot(eye)),
            c3: vec4(0.0, 0.0, 0.0, 1.0),
        }
    }

    #[cfg(target_arch = "spirv")]
    pub fn inverse(&self) -> Self {
        let mut result = Self::default();
        unsafe {
            asm! {
                r#"%extension = OpExtInstImport "GLSL.std.450""#,
                "%matrix = OpLoad _ {matrix}",
                "%result = OpExtInst typeof*{result} %extension 34 %matrix",
                "OpStore {result} %result",
                result = in(reg) &mut result,
                matrix = in(reg) self,
            }
        };
        result
    }

    #[cfg(target_arch = "spirv")]
    pub fn transpose(&self) -> Self {
        let mut result = Self::default();
        unsafe {
            asm! {
                r#"%extension = OpExtInstImport "GLSL.std.450""#,
                "%matrix = OpLoad _ {matrix}",
                "%result = OpTranspose typeof*{result} %matrix",
                "OpStore {result} %result",
                result = in(reg) &mut result,
                matrix = in(reg) self,
            }
        };
        result
    }
}

impl Mul<Mat44<f32>> for Vec4<f32> {
    type Output = Vec4<f32>;
    fn mul(self, rhs: Mat44<f32>) -> Self::Output {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%lhs = OpLoad _ {lhs}",
                    "%rhs = OpLoad _ {rhs}",
                    "%result = OpVectorTimesMatrix typeof*{result} %lhs %rhs",
                    "OpStore {result} %result",
                    result = in(reg) &mut result,
                    lhs = in(reg) &self,
                    rhs = in(reg) &rhs,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            Vec4 {
                x: self.dot(rhs.c0),
                y: self.dot(rhs.c1),
                z: self.dot(rhs.c2),
                w: self.dot(rhs.c3),
            }
        }
    }
}
