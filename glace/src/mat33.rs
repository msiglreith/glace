use super::Vec3;
use crate::std::scalar::Scalar;
use core::ops::Mul;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(target_arch = "spirv", spirv(matrix))]
pub struct Mat33<T> {
    pub c0: Vec3<T>,
    pub c1: Vec3<T>,
    pub c2: Vec3<T>,
}

impl<T: Scalar> Default for Mat33<T> {
    fn default() -> Self {
        Self {
            c0: Default::default(),
            c1: Default::default(),
            c2: Default::default(),
        }
    }
}

impl Mat33<f32> {
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

impl Mul<Mat33<f32>> for Vec3<f32> {
    type Output = Vec3<f32>;
    fn mul(self, rhs: Mat33<f32>) -> Self::Output {
        Vec3 {
            x: self.dot(rhs.c0),
            y: self.dot(rhs.c1),
            z: self.dot(rhs.c2),
        }
    }
}
