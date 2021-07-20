use super::{Vec3, Vec4};
use crate::std::scalar::Scalar;
use core::ops::Mul;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(target_arch = "spirv", spirv(matrix))]
pub struct Mat43<T> {
    pub c0: Vec4<T>,
    pub c1: Vec4<T>,
    pub c2: Vec4<T>,
}

impl<T: Scalar> Default for Mat43<T> {
    fn default() -> Self {
        Self {
            c0: Default::default(),
            c1: Default::default(),
            c2: Default::default(),
        }
    }
}

impl Mul<Mat43<f32>> for Vec4<f32> {
    type Output = Vec3<f32>;
    fn mul(self, rhs: Mat43<f32>) -> Self::Output {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::Output::default();
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
            Vec3 {
                x: self.dot(rhs.c0),
                y: self.dot(rhs.c1),
                z: self.dot(rhs.c2),
            }
        }
    }
}
