use core::default::Default;
use core::ops::Mul;
use spirv_std::{scalar::Scalar, vector::Vector};

#[derive(Debug, Copy, Clone)]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

unsafe impl<T: Scalar> Vector<T, 4> for Vec4<T> {}

impl<T: Scalar> Default for Vec4<T> {
    fn default() -> Self {
        Vec4 {
            x: Default::default(),
            y: Default::default(),
            z: Default::default(),
            w: Default::default(),
        }
    }
}

impl Vec4<f32> {
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

pub fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vec4 { x, y, z, w }
}

impl Mul<f32> for Vec4<f32> {
    type Output = Self;
    #[inline]
    fn mul(self, other: f32) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec = OpLoad typeof*{1} {1}",
                    "%scalar = OpLoad typeof*{2} {2}",
                    "%result = OpVectorTimesScalar typeof*{0} %vec %scalar",
                    "OpStore {0} %result",
                    in(reg) &mut result,
                    in(reg) &self,
                    in(reg) &other,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            Vec4 {
                x: self.x * other,
                y: self.y * other,
                z: self.z * other,
                w: self.w * other,
            }
        }
    }
}

impl Mul<Vec4<f32>> for f32 {
    type Output = Vec4<f32>;
    #[inline]
    fn mul(self, other: Vec4<f32>) -> Vec4<f32> {
        other * self
    }
}
