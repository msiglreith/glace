use crate::{vec2, vec3, Vec2, Vec3};
use core::default::Default;
use core::ops::{Add, BitXor, Mul, Shr};
use spirv_std::{scalar::Scalar, vector::Vector};

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl<T> Vec4<T> {
    pub fn xy(self) -> Vec2<T> {
        vec2(self.x, self.y)
    }

    pub fn xyz(self) -> Vec3<T> {
        vec3(self.x, self.y, self.z)
    }
}

impl Vec4<f32> {
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[cfg(target_feature = "GroupNonUniformQuad")]
    #[spirv_std_macros::gpu_only]
    pub fn quad_swap_horizontal(self) -> Self {
        crate::arch::subgroup::quad_swap_horizontal(self)
    }

    #[cfg(target_feature = "GroupNonUniformQuad")]
    #[spirv_std_macros::gpu_only]
    pub fn quad_swap_vertical(self) -> Self {
        crate::arch::subgroup::quad_swap_vertical(self)
    }

    #[cfg(target_feature = "GroupNonUniformQuad")]
    #[spirv_std_macros::gpu_only]
    pub fn quad_swap_diagonal(self) -> Self {
        crate::arch::subgroup::quad_swap_vertical(self)
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
                    "%vec = OpLoad _ {1}",
                    "%scalar = OpLoad _ {2}",
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

impl Add for Vec4<f32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl Vec4<u32> {
    pub fn wrapping_add(self, other: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec1 = OpLoad _ {1}",
                    "%vec2 = OpLoad _ {2}",
                    "%result = OpIAdd typeof*{0} %vec1 %vec2",
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
                x: self.x.wrapping_add(other.x),
                y: self.y.wrapping_add(other.y),
                z: self.z.wrapping_add(other.z),
                w: self.w.wrapping_add(other.w),
            }
        }
    }

    pub fn wrapping_mul(self, other: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec1 = OpLoad _ {1}",
                    "%vec2 = OpLoad _ {2}",
                    "%result = OpIMul typeof*{0} %vec1 %vec2",
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
                x: self.x.wrapping_mul(other.x),
                y: self.y.wrapping_mul(other.y),
                z: self.z.wrapping_mul(other.z),
                w: self.w.wrapping_mul(other.w),
            }
        }
    }
}

impl Shr for Vec4<u32> {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec = OpLoad _ {1}",
                    "%rhs = OpLoad _ {2}",
                    "%result = OpShiftRightLogical typeof*{0} %vec %rhs",
                    "OpStore {0} %result",
                    in(reg) &mut result,
                    in(reg) &self,
                    in(reg) &rhs,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            Vec4 {
                x: self.x >> rhs.x,
                y: self.y >> rhs.y,
                z: self.z >> rhs.z,
                w: self.w >> rhs.w,
            }
        }
    }
}

impl BitXor for Vec4<u32> {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec1 = OpLoad _ {1}",
                    "%vec2 = OpLoad _ {2}",
                    "%result = OpBitwiseXor typeof*{0} %vec1 %vec2",
                    "OpStore {0} %result",
                    in(reg) &mut result,
                    in(reg) &self,
                    in(reg) &rhs,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            Vec4 {
                x: self.x ^ rhs.x,
                y: self.y ^ rhs.y,
                z: self.z ^ rhs.z,
                w: self.w ^ rhs.w,
            }
        }
    }
}

impl Add for Vec4<u32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}
