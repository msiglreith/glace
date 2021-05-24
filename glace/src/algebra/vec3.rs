use core::default::Default;
use core::ops::{Add, BitXor, Mul, Shr, Sub};
use spirv_std::{scalar::Scalar, vector::Vector};

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(target_arch = "spirv", spirv(vector))]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

unsafe impl<T: Scalar> Vector<T, 3> for Vec3<T> {}

impl<T: Scalar> Default for Vec3<T> {
    fn default() -> Self {
        Vec3 {
            x: Default::default(),
            y: Default::default(),
            z: Default::default(),
        }
    }
}

pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3 { x, y, z }
}

impl Vec3<f32> {
    pub fn cross(self, rhs: Self) -> Self {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Vec3 {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
        }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl Vec3<u32> {
    pub fn wrapping_add(self, other: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec1 = OpLoad typeof*{1} {1}",
                    "%vec2 = OpLoad typeof*{2} {2}",
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
            Vec3 {
                x: self.x.wrapping_add(other.x),
                y: self.y.wrapping_add(other.y),
                z: self.z.wrapping_add(other.z),
            }
        }
    }

    pub fn wrapping_mul(self, other: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec1 = OpLoad typeof*{1} {1}",
                    "%vec2 = OpLoad typeof*{2} {2}",
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
            Vec3 {
                x: self.x.wrapping_mul(other.x),
                y: self.y.wrapping_mul(other.y),
                z: self.z.wrapping_mul(other.z),
            }
        }
    }
}

impl Shr for Vec3<u32> {
    type Output = Self;
    fn shr(self, rhs: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec = OpLoad typeof*{1} {1}",
                    "%rhs = OpLoad typeof*{2} {2}",
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
            Vec3 {
                x: self.x >> rhs.x,
                y: self.y >> rhs.y,
                z: self.z >> rhs.z,
            }
        }
    }
}

impl BitXor for Vec3<u32> {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::default();
            unsafe {
                asm! {
                    "%vec1 = OpLoad typeof*{1} {1}",
                    "%vec2 = OpLoad typeof*{2} {2}",
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
            Vec3 {
                x: self.x ^ rhs.x,
                y: self.y ^ rhs.y,
                z: self.z ^ rhs.z,
            }
        }
    }
}

impl Add for Vec3<u32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Vec3<f32> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<u32> for Vec3<u32> {
    type Output = Self;
    #[inline]
    fn mul(self, other: u32) -> Self {
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
            Vec3 {
                x: self.x * other,
                y: self.y * other,
                z: self.z * other,
            }
        }
    }
}

impl Mul<f32> for Vec3<f32> {
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
            Vec3 {
                x: self.x * other,
                y: self.y * other,
                z: self.z * other,
            }
        }
    }
}

impl Mul<Vec3<f32>> for f32 {
    type Output = Vec3<f32>;
    #[inline]
    fn mul(self, other: Vec3<f32>) -> Vec3<f32> {
        other * self
    }
}

impl Add for Vec3<f32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}
