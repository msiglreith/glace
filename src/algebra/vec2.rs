use core::ops::{Add, Mul, Sub, Div};
use super::Element;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(target_arch = "spirv", repr(simd))]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Element> Vec2<T> {
    pub const ZERO: Self = Vec2 {
        x: T::ZERO,
        y: T::ZERO,
    };
}

impl Vec2<f32> {
    #[inline]
    pub fn length(&self) -> f32 {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = 0.0;
            unsafe {
                asm! {
                    r#"%extension = OpExtInstImport "GLSL.std.450""#,
                    "%vec = OpLoad typeof*{1} {1}",
                    "%result = OpExtInst typeof*{0} %extension 66 %vec",
                    "OpStore {0} %result",
                    in(reg) &mut result,
                    in(reg) self,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            (self.x * self.x + self.y + self.y).sqrt()
        }
    }

    #[inline]
    pub fn dot(&self, other: Self) -> f32 {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = 0.0;
            unsafe {
                asm! {
                    "%vec1 = OpLoad typeof*{1} {1}",
                    "%vec2 = OpLoad typeof*{2} {2}",
                    "%result = OpDot typeof*{0} %vec1 %vec2",
                    "OpStore {0} %result",
                    in(reg) &mut result,
                    in(reg) self,
                    in(reg) &other,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            self.x * other.x + self.y + other.y
        }
    }

    #[inline]
    #[cfg(target_arch = "spirv")]
    pub fn fwidth(&self) -> Self {
        let mut result = Self::ZERO;
        unsafe {
            asm! {
                "%vec = OpLoad typeof*{1} {1}",
                "%result = OpFwidth typeof*{0} %vec",
                "OpStore {0} %result",
                in(reg) &mut result,
                in(reg) self,
            }
        };
        result
    }
}

impl Add for Vec2<f32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl Add for Vec2<i32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl Add for Vec2<u32> {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Vec2<f32> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl Sub for Vec2<i32> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl Sub for Vec2<u32> {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul for Vec2<f32> {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::ZERO;
            unsafe {
                asm! {
                    "%vec1 = OpLoad typeof*{1} {1}",
                    "%vec2 = OpLoad typeof*{2} {2}",
                    "%result = OpFMul typeof*{0} %vec1 %vec2",
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
            Vec2 {
                x: self.x * other.x,
                y: self.y * other.y,
            }
        }
    }
}

impl Mul<f32> for Vec2<f32> {
    type Output = Self;
    #[inline]
    fn mul(self, other: f32) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::ZERO;
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
            Vec2 {
                x: self.x * other,
                y: self.y * other,
            }
        }
    }
}

impl Mul<Vec2<f32>> for f32 {
    type Output = Vec2<f32>;
    #[inline]
    fn mul(self, other: Vec2<f32>) -> Vec2<f32> {
        other * self
    }
}

impl Div for Vec2<f32> {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = Self::ZERO;
            unsafe {
                asm! {
                    "%vec1 = OpLoad typeof*{1} {1}",
                    "%vec2 = OpLoad typeof*{2} {2}",
                    "%result = OpFDiv typeof*{0} %vec1 %vec2",
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
            Vec2 {
                x: self.x / other.x,
                y: self.y / other.y,
            }
        }
    }
}
