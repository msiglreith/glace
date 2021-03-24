//! Math operations which are currently not part of `spirv_std`.

pub trait Gl {
    fn clamp(self, lower: Self, upper: Self) -> Self;
}

impl Gl for f32 {
    #[inline]
    fn clamp(self, lower: f32, upper: f32) -> f32 {
        #[cfg(target_arch = "spirv")]
        {
            let mut result = 0.0;
            unsafe {
                asm! {
                    r#"%extension = OpExtInstImport "GLSL.std.450""#,
                    "%x = OpLoad typeof*{1} {1}",
                    "%lower = OpLoad typeof*{2} {2}",
                    "%upper = OpLoad typeof*{3} {3}",
                    "%result = OpExtInst typeof*{0} %extension 43 %x %lower %upper",
                    "OpStore {0} %result",
                    in(reg) &mut result,
                    in(reg) &self,
                    in(reg) &lower,
                    in(reg) &upper,
                }
            };
            result
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            self.max(lower).min(upper)
        }
    }
}
