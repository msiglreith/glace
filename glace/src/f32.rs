#[inline]
pub fn clamp(value: f32, lower: f32, upper: f32) -> f32 {
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
                in(reg) &value,
                in(reg) &lower,
                in(reg) &upper,
            }
        };
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        value.max(lower).min(upper)
    }
}

#[inline]
pub fn fabs(value: f32) -> f32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        crate::arch::glsl::fabs(value)
    }
    #[cfg(not(target_arch = "spirv"))]
    value.abs()
}

#[inline]
pub fn fsign(value: f32) -> f32 {
    #[cfg(target_arch = "spirv")]
    unsafe {
        crate::arch::glsl::fsign(value)
    }
    #[cfg(not(target_arch = "spirv"))]
    value.signum() // not equal in infinite/NaN case
}
