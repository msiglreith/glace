//! GLSL extended instructions.

#[spirv_std_macros::gpu_only]
pub unsafe fn round<T: Default>(value: T) -> T {
    let mut result = T::default();
    asm! {
        r#"%extension = OpExtInstImport "GLSL.std.450""#,
        "%value = OpLoad _ {value}",
        "%ret = OpExtInst typeof*{result} %extension 1 %value",
        "OpStore {result} %ret",
        result = in(reg) &mut result,
        value = in(reg) &value,
    };
    result
}

#[spirv_std_macros::gpu_only]
pub unsafe fn fabs<T: Default>(value: T) -> T {
    let mut result = T::default();
    asm! {
        r#"%extension = OpExtInstImport "GLSL.std.450""#,
        "%value = OpLoad _ {value}",
        "%ret = OpExtInst typeof*{result} %extension 4 %value",
        "OpStore {result} %ret",
        result = in(reg) &mut result,
        value = in(reg) &value,
    };
    result
}

#[spirv_std_macros::gpu_only]
pub unsafe fn fsign<T: Default>(value: T) -> T {
    let mut result = T::default();
    asm! {
        r#"%extension = OpExtInstImport "GLSL.std.450""#,
        "%value = OpLoad _ {value}",
        "%ret = OpExtInst typeof*{result} %extension 6 %value",
        "OpStore {result} %ret",
        result = in(reg) &mut result,
        value = in(reg) &value,
    };
    result
}
