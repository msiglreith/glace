#[cfg(target_feature = "GroupNonUniformQuad")]
#[spirv_std_macros::gpu_only]
pub fn quad_swap_horizontal<T: Default>(value: T) -> T {
    let mut result = T::default();
    let direction = 0u32;
    let scope = 3u32;
    unsafe {
        asm! {
            "%value = OpLoad _ {value}",
            "%ret = OpGroupNonUniformQuadSwap _ {scope} %value {direction}",
            "OpStore {result} %ret",
            result = in(reg) &mut result,
            value = in(reg) &value,
            direction = in(reg) direction,
            scope = in(reg) scope,
        }
    };
    result
}

#[cfg(target_feature = "GroupNonUniformQuad")]
#[spirv_std_macros::gpu_only]
pub fn quad_swap_vertical<T: Default>(value: T) -> T {
    let mut result = T::default();
    let direction = 1u32;
    let scope = 3u32;
    unsafe {
        asm! {
            "%value = OpLoad _ {value}",
            "%ret = OpGroupNonUniformQuadSwap _ {scope} %value {direction}",
            "OpStore {result} %ret",
            result = in(reg) &mut result,
            value = in(reg) &value,
            direction = in(reg) direction,
            scope = in(reg) scope,
        }
    };
    result
}

#[cfg(target_feature = "GroupNonUniformQuad")]
#[spirv_std_macros::gpu_only]
pub fn quad_swap_diagonal<T: Default>(value: T) -> T {
    let mut result = T::default();
    let direction = 2u32;
    let scope = 3u32;
    unsafe {
        asm! {
            "%value = OpLoad _ {value}",
            "%ret = OpGroupNonUniformQuadSwap _ {scope} %value {direction}",
            "OpStore {result} %ret",
            result = in(reg) &mut result,
            value = in(reg) &value,
            direction = in(reg) direction,
            scope = in(reg) scope,
        }
    };
    result
}
