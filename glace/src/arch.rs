pub mod glsl;
pub mod subgroup;

#[spirv_std_macros::gpu_only]
pub fn bitcast<T, U: Default>(value: T) -> U {
    let mut result = U::default();
    unsafe {
        asm! {
            "%value = OpLoad _ {value}",
            "%ret = OpBitcast typeof*{result} %value",
            "OpStore {result} %ret",
            result = in(reg) &mut result,
            value = in(reg) &value,
        }
    };
    result
}
