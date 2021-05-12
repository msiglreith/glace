#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm)]
#![register_attr(spirv)]

pub mod cubemap;
pub mod debug;
pub mod ocean;
pub mod pbr;
pub mod skybox;

use glace::{f32x2, u32x3, vec2, vec4};
use spirv_std::num_traits::Float;
use spirv_std::StorageImage2d;

#[spirv(compute(threads(64)))]
pub fn compute(
    #[spirv(global_invocation_id)] gid: u32x3,
    #[spirv(uniform_constant, binding = 0)] u_image: &StorageImage2d,
) {
    unsafe {
        u_image.write(vec2(gid.x, gid.y), vec4(1.0, 0.0, 0.0, 1.0));
    }
}

const RESOLUTION: usize = 512;

fn complex_mul(c0: f32x2, c1: f32x2) -> f32x2 {
    return vec2(c0.x * c1.x - c0.y * c1.y, c0.y * c1.x + c0.x * c1.y);
}

fn complex_add(c0: f32x2, c1: f32x2) -> f32x2 {
    return vec2(c0.x + c1.x, c0.y + c1.y);
}

fn complex_sub(c0: f32x2, c1: f32x2) -> f32x2 {
    return vec2(c0.x - c1.x, c0.y - c1.y);
}

#[spirv(compute(threads(256)))]
pub fn fft_row(
    #[spirv(global_invocation_id)] gid: u32x3,
    #[spirv(storage_buffer, binding = 0)] u_fft: &mut [f32x2],
    #[spirv(workgroup)] shared: &mut [[f32x2; RESOLUTION]; 2],
) {
    let index = gid.x as usize + RESOLUTION * gid.y as usize;
    shared[0][gid.x as usize] = u_fft[0];
    shared[0][gid.x as usize + 256] = vec2(0.0, 0.0); // u_fft[index + 256];

    // unsafe {
    //     asm! {
    //         "%u32 = OpTypeInt 32 0",
    //         "%execution = OpConstant %u32 2", // workgroup
    //         "%memory = OpConstant %u32 2", // workgroup
    //         "%semantics = OpConstant %u32 384", // acquire/release | workgroup mem
    //         "OpControlBarrier %execution %memory %semantics",
    //     }
    // }

    /*
    for i in 0..9 {
        let block_size = 1 << i;
        let src = i % 2;
        let dst = (i + 1) % 2;

        // butterfly
        {
            let index = gid.x as usize;
            let k = index & (block_size - 1);

            let in0 = shared[src][index];
            let in1 = shared[src][index + 256];

            let theta = core::f32::consts::PI * k as f32 / block_size as f32; // NOTE: not 2 * pi as stated in the paper!
            let c = vec2(theta.cos(), theta.sin());
            let temp = complex_mul(in1, c);

            let dest = (index << 1) - k;

            shared[dst][dest] = complex_add(in0, temp);
            shared[dst][dest + block_size] = complex_sub(in0, temp);
        }

        unsafe {
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%execution = OpConstant %u32 2", // workgroup
                "%memory = OpConstant %u32 2", // workgroup
                "%semantics = OpConstant %u32 384", // acquire/release | workgroup mem
                "OpControlBarrier %execution %memory %semantics",
            }
        }
    }
    */

    // u_fft.data[index] = shared[1][gid.x as usize];
    // u_fft.data[index + 256] = shared[1][gid.x as usize + 256];
}
