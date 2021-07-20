use crate::compute;
use crate::{
    f32x4, std,
    std::memory::{Scope, Semantics},
    u32x2, vec2,
};

fn workgroup_memory_barrier() {
    unsafe {
        std::arch::control_barrier::<
            { Scope::Subgroup as u32 },
            { Scope::Workgroup as u32 },
            { Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() },
        >();
    }
}

/// Requires threadgroup size of 256.
pub fn downsample_cs<S, R, D>(
    thread_local: u32,
    thread_group_xy: u32x2,
    mip_levels: u32,
    scratch: &mut [[f32x4; 16]; 16],
    src: S,
    reduce: R,
    mut dst: D,
) where
    S: Fn(u32x2) -> f32x4,
    R: Fn(f32x4, f32x4, f32x4, f32x4) -> f32x4,
    D: FnMut(u32x2, u32, f32x4),
{
    let quad_reduce = |value: f32x4| -> f32x4 {
        let v0 = value;
        let v1 = value.quad_swap_horizontal();
        let v2 = value.quad_swap_vertical();
        let v3 = value.quad_swap_diagonal();

        reduce(v0, v1, v2, v3)
    };

    let thread_local_xy = compute::reorder256_xyyxxyyx(thread_local);

    // mip 0 -> 1
    let quad0_xy = thread_group_xy * 64 + thread_local_xy * 2;
    let quad1_xy = thread_group_xy * 32 + thread_local_xy;

    let quad00_xy = quad0_xy;
    let q10 = reduce(
        src(quad00_xy),
        src(quad00_xy + vec2(0, 1)),
        src(quad00_xy + vec2(1, 0)),
        src(quad00_xy + vec2(1, 1)),
    );
    dst(quad1_xy, 0, q10);

    let quad01_xy = quad0_xy + vec2(32, 0);
    let q11 = reduce(
        src(quad01_xy),
        src(quad01_xy + vec2(0, 1)),
        src(quad01_xy + vec2(1, 0)),
        src(quad01_xy + vec2(1, 1)),
    );
    dst(quad1_xy + vec2(16, 0), 0, q11);

    let quad02_xy = quad0_xy + vec2(0, 32);
    let q12 = reduce(
        src(quad02_xy),
        src(quad02_xy + vec2(0, 1)),
        src(quad02_xy + vec2(1, 0)),
        src(quad02_xy + vec2(1, 1)),
    );
    dst(quad1_xy + vec2(0, 16), 0, q12);

    let quad03_xy = quad0_xy + vec2(32, 32);
    let q13 = reduce(
        src(quad03_xy),
        src(quad03_xy + vec2(0, 1)),
        src(quad03_xy + vec2(1, 0)),
        src(quad03_xy + vec2(1, 1)),
    );
    dst(quad1_xy + vec2(16, 16), 0, q13);

    if mip_levels < 2 {
        return;
    }

    // mip 2 -> 1
    let q20 = quad_reduce(q10);
    let q21 = quad_reduce(q11);
    let q22 = quad_reduce(q12);
    let q23 = quad_reduce(q13);

    if thread_local % 4 == 0 {
        let tile2_xy = thread_group_xy * 16;
        let quad2_local_xy = thread_local_xy / 2;

        let quad20_local_xy = quad2_local_xy;
        dst(tile2_xy + quad20_local_xy, 1, q20);
        scratch[quad2_local_xy.y as usize][quad2_local_xy.x as usize] = q20;

        let quad21_local_xy = quad2_local_xy + vec2(8, 0);
        dst(tile2_xy + quad21_local_xy, 1, q21);
        scratch[quad21_local_xy.y as usize][quad21_local_xy.x as usize] = q21;

        let quad22_local_xy = quad2_local_xy + vec2(0, 8);
        dst(tile2_xy + quad22_local_xy, 1, q22);
        scratch[quad22_local_xy.y as usize][quad22_local_xy.x as usize] = q22;

        let quad23_local_xy = quad2_local_xy + vec2(8, 8);
        dst(tile2_xy + quad23_local_xy, 1, q23);
        scratch[quad23_local_xy.y as usize][quad23_local_xy.x as usize] = q23;
    }

    if mip_levels < 3 {
        return;
    }

    // mip 2 -> 3
    workgroup_memory_barrier();

    let v = scratch[thread_local_xy.y as usize][thread_local_xy.x as usize];
    let q3 = quad_reduce(v);

    if thread_local % 4 == 0 {
        let quad3_xy = thread_group_xy * 8 + thread_local_xy / 2;
        dst(quad3_xy, 2, q3);
    }
}
