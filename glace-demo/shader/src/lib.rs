#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm)]
#![register_attr(spirv)]

mod cube;

use cube::{CubeImage, ImageCube};
use glace::{f32x2, f32x3, f32x4, f32x4x4, vec3, vec4};
use spirv_std::SampledImage;

#[spirv(block)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocalsPbr {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

#[spirv(vertex)]
pub fn mesh_vs(
    v_position_obj: f32x3,
    v_normal_obj: f32x3,
    v_texcoord: f32x2,
    v_tangent_obj: f32x4,
    #[spirv(position)] a_position: &mut f32x4,
    a_normal_world: &mut f32x3,
    a_texcoord: &mut f32x2,
    a_tangent_world: &mut f32x4,
    a_position_world: &mut f32x3,
    #[spirv(uniform, binding = 0)] u_locals: &mut LocalsPbr,
) {
    *a_normal_world = v_normal_obj;
    *a_texcoord = v_texcoord;
    *a_tangent_world = v_tangent_obj;

    let pos_obj = v_position_obj;
    let pos_world = vec4(pos_obj.x, pos_obj.y, pos_obj.z, 1.0);
    *a_position_world = vec3(pos_world.x, pos_world.y, pos_world.z);

    let pos_view = pos_world * u_locals.world_to_view;
    let pos_clip = pos_view * u_locals.view_to_clip;
    *a_position = pos_clip;
}
