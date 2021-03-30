#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm)]
#![register_attr(spirv)]

mod cube;

use cube::{CubeImage, ImageCube};
use glace::{f32x2, f32x3, f32x4, f32x4x4, geometry::Fullscreen, vec3, vec4};
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

#[spirv(block)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct LocalsSkybox {
    view_to_world: f32x4x4,
    clip_to_view: f32x4x4,
}

#[spirv(vertex)]
pub fn skybox_vs(
    #[spirv(vertex_id)] vert_id: i32,
    #[spirv(position)] a_position: &mut f32x4,
    a_view_dir: &mut f32x3,
    #[spirv(uniform, binding = 0)] u_locals: &mut LocalsSkybox,
) {
    let position_uv = Fullscreen::position(vert_id);
    let position_clip = vec4(position_uv.x, position_uv.y, 0.0, 1.0);
    let mut position_view = position_clip * u_locals.clip_to_view;
    position_view.w = 0.0;
    let position_world = position_view * u_locals.view_to_world;

    *a_view_dir = vec3(position_world.x, position_world.y, position_world.z);
    *a_position = position_clip;
}

#[spirv(fragment)]
pub fn skybox_fs(
    f_view_dir: f32x3,
    #[spirv(uniform_constant, binding = 0)] u_diffuse_map: &SampledImage<ImageCube>,
    output: &mut f32x4,
) {
    let sky: f32x4 = u_diffuse_map.sample(f_view_dir);
    *output = vec4(sky.x, sky.y, sky.z, 1.0);
}