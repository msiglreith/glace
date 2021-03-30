use crate::cubemap::cubemap_sample;
use glace::{f32x3, f32x4, f32x4x4, vec3, vec4};
use spirv_std::{Cubemap, SampledImage};

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
    #[spirv(uniform, binding = 0)] u_locals: &LocalsSkybox,
) {
    let position_uv = glace::geometry::Fullscreen::position(vert_id);
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
    #[spirv(uniform_constant, binding = 0)] u_diffuse_map: &SampledImage<Cubemap>,
    output: &mut f32x4,
) {
    let sky = cubemap_sample(u_diffuse_map, f_view_dir);
    *output = vec4(sky.x, sky.y, sky.z, 1.0);
}
