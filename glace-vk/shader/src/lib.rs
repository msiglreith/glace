#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm, abi_unadjusted)]
#![register_attr(spirv)]

use glace::{f32x2, f32x3, f32x4, f32x4x4, geometry::Fullscreen, vec3, vec4};
use spirv_std::{Sampler, Image2d, bindless::{resource_access, ArrayBuffer, Buffer, RenderResourceHandle, SimpleBuffer}};
use spirv_std::num_traits::Float;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct WorldData {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct InstanceData {
    sampler: RenderResourceHandle,
    albedo_map: RenderResourceHandle,
    normal_map: RenderResourceHandle,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GeometryData {
    v_position_obj: ArrayBuffer<f32x3>,
    v_normal_obj: ArrayBuffer<f32x3>,
    v_texcoord: ArrayBuffer<f32x2>,
    v_tangent_obj: ArrayBuffer<f32x4>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Constants {
    world: SimpleBuffer<WorldData>,
    geometry: SimpleBuffer<GeometryData>,
    instance: SimpleBuffer<InstanceData>,
}

#[spirv(vertex)]
pub fn mesh_vs(
    #[spirv(vertex_index)] vert_id: i32,
    #[spirv(position)] a_position: &mut f32x4,
    a_normal_world: &mut f32x3,
    a_texcoord: &mut f32x2,
    a_tangent_world: &mut f32x4,
    a_position_world: &mut f32x3,
    #[spirv(push_constant)] constants: &Constants,
) {
    let vertex_index = vert_id as u32;

    let u_world = constants.world.load();
    let u_geometry = constants.geometry.load();

    let v_position_obj = u_geometry.v_position_obj.load(vertex_index);
    let v_normal_obj = u_geometry.v_normal_obj.load(vertex_index);
    let v_texcoord = u_geometry.v_texcoord.load(vertex_index);
    let v_tangent_obj = u_geometry.v_tangent_obj.load(vertex_index);

    *a_normal_world = v_normal_obj;
    *a_texcoord = v_texcoord;
    *a_tangent_world = v_tangent_obj;

    let pos_world = vec4(v_position_obj.x, v_position_obj.y, v_position_obj.z, 1.0);
    *a_position_world = vec3(pos_world.x, pos_world.y, pos_world.z);

    let pos_view = pos_world * u_world.world_to_view;
    let pos_clip = pos_view * u_world.view_to_clip;

    *a_position = pos_clip;
}

#[spirv(fragment)]
pub unsafe fn mesh_fs(
    a_normal_world: f32x3,
    a_texcoord: f32x2,
    a_tangent_world: f32x4,
    a_position_world: f32x3,
    output: &mut f32x4,
    #[spirv(push_constant)] constants: &Constants,
) {
    let instance_data = constants.instance.load();

    let albedo: Image2d = instance_data.albedo_map.access();
    let sampler: Sampler = instance_data.sampler.access();

    let normal: f32x4 = albedo.sample(sampler, a_texcoord);
    *output = vec4(normal.x, normal.y, normal.z, 1.0);
}
