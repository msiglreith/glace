#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 v_position_obj;
layout (location = 1) in vec3 v_normal_obj;
layout (location = 2) in vec2 v_texcoord;
layout (location = 3) in vec4 v_tangent_obj;

layout (location = 0) out vec3 a_normal_world;
layout (location = 1) out vec2 a_texcoord;
layout (location = 2) out vec4 a_tangent_world;
// layout (location = 3) out vec3 a_position_world;

layout (binding = 0) uniform Locals {
    mat4 world_to_view;
    mat4 view_to_clip;
} u_locals;

void main() {
    a_normal_world = v_normal_obj;
    a_texcoord = v_texcoord;
    a_tangent_world = v_tangent_obj;

    gl_Position = vec4(v_position_obj, 1.0) * u_locals.world_to_view * u_locals.view_to_clip;
}

// pub fn main_vs(
//     v_position_obj: Input<f32x3>,
//     v_normal_obj: Input<f32x3>,
//     v_texcoord: Input<f32x2>,
//     v_tangent_obj: Input<f32x4>,
//     #[spirv(position)] mut a_position: Output<f32x4>,
//     mut a_normal_world: Output<f32x3>,
//     mut a_texcoord: Output<f32x2>,
//     mut a_tangent_world: Output<f32x4>,
//     mut a_position_world: Output<f32x3>,
//     #[spirv(binding = 0)] u_locals_fs: Uniform<LocalsPbrFs>,
//     #[spirv(binding = 1)] u_locals_vs: Uniform<LocalsPbrVs>,
// ) {
//     a_normal_world.store(v_normal_obj.load());
//     a_texcoord.store(v_texcoord.load());
//     a_tangent_world.store(v_tangent_obj.load());

//     let u_locals_vs = u_locals_vs.load();

//     let pos_obj = v_position_obj.load();
//     let pos_world = vec4(pos_obj.x, pos_obj.y, pos_obj.z, 1.0);
//     a_position_world.store(vec3(pos_world.x, pos_world.y, pos_world.z));

//     let pos_view = pos_world * u_locals_vs.world_to_view;
//     let pos_clip = pos_view * u_locals_vs.view_to_clip;
//     a_position.store(pos_clip);
// }
