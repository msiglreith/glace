#version 450 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 a_normal_world;
layout (location = 1) in vec2 a_texcoord;
layout (location = 2) in vec4 a_tangent_world;

layout (location = 0) out vec4 o_color;

layout (binding = 1) uniform texture2D u_maps[2];
layout (binding = 2) uniform sampler u_sampler;

void main() {
    vec3 normal_obj = texture(sampler2D(u_maps[1], u_sampler), a_texcoord).xyz;
    vec4 albedo = texture(sampler2D(u_maps[0], u_sampler), a_texcoord);
    o_color = vec4(normal_obj, 1.0);
}
