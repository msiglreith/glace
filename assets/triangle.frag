#version 450 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 a_normal_world;
layout (location = 1) in vec2 a_texcoord;
layout (location = 2) in vec4 a_tangent_world;

layout (location = 0) out vec4 o_color;

void main() {
    o_color = vec4(0.5 * a_normal_world + 0.5, 1.0);
}
