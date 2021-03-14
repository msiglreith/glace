#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 v_position_clip;
layout (location = 1) in vec4 v_color;

layout (location = 0) out vec4 a_color;

void main() {
    a_color = v_color;
    gl_Position = vec4(v_position_clip, 0.0, 1.0);
}