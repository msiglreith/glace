use glace::{f32x2, f32x4, vec2, vec4};
use spirv_std::{Image2d, SampledImage};

const QUAD_CLIP: [f32x4; 4] = [
    f32x4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    },
    f32x4 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    },
    f32x4 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        w: 1.0,
    },
    f32x4 {
        x: 1.0,
        y: 1.0,
        z: 0.0,
        w: 1.0,
    },
];

#[spirv(vertex)]
pub fn quad_vs(
    #[spirv(vertex_id)] vert_id: i32,
    a_uv: &mut f32x2,
    #[spirv(position)] a_position: &mut f32x4,
    #[spirv(uniform_constant, binding = 1)] u_position: &f32x2,
    #[spirv(uniform_constant, binding = 2)] u_size: &f32x2,
) {
    let pos_clip = QUAD_CLIP[vert_id as usize];
    *a_position = vec4(
        pos_clip.x * u_size.x + u_position.x,
        pos_clip.y * u_size.y + u_position.y,
        pos_clip.z,
        pos_clip.w,
    );
    *a_uv = vec2(pos_clip.x, pos_clip.y);
}

#[spirv(fragment)]
pub fn quad_fs(
    a_uv: f32x2,
    o_color: &mut f32x4,
    #[spirv(uniform_constant, binding = 0)] u_texture: &SampledImage<Image2d>,
) {
    *o_color = u_texture.sample(a_uv);
}
