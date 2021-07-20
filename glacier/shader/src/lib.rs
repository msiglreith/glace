#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm, abi_unadjusted)]
#![register_attr(spirv)]

use core::f32::consts::FRAC_1_PI;
use glace::{
    bindless::{resource_from_handle, Buffer},
    f32x2, f32x3, f32x3x3, f32x4, f32x4x3, f32x4x4,
    ray::Ray,
    std::{
        self,
        image::{Image2d, SampledImage, StorageImage2d},
        num_traits::Float,
        ray_tracing::{AccelerationStructure, CommittedIntersection, RayFlags, RayQuery},
    },
    u32x3, vec2, vec3, vec4,
};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct WorldData {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MaterialData {
    albedo_map: u64,
    normal_map: u64,
    albedo_color: f32x4,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct InstanceData {
    material: u32,
    geometry: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GeometryData {
    v_position_obj: Buffer,
    v_normal_obj: Buffer,
    v_texcoord: Buffer,
    v_tangent_obj: Buffer,
    indices: Buffer,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PathtraceConstants {
    world: Buffer,
    instance: Buffer,
    geometry: Buffer,
    material: Buffer,

    tlas: u64,
    target: u64,

    frame_id: u32,
}

fn gen_norm_f32(x: u32) -> f32 {
    const F32_1: u32 = 0x3f800000; // 1.0 as u32
    const MANTISSA_MASK: u32 = (1 << core::f32::MANTISSA_DIGITS) - 1;

    // [1.0, 2.0) -> [0.0, 1.0)
    glace::arch::bitcast::<_, f32>(F32_1 | (x & MANTISSA_MASK)) - 1.0
}

/// Source: Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, and Ryusuke Villemin,
///         Building an Orthonormal Basis, Revisited, Journal of Computer Graphics Techniques (JCGT), vol. 6, no. 1, 1-8, 2017
fn orthonormal_basis(normal: f32x3) -> f32x3x3 {
    let sign = if normal.z < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    f32x3x3 {
        c0: vec3(
            1.0 + sign * normal.x * normal.x * a,
            sign * b,
            -sign * normal.x,
        ),
        c1: vec3(b, sign + normal.y * normal.y * a, -normal.y),
        c2: normal,
    }
}

pub struct BsdfQuery {
    // From surface to light. space: solid angles
    pub w_i: f32x3,
    // Outgoing from surface. space: solid angles
    pub w_o: f32x3,
}

impl BsdfQuery {
    pub fn incoming(w_i: f32x3) -> Self {
        Self {
            w_i,
            w_o: vec3(0.0, 0.0, 0.0),
        }
    }
}
pub struct Lambert {
    albedo: f32x3,
}
impl Lambert {
    pub fn eval(&self) -> f32x3 {
        self.albedo * FRAC_1_PI
    }

    // brdf / sample_pdf * cos theta
    pub fn sample(&self, query: &mut BsdfQuery, u: f32x2) -> f32x3 {
        let (sample_pdf, sample) = glace::sample::hemisphere_cosine(u);
        query.w_o = sample;
        let cos_theta = query.w_o.z;

        self.albedo * FRAC_1_PI * (cos_theta / sample_pdf)
    }
}

#[spirv(compute(threads(16, 16)))]
pub unsafe fn pathtrace(
    #[spirv(global_invocation_id)] id: u32x3,
    #[spirv(push_constant)] constants: &PathtraceConstants,
) {
    let sample_offset = glace::hash::pcg::pcg3d(vec3(id.x, id.y, constants.frame_id));
    let dx = gen_norm_f32(sample_offset.x);
    let dy = gen_norm_f32(sample_offset.y);
    let sample = vec2(
        ((id.x as f32 + dx) / 1440.0) * 2.0 - 1.0,
        -(((id.y as f32 + dy) / 800.0) * 2.0 - 1.0),
    );
    let u_world: &WorldData = constants.world.load();

    let clip_to_view = u_world.view_to_clip.inverse();
    let view_to_world = u_world.world_to_view.inverse();

    let near = vec4(sample.x, sample.y, 0.0, 1.0) * clip_to_view * view_to_world;
    let far = vec4(sample.x, sample.y, 1.0, 1.0) * clip_to_view * view_to_world;

    let origin = vec3(near.x / near.w, near.y / near.w, near.z / near.w);
    let mut primary_ray = Ray {
        origin,
        direction: (vec3(far.x / far.w, far.y / far.w, far.z / far.w) - origin).normalize(),
    };

    let tlas: AccelerationStructure = resource_from_handle(constants.tlas);

    std::ray_query!(let mut primary);
    std::ray_query!(let mut shadow);

    let mut rng_idx = 0;
    let mut rng = || {
        let rnd = glace::hash::pcg::pcg4d(vec4(id.x, id.y, constants.frame_id, rng_idx));
        rng_idx += 1;
        vec4(
            gen_norm_f32(rnd.x),
            gen_norm_f32(rnd.y),
            gen_norm_f32(rnd.z),
            gen_norm_f32(rnd.w),
        )
    };

    let mut radiance = vec3(0.0, 0.0, 0.0);
    let mut attenuation = vec3(1.0, 1.0, 1.0);

    for _ in 0..4 {
        primary.initialize(
            &tlas,
            RayFlags::OPAQUE,
            0xFF,
            primary_ray.origin,
            0.0,
            primary_ray.direction,
            100000.0,
        );

        while primary.proceed() {}

        match primary.get_committed_intersection_type() {
            CommittedIntersection::Triangle => {
                let triangle_index = primary.get_committed_intersection_primitive_index();
                let instance_index = primary.get_committed_intersection_instance_custom_index();
                let geometry_index = primary.get_committed_intersection_geometry_index();
                let barycentrics: f32x2 = primary.get_committed_intersection_barycentrics();
                let u_obj_to_world: [f32x3; 4] =
                    primary.get_committed_intersection_object_to_world();
                let u_obj_to_world = f32x4x3 {
                    c0: vec4(
                        u_obj_to_world[0].x,
                        u_obj_to_world[1].x,
                        u_obj_to_world[2].x,
                        u_obj_to_world[3].x,
                    ),
                    c1: vec4(
                        u_obj_to_world[0].y,
                        u_obj_to_world[1].y,
                        u_obj_to_world[2].y,
                        u_obj_to_world[3].y,
                    ),
                    c2: vec4(
                        u_obj_to_world[0].z,
                        u_obj_to_world[1].z,
                        u_obj_to_world[2].z,
                        u_obj_to_world[3].z,
                    ),
                };

                let u_normal_to_world = f32x4x4 {
                    c0: u_obj_to_world.c0,
                    c1: u_obj_to_world.c1,
                    c2: u_obj_to_world.c2,
                    c3: vec4(0.0, 0.0, 0.0, 1.0),
                }
                .inverse()
                .transpose();

                let barycentrics = vec3(
                    1.0 - barycentrics.x - barycentrics.y,
                    barycentrics.x,
                    barycentrics.y,
                );

                let u_instance: &InstanceData = constants
                    .instance
                    .index((instance_index + geometry_index) as _);

                let u_geometry: &GeometryData = constants.geometry.index(u_instance.geometry as _);
                let indices: &[usize; 3] = u_geometry.indices.index(triangle_index as _);

                let position0: &[f32; 3] = u_geometry.v_position_obj.index(indices[0]);
                let position1: &[f32; 3] = u_geometry.v_position_obj.index(indices[1]);
                let position2: &[f32; 3] = u_geometry.v_position_obj.index(indices[2]);

                let normal0: &[f32; 3] = u_geometry.v_normal_obj.index(indices[0]);
                let normal1: &[f32; 3] = u_geometry.v_normal_obj.index(indices[1]);
                let normal2: &[f32; 3] = u_geometry.v_normal_obj.index(indices[2]);

                let uv = if u_geometry.v_texcoord.is_valid() {
                    let uv0: &[f32; 2] = u_geometry.v_texcoord.index(indices[0]);
                    let uv1: &[f32; 2] = u_geometry.v_texcoord.index(indices[1]);
                    let uv2: &[f32; 2] = u_geometry.v_texcoord.index(indices[2]);

                    f32x2::from(uv0) * barycentrics.x
                        + f32x2::from(uv1) * barycentrics.y
                        + f32x2::from(uv2) * barycentrics.z
                } else {
                    vec2(0.0, 0.0)
                };

                let geometric10 = f32x3::from(position1) - f32x3::from(position0);
                let geometric20 = f32x3::from(position2) - f32x3::from(position0);

                let position_obj = f32x3::from(position0) * barycentrics.x
                    + f32x3::from(position1) * barycentrics.y
                    + f32x3::from(position2) * barycentrics.z;

                let position_world = position_obj.w(1.0) * u_obj_to_world;

                let mut geometric_obj = geometric10.cross(geometric20);

                let mut normal_obj = f32x3::from(normal0) * barycentrics.x
                    + f32x3::from(normal1) * barycentrics.y
                    + f32x3::from(normal2) * barycentrics.z;

                if !primary.get_committed_intersection_front_face() {
                    geometric_obj = -geometric_obj;
                    normal_obj = -normal_obj;
                }

                let geometric_world = geometric_obj.w(0.0) * u_normal_to_world;
                let geometric_world = geometric_world.xyz().normalize();

                let normal_world = normal_obj.w(0.0) * u_normal_to_world;
                let normal_world = normal_world.xyz().normalize();

                let world_to_tangent = orthonormal_basis(normal_world);
                let tangent_to_world = world_to_tangent.transpose();

                let u_material: &MaterialData = constants.material.index(u_instance.material as _);
                let brdf = {
                    let mut albedo = u_material.albedo_color.xyz();
                    if u_material.albedo_map != 0 {
                        let albedo_tex: f32x4 =
                            resource_from_handle::<SampledImage<Image2d>>(u_material.albedo_map)
                                .sample_by_lod(uv, 0.0);
                        albedo = albedo * albedo_tex.xyz();
                    }

                    Lambert { albedo }
                };

                // next event estimation for global direction light
                let shadow_ray = Ray {
                    origin: position_world,
                    direction: vec3(-0.3, 1.0, 1.0).normalize(),
                }
                .geometric_offset(geometric_world);
                shadow.initialize(
                    &tlas,
                    RayFlags::OPAQUE | RayFlags::TERMINATE_ON_FIRST_HIT,
                    0xFF,
                    shadow_ray.origin,
                    0.0,
                    shadow_ray.direction,
                    100000.0,
                );
                while shadow.proceed() {}

                if shadow.get_committed_intersection_type() != CommittedIntersection::Triangle {
                    let n_dot_l = shadow_ray.direction.dot(normal_world);
                    // radiance += attenuation * brdf.eval() * n_dot_l;
                }

                // indirect light.
                let u = rng();
                let mut indirect_query = BsdfQuery::incoming(-primary_ray.direction);
                let indirect_sample = brdf.sample(&mut indirect_query, vec2(u.x, u.y));
                attenuation = attenuation * indirect_sample;

                // next ray to measure indirect light contribution along outgoing sample direction.
                primary_ray = Ray {
                    origin: shadow_ray.origin,
                    direction: indirect_query.w_o * tangent_to_world,
                };
            }
            _ => {
                radiance += attenuation * vec3(1.0, 1.0, 1.0); // sky color;
                break;
            }
        }
    }

    let target: StorageImage2d = unsafe { resource_from_handle(constants.target) };
    let target_texel = vec2(id.x, id.y);
    let history: f32x4 = target.read(target_texel);
    let inv_weight = 1.0 / (constants.frame_id + 1) as f32;
    let accumulated = vec3(
        srgb_to_linear(history.x),
        srgb_to_linear(history.y),
        srgb_to_linear(history.z),
    ) * (constants.frame_id as f32 * inv_weight)
        + radiance * inv_weight;
    target.write(
        target_texel,
        vec4(
            linear_to_srgb(accumulated.x),
            linear_to_srgb(accumulated.y),
            linear_to_srgb(accumulated.z),
            1.0,
        ),
    );
}

fn linear_to_srgb(linear: f32) -> f32 {
    if linear < 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_to_linear(srgb: f32) -> f32 {
    if srgb < 0.04045 {
        srgb / 12.92
    } else {
        ((srgb + 0.055) / 1.055).powf(2.4)
    }
}
