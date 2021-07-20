#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(asm, register_attr, repr_simd, abi_unadjusted),
    register_attr(spirv)
)]

mod mat33;
mod mat43;
mod mat44;
mod vec2;
mod vec3;
mod vec4;

pub use self::mat33::*;
pub use self::mat43::*;
pub use self::mat44::*;
pub use self::vec2::*;
pub use self::vec3::*;
pub use self::vec4::*;

pub use spirv_std as std;

pub mod arch;
pub mod bezier;
pub mod bindless;
pub mod compute;
pub mod geometry;
pub mod hash;
pub mod ray;
pub mod sample;
pub mod sdf;

pub mod f32;
pub mod u32;

#[allow(non_camel_case_types)]
pub type f32x3x3 = Mat33<f32>;
#[allow(non_camel_case_types)]
pub type f32x4x3 = Mat43<f32>;
#[allow(non_camel_case_types)]
pub type f32x4x4 = Mat44<f32>;

#[allow(non_camel_case_types)]
pub type f32x2 = Vec2<f32>;
#[allow(non_camel_case_types)]
pub type f32x3 = Vec3<f32>;
#[allow(non_camel_case_types)]
pub type f32x4 = Vec4<f32>;

#[allow(non_camel_case_types)]
pub type i32x2 = Vec2<i32>;
#[allow(non_camel_case_types)]
pub type i32x3 = Vec3<i32>;
#[allow(non_camel_case_types)]
pub type i32x4 = Vec4<i32>;

#[allow(non_camel_case_types)]
pub type u32x2 = Vec2<u32>;
#[allow(non_camel_case_types)]
pub type u32x3 = Vec3<u32>;
#[allow(non_camel_case_types)]
pub type u32x4 = Vec4<u32>;
