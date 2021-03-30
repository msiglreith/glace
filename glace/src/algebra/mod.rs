mod mat33;
mod mat44;
mod vec2;
mod vec3;
mod vec4;

pub use self::mat33::*;
pub use self::mat44::*;
pub use self::vec2::*;
pub use self::vec3::*;
pub use self::vec4::*;

#[allow(non_camel_case_types)]
pub type f32x3x3 = Mat33<f32>;
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
