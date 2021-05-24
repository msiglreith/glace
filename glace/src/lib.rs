#![cfg_attr(
    target_arch = "spirv",
    no_std,
    feature(asm, register_attr, repr_simd),
    register_attr(spirv)
)]

mod algebra;
mod ops;

pub use crate::algebra::*;
pub use crate::ops::*;

pub use spirv_std as std;

pub mod bezier;
pub mod geometry;
pub mod hash;
pub mod sdf;
