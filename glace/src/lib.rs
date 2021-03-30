#![cfg_attr(target_arch = "spirv", no_std, feature(asm, repr_simd))]

mod algebra;
mod ops;

pub use crate::algebra::*;
pub use crate::ops::*;

pub mod hash;
pub mod sdf;
pub mod geometry;
pub mod bezier;
