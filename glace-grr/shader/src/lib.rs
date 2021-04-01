#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items, register_attr, asm)]
#![register_attr(spirv)]

pub mod cubemap;
pub mod debug;
pub mod ocean;
pub mod pbr;
pub mod skybox;
