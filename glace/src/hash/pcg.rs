//! Permuted Congruential Generator.
//!
//! @article{Jarzynski2020Hash,
//!   author =       {Mark Jarzynski and Marc Olano},
//!   title =        {Hash Functions for GPU Rendering},
//!   year =         2020,
//!   month =        {October},
//!   day =          17,
//!   journal =      {Journal of Computer Graphics Techniques (JCGT)},
//!   volume =       9,
//!   number =       3,
//!   pages =        {20--38},
//!   url =          {http://jcgt.org/published/0009/03/02/},
//!   issn =         {2331-7418}
//! }

use crate::u32x3;

pub fn pcg3d(mut v: u32x3) -> u32x3 {
    v = v
        .wrapping_mul(u32x3 {
            x: 1664525u32,
            y: 1664525u32,
            z: 1664525u32,
        })
        .wrapping_add(u32x3 {
            x: 1013904223u32,
            y: 1013904223u32,
            z: 1013904223u32,
        });
    v.x = v.x.wrapping_add(v.y.wrapping_mul(v.z));
    v.y = v.y.wrapping_add(v.z.wrapping_mul(v.x));
    v.z = v.z.wrapping_add(v.x.wrapping_mul(v.y));
    v = v
        ^ (v >> u32x3 {
            x: 16u32,
            y: 16u32,
            z: 16u32,
        });
    v.x = v.x.wrapping_add(v.y.wrapping_mul(v.z));
    v.y = v.y.wrapping_add(v.z.wrapping_mul(v.x));
    v.z = v.z.wrapping_add(v.x.wrapping_mul(v.y));
    v
}
