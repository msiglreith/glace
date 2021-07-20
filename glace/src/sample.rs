use crate::{f32x2, f32x3, std::num_traits::Float};

pub fn hemisphere_cosine(u: f32x2) -> (f32, f32x3) {
    let ux_sqrt = u.x.sqrt();
    let uy_tau = u.y * core::f32::consts::TAU;

    let v = f32x3 {
        x: ux_sqrt * uy_tau.cos(),
        y: ux_sqrt * uy_tau.sin(),
        z: (1.0 - u.x).sqrt(),
    };
    let pdf = v.z * core::f32::consts::FRAC_1_PI;

    (pdf, v)
}
