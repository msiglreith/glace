use crate::f32x2;

pub struct Fullscreen;

impl Fullscreen {
    pub fn position(idx: i32) -> f32x2 {
        f32x2 {
            x: (((idx & 1) << 2) - 1) as f32,
            y: (((idx & 2) << 1) - 1) as f32,
        }
    }
}