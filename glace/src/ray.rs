use crate::Vec3;

#[derive(Copy, Clone)]
pub struct Ray<T> {
    pub origin: Vec3<T>,
    pub direction: Vec3<T>,
}

impl Ray<f32> {
    /// Offset ray origin to avoid self-intersection based on geometric normal.
    ///
    /// Source: Carsten Wächter and Nikolaus Binder, 'A Fast and Robust Method for Avoiding Self-Intersection',
    pub fn geometric_offset(self, normal: Vec3<f32>) -> Self {
        use crate::{arch::bitcast, f32::fabs, f32x3, i32x3, vec3};

        const ORIGIN: f32 = 1.0 / 32.0;
        const FLOAT_SCALE: f32 = 1.0 / 65536.0;
        const INT_SCALE: f32 = 256.0;

        let offset: i32x3 = vec3(
            (INT_SCALE * normal.x) as _,
            (INT_SCALE * normal.y) as _,
            (INT_SCALE * normal.z) as _,
        );
        let pos: f32x3 = vec3(
            bitcast(
                bitcast::<_, i32>(self.origin.x)
                    + if self.origin.x < 0.0 {
                        -offset.x
                    } else {
                        offset.x
                    },
            ),
            bitcast(
                bitcast::<_, i32>(self.origin.y)
                    + if self.origin.y < 0.0 {
                        -offset.y
                    } else {
                        offset.y
                    },
            ),
            bitcast(
                bitcast::<_, i32>(self.origin.z)
                    + if self.origin.z < 0.0 {
                        -offset.z
                    } else {
                        offset.z
                    },
            ),
        );

        Self {
            origin: vec3(
                if fabs(self.origin.x) < ORIGIN {
                    self.origin.x + FLOAT_SCALE * normal.x
                } else {
                    pos.x
                },
                if fabs(self.origin.y) < ORIGIN {
                    self.origin.y + FLOAT_SCALE * normal.y
                } else {
                    pos.y
                },
                if fabs(self.origin.z) < ORIGIN {
                    self.origin.z + FLOAT_SCALE * normal.z
                } else {
                    pos.z
                },
            ),
            direction: self.direction,
        }
    }
}
