use crate::u32x2;

#[cfg(target_feature = "GroupNonUniformQuad")]
pub mod downsample;

/// Invocation Id reordering (1D -> 2D).
///
/// y2x2x1y1y0x0: 64 -> (x2x1x0, y2y1y0); (8, 8)
/// Quad (x0-y0), Quad (y1-x1), Quad (x2-y2)
pub fn reorder64_yxxyyx(id: u32) -> u32x2 {
    let x2x1y0 = crate::u32::bitfield_extract(id, 2, 3);
    let x = crate::u32::bitfield_insert(x2x1y0, id, 0, 1);

    let y1y0 = crate::u32::bitfield_extract(id, 1, 2);
    let y2x2x1 = crate::u32::bitfield_extract(id, 3, 3);
    let y = crate::u32::bitfield_insert(y2x2x1, y1y0, 0, 2);

    u32x2 { x, y }
}

pub fn reorder256_xyyxxyyx(id: u32) -> u32x2 {
    let x2x1y0 = crate::u32::bitfield_extract(id, 2, 3);
    let x2x1x0 = crate::u32::bitfield_insert(x2x1y0, id, 0, 1);
    let x3y3y2x2 = crate::u32::bitfield_extract(id, 4, 4);
    let x = crate::u32::bitfield_insert(x3y3y2x2, x2x1x0, 0, 3);

    let y1y0 = crate::u32::bitfield_extract(id, 1, 2);
    let y3y2x2x1 = crate::u32::bitfield_extract(id, 3, 4);
    let y = crate::u32::bitfield_insert(y3y2x2x1, y1y0, 0, 2);

    u32x2 { x, y }
}
