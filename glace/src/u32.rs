pub fn bitfield_insert(value: u32, insert: u32, offset: usize, count: usize) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        let result: u32;
        unsafe {
            asm! {
                "{0} = OpBitFieldInsert _ {1} {2} {3} {4}",
                out(reg) result,
                in(reg) value,
                in(reg) insert,
                in(reg) offset,
                in(reg) count,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let mask = (1 << count) - 1;
        (value & !(mask << offset)) | ((insert & mask) << offset)
    }
}

pub fn bitfield_extract(value: u32, offset: usize, count: usize) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        let result: u32;
        unsafe {
            asm! {
                "{result} = OpBitFieldUExtract _ {value} {offset} {count}",
                result = out(reg) result,
                value = in(reg) value,
                offset = in(reg) offset,
                count = in(reg) count,
            }
        }
        result
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        (value >> offset) & ((1 << count) - 1)
    }
}
