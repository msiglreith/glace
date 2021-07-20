#[cfg(target_arch = "spirv")]
use spirv_std::RuntimeArray;

#[spirv_std_macros::gpu_only]
#[cfg_attr(target_arch = "spirv", spirv(resource_from_handle_intrinsic))]
pub unsafe extern "unadjusted" fn resource_from_handle<T>(_resource: u64) -> T {
    unimplemented!()
}

#[derive(Copy, Clone)]
pub struct Buffer(pub u64);

impl Buffer {
    pub fn is_valid(self) -> bool {
        self.0 != 0
    }
    #[spirv_std_macros::gpu_only]
    pub unsafe fn load<T: 'static>(self) -> &'static mut T {
        resource_from_handle::<&'static mut T>(self.0)
    }

    #[spirv_std_macros::gpu_only]
    pub unsafe fn index<T: 'static>(self, index: usize) -> &'static T {
        self.load::<RuntimeArray<T>>().index(index)
    }

    #[spirv_std_macros::gpu_only]
    pub unsafe fn index_mut<T: 'static>(self, index: usize) -> &'static mut T {
        self.load::<RuntimeArray<T>>().index_mut(index)
    }
}
