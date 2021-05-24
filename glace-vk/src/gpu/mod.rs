mod device;
mod instance;
mod swapchain;
mod descriptor;

pub use self::device::{DescriptorsDesc, Gpu};
pub use self::instance::Instance;
pub use self::swapchain::Swapchain;
pub use self::descriptor::{CpuDescriptor, Descriptors};

pub use ash::vk::{Buffer, BufferUsageFlags, ImageView, ImageLayout, AccelerationStructureKHR as AccelerationStructure};

use ash::vk;

/// View a slice as raw byte slice.
///
/// Reinterprets the passed data as raw memory.
/// Be aware of possible packing and aligning rules by Rust compared to OpenGL.
pub fn as_u8_slice<T>(data: &[T]) -> &[u8] {
    let len = std::mem::size_of::<T>() * data.len();
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) }
}

#[derive(Debug, Copy, Clone)]
pub struct GpuDescriptors {
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
}

#[derive(Debug, Copy, Clone)]
pub struct Layout {
    pub pipeline_layout: vk::PipelineLayout,
    pub samplers: GpuDescriptors,
}

#[derive(Debug, Copy, Clone)]
pub struct BufferDescriptor {
    pub handle: CpuDescriptor,
    pub buffer: Buffer,
    pub offset: u64,
    pub range: u64,
}

#[derive(Debug, Copy, Clone)]
pub struct ImageDescriptor {
    pub handle: CpuDescriptor,
    pub view: ImageView,
    pub layout: ImageLayout,
}

#[derive(Debug, Copy, Clone)]
pub struct AccelerationStructureDescriptor {
    pub handle: CpuDescriptor,
    pub acceleration_structure: AccelerationStructure,
}
