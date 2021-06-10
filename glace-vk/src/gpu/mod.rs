mod descriptor;
mod device;
mod instance;
mod swapchain;

pub use self::descriptor::{CpuDescriptor, Descriptors};
pub use self::device::{DescriptorsDesc, Gpu, Pool};
pub use self::instance::Instance;
pub use self::swapchain::Swapchain;

pub use ash::vk::{
    AccelerationStructureKHR as AccelerationStructure, AccessFlags2KHR as Access, Buffer,
    BufferUsageFlags, Image, ImageLayout, ImageView, PipelineStageFlags2KHR as Stage,
    Semaphore,
};

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

#[derive(Debug, Copy, Clone)]
pub struct MemoryAccess {
    pub access: Access,
    pub stage: Stage,
}

#[derive(Debug, Copy, Clone)]
pub struct MemoryBarrier {
    pub src: MemoryAccess,
    pub dst: MemoryAccess,
}

impl MemoryBarrier {
    pub fn full() -> Self {
        MemoryBarrier {
            src: MemoryAccess {
                access: Access::MEMORY_READ
                    | Access::MEMORY_WRITE,
                stage: Stage::ALL_COMMANDS,
            },
            dst: MemoryAccess {
                access: Access::MEMORY_READ
                    | Access::MEMORY_WRITE,
                stage: Stage::ALL_COMMANDS,
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ImageAccess {
    pub access: Access,
    pub stage: Stage,
    pub layout: ImageLayout,
}

#[derive(Debug, Copy, Clone)]
pub struct ImageBarrier {
    pub image: Image,
    pub range: vk::ImageSubresourceRange,
    pub src: ImageAccess,
    pub dst: ImageAccess,
}

#[derive(Debug, Copy, Clone)]
pub struct SemaphoreSubmit {
    pub semaphore: Semaphore,
    pub stage: Stage,
}

#[derive(Debug, Copy, Clone)]
pub struct Submit<'a> {
    pub waits: &'a [SemaphoreSubmit],
    pub signals: &'a [SemaphoreSubmit],
}
