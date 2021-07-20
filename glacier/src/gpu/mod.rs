mod device;
mod instance;
mod swapchain;

pub use self::device::{Gpu, Pool};
pub use self::instance::Instance;
pub use self::swapchain::Swapchain;

pub use ash::vk::{
    AccelerationStructureKHR as AccelerationStructure, AccessFlags2KHR as Access, Buffer,
    BufferUsageFlags, Image, ImageLayout, ImageUsageFlags, ImageView, Pipeline,
    PipelineStageFlags2KHR as Stage, Sampler, Semaphore, ShaderModule as Shader,
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
pub struct Layout {
    pub pipeline_layout: vk::PipelineLayout,
}

#[derive(Debug, Copy, Clone)]
pub enum BufferInit<'a> {
    Host { pool: Pool, data: &'a [u8] },
    None,
}

#[derive(Debug, Copy, Clone)]
pub enum ImageInit<'a> {
    Host {
        pool: Pool,
        aspect: vk::ImageAspectFlags,
        data: &'a [u8],
    },
    None,
}

pub enum GeometryBlas {
    Triangles {
        flags: vk::GeometryFlagsKHR,
        format: vk::Format,
        vertex_buffer: u64,
        vertex_stride: usize,
        num_vertices: usize,
        index_type: vk::IndexType,
        index_buffer: u64,
        num_indices: usize,
    },
}

pub struct GeometryTlas {
    pub flags: vk::GeometryFlagsKHR,
    pub instance_buffer: BufferView,
    pub num_instances: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct BufferView {
    pub buffer: Buffer,
    pub offset: u64,
    pub range: u64,
}

impl BufferView {
    pub fn whole(buffer: Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }

    pub unsafe fn handle(&self, gpu: &Gpu) -> vk::DeviceAddress {
        gpu.buffer_address(self.buffer) + self.offset
    }
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
                access: Access::MEMORY_READ | Access::MEMORY_WRITE,
                stage: Stage::ALL_COMMANDS,
            },
            dst: MemoryAccess {
                access: Access::MEMORY_READ | Access::MEMORY_WRITE,
                stage: Stage::ALL_COMMANDS,
            },
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

pub struct ImageDesc {
    pub ty: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub usage: ImageUsageFlags,
    pub mip_levels: usize,
    pub array_layers: usize,
    pub samples: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct Attachment {
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub samples: usize,
    pub src: (vk::ImageLayout, vk::AttachmentLoadOp),
    pub dst: (vk::ImageLayout, vk::AttachmentStoreOp),
}

#[derive(Debug, Copy, Clone)]
pub struct RenderPass {
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
}
