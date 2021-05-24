use glace::std::bindless::RenderResourceTag;
use crate::gpu::GpuDescriptors;

pub use glace::std::bindless::RenderResourceHandle as CpuDescriptor;

type CpuDescriptors = Vec<CpuDescriptor>;

pub struct Descriptors {
    free_handle: usize,
    tag: RenderResourceTag,

    cpu: CpuDescriptors,
    pub(crate) gpu: GpuDescriptors,
}

impl Descriptors {
    pub fn new(tag: RenderResourceTag, len: usize, gpu: GpuDescriptors) -> Self {
        let mut cpu = CpuDescriptors::with_capacity(len);
        for i in 0..len {
            cpu.push(CpuDescriptor::new(0, tag, i as u32 + 1));
        }

        Self {
            free_handle: 0,
            tag,
            cpu,
            gpu,
        }
    }

    fn invalid_index(&self) -> usize {
        self.cpu.len()
    }

    pub unsafe fn create(&mut self) -> CpuDescriptor {
        assert_ne!(self.free_handle, self.invalid_index()); // out of memory

        let idx = self.free_handle;
        let handle = self.cpu[self.free_handle];

        let version = (handle.version() + 1) % 64;
        let index = handle.index() as usize;

        assert_ne!(index, self.free_handle);
        self.free_handle = index;

        let handle = CpuDescriptor::new(version as _, self.tag, idx as _);
        self.cpu[idx] = handle;

        handle
    }

    pub unsafe fn is_valid(&self, handle: CpuDescriptor) -> bool {
        let idx = handle.index() as usize;
        if idx >= self.cpu.len() {
            return false;
        }

        self.cpu[idx] == handle
    }
}