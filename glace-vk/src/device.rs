use crate::instance::Instance;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use gpu_allocator::{VulkanAllocator, VulkanAllocatorCreateDesc};

pub struct Device {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub allocator: VulkanAllocator,
    pub cmd_pools: Vec<vk::CommandPool>,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
    pub timeline: vk::Semaphore,
}

impl Device {
    pub unsafe fn new(instance: &Instance, frame_buffering: usize) -> anyhow::Result<Self> {
        let (device, queue) = {
            let device_extensions = vec![khr::Swapchain::name().as_ptr()];
            let features = vk::PhysicalDeviceFeatures::builder();
            let mut features12 =
                vk::PhysicalDeviceVulkan12Features::builder().imageless_framebuffer(true);

            let queue_priorities = [1.0];
            let queue_descs = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(instance.family_index)
                .queue_priorities(&queue_priorities)
                .build()];
            let device_desc = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_descs)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&features)
                .push_next(&mut features12);

            let device =
                instance
                    .instance
                    .create_device(instance.physical_device, &device_desc, None)?;
            let queue = device.get_device_queue(instance.family_index, 0);

            (device, queue)
        };

        let allocator = VulkanAllocator::new(&VulkanAllocatorCreateDesc {
            instance: instance.instance.clone(),
            device: device.clone(),
            physical_device: instance.physical_device,
            debug_settings: Default::default(),
        });

        let cmd_pools = (0..frame_buffering)
            .map(|_| {
                let desc =
                    vk::CommandPoolCreateInfo::builder().queue_family_index(instance.family_index);
                device.create_command_pool(&desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let cmd_buffers = {
            let cmd_buffers = cmd_pools
                .iter()
                .map(|pool| {
                    let desc = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(*pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    device.allocate_command_buffers(&desc)
                })
                .collect::<Result<Vec<_>, _>>()?;
            cmd_buffers.into_iter().flatten().collect()
        };

        let timeline = {
            let mut timeline_desc = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let desc = vk::SemaphoreCreateInfo::builder().push_next(&mut timeline_desc);
            device.create_semaphore(&desc, None)?
        };

        Ok(Device {
            device,
            queue,
            allocator,
            cmd_pools,
            cmd_buffers,
            timeline,
        })
    }
}

impl std::ops::Deref for Device {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}
