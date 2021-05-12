use crate::instance::Instance;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, DeviceV1_2, InstanceV1_0},
    vk,
};
use gpu_allocator::{VulkanAllocator, VulkanAllocatorCreateDesc};

const NUM_LAYOUTS: u32 = 64;

#[derive(Debug, Copy, Clone)]
pub struct Resource {
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
}

#[derive(Debug, Copy, Clone)]
pub struct Layout {
    pub pipeline_layout: vk::PipelineLayout,
    pub samplers: Resource,
}

pub struct Gpu {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub allocator: VulkanAllocator,
    pub cmd_pools: Vec<vk::CommandPool>,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
    pub timeline: vk::Semaphore,
    pub descriptor_pool: vk::DescriptorPool,
    pub buffers: Resource,
    pub sampled_images: Resource,
    pub ext_sync2: khr::Synchronization2,
    frame_id: usize,
}

pub struct Descriptors {
    pub buffers: usize,
    pub images: usize,
    pub samplers: usize,
}

impl Gpu {
    pub unsafe fn new(
        instance: &Instance,
        frame_buffering: usize,
        descriptors: Descriptors,
    ) -> anyhow::Result<Self> {
        let (device, queue) = {
            let device_extensions = vec![
                khr::Swapchain::name().as_ptr(),
                khr::Synchronization2::name().as_ptr(),
            ];
            let features = vk::PhysicalDeviceFeatures::builder();
            let mut features11 = vk::PhysicalDeviceVulkan11Features::builder()
                .variable_pointers(true)
                .variable_pointers_storage_buffer(true);
            let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
                .imageless_framebuffer(true)
                .descriptor_indexing(true)
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .vulkan_memory_model(true);
            let mut features_sync2 =
                vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);

            let queue_priorities = [1.0];
            let queue_descs = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(instance.family_index)
                .queue_priorities(&queue_priorities)
                .build()];
            let device_desc = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_descs)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&features)
                .push_next(&mut features11)
                .push_next(&mut features12)
                .push_next(&mut features_sync2);

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

        let descriptor_pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: descriptors.buffers as _,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: descriptors.images as _,
                },
            ];
            let desc = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(NUM_LAYOUTS)
                .pool_sizes(&pool_sizes)
                .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
            device.create_descriptor_pool(&desc, None)?
        };

        let buffer_layout = {
            let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 1];
            let mut flag_desc = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_flags);

            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .binding(0)
                .descriptor_count(descriptors.buffers as _)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()];

            let desc = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut flag_desc);
            device.create_descriptor_set_layout(&desc, None)?
        };
        let sampled_image_layout = {
            let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 1];
            let mut flag_desc = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_flags);

            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .binding(0)
                .descriptor_count(descriptors.images as _)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()];

            let desc = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut flag_desc);
            device.create_descriptor_set_layout(&desc, None)?
        };

        let sets = {
            let layouts = [buffer_layout, sampled_image_layout];

            let desc = vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&layouts)
                .descriptor_pool(descriptor_pool);

            device.allocate_descriptor_sets(&desc)?
        };

        let timeline = {
            let mut timeline_desc = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let desc = vk::SemaphoreCreateInfo::builder().push_next(&mut timeline_desc);
            device.create_semaphore(&desc, None)?
        };

        let ext_sync2 = khr::Synchronization2::new(&instance.instance, &device);

        Ok(Self {
            device,
            queue,
            allocator,
            cmd_pools,
            cmd_buffers,
            timeline,
            descriptor_pool,
            buffers: Resource {
                layout: buffer_layout,
                set: sets[0],
            },
            sampled_images: Resource {
                layout: sampled_image_layout,
                set: sets[1],
            },
            ext_sync2,
            frame_id: 0,
        })
    }

    pub unsafe fn create_layout(&mut self, samplers: &[vk::Sampler], num_constants: u32) -> anyhow::Result<Layout> {
        if samplers.is_empty() {
            let set_layouts = [
                self.buffers.layout,
                self.sampled_images.layout,
            ];
            let push_constants = [vk::PushConstantRange::builder()
                .offset(0)
                .size(num_constants)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()];
            let desc = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constants);

            let pipeline_layout = self.create_pipeline_layout(&desc, None)?;

            Ok(Layout {
                pipeline_layout,
                samplers: Resource {
                    layout: vk::DescriptorSetLayout::null(),
                    set: vk::DescriptorSet::null(),
                }
            })
        } else {
            let sampler_layout = {
                let bindings = [vk::DescriptorSetLayoutBinding::builder()
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .binding(0)
                    .descriptor_count(samplers.len() as _)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .immutable_samplers(samplers)
                    .build()];

                let desc = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
                self.create_descriptor_set_layout(&desc, None)?
            };

            let layouts = [sampler_layout];
            let desc = vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&layouts)
                .descriptor_pool(self.descriptor_pool);
            let set = self.allocate_descriptor_sets(&desc)?;

            let set_layouts = [
                self.buffers.layout,
                self.sampled_images.layout,
                sampler_layout
            ];
            let push_constants = [vk::PushConstantRange::builder()
                .offset(0)
                .size(num_constants)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()];
            let desc = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constants);

            let pipeline_layout = self.create_pipeline_layout(&desc, None)?;

            Ok(Layout {
                pipeline_layout,
                samplers: Resource {
                    layout: sampler_layout,
                    set: set[0],
                }
            })
        }
    }

    pub unsafe fn acquire_cmd_buffer(&mut self) -> anyhow::Result<vk::CommandBuffer> {
        let frame_queue = self.cmd_pools.len();
        let frame_local = self.frame_id % frame_queue;
        if self.frame_id >= frame_queue {
            let semaphores = [self.timeline];
            let wait_values = [(self.frame_id - frame_queue + 1) as u64];
            let wait_info = vk::SemaphoreWaitInfo::builder()
                .semaphores(&semaphores)
                .values(&wait_values);
            self.device.wait_semaphores(&wait_info, !0)?;
            self.device.reset_command_pool(
                self.cmd_pools[frame_local],
                vk::CommandPoolResetFlags::empty(),
            )?;
        }

        let cmd_buffer = self.cmd_buffers[frame_local];
        let begin_desc = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(cmd_buffer, &begin_desc)?;

        self.frame_id += 1;

        Ok(cmd_buffer)
    }

    pub unsafe fn cmd_bind_descriptors(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        pipeline: vk::PipelineBindPoint,
        layout: Layout,
    ) {
        if layout.samplers.layout == vk::DescriptorSetLayout::null() {
            let sets = [self.buffers.set, self.sampled_images.set];
            self.cmd_bind_descriptor_sets(cmd_buffer, pipeline, layout.pipeline_layout, 0, &sets, &[]);
        } else {
            let sets = [self.buffers.set, self.sampled_images.set, layout.samplers.set];
            self.cmd_bind_descriptor_sets(cmd_buffer, pipeline, layout.pipeline_layout, 0, &sets, &[]);
        }
    }
}

impl std::ops::Deref for Gpu {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}
