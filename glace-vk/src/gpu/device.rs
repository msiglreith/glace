use crate::gpu;
use ash::{
    extensions::khr,
    version::{DeviceV1_0, DeviceV1_2, InstanceV1_0},
    vk,
};
use gpu_allocator::{VulkanAllocator, VulkanAllocatorCreateDesc};
use glace::std::bindless::RenderResourceTag;

const NUM_LAYOUTS: u32 = 64;

pub struct Extensions {
    pub sync2: khr::Synchronization2,
    #[cfg(feature = "raytrace")]
    pub accel_structure: khr::AccelerationStructure,
    #[cfg(feature = "raytrace")]
    pub deferred: khr::DeferredHostOperations,
}

pub struct Gpu {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub allocator: VulkanAllocator,
    pub cmd_pools: Vec<vk::CommandPool>,
    pub cmd_buffers: Vec<vk::CommandBuffer>,
    pub timeline: vk::Semaphore,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptors_buffer: gpu::Descriptors,
    pub descriptors_image: gpu::Descriptors,
    pub descriptors_accel: gpu::Descriptors,
    pub ext: Extensions,
    frame_id: usize,
}

pub struct DescriptorsDesc {
    pub buffers: usize,
    pub images: usize,
    pub acceleration_structures: usize,
}

impl Gpu {
    pub unsafe fn new(
        instance: &gpu::Instance,
        frame_buffering: usize,
        descriptors: DescriptorsDesc,
    ) -> anyhow::Result<Self> {
        let (device, queue) = {
            let mut device_extensions = vec![
                khr::Swapchain::name().as_ptr(),
                // khr::Synchronization2::name().as_ptr(),
            ];

            if cfg!(feature = "raytrace") {
                device_extensions.extend(&[
                    khr::DeferredHostOperations::name().as_ptr(),
                khr::AccelerationStructure::name().as_ptr(),
                ]);
            }

            let features = vk::PhysicalDeviceFeatures::builder();
            let mut features11 = vk::PhysicalDeviceVulkan11Features::builder()
                .variable_pointers(true)
                .variable_pointers_storage_buffer(true);
            let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
                .timeline_semaphore(true)
                .buffer_device_address(true)
                .descriptor_indexing(true)
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                .vulkan_memory_model(true);
            let mut features_sync2 =
                vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
            let mut features_ray_query =
                vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);
            let mut features_accel_structure =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder().acceleration_structure(true);

            let queue_priorities = [1.0];
            let queue_descs = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(instance.family_index)
                .queue_priorities(&queue_priorities)
                .build()];
            let mut device_desc = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_descs)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&features)
                .push_next(&mut features11)
                .push_next(&mut features12)
                // .push_next(&mut features_sync2)
                .push_next(&mut features_ray_query);

            if cfg!(feature = "raytrace") {
                device_desc = device_desc.push_next(&mut features_accel_structure);
            }

            let device =
                instance
                    .instance
                    .create_device(instance.physical_device, &device_desc, None)?;
            let queue = device.get_device_queue(instance.family_index, 0);

            (device, queue)
        };

        // extensions
        let ext_sync2 = khr::Synchronization2::new(&instance.instance, &device);

        #[cfg(feature = "raytrace")]
        let ext_accel_structure = khr::AccelerationStructure::new(&instance.instance, &device);
        #[cfg(feature = "raytrace")]
        let ext_deferred = khr::DeferredHostOperations::new(&instance.instance, &device);

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
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    descriptor_count: descriptors.acceleration_structures as _,
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
        let acceleration_structure_layout = {
            let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 1];
            let mut flag_desc = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_flags);

            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .binding(0)
                .descriptor_count(descriptors.acceleration_structures as _)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()];

            let desc = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut flag_desc);
            device.create_descriptor_set_layout(&desc, None)?
        };

        let sets = {
            let layouts = [buffer_layout, sampled_image_layout, acceleration_structure_layout];

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

        let gpu_buffers = gpu::GpuDescriptors {
            layout: buffer_layout,
            set: sets[0],
        };
        let descriptors_buffer = gpu::Descriptors::new(RenderResourceTag::Buffer, descriptors.buffers, gpu_buffers);

        let gpu_images = gpu::GpuDescriptors {
            layout: sampled_image_layout,
            set: sets[1],
        };
        let descriptors_image = gpu::Descriptors::new(RenderResourceTag::Texture, descriptors.images, gpu_images);

        let descriptors_accel = {
            let gpu_accel = gpu::GpuDescriptors {
                layout: acceleration_structure_layout,
                set: sets[2],
            };
            gpu::Descriptors::new(RenderResourceTag::Tlas, descriptors.acceleration_structures, gpu_accel)
        };

        Ok(Self {
            device,
            queue,
            allocator,
            cmd_pools,
            cmd_buffers,
            timeline,
            descriptor_pool,
            descriptors_buffer,
            descriptors_image,
            descriptors_accel,
            ext: Extensions {
                sync2: ext_sync2,
                #[cfg(feature = "raytrace")]
                accel_structure: ext_accel_structure,
                #[cfg(feature = "raytrace")]
                deferred: ext_deferred,
            },
            frame_id: 0,
        })
    }

    #[cfg(feature = "raytrace")]
    pub unsafe fn acceleration_structure_address(&self, acceleration_structure: gpu::AccelerationStructure) -> vk::DeviceAddress {
        let desc = vk::AccelerationStructureDeviceAddressInfoKHR::builder().acceleration_structure(acceleration_structure);
        self.ext.accel_structure.get_acceleration_structure_device_address(&desc)
    }

    pub unsafe fn buffer_address(&self, buffer: gpu::Buffer) -> vk::DeviceAddress {
        let desc = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
        self.get_buffer_device_address(&desc)
    }

    pub unsafe fn create_buffer_gpu<T>(
        &mut self,
        name: &str,
        len: usize,
        usage: gpu::BufferUsageFlags,
    ) -> anyhow::Result<gpu::Buffer> {
        let size = std::mem::size_of::<T>() * len;
        let desc = vk::BufferCreateInfo::builder().size(size as _).usage(usage);
        let buffer = self.create_buffer(&desc, None)?;
        let alloc_desc = gpu_allocator::AllocationCreateDesc {
            name,
            requirements: self.get_buffer_memory_requirements(buffer),
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
        };
        let allocation = self.allocator.allocate(&alloc_desc)?;
        self.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        Ok(buffer)
    }

    pub unsafe fn create_layout(
        &mut self,
        samplers: &[vk::Sampler],
        num_constants: u32,
    ) -> anyhow::Result<gpu::Layout> {
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
            self.descriptors_buffer.gpu.layout,
            self.descriptors_image.gpu.layout,
            sampler_layout,
            self.descriptors_accel.gpu.layout,
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

        Ok(gpu::Layout {
            pipeline_layout,
            samplers: gpu::GpuDescriptors {
                layout: sampler_layout,
                set: set[0],
            },
        })
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

    pub unsafe fn update_descriptors(
        &mut self,
        buffers: &[gpu::BufferDescriptor],
        images: &[gpu::ImageDescriptor],
        accels: &[gpu::AccelerationStructureDescriptor],
    ) {
        let mut updates = Vec::new();

        let mut buffer_infos = Vec::new();
        for buffer in buffers {

            buffer_infos.push(vk::DescriptorBufferInfo {
                buffer: buffer.buffer,
                offset: buffer.offset,
                range: buffer.range,
            });
        }
        for (i, buffer) in buffers.iter().enumerate() {
            assert!(self.descriptors_buffer.is_valid(buffer.handle));
            updates.push(
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .dst_set(self.descriptors_buffer.gpu.set)
                    .dst_binding(0)
                    .dst_array_element(buffer.handle.index())
                    .buffer_info(&buffer_infos[i .. i+1])
                    .build()
            )
        }

        let mut image_infos = Vec::new();
        for image in images {
            image_infos.push(vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                    image_view: image.view,
                    image_layout: image.layout,
            });
        }
        for (i, image) in images.iter().enumerate() {
            assert!(self.descriptors_image.is_valid(image.handle));
            updates.push(
                vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_set(self.descriptors_image.gpu.set)
                    .dst_binding(0)
                    .dst_array_element(image.handle.index())
                    .image_info(&image_infos[i .. i + 1])
                    .build(),
            )
        }


        let mut accel_handles = Vec::new();
        for accel in accels {
            accel_handles.push(accel.acceleration_structure);
        }
        let mut accel_infos = Vec::new();
        for i in 0..accels.len() {
            accel_infos.push(vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                .acceleration_structures(&accel_handles[i .. i + 1]).build()
            );
        }

        #[cfg(feature = "raytrace")]
        for (i, accel) in accels.iter().enumerate() {
            assert!(self.descriptors_accel.is_valid(accel.handle));
            let mut write = vk::WriteDescriptorSet::builder()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .dst_set(self.descriptors_accel.gpu.set)
                .dst_binding(0)
                .dst_array_element(accel.handle.index())
                .push_next(&mut accel_infos[i])
                .build();
            write.descriptor_count = 1;
            updates.push(write);
        }

        self.update_descriptor_sets(
            &updates,
            &[],
        );
    }

    pub unsafe fn cmd_bind_descriptors(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        pipeline: vk::PipelineBindPoint,
        layout: gpu::Layout,
    ) {
        let sets = [
            self.descriptors_buffer.gpu.set,
            self.descriptors_image.gpu.set,
            layout.samplers.set,
            self.descriptors_accel.gpu.set,
        ];
        self.cmd_bind_descriptor_sets(
            cmd_buffer,
            pipeline,
            layout.pipeline_layout,
            0,
            &sets,
            &[],
        );
    }
}

impl std::ops::Deref for Gpu {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}
