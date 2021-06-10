use crate::gpu;
use ash::{extensions::khr, vk, prelude::*};
use glace::std::bindless::RenderResourceTag;
use gpu_allocator::{VulkanAllocator, VulkanAllocatorCreateDesc};

const NUM_LAYOUTS: u32 = 64;

pub struct Extensions {
    pub sync2: Option<khr::Synchronization2>, // not supported by Nsight
    #[cfg(feature = "raytrace")]
    pub accel_structure: khr::AccelerationStructure,
    #[cfg(feature = "raytrace")]
    pub deferred: khr::DeferredHostOperations,
}

struct Shrine {
    buffers: Vec<vk::Buffer>,
    images: Vec<vk::Image>,
}

impl Shrine {
    fn new() -> Self {
        Shrine {
            buffers: Vec::new(),
            images: Vec::new(),
        }
    }
}

struct PoolData {
    cmd_pool: vk::CommandPool,
    cmd_buffer: vk::CommandBuffer,
    shrine: Shrine,
}

#[derive(Copy, Clone, Debug)]
pub struct Pool {
    pub id: usize,
    pub cmd_buffer: vk::CommandBuffer,
}

pub struct Gpu {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub allocator: VulkanAllocator,
    pools: Vec<PoolData>,
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
        let supports_sync2 = instance.supports_extension(khr::Synchronization2::name()) && !cfg!(feature = "nsight");

        let (device, queue) = {
            let mut device_extensions = vec![
                khr::Swapchain::name().as_ptr(),
            ];
            if supports_sync2 {
                device_extensions.push(khr::Synchronization2::name().as_ptr());
            }

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
                .shader_int8(true) // required as rust-gpu forcefully sets it
                .vulkan_memory_model(true);
            let mut features_sync2 =
                vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
            let mut features_ray_query =
                vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);
            let mut features_accel_structure =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                    .acceleration_structure(true);

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
                .push_next(&mut features_sync2)
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
        let ext_sync2 = supports_sync2.then(|| khr::Synchronization2::new(&instance.instance, &device));

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

        let pools = (0..frame_buffering)
            .map(|_| {
                let cmd_pool = {
                    let desc =
                        vk::CommandPoolCreateInfo::builder().queue_family_index(instance.family_index);
                    device.create_command_pool(&desc, None)?
                };
                let cmd_buffer = {
                    let desc = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    device.allocate_command_buffers(&desc)?[0]
                };

                Ok(PoolData {
                    cmd_pool,
                    cmd_buffer,
                    shrine: Shrine::new(),
                })
            })
            .collect::<Result<Vec<_>, vk::Result>>()?;

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
        let acceleration_structure_layout = if cfg!(feature = "raytrace") {
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
        } else {
            // dummy layout
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .binding(0)
                .descriptor_count(0)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build()];
            let desc = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings);
            device.create_descriptor_set_layout(&desc, None)?
        };

        let sets = {
            let layouts = [
                buffer_layout,
                sampled_image_layout,
                acceleration_structure_layout,
            ];

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
        let descriptors_buffer =
            gpu::Descriptors::new(RenderResourceTag::Buffer, descriptors.buffers, gpu_buffers);

        let gpu_images = gpu::GpuDescriptors {
            layout: sampled_image_layout,
            set: sets[1],
        };
        let descriptors_image =
            gpu::Descriptors::new(RenderResourceTag::Texture, descriptors.images, gpu_images);

        let descriptors_accel = {
            let gpu_accel = gpu::GpuDescriptors {
                layout: acceleration_structure_layout,
                set: sets[2],
            };
            gpu::Descriptors::new(
                RenderResourceTag::Tlas,
                descriptors.acceleration_structures,
                gpu_accel,
            )
        };

        Ok(Self {
            device,
            queue,
            allocator,
            pools,
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
    pub unsafe fn acceleration_structure_address(
        &self,
        acceleration_structure: gpu::AccelerationStructure,
    ) -> vk::DeviceAddress {
        let desc = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
            .acceleration_structure(acceleration_structure);
        self.ext
            .accel_structure
            .get_acceleration_structure_device_address(&desc)
    }

    #[cfg(feature = "raytrace")]
    pub unsafe fn buffer_address(&self, buffer: gpu::Buffer) -> vk::DeviceAddress {
        let desc = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
        self.get_buffer_device_address(&desc)
    }

    pub unsafe fn create_buffer_gpu(
        &mut self,
        name: &str,
        size: usize,
        usage: gpu::BufferUsageFlags,
    ) -> anyhow::Result<gpu::Buffer> {
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

    pub unsafe fn acquire_pool(&mut self) -> anyhow::Result<Pool> {
        let frame_queue = self.pools.len();
        let frame_local = self.frame_id % frame_queue;
        let pool = &mut self.pools[frame_local];

        if self.frame_id >= frame_queue {
            let semaphores = [self.timeline];
            let wait_values = [(self.frame_id - frame_queue + 1) as u64];
            let wait_info = vk::SemaphoreWaitInfo::builder()
                .semaphores(&semaphores)
                .values(&wait_values);
            self.device.wait_semaphores(&wait_info, !0)?;
            self.device.reset_command_pool(
                pool.cmd_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
            for buffer in pool.shrine.buffers.drain(..) {
                self.device.destroy_buffer(buffer, None);
            }
            for image in pool.shrine.images.drain(..) {
                self.device.destroy_image(image, None);
            }
        }

        let cmd_buffer = pool.cmd_buffer;
        let begin_desc = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(cmd_buffer, &begin_desc)?;

        let pool_handle = Pool {
            cmd_buffer,
            id: self.frame_id,
        };

        self.frame_id += 1;

        Ok(pool_handle)
    }

    pub unsafe fn submit_pool(&mut self, pool: Pool, submit: gpu::Submit) -> VkResult<()> {
        self.device.end_command_buffer(pool.cmd_buffer)?;

        match self.ext.sync2 {
            Some(ref sync2) => {
                let waits = submit.waits.iter().map(|desc| {
                    vk::SemaphoreSubmitInfoKHR::builder()
                        .semaphore(desc.semaphore)
                        .stage_mask(desc.stage)
                        .build()
                }).collect::<Box<[_]>>();

                let mut signals = submit.signals.iter().map(|desc| {
                    vk::SemaphoreSubmitInfoKHR::builder()
                        .semaphore(desc.semaphore)
                        .stage_mask(desc.stage)
                        .build()
                }).collect::<Vec<_>>();
                signals.push(
                    vk::SemaphoreSubmitInfoKHR::builder()
                        .semaphore(self.timeline)
                        .value(pool.id as u64 + 1)
                        .stage_mask(gpu::Stage::NONE)
                        .build()
                );

                let cmd_buffers = [
                    vk::CommandBufferSubmitInfoKHR::builder()
                        .command_buffer(pool.cmd_buffer)
                        .build()
                ];

                let desc = [
                    vk::SubmitInfo2KHR::builder()
                        .wait_semaphore_infos(&waits)
                        .signal_semaphore_infos(&signals)
                        .command_buffer_infos(&cmd_buffers)
                        .build()
                ];
                sync2.queue_submit2(self.queue, &desc, vk::Fence::null())
            }
            None => {
                let waits = submit.waits.iter().map(|desc| desc.semaphore).collect::<Box<[_]>>();
                let wait_stages = submit.waits.iter().map(|desc| map_stage(SyncScope::Second, desc.stage)).collect::<Box<[_]>>();
                let mut signals = submit.signals.iter().map(|desc| desc.semaphore).collect::<Vec<_>>();
                let cmd_buffer = [pool.cmd_buffer];

                let wait_values = vec![0; waits.len()];
                let mut signal_values = vec![0; signals.len()];

                signals.push(self.timeline);
                signal_values.push(pool.id as u64 + 1);

                let mut timeline_desc = vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(&wait_values)
                    .signal_semaphore_values(&signal_values);

                let desc = [
                    vk::SubmitInfo::builder()
                        .wait_semaphores(&waits)
                        .command_buffers(&cmd_buffer)
                        .wait_dst_stage_mask(&wait_stages)
                        .signal_semaphores(&signals)
                        .push_next(&mut timeline_desc)
                        .build()
                ];
                self.queue_submit(self.queue, &desc, vk::Fence::null())
            }
        }
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
                    .buffer_info(&buffer_infos[i..i + 1])
                    .build(),
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
                    .image_info(&image_infos[i..i + 1])
                    .build(),
            )
        }

        let mut accel_handles = Vec::new();
        for accel in accels {
            accel_handles.push(accel.acceleration_structure);
        }
        let mut accel_infos = Vec::new();
        for i in 0..accels.len() {
            accel_infos.push(
                vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                    .acceleration_structures(&accel_handles[i..i + 1])
                    .build(),
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

        self.update_descriptor_sets(&updates, &[]);
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
        self.cmd_bind_descriptor_sets(cmd_buffer, pipeline, layout.pipeline_layout, 0, &sets, &[]);
    }

    pub unsafe fn cmd_barriers(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        memory: &[gpu::MemoryBarrier],
        image: &[gpu::ImageBarrier],
    ) {
        match self.ext.sync2 {
            Some(ref sync2) => {
                let memory_barriers = memory
                    .iter()
                    .map(|barrier| {
                        vk::MemoryBarrier2KHR::builder()
                            .src_access_mask(barrier.src.access)
                            .dst_access_mask(barrier.dst.access)
                            .src_stage_mask(barrier.src.stage)
                            .dst_stage_mask(barrier.dst.stage)
                            .build()
                    })
                    .collect::<Box<[_]>>();

                let image_barriers = image
                    .iter()
                    .map(|barrier| {
                        vk::ImageMemoryBarrier2KHR::builder()
                            .image(barrier.image)
                            .subresource_range(barrier.range)
                            .old_layout(barrier.src.layout)
                            .new_layout(barrier.dst.layout)
                            .src_access_mask(barrier.src.access)
                            .dst_access_mask(barrier.dst.access)
                            .src_stage_mask(barrier.src.stage)
                            .dst_stage_mask(barrier.dst.stage)
                            .build()
                    })
                    .collect::<Box<[_]>>();

                let dependency = vk::DependencyInfoKHR::builder()
                    .memory_barriers(&memory_barriers)
                    .image_memory_barriers(&image_barriers);
                sync2.cmd_pipeline_barrier2(cmd_buffer, &dependency);
            }
            None => {
                let mut src_stage = vk::PipelineStageFlags::empty();
                let mut dst_stage = vk::PipelineStageFlags::empty();

                let mut memory_src_access = vk::AccessFlags::empty();
                let mut memory_dst_access = vk::AccessFlags::empty();
                for barrier in memory {
                    src_stage |= map_stage(SyncScope::First, barrier.src.stage);
                    dst_stage |= map_stage(SyncScope::Second, barrier.dst.stage);

                    memory_src_access |= map_access(barrier.src.access);
                    memory_dst_access |= map_access(barrier.dst.access);
                }

                let mut image_barriers = Vec::new();
                for barrier in image {
                    src_stage |= map_stage(SyncScope::First, barrier.src.stage);
                    dst_stage |= map_stage(SyncScope::Second, barrier.dst.stage);

                    image_barriers.push(vk::ImageMemoryBarrier::builder()
                        .src_access_mask(map_access(barrier.src.access))
                        .dst_access_mask(map_access(barrier.dst.access))
                        .old_layout(barrier.src.layout)
                        .new_layout(barrier.dst.layout)
                        .image(barrier.image)
                        .subresource_range(barrier.range)
                        .build()
                    );
                }

                let mut memory_barrier = vec![];
                if !(memory_src_access.is_empty() && memory_dst_access.is_empty()) {
                    memory_barrier.push(vk::MemoryBarrier::builder()
                        .src_access_mask(memory_src_access)
                        .dst_access_mask(memory_dst_access)
                        .build());
                }
                self.cmd_pipeline_barrier(
                    cmd_buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &memory_barrier,
                    &[],
                    &image_barriers,
                );
            }
        }
    }
}

impl std::ops::Deref for Gpu {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl std::ops::Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            self.device_wait_idle().unwrap();
        }
    }
}

enum SyncScope {
    First,
    Second,
}

fn map_stage(scope: SyncScope, stage: gpu::Stage) -> vk::PipelineStageFlags {
    let mut stage_flags = vk::PipelineStageFlags::from_raw((stage.as_raw() & 0x7fffffff) as _);

    if stage == gpu::Stage::NONE {
        stage_flags |= match scope {
            SyncScope::First => vk::PipelineStageFlags::TOP_OF_PIPE,
            SyncScope::Second => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        };
    }

    if stage.intersects(gpu::Stage::COPY | gpu::Stage::RESOLVE | gpu::Stage::BLIT | gpu::Stage::CLEAR) {
        stage_flags |= vk::PipelineStageFlags::TRANSFER;
    }
    if stage.intersects(gpu::Stage::INDEX_INPUT | gpu::Stage::VERTEX_ATTRIBUTE_INPUT) {
        stage_flags |= vk::PipelineStageFlags::VERTEX_INPUT;
    }

    // todo: pre-rasterization

    stage_flags
}

fn map_access(access: gpu::Access) -> vk::AccessFlags {
    let mut access_flags = vk::AccessFlags::from_raw((access.as_raw() & 0x7fffffff) as _);

    if access.contains(gpu::Access::SHADER_STORAGE_WRITE) {
        access_flags |= vk::AccessFlags::SHADER_WRITE;
    }
    if access.intersects(gpu::Access::SHADER_STORAGE_READ | gpu::Access::SHADER_SAMPLED_READ) {
        access_flags |= vk::AccessFlags::SHADER_READ;
    }

    access_flags
}