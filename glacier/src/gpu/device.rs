use crate::gpu;
use ash::{
    extensions::{ext, khr},
    prelude::*,
    vk::Handle,
    vk::{self, NvxImageViewHandleFn},
};
use gpu_allocator::{VulkanAllocator, VulkanAllocatorCreateDesc};
use std::{
    ffi::CString,
    path::{Path, PathBuf},
};

pub struct Extensions {
    pub sync2: Option<khr::Synchronization2>, // not well supported by nsight
    pub image_handle: vk::NvxImageViewHandleFn,
    pub debug_utils: Option<ext::DebugUtils>,
    pub accel_structure: khr::AccelerationStructure,
    pub deferred: khr::DeferredHostOperations,
    pub raytracing_pipeline: khr::RayTracingPipeline,
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
    pub ext: Extensions,
    frame_id: usize,
    rust_spv: PathBuf,
}

impl Gpu {
    pub unsafe fn new(instance: &gpu::Instance, frame_buffering: usize) -> anyhow::Result<Self> {
        let supports_sync2 = instance.supports_device_extension(khr::Synchronization2::name())
            && !cfg!(feature = "nsight");
        let supports_debug_utils = instance.supports_instance_extension(ext::DebugUtils::name());

        let (device, queue) = {
            let mut device_extensions = vec![khr::Swapchain::name().as_ptr()];
            if supports_sync2 {
                device_extensions.push(khr::Synchronization2::name().as_ptr());
            }

            device_extensions.extend(&[
                khr::DeferredHostOperations::name().as_ptr(),
                khr::AccelerationStructure::name().as_ptr(),
                vk::KhrRayQueryFn::name().as_ptr(),
            ]);

            device_extensions.push(NvxImageViewHandleFn::name().as_ptr());

            let features = vk::PhysicalDeviceFeatures::builder()
                .robust_buffer_access(true)
                .shader_storage_image_write_without_format(true)
                .shader_int64(true);
            let mut features11 = vk::PhysicalDeviceVulkan11Features::builder()
                .variable_pointers(true)
                .variable_pointers_storage_buffer(true);
            let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
                .timeline_semaphore(true)
                .buffer_device_address(true)
                .imageless_framebuffer(true)
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
            let device_desc = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_descs)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&features)
                .push_next(&mut features11)
                .push_next(&mut features12)
                .push_next(&mut features_sync2)
                .push_next(&mut features_ray_query)
                .push_next(&mut features_accel_structure);

            let device =
                instance
                    .instance
                    .create_device(instance.physical_device, &device_desc, None)?;
            let queue = device.get_device_queue(instance.family_index, 0);

            (device, queue)
        };

        // extensions
        let ext_sync2 =
            supports_sync2.then(|| khr::Synchronization2::new(&instance.instance, &device));
        let ext_debug_utils =
            supports_debug_utils.then(|| ext::DebugUtils::new(&instance.entry, &instance.instance));

        let ext_accel_structure = khr::AccelerationStructure::new(&instance.instance, &device);
        let ext_deferred = khr::DeferredHostOperations::new(&instance.instance, &device);
        let ext_raytracing_pipeline = khr::RayTracingPipeline::new(&instance.instance, &device);

        let ext_image_handke = vk::NvxImageViewHandleFn::load(|name| unsafe {
            std::mem::transmute(
                instance
                    .instance
                    .get_device_proc_addr(device.handle(), name.as_ptr()),
            )
        });

        let allocator = VulkanAllocator::new(&VulkanAllocatorCreateDesc {
            instance: instance.instance.clone(),
            device: device.clone(),
            physical_device: instance.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
        });

        let pools = (0..frame_buffering)
            .map(|_| {
                let cmd_pool = {
                    let desc = vk::CommandPoolCreateInfo::builder()
                        .queue_family_index(instance.family_index);
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

        let timeline = {
            let mut timeline_desc = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let desc = vk::SemaphoreCreateInfo::builder().push_next(&mut timeline_desc);
            device.create_semaphore(&desc, None)?
        };

        Ok(Self {
            device,
            queue,
            allocator,
            pools,
            timeline,
            ext: Extensions {
                sync2: ext_sync2,
                image_handle: ext_image_handke,
                debug_utils: ext_debug_utils,
                accel_structure: ext_accel_structure,
                deferred: ext_deferred,
                raytracing_pipeline: ext_raytracing_pipeline,
            },
            frame_id: 0,
            rust_spv: Path::new(env!("spv")).to_path_buf(),
        })
    }

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

    pub unsafe fn buffer_address(&self, buffer: gpu::Buffer) -> vk::DeviceAddress {
        let desc = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
        self.get_buffer_device_address(&desc)
    }

    pub unsafe fn sampled_image_address(
        &self,
        image: gpu::ImageView,
        sampler: gpu::Sampler,
    ) -> vk::DeviceAddress {
        let handle = vk::ImageViewHandleInfoNVX::builder()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_view(image)
            .sampler(sampler)
            .build();
        self.ext
            .image_handle
            .get_image_view_handle_nvx(self.handle(), &handle) as _
    }

    pub unsafe fn storage_image_address(&self, image: gpu::ImageView) -> vk::DeviceAddress {
        let handle = vk::ImageViewHandleInfoNVX::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_view(image)
            .build();
        self.ext
            .image_handle
            .get_image_view_handle_nvx(self.handle(), &handle) as _
    }

    pub fn retire_buffer(&mut self, pool: gpu::Pool, buffer: gpu::Buffer) {
        self.pools[pool.id].shrine.buffers.push(buffer);
    }

    pub fn retire_image(&mut self, pool: gpu::Pool, image: gpu::Image) {
        self.pools[pool.id].shrine.images.push(image);
    }

    pub unsafe fn create_blas(
        &mut self,
        pool: gpu::Pool,
        name: &str,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        geometry: &[gpu::GeometryBlas],
    ) -> anyhow::Result<gpu::AccelerationStructure> {
        let as_geometry = geometry
            .iter()
            .map(|geometry| match geometry {
                gpu::GeometryBlas::Triangles {
                    flags,
                    format,
                    vertex_buffer,
                    vertex_stride,
                    num_vertices,
                    index_type,
                    index_buffer,
                    ..
                } => vk::AccelerationStructureGeometryKHR::builder()
                    .flags(*flags)
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                            .vertex_format(*format)
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: *vertex_buffer,
                            })
                            .vertex_stride(*vertex_stride as _)
                            .max_vertex(*num_vertices as _)
                            .index_type(*index_type)
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: *index_buffer,
                            })
                            .build(),
                    })
                    .build(),
            })
            .collect::<Vec<_>>();

        let primitives = geometry
            .iter()
            .map(|geometry| match geometry {
                gpu::GeometryBlas::Triangles { num_indices, .. } => (*num_indices / 3) as u32,
            })
            .collect::<Vec<_>>();

        let build_geometry = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geometry);
        let blas_size = self
            .ext
            .accel_structure
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geometry,
                &primitives,
            );

        let blas_buffer = self.create_buffer_gpu(
            name,
            blas_size.acceleration_structure_size as _,
            gpu::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            gpu::BufferInit::None,
        )?;

        let blas_scratch_buffer = self.create_buffer_gpu(
            &format!("{} (scratch)", name),
            blas_size.build_scratch_size as _,
            gpu::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            gpu::BufferInit::None,
        )?;

        let blas = {
            let desc = vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(blas_buffer)
                .offset(0)
                .size(blas_size.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
            self.ext
                .accel_structure
                .create_acceleration_structure(&desc, None)?
        };

        let build_geometry = [vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(flags)
            .dst_acceleration_structure(blas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.buffer_address(blas_scratch_buffer),
            })
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geometry)
            .build()];

        let build_ranges = geometry
            .iter()
            .map(|geometry| match geometry {
                gpu::GeometryBlas::Triangles { num_indices, .. } => {
                    vk::AccelerationStructureBuildRangeInfoKHR {
                        primitive_count: (*num_indices / 3) as u32,
                        primitive_offset: 0,
                        first_vertex: 0,
                        transform_offset: 0,
                    }
                }
            })
            .collect::<Vec<_>>();

        self.ext.accel_structure.cmd_build_acceleration_structures(
            pool.cmd_buffer,
            &build_geometry,
            &[&build_ranges],
        );

        self.retire_buffer(pool, blas_scratch_buffer);

        Ok(blas)
    }

    pub unsafe fn create_tlas(
        &mut self,
        pool: gpu::Pool,
        name: &str,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        geometry: gpu::GeometryTlas,
    ) -> anyhow::Result<gpu::AccelerationStructure> {
        let as_geometry = [vk::AccelerationStructureGeometryKHR::builder()
            .flags(geometry.flags)
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: geometry.instance_buffer.handle(self),
                    })
                    .build(),
            })
            .build()];
        let primitives = [geometry.num_instances as _];
        let build_geometry = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geometry);
        let tlas_size = self
            .ext
            .accel_structure
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geometry,
                &primitives,
            );

        let tlas_buffer = self.create_buffer_gpu(
            name,
            tlas_size.acceleration_structure_size as _,
            gpu::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            gpu::BufferInit::None,
        )?;

        let tlas_scratch_buffer = self.create_buffer_gpu(
            &format!("{} (scratch)", name),
            tlas_size.build_scratch_size as _,
            gpu::BufferUsageFlags::empty(),
            gpu::BufferInit::None,
        )?;

        let tlas = {
            let desc = vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(tlas_buffer)
                .offset(0)
                .size(tlas_size.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
            self.ext
                .accel_structure
                .create_acceleration_structure(&desc, None)?
        };

        let build_geometry = [vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(flags)
            .dst_acceleration_structure(tlas)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.buffer_address(tlas_scratch_buffer),
            })
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geometry)
            .build()];
        let build_range = [vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: geometry.num_instances as _,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        }];

        self.ext.accel_structure.cmd_build_acceleration_structures(
            pool.cmd_buffer,
            &build_geometry,
            &[&build_range],
        );

        self.retire_buffer(pool, tlas_scratch_buffer);

        Ok(tlas)
    }

    pub unsafe fn create_renderpass(
        &mut self,
        extent: vk::Extent3D,
        color_attachments: &[gpu::Attachment],
        depth_attachment: Option<gpu::Attachment>,
    ) -> anyhow::Result<gpu::RenderPass> {
        let render_pass = {
            let mut subpass = vk::SubpassDescription::builder();

            let mut attachments = Vec::new();

            let mut color_attachment_refs = Vec::new();
            for (i, attachment) in color_attachments.iter().enumerate() {
                attachments.push(vk::AttachmentDescription {
                    format: attachment.format,
                    samples: vk::SampleCountFlags::from_raw(attachment.samples as _),
                    load_op: attachment.src.1,
                    store_op: attachment.dst.1,
                    final_layout: attachment.dst.0,
                    ..Default::default()
                });
                color_attachment_refs.push(vk::AttachmentReference {
                    attachment: i as _,
                    layout: attachment.src.0,
                });
            }
            subpass = subpass.color_attachments(&color_attachment_refs);

            let depth_attachment_ref = depth_attachment.map(|attachment| {
                attachments.push(vk::AttachmentDescription {
                    format: attachment.format,
                    samples: vk::SampleCountFlags::from_raw(attachment.samples as _),
                    load_op: attachment.src.1,
                    store_op: attachment.dst.1,
                    final_layout: attachment.dst.0,
                    ..Default::default()
                });
                vk::AttachmentReference {
                    attachment: attachments.len() as u32 - 1,
                    layout: attachment.src.0,
                }
            });
            if let Some(attachment) = depth_attachment_ref.as_ref() {
                subpass = subpass.depth_stencil_attachment(attachment);
            }

            let subpasses = [subpass
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build()];
            let desc = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);
            self.create_render_pass(&desc, None)?
        };

        let framebuffer = {
            let formats = {
                let mut formats = Vec::new();
                for attachment in color_attachments {
                    formats.push(attachment.format);
                }
                for attachment in depth_attachment {
                    formats.push(attachment.format);
                }
                formats
            };

            let images = {
                let mut images = Vec::new();
                for i in 0..color_attachments.len() {
                    images.push(
                        vk::FramebufferAttachmentImageInfo::builder()
                            .view_formats(&formats[i..i + 1])
                            .width(extent.width)
                            .height(extent.height)
                            .usage(
                                vk::ImageUsageFlags::COLOR_ATTACHMENT | color_attachments[i].usage,
                            )
                            .layer_count(extent.depth)
                            .build(),
                    );
                }
                if let Some(attachment) = depth_attachment {
                    let i = color_attachments.len();
                    images.push(
                        vk::FramebufferAttachmentImageInfo::builder()
                            .view_formats(&formats[i..i + 1])
                            .width(extent.width)
                            .height(extent.height)
                            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | attachment.usage)
                            .layer_count(extent.depth)
                            .build(),
                    );
                }
                images
            };
            let mut attachments =
                vk::FramebufferAttachmentsCreateInfo::builder().attachment_image_infos(&images);
            let mut desc = vk::FramebufferCreateInfo::builder()
                .flags(vk::FramebufferCreateFlags::IMAGELESS)
                .render_pass(render_pass)
                .width(extent.width)
                .height(extent.height)
                .layers(extent.depth)
                .push_next(&mut attachments);
            desc.attachment_count = images.len() as _;

            self.create_framebuffer(&desc, None)?
        };

        Ok(gpu::RenderPass {
            render_pass,
            framebuffer,
        })
    }

    pub unsafe fn create_buffer_gpu(
        &mut self,
        name: &str,
        size: usize,
        mut usage: gpu::BufferUsageFlags,
        initialization: gpu::BufferInit,
    ) -> anyhow::Result<gpu::Buffer> {
        usage |= gpu::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let buffer = {
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
            buffer
        };

        if let Some(ext) = &self.ext.debug_utils {
            let name = CString::new(name)?;
            let info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(vk::ObjectType::BUFFER)
                .object_handle(buffer.as_raw())
                .object_name(&name);
            ext.debug_utils_set_object_name(self.device.handle(), &info)?;
        }

        match initialization {
            gpu::BufferInit::Host { pool, data } => {
                let buffer_init = {
                    let desc = vk::BufferCreateInfo::builder()
                        .size(data.len() as _)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
                    let buffer = self.create_buffer(&desc, None)?;
                    let alloc_desc = gpu_allocator::AllocationCreateDesc {
                        name: &format!("{} (init)", name),
                        requirements: self.get_buffer_memory_requirements(buffer),
                        location: gpu_allocator::MemoryLocation::CpuToGpu,
                        linear: true,
                    };
                    let mut allocation = self.allocator.allocate(&alloc_desc)?;
                    {
                        let mapping = allocation.mapped_slice_mut().unwrap();
                        mapping[..data.len()].copy_from_slice(data);
                    }
                    self.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
                    buffer
                };

                self.cmd_copy_buffer(
                    pool.cmd_buffer,
                    buffer_init,
                    buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: data.len() as _,
                    }],
                );

                self.retire_buffer(pool, buffer_init);
            }
            gpu::BufferInit::None => (),
        }

        Ok(buffer)
    }

    pub unsafe fn create_image_gpu(
        &mut self,
        name: &str,
        desc: &gpu::ImageDesc,
        initialization: gpu::ImageInit,
    ) -> anyhow::Result<gpu::Image> {
        let image = {
            let desc = vk::ImageCreateInfo::builder()
                .image_type(desc.ty)
                .format(desc.format)
                .extent(desc.extent)
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT) // todo: hmm - image view create info list
                .mip_levels(desc.mip_levels as _)
                .array_layers(desc.array_layers as _)
                .samples(vk::SampleCountFlags::from_raw(desc.samples as _))
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(desc.usage);

            let image = self.create_image(&desc, None)?;
            let alloc_desc = gpu_allocator::AllocationCreateDesc {
                name,
                requirements: self.get_image_memory_requirements(image),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            };
            let allocation = self.allocator.allocate(&alloc_desc)?;
            self.bind_image_memory(image, allocation.memory(), allocation.offset())?;

            image
        };

        match initialization {
            gpu::ImageInit::Host { pool, aspect, data } => {
                let buffer_init = {
                    let desc = vk::BufferCreateInfo::builder()
                        .size(data.len() as _)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
                    let buffer = self.create_buffer(&desc, None)?;
                    let alloc_desc = gpu_allocator::AllocationCreateDesc {
                        name: &format!("{} (init)", name),
                        requirements: self.get_buffer_memory_requirements(buffer),
                        location: gpu_allocator::MemoryLocation::CpuToGpu,
                        linear: true,
                    };
                    let mut allocation = self.allocator.allocate(&alloc_desc)?;
                    {
                        let mapping = allocation.mapped_slice_mut().unwrap();
                        mapping[..data.len()].copy_from_slice(data);
                    }
                    self.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
                    buffer
                };

                let range = vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                self.cmd_barriers(
                    pool.cmd_buffer,
                    &[],
                    &[gpu::ImageBarrier {
                        image,
                        range,
                        src: gpu::ImageAccess {
                            access: gpu::Access::NONE,
                            stage: gpu::Stage::NONE,
                            layout: gpu::ImageLayout::UNDEFINED,
                        },
                        dst: gpu::ImageAccess {
                            access: gpu::Access::MEMORY_WRITE,
                            stage: gpu::Stage::COPY,
                            layout: gpu::ImageLayout::TRANSFER_DST_OPTIMAL,
                        },
                    }],
                );

                let copy = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: range.aspect_mask,
                        mip_level: range.base_mip_level,
                        base_array_layer: range.base_array_layer,
                        layer_count: range.layer_count,
                    },
                    image_extent: desc.extent,
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                };

                self.cmd_copy_buffer_to_image(
                    pool.cmd_buffer,
                    buffer_init,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                );

                self.retire_buffer(pool, buffer_init);
            }
            gpu::ImageInit::None => (),
        }

        Ok(image)
    }

    pub unsafe fn create_rust_shader(&mut self, name: &str) -> anyhow::Result<gpu::Shader> {
        let mut file = std::io::Cursor::new(std::fs::read(self.rust_spv.join(name))?);
        let code = ash::util::read_spv(&mut file)?;
        let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
        let shader = self.device.create_shader_module(&desc, None)?;
        Ok(shader)
    }

    pub unsafe fn create_layout(&mut self, num_constants: u32) -> anyhow::Result<gpu::Layout> {
        let mut push_constants = vec![];
        if num_constants > 0 {
            push_constants.push(
                vk::PushConstantRange::builder()
                    .offset(0)
                    .size(num_constants)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .build(),
            );
        }
        let desc = vk::PipelineLayoutCreateInfo::builder().push_constant_ranges(&push_constants);

        let pipeline_layout = self.create_pipeline_layout(&desc, None)?;

        Ok(gpu::Layout { pipeline_layout })
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
            self.device
                .reset_command_pool(pool.cmd_pool, vk::CommandPoolResetFlags::empty())?;
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
            // TODO: doesnt work very well with nsight ..

            // Some(ref sync2) => {
            //     let waits = submit
            //         .waits
            //         .iter()
            //         .map(|desc| {
            //             vk::SemaphoreSubmitInfoKHR::builder()
            //                 .semaphore(desc.semaphore)
            //                 .stage_mask(desc.stage)
            //                 .build()
            //         })
            //         .collect::<Box<[_]>>();

            //     let mut signals = submit
            //         .signals
            //         .iter()
            //         .map(|desc| {
            //             vk::SemaphoreSubmitInfoKHR::builder()
            //                 .semaphore(desc.semaphore)
            //                 .stage_mask(desc.stage)
            //                 .build()
            //         })
            //         .collect::<Vec<_>>();
            //     signals.push(
            //         vk::SemaphoreSubmitInfoKHR::builder()
            //             .semaphore(self.timeline)
            //             .value(pool.id as u64 + 1)
            //             .stage_mask(gpu::Stage::NONE)
            //             .build(),
            //     );

            //     let cmd_buffers = [vk::CommandBufferSubmitInfoKHR::builder()
            //         .command_buffer(pool.cmd_buffer)
            //         .build()];

            //     let desc = [vk::SubmitInfo2KHR::builder()
            //         .wait_semaphore_infos(&waits)
            //         .signal_semaphore_infos(&signals)
            //         .command_buffer_infos(&cmd_buffers)
            //         .build()];
            //     sync2.queue_submit2(self.queue, &desc, vk::Fence::null())
            // }
            _ => {
                let waits = submit
                    .waits
                    .iter()
                    .map(|desc| desc.semaphore)
                    .collect::<Box<[_]>>();
                let wait_stages = submit
                    .waits
                    .iter()
                    .map(|desc| map_stage(SyncScope::Second, desc.stage))
                    .collect::<Box<[_]>>();
                let mut signals = submit
                    .signals
                    .iter()
                    .map(|desc| desc.semaphore)
                    .collect::<Vec<_>>();
                let cmd_buffer = [pool.cmd_buffer];

                let wait_values = vec![0; waits.len()];
                let mut signal_values = vec![0; signals.len()];

                signals.push(self.timeline);
                signal_values.push(pool.id as u64 + 1);

                let mut timeline_desc = vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(&wait_values)
                    .signal_semaphore_values(&signal_values);

                let desc = [vk::SubmitInfo::builder()
                    .wait_semaphores(&waits)
                    .command_buffers(&cmd_buffer)
                    .wait_dst_stage_mask(&wait_stages)
                    .signal_semaphores(&signals)
                    .push_next(&mut timeline_desc)
                    .build()];
                self.queue_submit(self.queue, &desc, vk::Fence::null())
            }
        }
    }

    pub unsafe fn cmd_begin_renderpass(
        &mut self,
        cmd_buffer: vk::CommandBuffer,
        renderpass: gpu::RenderPass,
        attachments: &[(gpu::ImageView, vk::ClearValue)],
        area: vk::Rect2D,
    ) {
        let clear_values = attachments
            .iter()
            .map(|(_, clear)| *clear)
            .collect::<Vec<_>>();
        let attachments = attachments
            .iter()
            .map(|(view, _)| *view)
            .collect::<Vec<_>>();
        let mut render_pass_attachments =
            vk::RenderPassAttachmentBeginInfo::builder().attachments(&attachments);
        let begin_desc = vk::RenderPassBeginInfo::builder()
            .render_pass(renderpass.render_pass)
            .framebuffer(renderpass.framebuffer)
            .render_area(area)
            .clear_values(&clear_values)
            .push_next(&mut render_pass_attachments);

        self.cmd_begin_render_pass(cmd_buffer, &begin_desc, vk::SubpassContents::INLINE);
        self.cmd_set_scissor(cmd_buffer, 0, &[area]);
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

                    image_barriers.push(
                        vk::ImageMemoryBarrier::builder()
                            .src_access_mask(map_access(barrier.src.access))
                            .dst_access_mask(map_access(barrier.dst.access))
                            .old_layout(barrier.src.layout)
                            .new_layout(barrier.dst.layout)
                            .image(barrier.image)
                            .subresource_range(barrier.range)
                            .build(),
                    );
                }

                let mut memory_barrier = vec![];
                if !(memory_src_access.is_empty() && memory_dst_access.is_empty()) {
                    memory_barrier.push(
                        vk::MemoryBarrier::builder()
                            .src_access_mask(memory_src_access)
                            .dst_access_mask(memory_dst_access)
                            .build(),
                    );
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

    if stage
        .intersects(gpu::Stage::COPY | gpu::Stage::RESOLVE | gpu::Stage::BLIT | gpu::Stage::CLEAR)
    {
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
