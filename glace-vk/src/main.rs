use ash::vk;
use camera::{Camera, InputMap};
use glace::{f32x4x4, vec3};
use gpu_allocator::AllocationCreateDesc;
use std::{
    mem,
    path::{Path, PathBuf},
};
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod camera;
mod gpu;
mod pass;
// mod ktx;

#[repr(C)]
#[derive(Debug)]
struct WorldPbr {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

#[repr(C)]
#[derive(Debug)]
struct GeometryData {
    v_position_obj: gpu::CpuDescriptor,
    v_normal_obj: gpu::CpuDescriptor,
    v_texcoord: gpu::CpuDescriptor,
    v_tangent_obj: gpu::CpuDescriptor,
}

#[repr(C)]
#[derive(Debug)]
struct InstanceData {
    sampler: u32,
    albedo_map: gpu::CpuDescriptor,
    normal_map: gpu::CpuDescriptor,
}

unsafe fn load_png(
    gpu: &mut gpu::Gpu,
    pool: gpu::Pool,
    path: PathBuf,
    format: vk::Format,
) -> anyhow::Result<(vk::Image, vk::ImageView, vk::Buffer)> {
    let img = image::open(&Path::new(&path)).unwrap().to_rgba8();
    let img_width = img.width();
    let img_height = img.height();
    let img_data = img.into_raw();

    let desc = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: img_width as _,
            height: img_height as _,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST);

    let name = path.to_str().unwrap();
    let image = gpu.create_image(&desc, None)?;
    let alloc_desc = AllocationCreateDesc {
        name,
        requirements: gpu.get_image_memory_requirements(image),
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: false,
    };
    let allocation = gpu.allocator.allocate(&alloc_desc)?;
    gpu.bind_image_memory(image, allocation.memory(), allocation.offset())?;

    let desc = vk::BufferCreateInfo::builder()
        .size(img_data.len() as _)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    let buffer = gpu.create_buffer(&desc, None)?;
    let alloc_desc = AllocationCreateDesc {
        name,
        requirements: gpu.get_buffer_memory_requirements(buffer),
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
    };
    let mut allocation = gpu.allocator.allocate(&alloc_desc)?;
    {
        let mapping = allocation.mapped_slice_mut().unwrap();
        mapping[..img_data.len()].copy_from_slice(gpu::as_u8_slice(&img_data));
    }
    gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

    gpu.cmd_barriers(
        pool.cmd_buffer,
        &[],
        &[gpu::ImageBarrier {
            image,
            range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
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
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_extent: vk::Extent3D {
            width: img_width as _,
            height: img_height as _,
            depth: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
    };

    gpu.cmd_copy_buffer_to_image(
        pool.cmd_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[copy],
    );

    gpu.cmd_barriers(
        pool.cmd_buffer,
        &[],
        &[gpu::ImageBarrier {
            image,
            range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src: gpu::ImageAccess {
                access: gpu::Access::MEMORY_WRITE,
                stage: gpu::Stage::COPY,
                layout: gpu::ImageLayout::TRANSFER_DST_OPTIMAL,
            },
            dst: gpu::ImageAccess {
                access: gpu::Access::MEMORY_READ,
                stage: gpu::Stage::ALL_COMMANDS,
                layout: gpu::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        }],
    );

    let view = {
        let desc = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        gpu.create_image_view(&desc, None)?
    };

    Ok((image, view, buffer))
}

fn main() -> anyhow::Result<()> {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(LogicalSize::new(1440.0, 900.0))
        .build(&event_loop)?;

    let size = window.inner_size();
    let frames_in_flight: usize = 2;

    unsafe {
        let instance = gpu::Instance::new(&window)?;
        let mut gpu = gpu::Gpu::new(
            &instance,
            frames_in_flight,
            gpu::DescriptorsDesc {
                buffers: 1024,
                images: 1024,
                acceleration_structures: 512,
            },
        )?;
        let mut wsi = gpu::Swapchain::new(&instance, &gpu, size.width, size.height)?;

        let directory = Path::new("assets");

        let num_indices = 70_074;
        let num_vertex = 70_074;
        let bin = std::fs::read(directory.join("SciFiHelmet.bin"))?;

        let mesh_cpu = {
            let desc = vk::BufferCreateInfo::builder()
                .size(bin.len() as _)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let buffer = gpu.create_buffer(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Triangle Buffer (CPU)",
                requirements: gpu.get_buffer_memory_requirements(buffer),
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
            };
            let mut allocation = gpu.allocator.allocate(&alloc_desc)?;
            {
                let mapping = allocation.mapped_slice_mut().unwrap();
                mapping[..bin.len()].copy_from_slice(gpu::as_u8_slice(&bin));
            }
            gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let mesh_gpu = gpu.create_buffer_gpu(
            "mesh gpu",
            bin.len(),
            gpu::BufferUsageFlags::STORAGE_BUFFER
                | gpu::BufferUsageFlags::INDEX_BUFFER
                | gpu::BufferUsageFlags::TRANSFER_DST
                | gpu::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;

        let world_pbr_gpu = gpu.create_buffer_gpu(
            "pbr::world gpu",
            mem::size_of::<WorldPbr>(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
        )?;

        let geometry_data_gpu = gpu.create_buffer_gpu(
            "pbr::geometry gpu",
            mem::size_of::<GeometryData>(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
        )?;

        let instance_data_gpu = gpu.create_buffer_gpu(
            "pbr::instance gpu",
            mem::size_of::<InstanceData>(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
        )?;

        let upload_pool = gpu.acquire_pool().unwrap();

        gpu.cmd_copy_buffer(
            upload_pool.cmd_buffer,
            mesh_cpu,
            mesh_gpu,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: bin.len() as _,
            }],
        );

        let (_albedo, albedo_view, _albedo_buffer) = load_png(
            &mut gpu,
            upload_pool,
            directory.join("SciFiHelmet_BaseColor.png"),
            vk::Format::R8G8B8A8_SRGB,
        )?;
        let (_normal, normal_view, _normal_buffer) = load_png(
            &mut gpu,
            upload_pool,
            directory.join("SciFiHelmet_Normal.png"),
            vk::Format::R8G8B8A8_UNORM,
        )?;

        // acceleration structure
        #[cfg(feature = "raytrace")]
        {
            let geometry = [vk::AccelerationStructureGeometryKHR::builder()
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: gpu.buffer_address(mesh_gpu) + num_vertex * 4,
                        })
                        .vertex_stride(12)
                        .max_vertex(num_vertex as _)
                        .index_type(vk::IndexType::UINT32)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: gpu.buffer_address(mesh_gpu),
                        })
                        .build(),
                })
                .build()];
            let primitives = [num_indices / 3];
            let build_geometry = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry);
            let blas_size = gpu
                .ext
                .accel_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geometry,
                    &primitives,
                );

            let blas_buffer = gpu.create_buffer_gpu(
                "blas gpu",
                blas_size.acceleration_structure_size as _,
                gpu::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            )?;

            let blas_scratch_buffer = gpu.create_buffer_gpu(
                "blas (scratch) gpu",
                blas_size.build_scratch_size as _,
                gpu::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?;

            let blas = {
                let desc = vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(blas_buffer)
                    .offset(0)
                    .size(blas_size.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);
                gpu.ext
                    .accel_structure
                    .create_acceleration_structure(&desc, None)?
            };

            let build_geometry = [vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .dst_acceleration_structure(blas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: gpu.buffer_address(blas_scratch_buffer),
                })
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry)
                .build()];
            let build_range = [vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: num_indices / 3,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            }];

            gpu.ext.accel_structure.cmd_build_acceleration_structures(
                upload_pool.cmd_buffer,
                &build_geometry,
                &[&build_range],
            );

            // tlas
            let tlas_instances = [vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                },
                instance_custom_index_and_mask: 0xFF,
                instance_shader_binding_table_record_offset_and_flags: 0,
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: gpu.acceleration_structure_address(blas),
                },
            }];
            let tlas_instances_buffer = {
                let size = std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()
                    * tlas_instances.len();
                let desc = vk::BufferCreateInfo::builder()
                    .size(size as _)
                    .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
                let buffer = gpu.create_buffer(&desc, None)?;
                let alloc_desc = AllocationCreateDesc {
                    name: "tlas instances",
                    requirements: gpu.get_buffer_memory_requirements(buffer),
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true,
                };
                let mut allocation = gpu.allocator.allocate(&alloc_desc)?;
                {
                    let mapping = allocation.mapped_slice_mut().unwrap();
                    mapping[..size].copy_from_slice(gpu::as_u8_slice(&tlas_instances));
                }
                gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
                buffer
            };

            let geometry = [vk::AccelerationStructureGeometryKHR::builder()
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                        .array_of_pointers(false)
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: gpu.buffer_address(tlas_instances_buffer),
                        })
                        .build(),
                })
                .build()];
            let primitives = [1];
            let build_geometry = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry);
            let tlas_size = gpu
                .ext
                .accel_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geometry,
                    &primitives,
                );

            let tlas_buffer = gpu.create_buffer_gpu(
                "tlas gpu",
                tlas_size.acceleration_structure_size as _,
                gpu::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            )?;

            let tlas_scratch_buffer = gpu.create_buffer_gpu(
                "tlas (scratch) gpu",
                tlas_size.build_scratch_size as _,
                gpu::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            )?;

            let tlas = {
                let desc = vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(tlas_buffer)
                    .offset(0)
                    .size(tlas_size.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);
                gpu.ext
                    .accel_structure
                    .create_acceleration_structure(&desc, None)?
            };

            let build_geometry = [vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .dst_acceleration_structure(tlas)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: gpu.buffer_address(tlas_scratch_buffer),
                })
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry)
                .build()];
            let build_range = [vk::AccelerationStructureBuildRangeInfoKHR {
                primitive_count: 1,
                primitive_offset: 0,
                first_vertex: 0,
                transform_offset: 0,
            }];

            gpu.ext.accel_structure.cmd_build_acceleration_structures(
                upload_pool.cmd_buffer,
                &build_geometry,
                &[&build_range],
            );

            let tlas_handle = gpu.descriptors_accel.create();

            let accel_info = [gpu::AccelerationStructureDescriptor {
                handle: tlas_handle,
                acceleration_structure: tlas,
            }];
            gpu.update_descriptors(&[], &[], &accel_info);
        }

        gpu.submit_pool(upload_pool, gpu::Submit {
            waits: &[],
            signals: &[],
        }).unwrap();

        let depth_stencil_format = vk::Format::D32_SFLOAT;
        let depth_stencil_image = {
            let desc = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(depth_stencil_format)
                .extent(vk::Extent3D {
                    width: size.width as _,
                    height: size.height as _,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);

            let image = gpu.create_image(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Depth Buffer",
                requirements: gpu.get_image_memory_requirements(image),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            };
            let allocation = gpu.allocator.allocate(&alloc_desc)?;
            gpu.bind_image_memory(image, allocation.memory(), allocation.offset())?;
            image
        };

        let depth_stencil_view = {
            let view_desc = vk::ImageViewCreateInfo::builder()
                .image(depth_stencil_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(depth_stencil_format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            gpu.create_image_view(&view_desc, None)?
        };

        let mesh_pass = {
            let attachments = [
                vk::AttachmentDescription {
                    format: wsi.surface_format.format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    ..Default::default()
                },
                vk::AttachmentDescription {
                    format: depth_stencil_format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                },
            ];
            let color_attachments = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth_stencil_attachment = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };
            let subpasses = [vk::SubpassDescription::builder()
                .color_attachments(&color_attachments)
                .depth_stencil_attachment(&depth_stencil_attachment)
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build()];
            let desc = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);
            gpu.create_render_pass(&desc, None)?
        };

        let mesh_fbos = wsi
            .frame_rtvs
            .iter()
            .map(|rtv| {
                let attachments = [*rtv, depth_stencil_view];
                let desc = vk::FramebufferCreateInfo::builder()
                    .render_pass(mesh_pass)
                    .width(size.width as _)
                    .height(size.height as _)
                    .layers(1)
                    .attachments(&attachments);
                gpu.create_framebuffer(&desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mesh_sampler = {
            let desc = vk::SamplerCreateInfo::builder(); // TODO: linear
            gpu.create_sampler(&desc, None)?
        };

        let mesh_layout = gpu.create_layout(&[mesh_sampler], 12)?;

        let world_pbr_handle = gpu.descriptors_buffer.create();
        let geometry_data_handle = gpu.descriptors_buffer.create();
        let instance_data_handle = gpu.descriptors_buffer.create();
        let mesh_position_handle = gpu.descriptors_buffer.create();
        let mesh_normal_handle = gpu.descriptors_buffer.create();
        let mesh_texcoord_handle = gpu.descriptors_buffer.create();
        let mesh_tangent_handle = gpu.descriptors_buffer.create();

        let albedo_srv_handle = gpu.descriptors_image.create();
        let normal_srv_handle = gpu.descriptors_image.create();

        {
            let base_offset = num_indices as u64 * 4;
            let buffer_info = [
                gpu::BufferDescriptor {
                    handle: world_pbr_handle,
                    buffer: world_pbr_gpu,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                gpu::BufferDescriptor {
                    handle: geometry_data_handle,
                    buffer: geometry_data_gpu,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                gpu::BufferDescriptor {
                    handle: instance_data_handle,
                    buffer: instance_data_gpu,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                },
                gpu::BufferDescriptor {
                    handle: mesh_position_handle,
                    buffer: mesh_gpu,
                    offset: base_offset,
                    range: num_vertex * 12, // R32G32B32_SFLOAT
                },
                gpu::BufferDescriptor {
                    handle: mesh_normal_handle,
                    buffer: mesh_gpu,
                    offset: base_offset + num_vertex * 12,
                    range: num_vertex * 12, // R32G32B32_SFLOAT
                },
                gpu::BufferDescriptor {
                    handle: mesh_texcoord_handle,
                    buffer: mesh_gpu,
                    offset: base_offset + num_vertex * 40,
                    range: num_vertex * 8, // R32G32_SFLOAT
                },
                gpu::BufferDescriptor {
                    handle: mesh_tangent_handle,
                    buffer: mesh_gpu,
                    offset: base_offset + num_vertex * 24,
                    range: num_vertex * 16, // R32G32B32A32_SFLOAT
                },
            ];
            let image_info = [
                gpu::ImageDescriptor {
                    handle: albedo_srv_handle,
                    view: albedo_view,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                gpu::ImageDescriptor {
                    handle: normal_srv_handle,
                    view: normal_view,
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ];
            gpu.update_descriptors(&buffer_info, &image_info, &[]);
        }

        let spirv_dir = Path::new(env!("spv"));

        let mesh_vs = {
            let mut file = std::io::Cursor::new(std::fs::read(spirv_dir.join("mesh_vs"))?);
            // let mut file = File::open("out.spv" /*directory.join("triangle.vert.spv")*/)?;
            let code = ash::util::read_spv(&mut file)?;
            let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
            gpu.create_shader_module(&desc, None)?
        };

        let mesh_fs = {
            let mut file = std::io::Cursor::new(std::fs::read(spirv_dir.join("mesh_fs"))?);
            // let mut file = File::open(directory.join("triangle.frag.spv"))?;
            let code = ash::util::read_spv(&mut file)?;
            let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
            gpu.create_shader_module(&desc, None)?
        };

        let mesh_pipeline = {
            let entry_vs = std::ffi::CStr::from_bytes_with_nul(b"mesh_vs\0")?;
            let entry_fs = std::ffi::CStr::from_bytes_with_nul(b"mesh_fs\0")?;
            let stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(mesh_vs)
                    .name(&entry_vs)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(mesh_fs)
                    .name(&entry_fs)
                    .build(),
            ];

            let ia_desc = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let rasterizer_desc = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            let viewport_desc = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);
            let multisample_desc = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let color_blend_attachment = [vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::FALSE,
                src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ZERO,
                dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::all(),
            }];
            let color_blend_desc = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachment);
            let depth_stencil_desc = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_desc =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();
            let desc = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&stages)
                .input_assembly_state(&ia_desc)
                .vertex_input_state(&vertex_input)
                .rasterization_state(&rasterizer_desc)
                .viewport_state(&viewport_desc)
                .multisample_state(&multisample_desc)
                .color_blend_state(&color_blend_desc)
                .depth_stencil_state(&depth_stencil_desc)
                .dynamic_state(&dynamic_desc)
                .render_pass(mesh_pass)
                .subpass(0)
                .layout(mesh_layout.pipeline_layout)
                .build(); // TODO

            gpu.create_graphics_pipelines(vk::PipelineCache::null(), &[desc], None)
                .unwrap()
        };

        let mut camera = Camera::new(vec3(0.0, 0.0, 0.0), 0.0, 0.0);
        let mut input = InputMap::new();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                Event::LoopDestroyed => {
                    wsi.swapchain_fn.destroy_swapchain(wsi.swapchain, None);
                    instance.surface_fn.destroy_surface(instance.surface, None);
                }
                Event::DeviceEvent { event, .. } => match event {
                    DeviceEvent::MouseMotion { delta } => {
                        input.update_mouse_motion((delta.0 as _, delta.1 as _));
                    }
                    DeviceEvent::Button { state, .. } => {
                        input.update_mouse1(state);
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    camera.update(&input);
                    input.reset_delta();

                    let frame = wsi.acquire().unwrap();
                    let pool = gpu.acquire_pool().unwrap();

                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    let eye = camera.position() + camera.view_dir() * 4.5;
                    let center = camera.position();

                    let world = [WorldPbr {
                        world_to_view: f32x4x4::look_at_inv(eye, eye - center),
                        view_to_clip: f32x4x4::perspective(
                            std::f32::consts::PI * 0.25,
                            aspect,
                            0.1,
                            10000.0,
                        ),
                    }];
                    gpu.cmd_update_buffer(
                        pool.cmd_buffer,
                        world_pbr_gpu,
                        0,
                        gpu::as_u8_slice(&world),
                    );
                    let instance = [InstanceData {
                        sampler: 0,
                        albedo_map: albedo_srv_handle,
                        normal_map: normal_srv_handle,
                    }];
                    gpu.cmd_update_buffer(
                        pool.cmd_buffer,
                        instance_data_gpu,
                        0,
                        gpu::as_u8_slice(&instance),
                    );

                    let draw_params = [GeometryData {
                        v_position_obj: mesh_position_handle,
                        v_normal_obj: mesh_normal_handle,
                        v_texcoord: mesh_texcoord_handle,
                        v_tangent_obj: mesh_tangent_handle,
                    }];
                    gpu.cmd_update_buffer(
                        pool.cmd_buffer,
                        geometry_data_gpu,
                        0,
                        gpu::as_u8_slice(&draw_params),
                    );

                    gpu.cmd_barriers(
                        pool.cmd_buffer,
                        &[gpu::MemoryBarrier::full()],
                        &[],
                    );

                    let clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.5, 0.5, 0.5, 1.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];

                    let mesh_pass_begin_desc = vk::RenderPassBeginInfo::builder()
                        .render_pass(mesh_pass)
                        .framebuffer(mesh_fbos[frame.id])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: size.width as _,
                                height: size.height as _,
                            },
                        })
                        .clear_values(&clear_values);

                    gpu.cmd_begin_render_pass(
                        pool.cmd_buffer,
                        &mesh_pass_begin_desc,
                        vk::SubpassContents::INLINE,
                    );
                    gpu.cmd_bind_index_buffer(pool.cmd_buffer, mesh_gpu, 0, vk::IndexType::UINT32);
                    gpu.cmd_bind_pipeline(
                        pool.cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        mesh_pipeline[0],
                    );
                    gpu.cmd_set_scissor(
                        pool.cmd_buffer,
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: size.width as _,
                                height: size.height as _,
                            },
                        }],
                    );
                    gpu.cmd_set_viewport(
                        pool.cmd_buffer,
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: size.height as _,
                            width: size.width as _,
                            height: -(size.height as f32),
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );
                    gpu.cmd_bind_descriptors(
                        pool.cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        mesh_layout,
                    );

                    let constants = [world_pbr_handle, geometry_data_handle, instance_data_handle];
                    gpu.cmd_push_constants(
                        pool.cmd_buffer,
                        mesh_layout.pipeline_layout,
                        vk::ShaderStageFlags::ALL,
                        0,
                        gpu::as_u8_slice(&constants),
                    );
                    gpu.cmd_draw_indexed(pool.cmd_buffer, num_indices as _, 1, 0, 0, 0);
                    gpu.cmd_end_render_pass(pool.cmd_buffer);

                    gpu.submit_pool(pool, gpu::Submit {
                        waits: &[gpu::SemaphoreSubmit {
                            semaphore: frame.acquire,
                            stage: gpu::Stage::COLOR_ATTACHMENT_OUTPUT,
                        }],
                        signals: &[gpu::SemaphoreSubmit {
                            semaphore: frame.present,
                            stage: gpu::Stage::COLOR_ATTACHMENT_OUTPUT,
                        }],
                    }).unwrap();

                    wsi.present(&gpu, frame).unwrap();
                }
                _ => (),
            }
        })
    }
}
