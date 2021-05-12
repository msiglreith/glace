use ash::{version::DeviceV1_0, vk};
use camera::{Camera, InputMap};
use glace::{f32x4x4, vec3};
use gpu_allocator::AllocationCreateDesc;
use std::{fs::File, mem, path::Path};
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod camera;
mod device;
mod instance;
mod pass;
mod swapchain;
// mod ktx;

#[repr(C)]
#[derive(Debug)]
struct LocalsPbr {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

/// View a slice as raw byte slice.
///
/// Reinterprets the passed data as raw memory.
/// Be aware of possible packing and aligning rules by Rust compared to OpenGL.
fn as_u8_slice<T>(data: &[T]) -> &[u8] {
    let len = std::mem::size_of::<T>() * data.len();
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) }
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
        let instance = instance::Instance::new(&window)?;
        let mut gpu = device::Gpu::new(
            &instance,
            frames_in_flight,
            device::Descriptors {
                buffers: 1024,
                images: 1024,
                samplers: 128,
            },
        )?;
        let mut wsi = swapchain::Swapchain::new(&instance, &gpu, size.width, size.height)?;

        let directory = Path::new("assets");

        let num_indices = 70_074;
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
                mapping[..bin.len()].copy_from_slice(as_u8_slice(&bin));
            }
            gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let mesh_gpu = {
            let desc = vk::BufferCreateInfo::builder().size(bin.len() as _).usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let buffer = gpu.create_buffer(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Triangle Buffer (GPU)",
                requirements: gpu.get_buffer_memory_requirements(buffer),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
            };
            let allocation = gpu.allocator.allocate(&alloc_desc)?;
            gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let locals_pbr_gpu = {
            let desc = vk::BufferCreateInfo::builder()
                .size(mem::size_of::<LocalsPbr>() as _)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
            let buffer = gpu.create_buffer(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Locals PBR (GPU)",
                requirements: gpu.get_buffer_memory_requirements(buffer),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
            };
            let allocation = gpu.allocator.allocate(&alloc_desc)?;
            gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let upload_cmd_buffer = gpu.acquire_cmd_buffer().unwrap();

        gpu.cmd_copy_buffer(
            upload_cmd_buffer,
            mesh_cpu,
            mesh_gpu,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: bin.len() as _,
            }],
        );

        let mut load_png = |name: &str,
                            format: vk::Format,
                            _downsample: bool|
         -> anyhow::Result<(vk::Image, vk::ImageView, vk::Buffer)> {
            let path = directory.join(name);
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
                mapping[..img_data.len()].copy_from_slice(as_u8_slice(&img_data));
            }
            gpu.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            let pre_transfer_barrier = [vk::ImageMemoryBarrier2KHR::builder()
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags2KHR::ACCESS_2_NONE)
                .src_stage_mask(vk::PipelineStageFlags2KHR::PIPELINE_STAGE_2_NONE)
                .dst_access_mask(vk::AccessFlags2KHR::ACCESS_2_MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2KHR::PIPELINE_STAGE_2_COPY)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .build()];

            let pre_transfer_dep =
                vk::DependencyInfoKHR::builder().image_memory_barriers(&pre_transfer_barrier);
            gpu.ext_sync2
                .cmd_pipeline_barrier2(upload_cmd_buffer, &pre_transfer_dep);

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
                upload_cmd_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );

            let post_transfer_barrier = [vk::ImageMemoryBarrier::builder()
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .build()];
            gpu.cmd_pipeline_barrier(
                upload_cmd_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &post_transfer_barrier,
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
        };

        let (albedo, albedo_view, albedo_buffer) =
            load_png("SciFiHelmet_BaseColor.png", vk::Format::R8G8B8A8_SRGB, true)?;
        let (normal, normal_view, normal_buffer) =
            load_png("SciFiHelmet_Normal.png", vk::Format::R8G8B8A8_UNORM, true)?;

        gpu.end_command_buffer(upload_cmd_buffer)?;
        let upload_buffers = [upload_cmd_buffer];
        let upload_submit = vk::SubmitInfo::builder()
            .command_buffers(&upload_buffers)
            .build();
        gpu.queue_submit(gpu.queue, &[upload_submit], vk::Fence::null())?;

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

        let mesh_fbo = {
            let image_formats0 = [wsi.surface_format.format];
            let image_formats1 = [depth_stencil_format];
            let images = [
                vk::FramebufferAttachmentImageInfo::builder()
                    .view_formats(&image_formats0)
                    .width(size.width as _)
                    .height(size.height as _)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .layer_count(1)
                    .build(),
                vk::FramebufferAttachmentImageInfo::builder()
                    .view_formats(&image_formats1)
                    .width(size.width as _)
                    .height(size.height as _)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .layer_count(1)
                    .build(),
            ];
            let mut attachments =
                vk::FramebufferAttachmentsCreateInfo::builder().attachment_image_infos(&images);
            let mut desc = vk::FramebufferCreateInfo::builder()
                .flags(vk::FramebufferCreateFlags::IMAGELESS)
                .render_pass(mesh_pass)
                .width(size.width as _)
                .height(size.height as _)
                .layers(1)
                .push_next(&mut attachments);
            desc.attachment_count = images.len() as _;
            gpu.create_framebuffer(&desc, None)?
        };

        let mesh_sampler = {
            let desc = vk::SamplerCreateInfo::builder(); // TODO: linear
            gpu.create_sampler(&desc, None)?
        };

        let mesh_layout = gpu.create_layout(&[mesh_sampler], 4)?;

        {
            let buffer_info = [vk::DescriptorBufferInfo {
                buffer: locals_pbr_gpu,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let image_info = [
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: albedo_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view: normal_view,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ];
            gpu.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::builder()
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_set(gpu.buffers.set)
                        .dst_binding(0)
                        .buffer_info(&buffer_info)
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .dst_set(gpu.sampled_images.set)
                        .dst_binding(0)
                        .image_info(&image_info)
                        .build(),
                ],
                &[],
            );
        }

        let spirv_dir = Path::new(env!("spv"));

        let shader = {
            let mut file = std::io::Cursor::new(std::fs::read(spirv_dir.join("mesh_vs"))?);
            // let mut file = File::open("out.spv" /*directory.join("triangle.vert.spv")*/)?;
            let code = ash::util::read_spv(&mut file)?;
            let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
            gpu.create_shader_module(&desc, None)?
        };

        let mesh_fs = {
            let mut file = File::open(directory.join("triangle.frag.spv"))?;
            let code = ash::util::read_spv(&mut file)?;
            let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
            gpu.create_shader_module(&desc, None)?
        };

        let mesh_pipeline = {
            let entry_vs = std::ffi::CStr::from_bytes_with_nul(b"mesh_vs\0")?;
            let entry_fs = std::ffi::CStr::from_bytes_with_nul(b"main\0")?;
            let stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(shader)
                    .name(&entry_vs)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(mesh_fs)
                    .name(&entry_fs)
                    .build(),
            ];

            let attrib_desc = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 1,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 2,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 3,
                    binding: 3,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: 0,
                },
            ];
            let binding_desc = [
                vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: 3 * 4, // R32G32B32_SFLOAT
                    input_rate: vk::VertexInputRate::VERTEX,
                },
                vk::VertexInputBindingDescription {
                    binding: 1,
                    stride: 3 * 4, // R32G32B32_SFLOAT
                    input_rate: vk::VertexInputRate::VERTEX,
                },
                vk::VertexInputBindingDescription {
                    binding: 2,
                    stride: 2 * 4, // R32G32_SFLOAT
                    input_rate: vk::VertexInputRate::VERTEX,
                },
                vk::VertexInputBindingDescription {
                    binding: 3,
                    stride: 4 * 4, // R32G32B32A32_SFLOAT
                    input_rate: vk::VertexInputRate::VERTEX,
                },
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&attrib_desc)
                .vertex_binding_descriptions(&binding_desc);
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
            let desc = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&ia_desc)
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

        let render_semaphores = (0..frames_in_flight)
            .map(|_| {
                let desc = vk::SemaphoreCreateInfo::builder();
                gpu.create_semaphore(&desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut camera = Camera::new(vec3(0.0, 0.0, 0.0), 0.0, 0.0);
        let mut input = InputMap::new();

        let mut frame_index = 0;

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
                    let frame_local = frame_index % frames_in_flight;

                    camera.update(&input);
                    input.reset_delta();

                    let image_index = wsi.acquire().unwrap();
                    let main_cmd_buffer = gpu.acquire_cmd_buffer().unwrap();

                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    let eye = camera.position() + camera.view_dir() * 4.5;
                    let center = camera.position();

                    let locals = LocalsPbr {
                        world_to_view: f32x4x4::look_at_inv(eye, eye - center),
                        view_to_clip: f32x4x4::perspective(
                            std::f32::consts::PI * 0.25,
                            aspect,
                            0.1,
                            10000.0,
                        ),
                    };
                    gpu.cmd_update_buffer(
                        main_cmd_buffer,
                        locals_pbr_gpu,
                        0,
                        as_u8_slice(&[locals]),
                    );

                    let mem_barrier = [vk::MemoryBarrier2KHR::builder()
                        .src_access_mask(
                            vk::AccessFlags2KHR::ACCESS_2_MEMORY_READ
                                | vk::AccessFlags2KHR::ACCESS_2_MEMORY_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2KHR::ACCESS_2_MEMORY_READ
                                | vk::AccessFlags2KHR::ACCESS_2_MEMORY_WRITE,
                        )
                        .src_stage_mask(vk::PipelineStageFlags2KHR::PIPELINE_STAGE_2_ALL_COMMANDS)
                        .dst_stage_mask(vk::PipelineStageFlags2KHR::PIPELINE_STAGE_2_ALL_COMMANDS)
                        .build()];
                    let full_barrier_dep =
                        vk::DependencyInfoKHR::builder().memory_barriers(&mem_barrier);
                    gpu.ext_sync2
                        .cmd_pipeline_barrier2(main_cmd_buffer, &full_barrier_dep);

                    let clear_values = [
                        vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 0.0],
                            },
                        },
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    ];

                    let attachments = [wsi.frame_rtvs[image_index], depth_stencil_view];
                    let mut render_pass_attachments =
                        vk::RenderPassAttachmentBeginInfo::builder().attachments(&attachments);
                    let mesh_pass_begin_desc = vk::RenderPassBeginInfo::builder()
                        .render_pass(mesh_pass)
                        .framebuffer(mesh_fbo)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: size.width as _,
                                height: size.height as _,
                            },
                        })
                        .clear_values(&clear_values)
                        .push_next(&mut render_pass_attachments);
                    gpu.cmd_begin_render_pass(
                        main_cmd_buffer,
                        &mesh_pass_begin_desc,
                        vk::SubpassContents::INLINE,
                    );

                    let base_offset = num_indices as u64 * 4;
                    gpu.cmd_bind_vertex_buffers(
                        main_cmd_buffer,
                        0,
                        &[mesh_gpu, mesh_gpu, mesh_gpu, mesh_gpu],
                        &[
                            base_offset,
                            base_offset + 840_888,
                            base_offset + 2_802_960,
                            base_offset + 1_681_776,
                        ],
                    );
                    gpu.cmd_bind_index_buffer(main_cmd_buffer, mesh_gpu, 0, vk::IndexType::UINT32);
                    gpu.cmd_bind_pipeline(
                        main_cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        mesh_pipeline[0],
                    );
                    gpu.cmd_set_scissor(
                        main_cmd_buffer,
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
                        main_cmd_buffer,
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
                        main_cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        mesh_layout,
                    );
                    gpu.cmd_push_constants(
                        main_cmd_buffer,
                        mesh_layout.pipeline_layout,
                        vk::ShaderStageFlags::ALL,
                        0,
                        &[0, 0, 0, 0],
                    );
                    gpu.cmd_draw_indexed(main_cmd_buffer, num_indices as _, 1, 0, 0, 0);
                    gpu.cmd_end_render_pass(main_cmd_buffer);

                    gpu.end_command_buffer(main_cmd_buffer).unwrap();

                    let main_waits = [wsi.frame_semaphores[image_index]];
                    let main_signals = [gpu.timeline, render_semaphores[frame_local]];
                    let main_stages = [vk::PipelineStageFlags::BOTTOM_OF_PIPE]; // TODO
                    let main_buffers = [main_cmd_buffer];

                    let main_waits_values = [0];
                    let main_signals_values = [frame_index as u64 + 1, 0];
                    let mut timeline_submit = vk::TimelineSemaphoreSubmitInfo::builder()
                        .wait_semaphore_values(&main_waits_values)
                        .signal_semaphore_values(&main_signals_values);
                    let main_submit = vk::SubmitInfo::builder()
                        .wait_semaphores(&main_waits)
                        .wait_dst_stage_mask(&main_stages)
                        .signal_semaphores(&main_signals)
                        .command_buffers(&main_buffers)
                        .push_next(&mut timeline_submit)
                        .build();
                    gpu.queue_submit(gpu.queue, &[main_submit], vk::Fence::null())
                        .unwrap();

                    let present_wait = [render_semaphores[frame_local]];
                    let present_swapchains = [wsi.swapchain];
                    let present_images = [image_index as u32];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&present_wait)
                        .swapchains(&present_swapchains)
                        .image_indices(&present_images);
                    wsi.swapchain_fn
                        .queue_present(gpu.queue, &present_info)
                        .unwrap();

                    frame_index += 1;
                }
                _ => (),
            }
        })
    }
}
