use ash::{
    extensions::khr,
    version::{DeviceV1_0, DeviceV1_2, EntryV1_0, InstanceV1_0},
    vk,
};
use camera::{Camera, InputMap};
use flink::{f32x4x4, vec3};
use gpu_allocator::{AllocationCreateDesc, VulkanAllocator, VulkanAllocatorCreateDesc};
use std::{fs::File, mem, path::Path};
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod camera;
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
        let entry = ash::Entry::new()?;
        let surface_extensions = ash_window::enumerate_required_extensions(&window)?;
        let instance_extensions = surface_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_version(1, 2, 0));
        let instance_desc = vk::InstanceCreateInfo::builder()
            .application_info(&app_desc)
            .enabled_extension_names(&instance_extensions);
        let instance = entry.create_instance(&instance_desc, None)?;

        let surface = ash_window::create_surface(&entry, &instance, &window, None)?;
        let surface_fn = khr::Surface::new(&entry, &instance);

        let (physical_device, device_id, family_index, _family_properties) = instance
            .enumerate_physical_devices()?
            .into_iter()
            .enumerate()
            .find_map(|(device_id, device)| {
                instance
                    .get_physical_device_queue_family_properties(device)
                    .into_iter()
                    .enumerate()
                    .find(|(i, family)| {
                        let universal = family
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
                        let surface_support = surface_fn
                            .get_physical_device_surface_support(device, *i as _, surface)
                            .unwrap();

                        universal && surface_support
                    })
                    .map(|(index, family)| (device, device_id, index as u32, family))
            })
            .unwrap();

        let (device, queue) = {
            let device_extensions = vec![khr::Swapchain::name().as_ptr()];
            let features = vk::PhysicalDeviceFeatures::builder();
            let mut features12 =
                vk::PhysicalDeviceVulkan12Features::builder().imageless_framebuffer(true);

            let queue_priorities = [1.0];
            let queue_descs = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(family_index)
                .queue_priorities(&queue_priorities)
                .build()];
            let device_desc = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_descs)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&features)
                .push_next(&mut features12);

            let device = instance.create_device(physical_device, &device_desc, None)?;
            let queue = device.get_device_queue(family_index, 0);

            (device, queue)
        };

        let swapchain_fn = khr::Swapchain::new(&instance, &device);
        let (swapchain, surface_format) = {
            let surface_capabilities =
                surface_fn.get_physical_device_surface_capabilities(physical_device, surface)?;
            let surface_formats =
                surface_fn.get_physical_device_surface_formats(physical_device, surface)?;

            let surface_format = surface_formats
                .into_iter()
                .map(|format| match format.format {
                    vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                        format: vk::Format::R8G8B8_SRGB,
                        color_space: format.color_space,
                    },
                    _ => format,
                })
                .next()
                .unwrap();

            let swapchain_desc = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(2)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(vk::Extent2D {
                    width: size.width as _,
                    height: size.height as _,
                })
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true);
            let swapchain = swapchain_fn.create_swapchain(&swapchain_desc, None)?;

            (swapchain, surface_format)
        };

        let frame_images = swapchain_fn.get_swapchain_images(swapchain)?;
        let mut frame_semaphores = (0..frame_images.len())
            .map(|_| {
                let desc = vk::SemaphoreCreateInfo::builder();
                device.create_semaphore(&desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let frame_rtvs = frame_images
            .iter()
            .map(|image| {
                let view_desc = vk::ImageViewCreateInfo::builder()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                device.create_image_view(&view_desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Semaphore to use for next swapchain acquire operation.
        // Will be cycled through with `frame_semaphores`.
        let mut acquire_semaphore = {
            let desc = vk::SemaphoreCreateInfo::builder();
            device.create_semaphore(&desc, None)?
        };

        let mut allocator = VulkanAllocator::new(&VulkanAllocatorCreateDesc {
            instance,
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
        });

        let directory = Path::new("assets");

        let num_indices = 70_074;
        let bin = std::fs::read(directory.join("SciFiHelmet.bin"))?;

        let mesh_cpu = {
            let desc = vk::BufferCreateInfo::builder()
                .size(bin.len() as _)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let buffer = device.create_buffer(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Triangle Buffer (CPU)",
                requirements: device.get_buffer_memory_requirements(buffer),
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
            };
            let mut allocation = allocator.allocate(&alloc_desc)?;
            {
                let mapping = allocation.mapped_slice_mut().unwrap();
                mapping[..bin.len()].copy_from_slice(as_u8_slice(&bin));
            }
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let mesh_gpu = {
            let desc = vk::BufferCreateInfo::builder().size(bin.len() as _).usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let buffer = device.create_buffer(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Triangle Buffer (GPU)",
                requirements: device.get_buffer_memory_requirements(buffer),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
            };
            let allocation = allocator.allocate(&alloc_desc)?;
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let locals_pbr_gpu = {
            let desc = vk::BufferCreateInfo::builder()
                .size(mem::size_of::<LocalsPbr>() as _)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
            let buffer = device.create_buffer(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Locals PBR (GPU)",
                requirements: device.get_buffer_memory_requirements(buffer),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: true,
            };
            let allocation = allocator.allocate(&alloc_desc)?;
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            buffer
        };

        let mesh_vs = {
            let mut file = File::open("assets/triangle.vert.spv")?;
            let code = ash::util::read_spv(&mut file)?;
            let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
            device.create_shader_module(&desc, None)?
        };
        let mesh_fs = {
            let mut file = File::open("assets/triangle.frag.spv")?;
            let code = ash::util::read_spv(&mut file)?;
            let desc = vk::ShaderModuleCreateInfo::builder().code(&code);
            device.create_shader_module(&desc, None)?
        };

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

            let image = device.create_image(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Depth Buffer",
                requirements: device.get_image_memory_requirements(image),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            };
            let allocation = allocator.allocate(&alloc_desc)?;
            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
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
            device.create_image_view(&view_desc, None)?
        };

        let mesh_pass = {
            let attachments = [
                vk::AttachmentDescription {
                    format: surface_format.format,
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
                    final_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                },
            ];
            let color_attachments = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let depth_stencil_attachment = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            };
            let subpasses = [vk::SubpassDescription::builder()
                .color_attachments(&color_attachments)
                .depth_stencil_attachment(&depth_stencil_attachment)
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .build()];
            let desc = vk::RenderPassCreateInfo::builder()
                .attachments(&attachments)
                .subpasses(&subpasses);
            device.create_render_pass(&desc, None)?
        };

        let mesh_fbo = {
            let image_formats0 = [surface_format.format];
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
            device.create_framebuffer(&desc, None)?
        };

        let mesh_set_layout = {
            let bindings = [vk::DescriptorSetLayoutBinding::builder()
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build()];
            let desc = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            device.create_descriptor_set_layout(&desc, None)?
        };

        let mesh_layout = {
            let set_layouts = [mesh_set_layout];
            let desc = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
            device.create_pipeline_layout(&desc, None)?
        };

        let mesh_set = {
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            }];
            let desc = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let pool = device.create_descriptor_pool(&desc, None)?;

            let layouts = [mesh_set_layout];
            let desc = vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&layouts)
                .descriptor_pool(pool);
            let set = device.allocate_descriptor_sets(&desc)?;

            let buffer_info = [vk::DescriptorBufferInfo {
                buffer: locals_pbr_gpu,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(set[0])
                    .buffer_info(&buffer_info)
                    .build()],
                &[],
            );

            set
        };

        let mesh_pipeline = {
            let entry_vs = std::ffi::CStr::from_bytes_with_nul(b"main\0")?;
            let entry_fs = std::ffi::CStr::from_bytes_with_nul(b"main\0")?;
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
                .layout(mesh_layout)
                .build(); // TODO

            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[desc], None)
                .unwrap()
        };

        let main_cmd_pools = (0..frames_in_flight)
            .map(|_| {
                let desc = vk::CommandPoolCreateInfo::builder().queue_family_index(family_index);
                device.create_command_pool(&desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let main_cmd_buffers = main_cmd_pools
            .iter()
            .map(|pool| {
                let desc = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                device.allocate_command_buffers(&desc)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let main_semaphore = {
            let mut timeline_desc = vk::SemaphoreTypeCreateInfo::builder()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let desc = vk::SemaphoreCreateInfo::builder().push_next(&mut timeline_desc);
            device.create_semaphore(&desc, None)?
        };

        let render_semaphores = (0..frames_in_flight)
            .map(|_| {
                let desc = vk::SemaphoreCreateInfo::builder();
                device.create_semaphore(&desc, None)
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
                    swapchain_fn.destroy_swapchain(swapchain, None);
                    surface_fn.destroy_surface(surface, None);
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

                    let image_index = {
                        let mut index = 0;
                        let desc = vk::AcquireNextImageInfoKHR::builder()
                            .swapchain(swapchain)
                            .timeout(!0)
                            .fence(vk::Fence::null())
                            .semaphore(acquire_semaphore)
                            .device_mask(1u32 << device_id)
                            .build();
                        let result = swapchain_fn.fp().acquire_next_image2_khr(
                            device.handle(),
                            &desc,
                            &mut index,
                        );
                        match result {
                            vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => index as usize,
                            _ => return,
                        }
                    };

                    std::mem::swap(&mut frame_semaphores[image_index], &mut acquire_semaphore);

                    let frame_local = frame_index % frames_in_flight;

                    if frame_index >= frames_in_flight {
                        let semaphores = [main_semaphore];
                        let wait_values = [(frame_index - frames_in_flight + 1) as u64];
                        let wait_info = vk::SemaphoreWaitInfo::builder()
                            .semaphores(&semaphores)
                            .values(&wait_values);
                        device.wait_semaphores(&wait_info, !0).unwrap();
                        device
                            .reset_command_pool(
                                main_cmd_pools[frame_local],
                                vk::CommandPoolResetFlags::empty(),
                            )
                            .unwrap();
                    }

                    let begin_desc = vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    let main_cmd_buffer = main_cmd_buffers[frame_local][0];
                    device
                        .begin_command_buffer(main_cmd_buffer, &begin_desc)
                        .unwrap();

                    if frame_index == 0 {
                        device.cmd_copy_buffer(
                            main_cmd_buffer,
                            mesh_cpu,
                            mesh_gpu,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: bin.len() as _,
                            }],
                        );
                    }

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
                    device.cmd_update_buffer(
                        main_cmd_buffer,
                        locals_pbr_gpu,
                        0,
                        as_u8_slice(&[locals]),
                    );

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

                    let attachments = [frame_rtvs[image_index], depth_stencil_view];
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
                    device.cmd_begin_render_pass(
                        main_cmd_buffer,
                        &mesh_pass_begin_desc,
                        vk::SubpassContents::INLINE,
                    );

                    let base_offset = num_indices as u64 * 4;
                    device.cmd_bind_vertex_buffers(
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
                    device.cmd_bind_index_buffer(
                        main_cmd_buffer,
                        mesh_gpu,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_bind_pipeline(
                        main_cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        mesh_pipeline[0],
                    );
                    device.cmd_set_scissor(
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
                    device.cmd_set_viewport(
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
                    device.cmd_bind_descriptor_sets(
                        main_cmd_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        mesh_layout,
                        0,
                        &mesh_set,
                        &[],
                    );
                    device.cmd_draw_indexed(main_cmd_buffer, num_indices as _, 1, 0, 0, 0);
                    device.cmd_end_render_pass(main_cmd_buffer);

                    device.end_command_buffer(main_cmd_buffer).unwrap();

                    let main_waits = [frame_semaphores[image_index]];
                    let main_signals = [main_semaphore, render_semaphores[frame_local]];
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
                    device
                        .queue_submit(queue, &[main_submit], vk::Fence::null())
                        .unwrap();

                    let present_wait = [render_semaphores[frame_local]];
                    let present_swapchains = [swapchain];
                    let present_images = [image_index as u32];
                    let present_info = vk::PresentInfoKHR::builder()
                        .wait_semaphores(&present_wait)
                        .swapchains(&present_swapchains)
                        .image_indices(&present_images);
                    swapchain_fn.queue_present(queue, &present_info).unwrap();

                    frame_index += 1;
                }
                _ => (),
            }
        })
    }
}
