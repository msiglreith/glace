use ash::vk;
use glace::{f32x4, f32x4x4, vec3, vec4};
use glacier::gpu;
use gpu_allocator::AllocationCreateDesc;
use std::{collections::HashMap, ffi::CString, hash::Hash, mem, ops::Range};
use winit::{
    dpi::LogicalSize,
    event::{ButtonId, DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Input {
    Button(ButtonId),
    Key(VirtualKeyCode),
}

pub type InputMap = HashMap<Input, ElementState>;

#[repr(C)]
#[derive(Debug)]
struct WorldPbr {
    world_to_view: f32x4x4,
    view_to_clip: f32x4x4,
}

#[repr(C)]
#[derive(Debug)]
struct GeometryData {
    v_position_obj: u64,
    v_normal_obj: u64,
    v_texcoord: u64,
    v_tangent_obj: u64,
    indices: u64,
}

#[repr(C)]
#[derive(Debug)]
struct MaterialData {
    albedo_map: u64,
    normal_map: u64,
    albedo_color: f32x4,
}

#[repr(C)]
#[derive(Debug)]
struct InstanceData {
    material: u32,
    geometry: u32,
}

fn map_format(format: gltf::image::Format, srgb: bool) -> vk::Format {
    match (srgb, format) {
        (false, gltf::image::Format::R8G8B8) | // preprocessed to rgba
        (false, gltf::image::Format::R8G8B8A8) => vk::Format::R8G8B8A8_UNORM,
        (true, gltf::image::Format::R8G8B8) | // preprocessed to rgba
        (true, gltf::image::Format::R8G8B8A8) => vk::Format::R8G8B8A8_SRGB,
        fmt => {
            dbg!(fmt);
            unimplemented!()
        },
    }
}

fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Box<[_]>>();
    if args.len() < 2 {
        return Err(anyhow::anyhow!("expected '<scene>' argument"));
    }
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("glace")
        .with_inner_size(LogicalSize::new(1440.0, 800.0))
        .build(&event_loop)?;

    unsafe {
        let instance = gpu::Instance::new(&window)?;
        let mut gpu = gpu::Gpu::new(&instance, 2)?;

        let size = dbg!(window.inner_size());
        let mut wsi = gpu::Swapchain::new(
            &instance,
            &gpu,
            size.width,
            size.height,
            vk::PresentModeKHR::IMMEDIATE,
        )?;

        let asset = &args[1];

        let (document, gltf_buffers, mut gltf_images) = gltf::import(asset).unwrap();

        let upload_pool = gpu.acquire_pool().unwrap();

        let world_pbr = gpu.create_buffer_gpu(
            "mesh::world",
            mem::size_of::<WorldPbr>(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
            gpu::BufferInit::None,
        )?;

        // gltf loading
        println!("load buffers");
        let buffers = gltf_buffers
            .iter()
            .map(|buffer| {
                gpu.create_buffer_gpu(
                    "",
                    buffer.len(),
                    gpu::BufferUsageFlags::STORAGE_BUFFER
                        | gpu::BufferUsageFlags::INDEX_BUFFER
                        | gpu::BufferUsageFlags::TRANSFER_DST
                        | gpu::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    gpu::BufferInit::Host {
                        pool: upload_pool,
                        data: &buffer,
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        println!("process images");
        for image in &mut gltf_images {
            match image.format {
                gltf::image::Format::R8G8B8 => {
                    // expand rgb -> rgba as not well supported on all platforms
                    let mut rgba = Vec::with_capacity(image.pixels.len() / 3 * 4);
                    for i in 0..image.pixels.len() / 3 {
                        rgba.push(image.pixels[3 * i]);
                        rgba.push(image.pixels[3 * i + 1]);
                        rgba.push(image.pixels[3 * i + 2]);
                        rgba.push(0xFF);
                    }
                    image.pixels = rgba;
                }
                _ => (),
            }
        }
        println!("load images");
        let images = gltf_images
            .into_iter()
            .map(|image| {
                let image_format = image.format;
                let format = map_format(image_format, false);
                let image = gpu.create_image_gpu(
                    "",
                    &gpu::ImageDesc {
                        ty: vk::ImageType::TYPE_2D,
                        format,
                        extent: vk::Extent3D {
                            width: image.width,
                            height: image.height,
                            depth: 1,
                        },
                        usage: vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::STORAGE,
                        mip_levels: 1,
                        array_layers: 1,
                        samples: 1,
                    },
                    gpu::ImageInit::Host {
                        pool: upload_pool,
                        data: &image.pixels,
                        aspect: vk::ImageAspectFlags::COLOR,
                    },
                )?;

                gpu.cmd_barriers(
                    upload_pool.cmd_buffer,
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
                            layout: gpu::ImageLayout::GENERAL,
                        },
                    }],
                );

                Ok((image, image_format))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut geometry = Vec::new();

        struct Primitive {
            material: usize,
        }
        let mut primitives = Vec::new();

        struct Mesh {
            primitives: Range<usize>,
            blas: gpu::AccelerationStructure,
        }
        let mut meshes = HashMap::new();

        for mesh in document.meshes() {
            println!("load mesh: {:?}", mesh.name());
            let mut geometry_blas = Vec::new();

            let primitive_start = geometry.len();
            for primitive in mesh.primitives() {
                let material = if let Some(material) = primitive.material().index() {
                    material
                } else {
                    continue;
                };

                let indices = primitive.indices().expect("no index buffer");

                let view = indices.view().unwrap();
                let (index_ty, index_size) = match indices.data_type() {
                    gltf::accessor::DataType::U16 => (vk::IndexType::UINT16, 2),
                    gltf::accessor::DataType::U32 => (vk::IndexType::UINT32, 4),
                    _ => unimplemented!(),
                };

                let indices_handle = {
                    let mut indices_u32 = Vec::new();
                    let view = indices.view().unwrap();
                    let buffer = &gltf_buffers[view.buffer().index() as usize];

                    let offset = (view.offset() + indices.offset()) as isize;

                    match index_ty {
                        vk::IndexType::UINT16 => {
                            let indices: &[u16] = {
                                std::slice::from_raw_parts(
                                    buffer.as_ptr().offset(offset) as _,
                                    indices.count(),
                                )
                            };
                            for index in indices {
                                indices_u32.push(*index as u32);
                            }
                        }
                        vk::IndexType::UINT32 => {
                            let indices: &[u32] = {
                                std::slice::from_raw_parts(
                                    buffer.as_ptr().offset(offset) as _,
                                    indices.count(),
                                )
                            };
                            for index in indices {
                                indices_u32.push(*index);
                            }
                        }
                        _ => unimplemented!(),
                    };

                    let buffer = gpu.create_buffer_gpu(
                        &format!("{:?}::indices", mesh.name()),
                        indices_u32.len() * 4,
                        gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
                        gpu::BufferInit::Host {
                            pool: upload_pool,
                            data: gpu::as_u8_slice(&indices_u32),
                        },
                    )?;
                    gpu.buffer_address(buffer)
                };

                let map_accessor = |accessor: &gltf::Accessor| -> vk::DeviceAddress {
                    let view = accessor.view().unwrap();
                    gpu.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder()
                            .buffer(buffers[view.buffer().index() as usize]),
                    ) + (view.offset() + accessor.offset()) as u64
                };

                let index_buffer = gpu::BufferView {
                    buffer: buffers[view.buffer().index()],
                    offset: (view.offset() + indices.offset()) as _,
                    range: (indices.size() * indices.count()) as _,
                };

                let position_acessor = primitive
                    .get(&gltf::Semantic::Positions)
                    .expect("no position data");
                let num_vertices = position_acessor.count();
                let position_handle = map_accessor(&position_acessor);

                let normal_handle = map_accessor(
                    &primitive
                        .get(&gltf::Semantic::Normals)
                        .expect("no normal data"),
                );

                let tangent_handle = primitive
                    .get(&gltf::Semantic::Tangents)
                    .map_or(0, |accessor| map_accessor(&accessor));

                let texcoord_handle = primitive
                    .get(&gltf::Semantic::TexCoords(0))
                    .map_or(0, |accessor| map_accessor(&accessor));

                geometry.push(GeometryData {
                    v_position_obj: position_handle,
                    v_normal_obj: normal_handle,
                    v_texcoord: texcoord_handle,
                    v_tangent_obj: tangent_handle,
                    indices: indices_handle,
                });

                primitives.push(Primitive { material });

                geometry_blas.push(gpu::GeometryBlas::Triangles {
                    flags: vk::GeometryFlagsKHR::OPAQUE,
                    format: vk::Format::R32G32B32_SFLOAT,
                    vertex_buffer: position_handle,
                    vertex_stride: 12,
                    num_vertices: num_vertices as _,
                    index_type: index_ty,
                    index_buffer: index_buffer.handle(&gpu),
                    num_indices: indices.count() as _,
                });
            }
            let primitive_end = geometry.len();

            if primitive_start == primitive_end {
                continue;
            }

            let blas = gpu.create_blas(
                upload_pool,
                "blas",
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                &geometry_blas,
            )?;

            meshes.insert(
                mesh.index(),
                Mesh {
                    primitives: primitive_start..primitive_end,
                    blas,
                },
            );
        }
        let geometry_data = gpu.create_buffer_gpu(
            "mesh::geometry",
            mem::size_of::<GeometryData>() * geometry.len(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
            gpu::BufferInit::Host {
                pool: upload_pool,
                data: gpu::as_u8_slice(&geometry),
            },
        )?;

        let mesh_sampler = {
            let desc = vk::SamplerCreateInfo::builder(); // TODO: linear
            gpu.create_sampler(&desc, None)?
        };

        println!("load materials");

        let mut materials = Vec::new();
        for material in document.materials() {
            let create_view = |image, format| -> anyhow::Result<u64> {
                let view_srv = {
                    let mut view_usage =
                        vk::ImageViewUsageCreateInfo::builder().usage(vk::ImageUsageFlags::SAMPLED);
                    let desc = vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .push_next(&mut view_usage);

                    gpu.create_image_view(&desc, None)?
                };
                Ok(gpu.sampled_image_address(view_srv, mesh_sampler))
            };

            let albedo_map = material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map_or(Ok(0), |texture| {
                    let tex = texture.texture().index();
                    create_view(images[tex].0, map_format(images[tex].1, true))
                })?;
            let normal_map = material.normal_texture().map_or(Ok(0), |texture| {
                let tex = texture.texture().index();
                create_view(images[tex].0, map_format(images[tex].1, false))
            })?;
            let albedo_factor = material.pbr_metallic_roughness().base_color_factor();

            materials.push(MaterialData {
                albedo_map,
                normal_map,
                albedo_color: vec4(
                    albedo_factor[0],
                    albedo_factor[1],
                    albedo_factor[2],
                    albedo_factor[3],
                ),
            });
        }
        let material_data = gpu.create_buffer_gpu(
            "mesh::material",
            mem::size_of::<MaterialData>() * materials.len(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
            gpu::BufferInit::Host {
                pool: upload_pool,
                data: gpu::as_u8_slice(&materials),
            },
        )?;

        let mut instances = Vec::new();
        let mut transforms = Vec::new();
        let mut tlas_instances = Vec::new();

        struct Node {
            mesh_id: usize,
            transform: f32x4x4,
        }

        let mut nodes = Vec::new();

        fn traverse_node(node: gltf::Node, nodes: &mut Vec<Node>, meshes: &HashMap<usize, Mesh>) {
            let m = node.transform().matrix();
            let transform = f32x4x4 {
                c0: vec4(m[0][0], m[1][0], m[2][0], m[3][0]),
                c1: vec4(m[0][1], m[1][1], m[2][1], m[3][1]),
                c2: vec4(m[0][2], m[1][2], m[2][2], m[3][2]),
                c3: vec4(m[0][3], m[1][3], m[2][3], m[3][3]),
            };
            if let Some(mesh) = node.mesh() {
                let mesh_id = mesh.index();
                if meshes.contains_key(&mesh_id) {
                    nodes.push(Node { mesh_id, transform });
                }
            }

            // todo: propagate transform
            for child in node.children() {
                traverse_node(child, nodes, meshes);
            }
        }

        for node in document.default_scene().unwrap().nodes() {
            traverse_node(node, &mut nodes, &meshes);
        }

        for node in &nodes {
            let mesh = &meshes[&node.mesh_id];

            let idx = instances.len() as u32;
            for prim_id in mesh.primitives.clone() {
                instances.push(InstanceData {
                    material: primitives[prim_id].material as _,
                    geometry: prim_id as _,
                });
                transforms.push(node.transform);
            }

            tlas_instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [
                        node.transform.c0.x,
                        node.transform.c0.y,
                        node.transform.c0.z,
                        node.transform.c0.w,
                        node.transform.c1.x,
                        node.transform.c1.y,
                        node.transform.c1.z,
                        node.transform.c1.w,
                        node.transform.c2.x,
                        node.transform.c2.y,
                        node.transform.c2.z,
                        node.transform.c2.w,
                    ],
                },
                instance_custom_index_and_mask: (0xFF << 24) | idx,
                instance_shader_binding_table_record_offset_and_flags: 0,
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: gpu.acceleration_structure_address(mesh.blas),
                },
            });
        }

        let instance_data = gpu.create_buffer_gpu(
            "mesh::instance",
            mem::size_of::<InstanceData>() * instances.len(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
            gpu::BufferInit::Host {
                pool: upload_pool,
                data: gpu::as_u8_slice(&instances),
            },
        )?;
        let _transform_data = gpu.create_buffer_gpu(
            "mesh::transform",
            mem::size_of::<f32x4x4>() * transforms.len(),
            gpu::BufferUsageFlags::STORAGE_BUFFER | gpu::BufferUsageFlags::TRANSFER_DST,
            gpu::BufferInit::Host {
                pool: upload_pool,
                data: gpu::as_u8_slice(&transforms),
            },
        )?;

        let tlas_instances_buffer = gpu.create_buffer_gpu(
            "tlas instances",
            tlas_instances.len() * mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
            gpu::BufferUsageFlags::TRANSFER_DST
                | gpu::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            gpu::BufferInit::Host {
                pool: upload_pool,
                data: gpu::as_u8_slice(&tlas_instances),
            },
        )?;

        let tlas = gpu.create_tlas(
            upload_pool,
            "tlas",
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            gpu::GeometryTlas {
                flags: vk::GeometryFlagsKHR::OPAQUE,
                instance_buffer: gpu::BufferView::whole(tlas_instances_buffer),
                num_instances: tlas_instances.len() as _,
            },
        )?;

        let ray_target_format = vk::Format::R32G32B32A32_SFLOAT;
        let ray_target = {
            let desc = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(ray_target_format)
                .extent(vk::Extent3D {
                    width: size.width as _,
                    height: size.height as _,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC);

            let image = gpu.create_image(&desc, None)?;
            let alloc_desc = AllocationCreateDesc {
                name: "Ray Target",
                requirements: gpu.get_image_memory_requirements(image),
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
            };
            let allocation = gpu.allocator.allocate(&alloc_desc)?;
            gpu.bind_image_memory(image, allocation.memory(), allocation.offset())?;
            image
        };
        let ray_target_view = {
            let view_desc = vk::ImageViewCreateInfo::builder()
                .image(ray_target)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(ray_target_format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            gpu.create_image_view(&view_desc, None)?
        };

        gpu.cmd_barriers(
            upload_pool.cmd_buffer,
            &[],
            &[gpu::ImageBarrier {
                image: ray_target,
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
                    stage: gpu::Stage::COMPUTE_SHADER,
                    layout: gpu::ImageLayout::GENERAL,
                },
            }],
        );

        gpu.submit_pool(
            upload_pool,
            gpu::Submit {
                waits: &[],
                signals: &[],
            },
        )
        .unwrap();

        #[repr(C)]
        struct PathtraceConstants {
            world: u64,
            instance: u64,
            geometry: u64,
            material: u64,
            tlas: u64,
            ray_target: u64,
            frame_id: u32,
        }
        let ray_layout = gpu.create_layout(mem::size_of::<PathtraceConstants>() as _)?;

        let pathtrace_pipeline = {
            let pathtrace = gpu.create_rust_shader("pathtrace")?;
            let entry_pathtrace = CString::new("pathtrace")?;

            let stage = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(pathtrace)
                .name(&entry_pathtrace)
                .build();
            let desc = vk::ComputePipelineCreateInfo::builder()
                .stage(stage)
                .layout(ray_layout.pipeline_layout)
                .build();

            gpu.create_compute_pipelines(vk::PipelineCache::null(), &[desc], None)
                .unwrap()[0]
        };

        use dolly::prelude::*;

        // let mut camera = Camera::new(vec3(0.0, 1.0, 0.0), 0.0, 0.0);
        let mut camera = CameraRig::builder()
            .with(Position::new(dolly::glam::Vec3::new(0.0, 1.0, 3.0)))
            .with(YawPitch::new().yaw_degrees(180.0))
            .with(Smooth::new_position_rotation(1.0, 1.0))
            .build();

        let mut input = InputMap::new();
        let mut actions = HashMap::new();
        actions.insert("forward", 0.0);
        actions.insert("right", 0.0);
        actions.insert("d_yaw", 0.0);
        actions.insert("d_pitch", 0.0);

        enum View {
            Raytrace,
        }
        let view = View::Raytrace;
        let mut global_frame_id = 0;
        let mut now = std::time::Instant::now();

        let mut prev_transform = camera.final_transform;

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
                    DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                        if let Some(ElementState::Pressed) = input.get(&Input::Button(1)) {
                            *actions.get_mut("d_yaw").unwrap() -= dx as f32;
                            *actions.get_mut("d_pitch").unwrap() += dy as f32;
                        }
                    }
                    DeviceEvent::Button { state, button } => {
                        input.insert(Input::Button(button), state);
                    }
                    DeviceEvent::Key(key) => {
                        if let Some(key_code) = key.virtual_keycode {
                            let prev_state = input.insert(Input::Key(key_code), key.state);

                            let factor = if prev_state == Some(key.state) {
                                0.0 // repeat
                            } else {
                                match key.state {
                                    ElementState::Pressed => 1.0,
                                    ElementState::Released => -1.0,
                                }
                            };
                            match key_code {
                                VirtualKeyCode::W => *actions.get_mut("forward").unwrap() += factor,
                                VirtualKeyCode::S => *actions.get_mut("forward").unwrap() -= factor,
                                VirtualKeyCode::D => *actions.get_mut("right").unwrap() += factor,
                                VirtualKeyCode::A => *actions.get_mut("right").unwrap() -= factor,
                                _ => (),
                            }
                        }
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    let dt = now.elapsed().as_secs_f32();
                    now = std::time::Instant::now();

                    let move_dir = -actions["forward"] * camera.final_transform.forward()
                        - actions["right"] * camera.final_transform.right();

                    camera
                        .driver_mut::<YawPitch>()
                        .rotate_yaw_pitch(0.15 * actions["d_yaw"], 0.15 * actions["d_pitch"]);
                    camera
                        .driver_mut::<Position>()
                        .translate(10.0 * move_dir * dt);
                    let transform = camera.update(dt);

                    actions.insert("d_yaw", 0.0);
                    actions.insert("d_pitch", 0.0);

                    if !transform
                        .position
                        .abs_diff_eq(prev_transform.position, 1.0e-4)
                        || !transform
                            .rotation
                            .abs_diff_eq(prev_transform.rotation, 1.0e-5)
                    {
                        global_frame_id = 0;
                    }

                    prev_transform = transform;

                    let frame = wsi.acquire().unwrap();
                    let pool = gpu.acquire_pool().unwrap();

                    let size = window.inner_size();
                    let aspect = size.width as f32 / size.height as f32;

                    let dir = transform.forward();
                    let eye = transform.position;

                    let world = [WorldPbr {
                        world_to_view: f32x4x4::look_at_inv(
                            vec3(eye.x, eye.y, eye.z),
                            vec3(dir.x, dir.y, dir.z),
                        ),
                        view_to_clip: f32x4x4::perspective(
                            std::f32::consts::PI * 0.25,
                            aspect,
                            0.1,
                            10000.0,
                        ),
                    }];
                    gpu.cmd_update_buffer(pool.cmd_buffer, world_pbr, 0, gpu::as_u8_slice(&world));

                    gpu.cmd_barriers(pool.cmd_buffer, &[gpu::MemoryBarrier::full()], &[]);

                    let world_addr = gpu.buffer_address(world_pbr);
                    let geometry_addr = gpu.buffer_address(geometry_data);
                    let material_addr = gpu.buffer_address(material_data);
                    let instance_addr = gpu.buffer_address(instance_data);

                    gpu.cmd_push_constants(
                        pool.cmd_buffer,
                        ray_layout.pipeline_layout,
                        vk::ShaderStageFlags::ALL,
                        0,
                        gpu::as_u8_slice(&[PathtraceConstants {
                            world: world_addr,
                            instance: instance_addr,
                            geometry: geometry_addr,
                            material: material_addr,
                            tlas: gpu.acceleration_structure_address(tlas),
                            ray_target: gpu.storage_image_address(ray_target_view),
                            frame_id: global_frame_id,
                        }]),
                    );
                    gpu.cmd_bind_pipeline(
                        pool.cmd_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pathtrace_pipeline,
                    );
                    gpu.cmd_dispatch(pool.cmd_buffer, size.width / 16, size.height / 16, 1);

                    // Blit ray target -> swapchain
                    gpu.cmd_barriers(
                        pool.cmd_buffer,
                        &[],
                        &[
                            gpu::ImageBarrier {
                                image: ray_target,
                                range: vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                src: gpu::ImageAccess {
                                    access: gpu::Access::SHADER_WRITE,
                                    stage: gpu::Stage::COMPUTE_SHADER,
                                    layout: gpu::ImageLayout::GENERAL,
                                },
                                dst: gpu::ImageAccess {
                                    access: gpu::Access::TRANSFER_READ,
                                    stage: gpu::Stage::BLIT,
                                    layout: gpu::ImageLayout::GENERAL,
                                },
                            },
                            gpu::ImageBarrier {
                                image: wsi.frame_images[frame.id],
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
                                    layout: gpu::ImageLayout::PRESENT_SRC_KHR,
                                },
                                dst: gpu::ImageAccess {
                                    access: gpu::Access::TRANSFER_WRITE,
                                    stage: gpu::Stage::BLIT,
                                    layout: gpu::ImageLayout::GENERAL,
                                },
                            },
                        ],
                    );

                    match view {
                        View::Raytrace => {
                            gpu.cmd_blit_image(
                                pool.cmd_buffer,
                                ray_target,
                                gpu::ImageLayout::GENERAL,
                                wsi.frame_images[frame.id],
                                gpu::ImageLayout::GENERAL,
                                &[vk::ImageBlit {
                                    src_subresource: vk::ImageSubresourceLayers {
                                        aspect_mask: vk::ImageAspectFlags::COLOR,
                                        mip_level: 0,
                                        base_array_layer: 0,
                                        layer_count: 1,
                                    },
                                    src_offsets: [
                                        vk::Offset3D { x: 0, y: 0, z: 0 },
                                        vk::Offset3D {
                                            x: size.width as _,
                                            y: size.height as _,
                                            z: 1,
                                        },
                                    ],
                                    dst_subresource: vk::ImageSubresourceLayers {
                                        aspect_mask: vk::ImageAspectFlags::COLOR,
                                        mip_level: 0,
                                        base_array_layer: 0,
                                        layer_count: 1,
                                    },
                                    dst_offsets: [
                                        vk::Offset3D { x: 0, y: 0, z: 0 },
                                        vk::Offset3D {
                                            x: size.width as _,
                                            y: size.height as _,
                                            z: 1,
                                        },
                                    ],
                                }],
                                vk::Filter::LINEAR,
                            );
                        }
                    }

                    gpu.cmd_barriers(
                        pool.cmd_buffer,
                        &[],
                        &[
                            gpu::ImageBarrier {
                                image: ray_target,
                                range: vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                src: gpu::ImageAccess {
                                    access: gpu::Access::TRANSFER_READ,
                                    stage: gpu::Stage::BLIT,
                                    layout: gpu::ImageLayout::GENERAL,
                                },
                                dst: gpu::ImageAccess {
                                    access: gpu::Access::SHADER_WRITE,
                                    stage: gpu::Stage::COMPUTE_SHADER,
                                    layout: gpu::ImageLayout::GENERAL,
                                },
                            },
                            gpu::ImageBarrier {
                                image: wsi.frame_images[frame.id],
                                range: vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                src: gpu::ImageAccess {
                                    access: gpu::Access::TRANSFER_WRITE,
                                    stage: gpu::Stage::BLIT,
                                    layout: gpu::ImageLayout::GENERAL,
                                },
                                dst: gpu::ImageAccess {
                                    access: gpu::Access::NONE,
                                    stage: gpu::Stage::empty(),
                                    layout: gpu::ImageLayout::PRESENT_SRC_KHR,
                                },
                            },
                        ],
                    );

                    gpu.submit_pool(
                        pool,
                        gpu::Submit {
                            waits: &[gpu::SemaphoreSubmit {
                                semaphore: frame.acquire,
                                stage: gpu::Stage::COLOR_ATTACHMENT_OUTPUT,
                            }],
                            signals: &[gpu::SemaphoreSubmit {
                                semaphore: frame.present,
                                stage: gpu::Stage::COLOR_ATTACHMENT_OUTPUT,
                            }],
                        },
                    )
                    .unwrap();

                    wsi.present(&gpu, frame).unwrap();

                    global_frame_id += 1;
                }
                _ => (),
            }
        })
    }
}
