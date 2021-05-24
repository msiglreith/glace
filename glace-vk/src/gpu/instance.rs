use ash::{
    extensions::khr,
    version::{EntryV1_0, InstanceV1_0},
    vk,
};

pub struct Instance {
    #[allow(dead_code)]
    entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface: ash::vk::SurfaceKHR,
    pub surface_fn: khr::Surface,
    pub physical_device: ash::vk::PhysicalDevice,
    pub device_id: usize,
    pub family_index: u32,
}

impl Instance {
    pub unsafe fn new(window: &winit::window::Window) -> anyhow::Result<Self> {
        let entry = ash::Entry::new()?;
        let surface_extensions = ash_window::enumerate_required_extensions(window)?;
        let instance_extensions = surface_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();
        let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_version(1, 2, 0));
        let instance_desc = vk::InstanceCreateInfo::builder()
            .application_info(&app_desc)
            .enabled_extension_names(&instance_extensions);
        let instance = entry.create_instance(&instance_desc, None)?;

        let surface = ash_window::create_surface(&entry, &instance, window, None)?;
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

        Ok(Instance {
            entry,
            instance,
            surface,
            surface_fn,
            physical_device,
            device_id,
            family_index,
        })
    }
}
