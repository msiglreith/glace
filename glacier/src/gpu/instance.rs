use ash::{
    extensions::{ext, khr},
    vk,
};
use std::ffi::CStr;

pub struct Instance {
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface: ash::vk::SurfaceKHR,
    pub surface_fn: khr::Surface,
    pub physical_device: ash::vk::PhysicalDevice,
    pub device_id: usize,
    pub family_index: u32,
    instance_extensions: Vec<vk::ExtensionProperties>,
    device_extensions: Vec<vk::ExtensionProperties>,
}

impl Instance {
    pub unsafe fn new(window: &winit::window::Window) -> anyhow::Result<Self> {
        let entry = ash::Entry::new()?;
        let instance_extensions = entry.enumerate_instance_extension_properties()?;
        let supports_extension = |extension| -> bool {
            instance_extensions
                .iter()
                .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == extension)
        };

        let surface_extensions = ash_window::enumerate_required_extensions(window)?;
        let mut extensions = surface_extensions
            .iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let supports_debug_utils = supports_extension(ext::DebugUtils::name());
        if supports_debug_utils {
            extensions.push(ext::DebugUtils::name().as_ptr());
        }

        let app_desc = vk::ApplicationInfo::builder().api_version(vk::make_api_version(0, 1, 2, 0));
        let instance_desc = vk::InstanceCreateInfo::builder()
            .application_info(&app_desc)
            .enabled_extension_names(&extensions);
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

        let device_extensions = instance.enumerate_device_extension_properties(physical_device)?;

        Ok(Instance {
            entry,
            instance,
            surface,
            surface_fn,
            physical_device,
            device_id,
            family_index,
            instance_extensions,
            device_extensions,
        })
    }

    pub unsafe fn supports_instance_extension(&self, extension: &CStr) -> bool {
        self.instance_extensions
            .iter()
            .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == extension)
    }

    pub unsafe fn supports_device_extension(&self, extension: &CStr) -> bool {
        self.device_extensions
            .iter()
            .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == extension)
    }
}
