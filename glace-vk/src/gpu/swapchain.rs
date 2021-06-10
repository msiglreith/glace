use crate::gpu::{Gpu, Instance};
use ash::{extensions::khr, prelude::*, vk};

#[derive(Debug)]
pub struct Frame {
    pub id: usize,
    pub acquire: vk::Semaphore,
    pub present: vk::Semaphore,
}

pub struct Swapchain {
    device_id: usize,
    pub swapchain_fn: khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub acquire_semaphore: vk::Semaphore,
    pub frame_semaphores: Vec<vk::Semaphore>,
    pub present_semaphores: Vec<vk::Semaphore>,
    pub frame_rtvs: Vec<vk::ImageView>,
}

impl Swapchain {
    pub unsafe fn new(
        instance: &Instance,
        device: &Gpu,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Self> {
        let swapchain_fn = khr::Swapchain::new(&instance.instance, &device.device);
        let (swapchain, surface_format) = {
            let surface_capabilities = instance
                .surface_fn
                .get_physical_device_surface_capabilities(
                    instance.physical_device,
                    instance.surface,
                )?;
            let surface_formats = instance
                .surface_fn
                .get_physical_device_surface_formats(instance.physical_device, instance.surface)?;

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
                .surface(instance.surface)
                .min_image_count(2)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(vk::Extent2D {
                    width: width,
                    height: height,
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
        let frame_semaphores = (0..frame_images.len())
            .map(|_| {
                let desc = vk::SemaphoreCreateInfo::builder();
                device.create_semaphore(&desc, None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let present_semaphores = (0..frame_images.len())
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
        let acquire_semaphore = {
            let desc = vk::SemaphoreCreateInfo::builder();
            device.create_semaphore(&desc, None)?
        };

        Ok(Swapchain {
            device_id: instance.device_id,
            swapchain,
            swapchain_fn,
            surface_format,
            acquire_semaphore,
            frame_semaphores,
            present_semaphores,
            frame_rtvs,
        })
    }

    pub unsafe fn acquire(&mut self) -> VkResult<Frame> {
        let mut index = 0;
        let desc = vk::AcquireNextImageInfoKHR::builder()
            .swapchain(self.swapchain)
            .timeout(!0)
            .fence(vk::Fence::null())
            .semaphore(self.acquire_semaphore)
            .device_mask(1u32 << self.device_id)
            .build();
        let result = self.swapchain_fn.fp().acquire_next_image2_khr(
            self.swapchain_fn.device(),
            &desc,
            &mut index,
        );
        let index = match result {
            vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => index as usize,
            _ => return VkResult::Err(result),
        };

        let frame = Frame {
            id: index,
            acquire: self.acquire_semaphore,
            present: self.present_semaphores[index],
        };

        std::mem::swap(
            &mut self.frame_semaphores[index],
            &mut self.acquire_semaphore,
        );

        VkResult::Ok(frame)
    }

    pub unsafe fn present(&mut self, gpu: &Gpu, frame: Frame) -> VkResult<()> {
        let present_wait = [frame.present];
        let present_swapchains = [self.swapchain];
        let present_images = [frame.id as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&present_wait)
            .swapchains(&present_swapchains)
            .image_indices(&present_images);
        self.swapchain_fn
            .queue_present(gpu.queue, &present_info)?;
        Ok(())
    }
}
