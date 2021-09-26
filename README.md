# glace

`rust-gpu` compatible library for Computer Graphics. (work in progress!)

## glacier

Vulkan 1.2 renderer using `rust-gpu` shaders including basic gltf support (might change in the future).

Currently only supports **Nvidia RTX or newer** GPUs due to raytracing requirement and (even more constraining) bindless textures handles (`SPV_NV_bindless_texture`). See [`Descriptorless Rendering in Vulkan`](https://msiglreith.gitbook.io/blog/descriptorless-rendering-in-vulkan) for the rationale of choosing this extension.