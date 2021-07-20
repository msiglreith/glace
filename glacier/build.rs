use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> anyhow::Result<()> {
    let result = SpirvBuilder::new("shader", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::DependencyOnly)
        .multimodule(true)
        .capability(Capability::Int8)
        .capability(Capability::Int64)
        .capability(Capability::GroupNonUniformQuad)
        .capability(Capability::StorageImageWriteWithoutFormat)
        .capability(Capability::RayQueryKHR)
        .capability(Capability::PhysicalStorageBufferAddresses)
        .capability(Capability::BindlessTextureNV)
        .extension("SPV_KHR_ray_query")
        .extension("SPV_NV_bindless_texture")
        .build()?;
    let directory = result
        .module
        .unwrap_multi()
        .iter()
        .next()
        .and_then(|(_, path)| path.parent())
        .unwrap();
    println!("cargo:rustc-env=spv={}", directory.to_str().unwrap());
    Ok(())
}
