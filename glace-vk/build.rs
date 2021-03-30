use spirv_builder::{MemoryModel, SpirvBuilder};

fn main() -> anyhow::Result<()> {
    SpirvBuilder::new("shader")
        .spirv_version(1, 5)
        .memory_model(MemoryModel::Vulkan)
        .build()?;
    Ok(())
}
