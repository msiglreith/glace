use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> anyhow::Result<()> {
    let result = SpirvBuilder::new("shader", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::DependencyOnly)
        .bindless(true)
        .multimodule(true)
        .capability(Capability::Int8)
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
