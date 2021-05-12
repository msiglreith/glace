use spirv_builder::SpirvBuilder;

fn main() -> anyhow::Result<()> {
    let result = SpirvBuilder::new("shader", "spirv-unknown-spv1.5")
        .print_metadata(false)
        .bindless(true)
        .build_multimodule()?;
    let directory = result
        .values()
        .next()
        .and_then(|path| path.parent())
        .unwrap();
    println!("cargo:rerun-if-changed=shader");
    println!("cargo:rustc-env=spv={}", directory.to_str().unwrap());
    Ok(())
}
