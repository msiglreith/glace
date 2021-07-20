use structopt::StructOpt;

/// Create a SPIR-V binary module from SPIR-V assembly text
#[derive(StructOpt)]
struct Args {
    /// Set the output filename. Use '-' for stdout.
    #[structopt(short, default_value = "out.spv")]
    output: String,
    /// The input file. Use '-' for stdin.
    #[structopt(name = "FILE")]
    input: String,
}

fn main() {
    use spirv_tools::assembler::{self, Assembler};

    let args = Args::from_args();

    let contents = if args.input == "-" {
        use std::io::Read;
        let mut v = Vec::with_capacity(1024);
        std::io::stdin()
            .read_to_end(&mut v)
            .expect("failed to read stdin");
        v
    } else {
        std::fs::read(&args.input).expect("failed to read input file")
    };
    let contents = ash::util::read_spv(&mut std::io::Cursor::new(&contents[..])).unwrap();

    let opts = assembler::DisassembleOptions::default();

    let assembler =
        assembler::compiled::CompiledAssembler::with_env(spirv_tools::TargetEnv::default());

    match assembler.disassemble(contents, opts) {
        Ok(binary) => {
            if let Some(str) = binary {
                if args.output == "-" {
                    use std::io::Write;
                    std::io::stdout()
                        .lock()
                        .write_all(str.as_bytes())
                        .expect("failed to write binary to stdout");
                } else {
                    std::fs::write(args.output, &str).expect("failed to write binary");
                }
            }
        }
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}
