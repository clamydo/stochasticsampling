extern crate vergen;

use vergen::{ConstantsFlags, Result, Vergen};

fn main() {
    gen_constants().expect("Unable to generate vergen constants!");
    // println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    // println!("cargo:rustc-link-lib=static=fftw3");
}

fn gen_constants() -> Result<()> {
    let vergen = Vergen::new(ConstantsFlags::all())?;

    for (k, v) in vergen.build_info() {
        println!("cargo:rustc-env={}={}", k.name(), v);
    }

    Ok(())
}