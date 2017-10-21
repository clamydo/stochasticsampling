extern crate vergen;

use vergen::{SHORT_SHA, vergen};

fn main() {
    vergen(SHORT_SHA).unwrap();
    //println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    //println!("cargo:rustc-link-lib=static=fftw3");
}
