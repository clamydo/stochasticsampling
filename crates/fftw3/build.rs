use std::env;
use std::path::Path;

fn main() {
    if cfg!(feature = "fftw-threaded") && cfg!(feature = "fftw-static") {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        println!(
            "cargo:rustc-link-search=native={}",
            Path::new(&dir)
                .join("external/fftw-3.3.7/build/lib")
                .display()
        );
        println!("cargo:rustc-link-lib=static=fftw3_threads");
    }
}
