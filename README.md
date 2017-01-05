[![build status](https://gitlab.physik.uni-mainz.de/fkoessel/mc-kinetics/badges/master/build.svg)](https://gitlab.physik.uni-mainz.de/fkoessel/mc-kinetics/commits/master)

Implementation of a Monte-Carlo like statistical sampling method for a time integration of a Fokker-Planck equation, coupled to a hydrodynamic field.

# Build prerequisits
* rustc >= 1.16-nightly
* FFTW3 3.x
* libclang 3.x

For a minimal build environment, have a look into the Docker image defined by `test/CI/Dockerfile`.

Due to the use of serde and syntax extension in testing code, rust nightly is
necessary at the moment.

# Build
If all prerequisits are fulfilled, just build with
```
env RUSTFLAGS="-C target-cpu=native" cargo build --release
```

# Documentation
Usage instructions can be found when

Run `cargo doc`. Or `make_documentation.sh` in the root directory to include also non public functions into the documentation.
