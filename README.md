[![build status](https://gitlab.physik.uni-mainz.de/fkoessel/mc-kinetics/badges/master/build.svg)](https://gitlab.physik.uni-mainz.de/fkoessel/mc-kinetics/commits/master)

Implementation of a Monte-Carlo like statistical sampling method for a time integration of a Fokker-Planck equation, coupled to a hydrodynamic field.

# Build prerequisits
* rustc 1.x-nightly
* FFTW3 3.x
* OpenMPI 1.10+ or MPICH 3.1+
* libclang 3.x

For a minimal building environment, have a look into the Docker image defined by `test/CI/Dockerfile`.

# Build
If all prerequisits are fulfilled, just build with
```
env RUSTFLAGS="-C target-cpu=native" cargo build --release
```

*Note*: Due to a compiler bug, this only works for `nightly-2016-11-06` at the
moment. You can fix a nightly compiler version with
```
rustup override set nightly-2016-11-06
```

# Documentation
Run `cargo doc`. Or `make_documentation.sh` in the root directory to include also non public functions into the documentation.
