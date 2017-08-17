[![build status](https://gitlab.physik.uni-mainz.de/fkoessel/mc-kinetics/badges/master/build.svg)](https://gitlab.physik.uni-mainz.de/fkoessel/mc-kinetics/commits/master)

Implementation of a Monte-Carlo like statistical sampling method for a time
integration of a Fokker-Planck equation, coupled to a hydrodynamic field.

# Build prerequisits
* rustc >= 1.21-nightly
* FFTW3 3.x
* libclang 3.x

For a minimal build environment, have a look into the Docker image defined by
`test/CI/Dockerfile`.

Due to the use of serde and syntax extension in testing code, rust nightly is
necessary at the moment.

# Build
If all prerequisits are fulfilled, build with
```
env RUSTFLAGS="-C target-cpu=native" cargo build --release
```
to get performance benefits from auto-vectorization.

# Documentation
Usage instructions can be found when

Run `cargo doc`. Or `make_documentation.sh` in the root directory to include
also non public functions into the documentation.

# Output format

By default the simulation output consistst of concatinated
[LZMA](https://tukaani.org/xz/) compressed blobs in the
[MessagePack](https://msgpack.org/) format. It also includes an uncompressed
header encoding the simulation parameters at the beginning of the output file.

A byte `buffer` containing such a compressed blob at an offset `offset` and size
`size` can be trivially read with python with

```python
import lzma
import msgpack

with open('output.msgpack.lzma', 'rb') as f:
    f.seek(offset)
    buffer = f.read(size)
    buffer = lzma.decompress(buffer)
    msgpack.unpackb(buffer, encoding='utf-8')

```
