Implementation of a Monte-Carlo like statistical sampling method for a time
integration of a Fokker-Planck equation, coupled to a hydrodynamic field.

# TERMS OF USE

In addition to the terms of the GPLv3, any publications that make use of this work or derived work must cite http://doi.org/10.25358/openscience-3103.

# Build prerequisits
* rustc >= 1.21-nightly
* FFTW3 3.x
* libclang 3.x

For a minimal build environment, have a look into the Docker image defined by
`test/CI/Dockerfile`.

Due to the use of serde and syntax extension in testing code, rust nightly is
necessary at the moment.

# Build
If all prerequisites are fulfilled, build with
```
env RUSTFLAGS="-C target-cpu=native" cargo build --release
```
to get performance benefits from auto-vectorization.

## Dependencies
Compile FFTW3 with
```
./configure --enable-threads --enable-sse2 --enable-avx --enable-avx2 --enable-avx512 --prefix fftw-3.3.7/build CFLAGS="-march=native"
```

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
# Profiling
One way to optain a runtime profile is using perf:
```
env RAYON_NUM_THREADS=1 perf record --call-graph=lbr <simulation>
```
or
```
env RAYON_NUM_THREADS=1 perf record -F 99 -b --call-graph=dwarf <simulation>
```
Using `frame-pointer` is not reliable since the are often omitted in builds (as are they in `opt-level >= 2` in rust). `lbr` is faster, but only available on Intel Haswell CPUs and later. On Skylake it is limited to a call depth of 16. `dwarf` results in rich information, but produces quiet heavy profiles. It might be necessary to reduce the sampling rate, for example `-F 99`, in `perf`.

In case `flamegraph` tools are installed, a flamegraph can be produces from the perf profile with
```
perf script | stackcollapse-perf | flamegraph > flame.svg
```
