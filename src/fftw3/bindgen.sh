env C_INCLUDE_PATH="/localscratch/fkoessel/.local/mpich/include" bindgen --convert-macros --builtins --link fftw3 /usr/include/fftw3.h > fftw3.rs
# env C_INCLUDE_PATH="/localscratch/fkoessel/.local/mpich/include" bindgen --convert-macros --builtins --link fftw3 /usr/include/fftw3-mpi.h > fftw3-mpi.rs
