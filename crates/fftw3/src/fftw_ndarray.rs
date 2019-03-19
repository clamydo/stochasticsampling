use ndarray::{ArrayViewMut, Ix2, Ix3};
use num_complex::Complex;
use std::mem;
use crate::Float;

#[cfg(feature = "single")]
use ::fftw3_ffi::fftwf_malloc as fftw_malloc;
#[cfg(not(feature = "single"))]
use ::fftw3_ffi::fftw_malloc as fftw_malloc;

#[cfg(feature = "single")]
use ::fftw3_ffi::fftwf_free as fftw_free;
#[cfg(not(feature = "single"))]
use ::fftw3_ffi::fftw_free as fftw_free;

/// FFTW3 allocated `ArrayView` for 2D data.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation.
pub struct FFTData2D<'a> {
    pub data: ArrayViewMut<'a, Complex<Float>, Ix2>,
}

impl<'a> FFTData2D<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: Ix2) -> FFTData2D<'a> {
        let data;
        unsafe {
            let ptr =
                fftw_malloc(shape[0] * shape[1] * mem::size_of::<Complex<Float>>());
            data = ArrayViewMut::from_shape_ptr(shape, ptr as *mut Complex<Float>);
        }

        FFTData2D { data: data }
    }
}

/// Automatically free memory, if goes out of scope.
impl<'a> Drop for FFTData2D<'a> {
    fn drop(&mut self) {
        unsafe { fftw_free(self.data.as_ptr() as *mut _) }
    }
}

/// FFTW3 allocated `ArrayView` for 3D data.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation.
pub struct FFTData3D<'a> {
    pub data: ArrayViewMut<'a, Complex<Float>, Ix3>,
}

impl<'a> FFTData3D<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: Ix3) -> FFTData3D<'a> {
        let data;
        unsafe {
            let ptr = fftw_malloc(
                shape[0] * shape[1] * shape[2] * mem::size_of::<Complex<Float>>(),
            );
            data = ArrayViewMut::from_shape_ptr(shape, ptr as *mut Complex<Float>);
        }

        FFTData3D { data: data }
    }
}

/// Automatically free memory, if goes out of scope.
impl<'a> Drop for FFTData3D<'a> {
    fn drop(&mut self) {
        unsafe { fftw_free(self.data.as_ptr() as *mut _) }
    }
}
