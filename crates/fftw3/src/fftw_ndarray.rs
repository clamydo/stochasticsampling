use complex::Complex;
use ndarray::{ArrayViewMut, Ix2};
use std::mem;

/// FFTW3 allocated `ArrayView` for 2D data.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation.
pub struct FFTData2D<'a> {
    pub data: ArrayViewMut<'a, Complex<f64>, Ix2>,
}


impl<'a> FFTData2D<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: Ix2) -> FFTData2D<'a> {
        let data;
        unsafe {
            let ptr = ::fftw3_ffi::fftw_malloc(shape[0] * shape[1] *
                                               mem::size_of::<Complex<f64>>());
            data = ArrayViewMut::from_shape_ptr(shape, ptr as *mut Complex<f64>);
        }

        FFTData2D { data: data }
    }
}

/// Automatically free memory, if goes out of scope.
impl<'a> Drop for FFTData2D<'a> {
    fn drop(&mut self) {
        unsafe { ::fftw3_ffi::fftw_free(self.data.as_ptr() as *mut _) }
    }
}
