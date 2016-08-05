use std::mem;
use ndarray::{ArrayViewMut, Ix};
use super::complex::Complex64;

/// FFTW3 allocated `ArrayView` for 2D data.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation.
pub struct FFTData2D<'a> {
    pub data: ArrayViewMut<'a, Complex64, (Ix, Ix)>,
}


impl<'a> FFTData2D<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: (Ix, Ix)) -> FFTData2D<'a> {
        let data;
        unsafe {
            let ptrin = super::fftw3::fftw_malloc(shape.0 * shape.1 * mem::size_of::<Complex64>());
            data = ArrayViewMut::from_shape_ptr(shape, ptrin as *mut Complex64);
        }

        FFTData2D { data: data }
    }
}

/// Automatically free memory, if goes out of scope.
impl<'a> Drop for FFTData2D<'a> {
    fn drop(&mut self) {
        unsafe { super::fftw3::fftw_free(self.data.as_ptr() as *mut _) }
    }
}
