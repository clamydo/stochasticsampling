use complex::Complex;
use ndarray::{ArrayViewMut, Ix};
use std::mem;

/// FFTW3 allocated `ArrayView` for 2D data.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation.
pub struct FFTData2D<'a> {
    pub data: ArrayViewMut<'a, Complex<f64>, (Ix, Ix)>,
}


impl<'a> FFTData2D<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: (Ix, Ix)) -> FFTData2D<'a> {
        let data;
        unsafe {
            let ptr = ::fftw3_ffi::fftw_malloc(shape.0 * shape.1 * mem::size_of::<Complex<f64>>());
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

/// FFTW3 allocated `ArrayView` for 2D vector field.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation. Since the Fourier transform needs
/// to be done component wise, the component-fields of the matrix are allocated
/// one after another in memory (). The first to axis iterate through the the
/// components.
#[derive(Debug)]
pub struct FFTData2DVectorField<'a> {
    pub data: ArrayViewMut<'a, Complex<f64>, (Ix, Ix, Ix)>,
}


impl<'a> FFTData2DVectorField<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: (Ix, Ix, Ix)) -> FFTData2DVectorField<'a> {
        let data;
        unsafe {
            let ptr = ::fftw3_ffi::fftw_malloc(shape.0 * shape.1 * shape.2 * shape.3 *
                                               mem::size_of::<Complex<f64>>());
            data = ArrayViewMut::from_shape_ptr(shape, ptr as *mut Complex<f64>);
        }

        FFTData2DVectorField { data: data }
    }
}

/// Automatically free memory, if goes out of scope.
impl<'a> Drop for FFTData2DVectorField<'a> {
    fn drop(&mut self) {
        unsafe { ::fftw3_ffi::fftw_free(self.data.as_ptr() as *mut _) }
    }
}

/// FFTW3 allocated `ArrayView` for 2D matrix field.
///
/// The memory is allocated by FFTW3 in order to align it properly. This is
/// needed for SIMD. See FFTW3 documentation. Since the Fourier transform needs
/// to be done component wise, the component-fields of the matrix are allocated
/// one after another in memory (). The first to axis iterate through the the
/// components.
#[derive(Debug)]
pub struct FFTData2DMatrixField<'a> {
    pub data: ArrayViewMut<'a, Complex<f64>, (Ix, Ix, Ix, Ix)>,
}


impl<'a> FFTData2DMatrixField<'a> {
    /// Return a new FFTData2 instance. Only considers the first
    pub fn new(shape: (Ix, Ix, Ix, Ix)) -> FFTData2DMatrixField<'a> {
        let data;
        unsafe {
            let ptr = ::fftw3_ffi::fftw_malloc(shape.0 * shape.1 * shape.2 * shape.3 *
                                               mem::size_of::<Complex<f64>>());
            data = ArrayViewMut::from_shape_ptr(shape, ptr as *mut Complex<f64>);
        }

        FFTData2DMatrixField { data: data }
    }
}

/// Automatically free memory, if goes out of scope.
impl<'a> Drop for FFTData2DMatrixField<'a> {
    fn drop(&mut self) {
        unsafe { ::fftw3_ffi::fftw_free(self.data.as_ptr() as *mut _) }
    }
}
