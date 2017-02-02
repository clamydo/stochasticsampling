use complex::Complex;
use ndarray::{ArrayViewMut, Ix2};
use std;
use std::sync::Arc;

pub type FFTWComplex = ::fftw3_ffi::fftw_complex;

/// Wrapper to manage the state of an FFTW3 plan.
pub struct FFTPlan {
    plan: Arc<::fftw3_ffi::fftw_plan>,
}

unsafe impl Send for FFTPlan {}

pub enum FFTDirection {
    Forward = ::fftw3_ffi::FFTW_FORWARD as isize,
    Backward = ::fftw3_ffi::FFTW_BACKWARD as isize,
}

pub enum FFTFlags {
    Estimate = ::fftw3_ffi::FFTW_ESTIMATE as isize,
    Measure = ::fftw3_ffi::FFTW_MEASURE as isize,
}

impl FFTPlan {
    /// Create a new FFTW3 complex to complex plan.
    /// INFO: According to FFTW3 documentation r2c/c2r can be more efficient.
    /// WARNING: This is an unormalized transformation. A forwards and
    /// backwards transformation will lead to input data scaled by the number
    /// of elements.
    pub fn new_c2c(ina: &mut ArrayViewMut<Complex<f64>, Ix2>,
                   outa: &mut ArrayViewMut<Complex<f64>, Ix2>,
                   direction: FFTDirection,
                   flags: FFTFlags)
                   -> Option<FFTPlan> {

        let (n0, n1) = ina.dim();
        let inp = ina.as_ptr() as *mut FFTWComplex;
        let outp = outa.as_ptr() as *mut FFTWComplex;

        let plan;
        // WARNING: Not thread safe!
        unsafe {
            plan = ::fftw3_ffi::fftw_plan_dft_2d(n0 as std::os::raw::c_int,
                                                 n1 as std::os::raw::c_int,
                                                 inp,
                                                 outp,
                                                 direction as std::os::raw::c_int,
                                                 flags as std::os::raw::c_uint);
        }

        if plan.is_null() {
            None
        } else {
            Some(FFTPlan { plan: Arc::new(plan) })
        }
    }

    /// Create a new FFTW3 complex to complex plan for an inplace
    /// transformation.
    /// WARNING: This is an unormalized transformation. A forwards and
    /// backwards transformation will lead to input data scaled by the number
    /// of elements.
    /// TODO: Write test. Return Result, not Option.
    pub fn new_c2c_inplace(arr: &mut ArrayViewMut<Complex<f64>, Ix2>,
                           direction: FFTDirection,
                           flags: FFTFlags)
                           -> Option<FFTPlan> {

        let (n0, n1) = arr.dim();
        let p = arr.as_ptr() as *mut FFTWComplex;

        let plan;
        unsafe {
            plan = ::fftw3_ffi::fftw_plan_dft_2d(n0 as std::os::raw::c_int,
                                                 n1 as std::os::raw::c_int,
                                                 p,
                                                 p,
                                                 direction as std::os::raw::c_int,
                                                 flags as std::os::raw::c_uint);
        }

        if plan.is_null() {
            None
        } else {
            Some(FFTPlan { plan: Arc::new(plan) })
        }
    }


    /// Execute FFTW# plan for associated given input and output.
    pub fn execute(&self) {
        unsafe { ::fftw3_ffi::fftw_execute(*self.plan) }
    }

    /// Reuse plan for different arrays
    /// TODO: Write test.
    pub fn reexecute(&self,
                     ina: &mut ArrayViewMut<Complex<f64>, Ix2>,
                     outa: &mut ArrayViewMut<Complex<f64>, Ix2>) {

        let inp = ina.as_ptr() as *mut FFTWComplex;
        let outp = outa.as_ptr() as *mut FFTWComplex;
        unsafe {
            ::fftw3_ffi::fftw_execute_dft(*self.plan, inp, outp);
        }
    }
}

/// Automatically destroy FFTW3 plan, when going out of scope.
impl Drop for FFTPlan {
    fn drop(&mut self) {
        unsafe {
            ::fftw3_ffi::fftw_destroy_plan(*self.plan);
        }
    }
}
