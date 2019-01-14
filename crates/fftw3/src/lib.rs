#[cfg(test)]
#[macro_use]
extern crate quickcheck;

extern crate ndarray;
extern crate num_complex;

pub mod fft;
#[cfg(feature = "default")]
mod fftw3_ffi;
#[cfg(feature = "fftw-threaded")]
mod fftw3_threads_ffi;
pub mod fftw_ndarray;

#[cfg(test)]
mod tests {
    use fft;
    use fft::FFTPlan;
    use fftw_ndarray::{FFTData2D, FFTData3D};
    use ndarray;
    use num_complex::Complex;
    use std::f64::EPSILON;

    /// Transforming for and back should be an identity operation (except for
    /// normalization factors)
    /// WARNING: Not thread safe. Run with `env RUST_TEST_THREADS=1 cargo test`.
    #[test]
    fn test_fft_identity_2d() {
        let shape = ndarray::Dim([11usize, 11]);

        let mut input = FFTData2D::new(shape);
        let mut fft = FFTData2D::new(shape);
        let mut ifft = FFTData2D::new(shape);

        // zero out input array
        input.data.fill(Complex::new(0., 0.));

        // define a real rectangle in the middle of the array
        input.data[[4, 4]] = Complex::new(1., 0.);
        input.data[[4, 5]] = Complex::new(1., 0.);
        input.data[[4, 6]] = Complex::new(1., 0.);
        input.data[[5, 4]] = Complex::new(1., 0.);
        input.data[[5, 5]] = Complex::new(1., 0.);
        input.data[[5, 6]] = Complex::new(1., 0.);
        input.data[[6, 4]] = Complex::new(1., 0.);
        input.data[[6, 5]] = Complex::new(1., 0.);
        input.data[[6, 6]] = Complex::new(1., 0.);

        let plan_forward = FFTPlan::new_c2c_2d(
            &mut input.data,
            &mut fft.data,
            fft::FFTDirection::Forward,
            fft::FFTFlags::Measure,
        ).unwrap();
        let plan_backward = FFTPlan::new_c2c_2d(
            &mut fft.data,
            &mut ifft.data,
            fft::FFTDirection::Backward,
            fft::FFTFlags::Measure,
        ).unwrap();

        plan_forward.execute();
        plan_backward.execute();

        for i in input.data.iter().zip(ifft.data.iter()) {
            let (left, right) = i;

            // Since transform is not normalized, we have to divide by the number of
            // elements to get back the original input
            let diff = *left - *right / (shape[0] * shape[1]) as f64;

            // Compare input to identity operation. Compare to machine precision to take
            // numerical roundoff errors into account.
            assert!(
                diff.re.abs() <= EPSILON,
                "Difference of real parts: {:e} should be smaller than machine EPSILON = {:e}",
                diff.re.abs(),
                EPSILON
            );
            assert!(
                diff.im.abs() <= EPSILON,
                "Difference of imaginary parts: {:e} should be smaller than machine EPSILON \
                 = {:e}",
                diff.im.abs(),
                EPSILON
            );
        }
    }

    /// Transforming for and back should be an identity operation (except for
    /// normalization factors)
    /// WARNING: Not thread safe. Run with `env RUST_TEST_THREADS=1 cargo test`.
    #[test]
    fn test_fft_identity_3d() {
        let shape = ndarray::Dim([11usize, 11, 11]);

        let mut input = FFTData3D::new(shape);
        let mut fft = FFTData3D::new(shape);
        let mut ifft = FFTData3D::new(shape);

        // zero out input array
        input.data.fill(Complex::new(0., 0.));

        // define a real rectangle in the middle of the array
        input.data[[4, 4, 0]] = Complex::new(1., 0.);
        input.data[[4, 5, 0]] = Complex::new(1., 0.);
        input.data[[4, 6, 0]] = Complex::new(1., 0.);
        input.data[[5, 4, 0]] = Complex::new(1., 0.);
        input.data[[5, 5, 0]] = Complex::new(1., 0.);
        input.data[[5, 6, 0]] = Complex::new(1., 0.);
        input.data[[6, 4, 0]] = Complex::new(1., 0.);
        input.data[[6, 5, 0]] = Complex::new(1., 0.);
        input.data[[6, 6, 0]] = Complex::new(1., 0.);

        let plan_forward = FFTPlan::new_c2c_3d(
            &mut input.data,
            &mut fft.data,
            fft::FFTDirection::Forward,
            fft::FFTFlags::Measure,
        ).unwrap();
        let plan_backward = FFTPlan::new_c2c_3d(
            &mut fft.data,
            &mut ifft.data,
            fft::FFTDirection::Backward,
            fft::FFTFlags::Measure,
        ).unwrap();

        plan_forward.execute();
        plan_backward.execute();

        for i in input.data.iter().zip(ifft.data.iter()) {
            let (left, right) = i;

            // Since transform is not normalized, we have to divide by the number of
            // elements to get back the original input
            let diff = *left - *right / (shape[0] * shape[1] * shape[2]) as f64;

            // Compare input to identity operation. Compare to machine precision to take
            // numerical roundoff errors into account.
            assert!(
                diff.re.abs() <= EPSILON,
                "Difference of real parts: {:e} should be smaller than machine EPSILON = {:e}",
                diff.re.abs(),
                EPSILON
            );
            assert!(
                diff.im.abs() <= EPSILON,
                "Difference of imaginary parts: {:e} should be smaller than machine EPSILON \
                 = {:e}",
                diff.im.abs(),
                EPSILON
            );
        }
    }
}
