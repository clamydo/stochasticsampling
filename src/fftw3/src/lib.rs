extern crate ndarray;

mod fftw3_ffi;
pub mod fftw_ndarray;
pub mod fft;
pub mod complex;

#[cfg(test)]
mod tests {
    use std::f64::EPSILON;
    use fft;
    use fft::FFTPlan;
    use fftw_ndarray::FFTData2D;
    use complex::Complex64;

    #[test]
    fn fft_test() {
        let shape = (11usize, 11);
        let mut input = FFTData2D::new(shape);
        let mut fft = FFTData2D::new(shape);
        let mut ifft = FFTData2D::new(shape);

        // zero out input array
        input.data.map_inplace(|x| *x = Complex64([0., 0.]));

        // define a real rectangle in the middle of the array
        input.data[[4, 4]] = Complex64([1., 0.]);
        input.data[[4, 5]] = Complex64([1., 0.]);
        input.data[[4, 6]] = Complex64([1., 0.]);
        input.data[[5, 4]] = Complex64([1., 0.]);
        input.data[[5, 5]] = Complex64([1., 0.]);
        input.data[[5, 6]] = Complex64([1., 0.]);
        input.data[[6, 4]] = Complex64([1., 0.]);
        input.data[[6, 5]] = Complex64([1., 0.]);
        input.data[[6, 6]] = Complex64([1., 0.]);

        let plan_forward = FFTPlan::new_c2c(&mut input.data,
                                            &mut fft.data,
                                            fft::FFTDirection::Forward,
                                            fft::FFTFlags::Measure);
        let plan_backward = FFTPlan::new_c2c(&mut fft.data,
                                             &mut ifft.data,
                                             fft::FFTDirection::Backward,
                                             fft::FFTFlags::Measure);

        plan_forward.execute();
        plan_backward.execute();

        for i in input.data.iter().zip(ifft.data.iter()) {
            let (left, right) = i;

            let Complex64(diff) = *left - *right / (shape.0 * shape.1) as f64;
            assert!(diff[0].abs() <= EPSILON,
                    "Difference of real part: {:e} should be smaller than machine EPSILON = {:e}",
                    diff[0].abs(),
                    EPSILON);
            assert!(diff[1].abs() <= EPSILON,
                    "Difference of imaginary part: {:e} should be smaller than machine EPSILON = \
                     {:e}",
                    diff[1].abs(),
                    EPSILON);
        }
    }

}
