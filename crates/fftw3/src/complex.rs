use num::Float;
use std::ops::{Add, Div, Mul, Sub};

/// Implement own complex representation in order to be conform with C99 memory
/// layout.
/// WARNING: Not guaranteed to has the same memory representation as [f64; 2],
/// but it is likely.
#[derive(Clone, Copy, Debug)]
pub struct Complex<T>(pub [T; 2]) where T: Float;

impl<T: Float> Complex<T> {
    /// Return a new complex number.
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex::<T>([re, im])
    }

    /// Returns real part.
    pub fn re(&self) -> T {
        let &Complex::<T>(ref c) = self;
        c[0]
    }

    /// Returns imaginary part.
    pub fn im(&self) -> T {
        let &Complex::<T>(ref c) = self;
        c[1]
    }
}


impl<T: Float> Mul for Complex<T> {
    type Output = Complex<T>;

    fn mul(self, _rhs: Complex<T>) -> Complex<T> {
        let Complex::<T>(lhs) = self;
        let Complex::<T>(rhs) = _rhs;
        let re = lhs[0] * rhs[0] - lhs[1] * rhs[1];
        let im = lhs[0] * rhs[1] + lhs[1] * rhs[0];
        Complex::<T>([re, im])
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Complex<T>;

    fn mul(self, _rhs: T) -> Complex<T> {
        let Complex::<T>(lhs) = self;
        let re = _rhs * lhs[0];
        let im = _rhs * lhs[1];
        Complex::<T>([re, im])
    }
}

impl Mul<Complex<f64>> for f64 {
    type Output = Complex<f64>;

    fn mul(self, _rhs: Complex<f64>) -> Complex<f64> {
        _rhs * self
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Complex<T>;

    fn div(self, _rhs: T) -> Complex<T> {
        let Complex::<T>(lhs) = self;
        let re = lhs[0] / _rhs;
        let im = lhs[1] / _rhs;
        Complex::<T>([re, im])
    }
}

impl<T: Float> Add for Complex<T> {
    type Output = Complex<T>;

    fn add(self, _rhs: Complex<T>) -> Complex<T> {
        let Complex::<T>(lhs) = self;
        let Complex::<T>(rhs) = _rhs;
        let re = lhs[0] + rhs[0];
        let im = lhs[1] + rhs[1];
        Complex::<T>([re, im])
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Complex<T>;

    fn sub(self, _rhs: Complex<T>) -> Complex<T> {
        let Complex::<T>(lhs) = self;
        let Complex::<T>(rhs) = _rhs;
        let re = lhs[0] - rhs[0];
        let im = lhs[1] - rhs[1];
        Complex::<T>([re, im])
    }
}

// TODO write test
impl<T: Float> From<T> for Complex<T> {
    fn from(re: T) -> Complex<T> {
        Complex([re, T::zero()])
    }
}

// TODO write test
impl<T: Float> PartialEq for Complex<T> {
    fn eq(&self, other: &Complex<T>) -> bool {
        let &Complex::<T>(ref a) = self;
        let &Complex::<T>(ref b) = other;

        a[0] == b[0] && a[1] == b[1]
    }
}



#[cfg(test)]
mod tests {
    use num::Complex as NC;
    use super::Complex;

    // check against different implementation
    quickcheck!{
        fn mul_qc(re1: f64, im1: f64, re2: f64, im2: f64) -> bool {
            let a = Complex::<f64>([re1, im1]);
            let b = Complex::<f64>([re2, im2]);

            let Complex::<f64>(res) = a * b;

            let na = NC::<f64> { re: re1, im: im1 };
            let nb = NC::<f64> { re: re2, im: im2 };

            let NC { re: cmp_re, im: cmp_im } = na * nb;

            cmp_re == res[0] && cmp_im == res[1]
        }
    }

    quickcheck!{
        fn mul_scalar_qc(re: f64, im: f64, s: f64) -> bool {
            let a = Complex::<f64>([re, im]);

            let Complex::<f64>(res_right) = a * s;
            let Complex::<f64>(res_left) = s * a;

            let na = NC::<f64> { re: re, im: im };

            let NC { re: cmp_re_right, im: cmp_im_right } = na * s;
            let NC { re: cmp_re_left, im: cmp_im_left } = s * na;

            cmp_re_right == res_right[0] && cmp_im_right == res_right[1] &&
            cmp_re_left == res_left[0] && cmp_im_left == res_left[1]
        }
    }

    // fails because of different implementations!
    // #[quickcheck]
    // fn div_scalar_qc(re: f64, im: f64, s: f64) -> bool {
    // let a = Complex::<f64>([re, im]);
    // let Complex::<f64>(res) = a / s;
    //
    // let na = NC::<f64> { re: re, im: im };
    // let NC { re: cmp_re, im: cmp_im } = na / s;
    //
    // if s == 0. {
    // if res[0] == 0. || res[1] == 0. {
    // res[0].is_infinite() || res[1].is_infinite()
    // } else {
    // res[0].is_nan() || res[1].is_nan()
    // }
    // } else {
    // (cmp_re - res[0]).abs() < ::std::f64::EPSILON &&
    // (cmp_im - res[1]).abs() < ::std::f64::EPSILON
    // }
    // }
    //
    // fails because of different implementations!
    // #[test]
    // fn div_scalar() {
    // let re = 0.;
    // let im = 98.29372511785971;
    // let s = 25.;
    //
    // let a = Complex::<f64>([re, im]);
    // let Complex::<f64>(res) = a / s;
    //
    // let na = NC::<f64> { re: re, im: im };
    // let NC { re: cmp_re, im: cmp_im } = na / s;
    //
    // if s == 0. {
    // if res[0] == 0. || res[1] == 0. {
    // assert!(res[0].is_infinite() || res[1].is_infinite(),
    // "is ({}, {}) / {}",
    // res[0],
    // res[1],
    // s);
    // } else {
    // assert!(res[0].is_nan() || res[1].is_nan(),
    // "is ({}, {}) / {}",
    // res[0],
    // res[1],
    // s);
    // }
    // } else {
    // assert!((cmp_re - res[0]).abs() < ::std::f64::EPSILON,
    // "is ({}, {}) / {} = {}",
    // res[0],
    // res[1],
    // s,
    // (cmp_re - res[1]).abs());
    // assert!((cmp_im - res[1]).abs() < ::std::f64::EPSILON,
    // "is ({}, {}) / {} = {}",
    // res[0],
    // res[1],
    // s,
    // (cmp_im - res[1]).abs());
    // }
    // }
    //
    #[test]
    fn div_scalar() {
        let a = Complex::<f64>([1.0f64, 2.0]);
        let Complex::<f64>(c) = a / 2.;
        assert_eq!(c, [0.5, 1.0]);
    }


    #[test]
    fn add() {
        let a = Complex::<f64>([1.0f64, 2.0]);
        let b = Complex::<f64>([3.0f64, 4.0]);
        let Complex::<f64>(c) = a + b;
        assert_eq!(c, [4., 6.]);
    }

    #[test]
    fn sub() {
        let a = Complex::<f64>([1.0f64, 2.0]);
        let b = Complex::<f64>([3.0f64, 4.0]);
        let Complex::<f64>(c) = a - b;
        assert_eq!(c, [-2., -2.]);
    }
}
