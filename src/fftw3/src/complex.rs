use std::ops::{Add, Sub, Mul, Div};
use num::Float;

/// Implement own complex representation in order to be conform with C99 memory
/// layout.
/// WARNING: Not guaranteed to has the same memory representation as [f64; 2],
/// but it is likely.
#[derive(Clone, Copy)]
pub struct Complex<T>(pub [T; 2]) where T: Float;

impl<T: Float> Complex<T> {
    /// Return a new complex number.
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex::<T>([re, im])
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

#[cfg(test)]
mod tests {
    use super::Complex;

    #[test]
    fn mul() {
        let a = Complex::<f64>([1.0f64, 2.0]);
        let b = Complex::<f64>([3.0f64, 4.0]);
        let Complex::<f64>(c) = a * b;
        assert_eq!(c, [-5., 10.]);
    }

    #[test]
    fn mul_scalar() {
        let a = Complex::<f64>([1.0f64, 2.0]);
        let Complex::<f64>(c) = a * 5.;
        assert_eq!(c, [5., 10.]);
        let Complex::<f64>(d) = 5. * a;
        assert_eq!(d, [5., 10.]);
    }

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
