use std::ops::{Add, Sub, Mul, Div};

/// Implement own complex representation in order to be conform with C99 memory
/// layout.
/// WARNING: Not guaranteed to has the same memory representation as [f64; 2]
#[derive(Clone, Copy)]
pub struct Complex64(pub [f64; 2]);


impl Mul for Complex64 {
    type Output = Complex64;

    fn mul(self, _rhs: Complex64) -> Complex64 {
        let Complex64(lhs) = self;
        let Complex64(rhs) = _rhs;
        let re = lhs[0] * rhs[0] - lhs[1] * rhs[1];
        let im = lhs[0] * rhs[1] + lhs[1] * rhs[0];
        Complex64([re, im])
    }
}

impl Mul<f64> for Complex64 {
    type Output = Complex64;

    fn mul(self, _rhs: f64) -> Complex64 {
        let Complex64(lhs) = self;
        let re = _rhs * lhs[0];
        let im = _rhs * lhs[1];
        Complex64([re, im])
    }
}

impl Mul<Complex64> for f64 {
    type Output = Complex64;

    fn mul(self, _rhs: Complex64) -> Complex64 {
        _rhs * self
    }
}

impl Div<f64> for Complex64 {
    type Output = Complex64;

    fn div(self, _rhs: f64) -> Complex64 {
        let Complex64(lhs) = self;
        let re = lhs[0] / _rhs;
        let im = lhs[1] / _rhs;
        Complex64([re, im])
    }
}

impl Add for Complex64 {
    type Output = Complex64;

    fn add(self, _rhs: Complex64) -> Complex64 {
        let Complex64(lhs) = self;
        let Complex64(rhs) = _rhs;
        let re = lhs[0] + rhs[0];
        let im = lhs[1] + rhs[1];
        Complex64([re, im])
    }
}

impl Sub for Complex64 {
    type Output = Complex64;

    fn sub(self, _rhs: Complex64) -> Complex64 {
        let Complex64(lhs) = self;
        let Complex64(rhs) = _rhs;
        let re = lhs[0] - rhs[0];
        let im = lhs[1] - rhs[1];
        Complex64([re, im])
    }
}

#[cfg(test)]
mod tests {
    use super::Complex64;

    #[test]
    fn mul() {
        let a = Complex64([1.0f64, 2.0]);
        let b = Complex64([3.0f64, 4.0]);
        let Complex64(c) = a * b;
        assert_eq!(c, [-5., 10.]);
    }

    #[test]
    fn mul_scalar() {
        let a = Complex64([1.0f64, 2.0]);
        let Complex64(c) = a * 5.;
        assert_eq!(c, [5., 10.]);
        let Complex64(d) = 5. * a;
        assert_eq!(d, [5., 10.]);
    }

    #[test]
    fn div_scalar() {
        let a = Complex64([1.0f64, 2.0]);
        let Complex64(c) = a / 2.;
        assert_eq!(c, [0.5, 1.0]);
    }

    #[test]
    fn add() {
        let a = Complex64([1.0f64, 2.0]);
        let b = Complex64([3.0f64, 4.0]);
        let Complex64(c) = a + b;
        assert_eq!(c, [4., 6.]);
    }

    #[test]
    fn sub() {
        let a = Complex64([1.0f64, 2.0]);
        let b = Complex64([3.0f64, 4.0]);
        let Complex64(c) = a - b;
        assert_eq!(c, [-2., -2.]);
    }
}
