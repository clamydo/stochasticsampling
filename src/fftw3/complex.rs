use std::ops::{Add, Mul};

/// Implement own complex representation in order to be conform with C99 memory
/// layout.
/// WARNING: Not guaranteed to has the same memory representation as [f64; 2]
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

#[cfg(test)]
mod tests {
    use super::Complex64;

    #[test]
    fn add() {
        let a = Complex64([1.0f64, 2.0]);
        let b = Complex64([3.0f64, 4.0]);
        let Complex64(c) = a + b;
        assert_eq!(c, [4., 6.]);
    }

    #[test]
    fn mul() {
        let a = Complex64([1.0f64, 2.0]);
        let b = Complex64([3.0f64, 4.0]);
        let Complex64(c) = a * b;
        assert_eq!(c, [-5., 10.]);
    }
}
