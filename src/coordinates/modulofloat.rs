//! This implements a positive, modulo 1 floating point type `Mf64`, based on
//! `f64`.
//!
//! All values are restricted to the interval [0,1). The type can only be
//! constructed by the
//! `new`-method, so that no invalid values can be created.
//!
//! # Examples
//! ```
//! use ::stochasticsampling::coordinates::modulofloat::Mf64;
//! let a = Mf64::new(-1.125);
//! println!("{:?}", a);
//!
//! assert_eq!(*(a + 22.5).as_ref(), 0.375);
//! assert_eq!(*(a * -22.0).as_ref(), 0.75);
//! ```

use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;


#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Mf64(f64);

impl Mf64 {
    /// Construct a modulo float from a float.
    pub fn new(f: f64) -> Mf64 {
        if f.is_nan() || f.is_infinite() {
            panic!()
        }

        if f >= 1. || f < 0. { Mf64(f - f.floor()) } else { Mf64(f) }
    }
}

/// Implement + operator with modulo 1. This represents the periodic boundary
/// condidtions. It implicitly assumes everyhing is normalised the size of the
/// box.
impl Add for Mf64 {
    type Output = Mf64;

    fn add(self, other: Mf64) -> Mf64 {
        let Mf64(a) = self;
        let Mf64(b) = other;
        Mf64::new(a + b)
    }
}

impl Add<f64> for Mf64 {
    type Output = Mf64;

    fn add(self, other: f64) -> Mf64 {
        let Mf64(a) = self;
        Mf64::new(a + other)
    }
}

/// Implement - operator with modulo 1. This represents the periodic boundary
/// condidtions. It implicitly assumes everyhing is normalised the size of the
/// box.
impl Sub for Mf64 {
    type Output = Mf64;

    fn sub(self, rhs: Mf64) -> Mf64 {
        let Mf64(a) = self;
        let Mf64(b) = rhs;
        Mf64::new(a - b)
    }
}

impl Sub<f64> for Mf64 {
    type Output = Mf64;

    fn sub(self, rhs: f64) -> Mf64 {
        let Mf64(a) = self;
        Mf64::new(a - rhs)
    }
}

/// Implement * operator with modulo 1. This represents the periodic boundary
/// condidtions. It implicitly assumes everyhing is normalised the size of the
/// box.
impl Mul for Mf64 {
    type Output = Mf64;

    fn mul(self, rhs: Mf64) -> Mf64 {
        let Mf64(a) = self;
        let Mf64(b) = rhs;
        Mf64::new(a * b)
    }
}

impl Mul<f64> for Mf64 {
    type Output = Mf64;

    fn mul(self, rhs: f64) -> Mf64 {
        let Mf64(a) = self;
        Mf64::new(a * rhs)
    }
}

/// Implement `From` trait for easy conversion.
/// Converts into Mf64 from a f64 by copying it.
impl From<f64> for Mf64 {
    fn from(float: f64) -> Self {
        Mf64::new(float)
    }
}

/// Implement `Into` trait for easy conversion.
/// Converts by value to an f64.
impl Into<f64> for Mf64 {
    fn into(self) -> f64 {
        self.0
    }
}

/// Implement `AsRef` for easy conversion.
/// Converts a Mf64 into reference of enclosed f64.
impl AsRef<f64> for Mf64 {
    fn as_ref(&self) -> &f64 {
        &self.0
    }
}

/// Implement `AsMut` for easy conversion.
/// Converts a Mf64 into mutable reference of enclosed f64.
impl AsMut<f64> for Mf64 {
    fn as_mut(&mut self) -> &mut f64 {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;
    use quickcheck::TestResult;

    #[test]
    #[should_panic]
    fn new_nan_panic() {
        Mf64::new(f64::NAN);
    }

    #[test]
    #[should_panic]
    fn new_inf_panic() {
        Mf64::new(f64::INFINITY);
    }

    #[test]
    #[should_panic]
    fn new_neginf_panic() {
        Mf64::new(f64::NEG_INFINITY);
    }

    #[test]
    fn new() {
        let input = [0., -0., 1., -1.];
        let output = [0., 0., 0., 0.];

        for (i, o) in input.into_iter().zip(output.into_iter()) {
            let Mf64(a) = Mf64::new(*i);
            assert!(a == *o, "a = {}, b ={}", a, *o);
        }
    }

    #[quickcheck]
    #[ignore]
    fn new_invariant_qc(f: f64) -> TestResult {
        if f > 1. || f < 0. {
            TestResult::discard()
        } else {
            let Mf64(a) = Mf64::new(f);
            TestResult::from_bool(a == f)
        }
    }

    #[quickcheck]
    #[ignore]
    fn new_range_qc(f: f64) -> bool {
        let Mf64(a) = Mf64::new(f);
        0. <= a && a < 1.
    }

    #[test]
    fn addition() {
        let lhs = [0.3, 1., 0., 0., 1.];
        let rhs = [0.7, 1., 0., 1., 0.];

        assert_eq!(*((Mf64::new(-1.125) + 22.5).as_ref()), 0.375);

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let a = Mf64::new(*l);
            let b = Mf64::new(*r);
            let Mf64(c) = a + b;
            let Mf64(d) = a + *r;

            assert!(0. <= c && c < 1., "a = {:?}, b = {:?}, c = {}", a, b, c);
            assert!(0. <= d && d < 1., "a = {:?}, r = {}, d = {}", a, *r, d);
            assert_eq!(c, 0.);
            assert_eq!(d, 0.);
        }
    }

    #[quickcheck]
    #[ignore]
    fn addition_range_qc(lhs: f64, rhs: f64) -> bool {
        let a = Mf64::new(lhs);
        let b = Mf64::new(rhs);
        let Mf64(c) = a + b;
        0. <= c && c < 1.
    }

    #[test]
    fn substraction() {
        let lhs = [3.7, 0., 1., 1., 0.];
        let rhs = [6.7, 0., 1., 0., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let a = Mf64::new(*l);
            let b = Mf64::new(*r);
            let Mf64(c) = a - b;
            let Mf64(d) = a - *r;

            assert!(0. <= c && c < 1., "a = {:?}, b = {:?}, c = {}", a, b, c);
            assert!(0. <= d && d < 1., "a = {:?}, r = {}, d = {}", a, *r, d);
            assert_eq!(c, 0.);
            assert_eq!(d, 0.);
        }
    }

    #[quickcheck]
    #[ignore]
    fn substraction_range_qc(lhs: f64, rhs: f64) -> bool {
        let a = Mf64::new(lhs);
        let b = Mf64::new(rhs);
        let Mf64(c) = a - b;
        0. <= c && c < 1.
    }

    #[test]
    fn multiplication() {
        assert_eq!(*(Mf64::new(-1.125) * -22.0).as_ref(), 0.75);
    }

    #[quickcheck]
    #[ignore]
    fn multiplication_range_qc(lhs: f64, rhs: f64) -> bool {
        let a = Mf64::new(lhs);
        let Mf64(b) = a * rhs;
        0. <= b && b < 1.
    }
}
