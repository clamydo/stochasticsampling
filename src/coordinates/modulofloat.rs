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
//! let a = Mf64::new(-1.125, 1.0);
//! println!("{:?}", a);
//!
//! assert_eq!(*(a + 22.5).as_ref(), 0.375);
//! assert_eq!(*(a * -22.0).as_ref(), 0.75);
//! ```

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};


#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Mf64 {
    pub v: f64, // value
    pub m: f64, // divisor
}

/// WARNING! Unsafe because having a divisor `m <= 0` gives unwanted results.
/// Make sure, not to do that!
unsafe fn modulo(f: f64, m: f64) -> f64 {
    f - (f / m).floor() * m
}

impl Mf64 {
    /// Construct a modulo float from a float.
    pub fn new(f: f64, div: f64) -> Mf64 {
        if f.is_nan() || f.is_infinite() {
            panic!("Values of NAN or INF are not allowed.")
        }
        if div <= 0.0 {
            panic!("Divisor must not be <= 0.0.")
        }

        Mf64 {
            v: unsafe { modulo(f, div) },
            m: div,
        }
    }

    /// Construct a modulo float from a float.
    /// Unsafe because having NAN, INFINITE for `f` or a value `div <= 0` for
    /// the divisor will lead to unwanted results.
    pub unsafe fn unchecked_new(f: f64, div: f64) -> Mf64 {
        Mf64 {
            v: modulo(f, div),
            m: div,
        }
    }
}

/// Implement + operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b + c mod d = (a + c)  mod b`!
impl Add for Mf64 {
    type Output = Mf64;

    fn add(self, other: Mf64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v + other.v, self.m) }
    }
}

impl Add<f64> for Mf64 {
    type Output = Mf64;

    fn add(self, other: f64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v + other, self.m) }
    }
}

// Implement inplace adding a value
impl AddAssign for Mf64 {
    fn add_assign(&mut self, _rhs: Mf64) {
        self.v = unsafe { modulo(self.v + _rhs.v, self.m) };
    }
}

impl AddAssign<f64> for Mf64 {
    fn add_assign(&mut self, _rhs: f64) {
        self.v = unsafe { modulo(self.v + _rhs, self.m) };
    }
}

/// Implement - operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b - c mod d = (a - c)  mod b`!
impl Sub for Mf64 {
    type Output = Mf64;

    fn sub(self, rhs: Mf64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v - rhs.v, self.m) }
    }
}

impl Sub<f64> for Mf64 {
    type Output = Mf64;

    fn sub(self, rhs: f64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v - rhs, self.m) }
    }
}

/// Implement * operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b * c mod d = (a * c)  mod b`!
impl Mul for Mf64 {
    type Output = Mf64;

    fn mul(self, rhs: Mf64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v * rhs.v, self.m) }
    }
}

impl Mul<f64> for Mf64 {
    type Output = Mf64;

    fn mul(self, rhs: f64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v * rhs, self.m) }
    }
}

/// Implement / operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b / c mod d = (a / c)  mod b`!
impl Div for Mf64 {
    type Output = Mf64;

    fn div(self, rhs: Mf64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v / rhs.v, self.m) }
    }
}

impl Div<f64> for Mf64 {
    type Output = Mf64;

    fn div(self, rhs: f64) -> Mf64 {
        unsafe { Mf64::unchecked_new(self.v / rhs, self.m) }
    }
}


/// Implement `Into` trait for easy conversion.
/// Converts by value to an f64.
impl Into<f64> for Mf64 {
    fn into(self) -> f64 {
        self.v
    }
}

/// Implement `AsRef` for easy conversion.
/// Converts a Mf64 into reference of enclosed f64.
impl AsRef<f64> for Mf64 {
    fn as_ref(&self) -> &f64 {
        &self.v
    }
}

/// Implement `AsMut` for easy conversion.
/// Converts a Mf64 into mutable reference of enclosed f64.
impl AsMut<f64> for Mf64 {
    fn as_mut(&mut self) -> &mut f64 {
        &mut self.v
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::TestResult;
    use std::f64;
    use super::*;

    #[test]
    #[should_panic]
    fn new_nan_panic() {
        Mf64::new(f64::NAN, 1.);
    }

    #[test]
    #[should_panic]
    fn new_inf_panic() {
        Mf64::new(f64::INFINITY, 1.);
    }

    #[test]
    #[should_panic]
    fn new_neginf_panic() {
        Mf64::new(f64::NEG_INFINITY, 1.);
    }

    #[test]
    #[should_panic]
    fn new_negdiv_panic() {
        Mf64::new(f64::NEG_INFINITY, 0.);
        Mf64::new(f64::NEG_INFINITY, -1.);
    }

    #[test]
    fn new() {
        let input = [0., -0., 1., -1., -0.3];
        let output = [0., 0., 0., 0., 0.7];

        for (i, o) in input.into_iter().zip(output.into_iter()) {
            let a = Mf64::new(*i, 1.);
            assert!(a.v == *o, "a = {}, b ={}", a.v, *o);
        }
    }

    #[quickcheck]
    #[ignore]
    fn new_invariant_qc(f: f64) -> TestResult {
        if f > 1. || f < 0. {
            TestResult::discard()
        } else {
            let a = Mf64::new(f, 1.);
            TestResult::from_bool(a.v == f)
        }
    }

    #[quickcheck]
    #[ignore]
    fn new_range_qc(f: f64) -> bool {
        let a = Mf64::new(f, 1.);
        0. <= a.v && a.v < 1.
    }

    #[test]
    fn addition_assign() {
        let lhs = [0.3, 1., 0., 0., 1.];
        let rhs = [0.7, 1., 0., 1., 0.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);
            let b = Mf64::new(*r, 1.);
            a += b;

            assert!(0. <= a.v && a.v < 1., "a = {} mod {}", a.v, a.m);
            assert_eq!(a.v, 0.);
            assert_eq!(a.v, 0.);
        }
    }

    #[test]
    fn scalar_addition_assign() {
        let lhs = [0.3, 1., 0., 0., 1.];
        let rhs = [0.7, 1., 0., 1., 0.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);
            a += *r;

            assert!(0. <= a.v && a.v < 1.);
            assert_eq!(a.v, 0.);
            assert_eq!(a.v, 0.);
        }
    }

    #[test]
    fn addition() {
        let lhs = [0.3, 1., 0., 0., 1.];
        let rhs = [0.7, 1., 0., 1., 0.];

        assert_eq!(*((Mf64::new(-1.125, 1.) + 22.5).as_ref()), 0.375);

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let a = Mf64::new(*l, 1.);
            let b = Mf64::new(*r, 1.);
            let c = a + b;
            let d = a + *r;

            assert!(0. <= c.v && c.v < 1.,
                    "a = {:?}, b = {:?}, c = {:?}",
                    a,
                    b,
                    c);
            assert!(0. <= d.v && d.v < 1.,
                    "a = {:?}, r = {}, d = {:?}",
                    a,
                    *r,
                    d);
            assert_eq!(c.v, 0.);
            assert_eq!(d.v, 0.);
        }
    }

    #[quickcheck]
    #[ignore]
    fn addition_range_qc(lhs: f64, rhs: f64, range: f64) -> TestResult {
        if range <= 0.0 {
            TestResult::discard()
        } else {
            let a = Mf64::new(lhs, range);
            let b = Mf64::new(rhs, range);
            let c = a + b;
            TestResult::from_bool(0. <= c.v && c.v < range)
        }
    }

    #[test]
    fn substraction() {
        let lhs = [3.7, 0., 1., 1., 0.];
        let rhs = [6.7, 0., 1., 0., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let a = Mf64::new(*l, 1.);
            let b = Mf64::new(*r, 1.);
            let c = a - b;
            let d = a - *r;

            assert!(0. <= c.v && c.v < 1.,
                    "a = {:?}, b = {:?}, c = {:?}",
                    a,
                    b,
                    c);
            assert!(0. <= d.v && d.v < 1.,
                    "a = {:?}, r = {}, d = {:?}",
                    a,
                    *r,
                    d);
            assert_eq!(c.v, 0.);
            assert_eq!(d.v, 0.);
        }
    }

    #[quickcheck]
    #[ignore]
    fn substraction_range_qc(lhs: f64, rhs: f64) -> bool {
        let a = Mf64::new(lhs, 1.);
        let b = Mf64::new(rhs, 1.);
        let c = a - b;
        0. <= c.v && c.v < 1.
    }

    #[test]
    fn multiplication() {
        assert_eq!(*(Mf64::new(-1.125, 1.) * -22.0).as_ref(), 0.75);
    }

    #[quickcheck]
    #[ignore]
    fn multiplication_range_qc(lhs: f64, rhs: f64) -> bool {
        let a = Mf64::new(lhs, 1.);
        let b = a * rhs;
        0. <= b.v && b.v < 1.
    }
}
