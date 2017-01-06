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

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::Visitor;
use std::f64;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// This structure represents a modulo type, with modulus m.
#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Mf64 {
    /// value
    pub v: f64,
    /// divisor/modulus
    pub m: f64,
}

// Restricts value `f` to interval of size `[0,m)`. Additional modulo %
// operation to
// prevent edge case documented in the test.
/// WARNING! Unsafe because having a divisor `m <= 0` gives unwanted results.
/// Make sure, not to do that!
fn modulo(f: f64, m: f64) -> f64 {
    (f - (f / m).floor() * m) % m
}

impl Mf64 {
    /// Construct a modulo float from a float.
    /// Unsafe because having NAN, INFINITE for `f` or a value `div <= 0` for
    /// the divisor will lead to unwanted results.
    fn unchecked_new(f: f64, div: f64) -> Mf64 {
        Mf64 {
            v: modulo(f, div),
            m: div,
        }
    }

    /// Construct a modulo float from a float.
    pub fn new(f: f64, div: f64) -> Mf64 {
        if f.is_nan() || f.is_infinite() {
            panic!("Values of NAN or INF are not allowed.")
        }
        if div <= 0.0 {
            panic!("Divisor must not be <= 0.0.")
        }

        Mf64::unchecked_new(f, div)
    }
}

/// Implement + operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b + c mod d = (a + c)  mod b`!
impl Add for Mf64 {
    type Output = Mf64;

    fn add(self, other: Mf64) -> Mf64 {
        Mf64::unchecked_new(self.v + other.v, self.m)
    }
}

impl Add<f64> for Mf64 {
    type Output = Mf64;

    fn add(self, other: f64) -> Mf64 {
        Mf64::unchecked_new(self.v + other, self.m)
    }
}

// Implement inplace adding a value
impl AddAssign for Mf64 {
    fn add_assign(&mut self, _rhs: Mf64) {
        self.v = modulo(self.v + _rhs.v, self.m);
    }
}

impl AddAssign<f64> for Mf64 {
    fn add_assign(&mut self, _rhs: f64) {
        self.v = modulo(self.v + _rhs, self.m);
    }
}

/// Implement - operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b - c mod d = (a - c)  mod b`!
impl Sub for Mf64 {
    type Output = Mf64;

    fn sub(self, rhs: Mf64) -> Mf64 {
        Mf64::unchecked_new(self.v - rhs.v, self.m)
    }
}

impl Sub<f64> for Mf64 {
    type Output = Mf64;

    fn sub(self, rhs: f64) -> Mf64 {
        Mf64::unchecked_new(self.v - rhs, self.m)
    }
}

// Implement inplace subtraction
impl SubAssign for Mf64 {
    fn sub_assign(&mut self, _rhs: Mf64) {
        self.v = modulo(self.v - _rhs.v, self.m);
    }
}

impl SubAssign<f64> for Mf64 {
    fn sub_assign(&mut self, _rhs: f64) {
        self.v = modulo(self.v - _rhs, self.m);
    }
}

/// Implement * operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b * c mod d = (a * c)  mod b`!
impl Mul for Mf64 {
    type Output = Mf64;

    fn mul(self, rhs: Mf64) -> Mf64 {
        Mf64::unchecked_new(self.v * rhs.v, self.m)
    }
}

impl Mul<f64> for Mf64 {
    type Output = Mf64;

    fn mul(self, rhs: f64) -> Mf64 {
        Mf64::unchecked_new(self.v * rhs, self.m)
    }
}

// Implement inplace multiplication
impl MulAssign for Mf64 {
    fn mul_assign(&mut self, _rhs: Mf64) {
        self.v = modulo(self.v * _rhs.v, self.m);
    }
}

impl MulAssign<f64> for Mf64 {
    fn mul_assign(&mut self, _rhs: f64) {
        self.v = modulo(self.v * _rhs, self.m);
    }
}

/// Implement / operator for modulo. **Caution**: This operation is not
/// commutative: `a mod b / c mod d = (a / c)  mod b`!
impl Div for Mf64 {
    type Output = Mf64;

    fn div(self, rhs: Mf64) -> Mf64 {
        Mf64::unchecked_new(self.v / rhs.v, self.m)
    }
}

impl Div<f64> for Mf64 {
    type Output = Mf64;

    fn div(self, rhs: f64) -> Mf64 {
        Mf64::unchecked_new(self.v / rhs, self.m)
    }
}

// Implement inplace multiplication
impl DivAssign for Mf64 {
    fn div_assign(&mut self, _rhs: Mf64) {
        self.v = modulo(self.v / _rhs.v, self.m);
    }
}

impl DivAssign<f64> for Mf64 {
    fn div_assign(&mut self, _rhs: f64) {
        self.v = modulo(self.v / _rhs, self.m);
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

/// Implement serde's Serialize in order to serialize only to a float.
/// This skips the modulo quotient.
impl Serialize for Mf64 {
    fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
        where S: Serializer
    {
        serializer.serialize_f64((*self).v)
    }
}

/// Implement a custom Deserialize trait, that deserializes a value to a Mf64
/// with default modulo quotient.
/// WARNING: Modulo quotient of zero means, that this is just a float, without
/// wrapping! This means, `Mf64` can also have a negative value for `v`!
impl Deserialize for Mf64 {
    fn deserialize<D>(deserializer: &mut D) -> Result<Mf64, D::Error>
        where D: Deserializer
    {
        struct F64Visitor;

        impl Visitor for F64Visitor {
            type Value = f64;

            fn visit_f64<E>(&mut self, value: f64) -> Result<f64, E>
                where E: ::serde::de::Error
            {
                Ok(value as f64)
            }
        }

        match deserializer.deserialize_f64(F64Visitor) {
            Err(e) => Err(e),
            Ok(v) => {
                Ok(Mf64 {
                    v: v,
                    m: f64::default(),
                })
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use quickcheck::TestResult;
    use std::f64;
    use super::*;
    use test;
    use test::Bencher;

    #[test]
    fn modulo_test() {
        let input = [[2. * ::std::f64::consts::PI, 2. * ::std::f64::consts::PI],
                     [-4.440892098500626e-16, 2. * ::std::f64::consts::PI]];
        let output = [0., 0.];

        for (i, o) in input.iter().zip(output.iter()) {
            let a = modulo(i[0], i[1]);
            assert!(a == *o,
                    "in: {} mod {}, out: {}, expected: {}",
                    i[0],
                    i[1],
                    a,
                    *o);
        }
    }

    #[bench]
    fn bench_modulo(b: &mut Bencher) {
        let m = test::black_box(1.);
        b.iter(|| for i in 1..1000 {
            modulo(i as f64, m);
        });
    }

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
            assert!(a.v == *o,
                    "in = {} mod 1, got: {}, expected ={}",
                    *i,
                    a.v,
                    *o);
        }
    }

    #[ignore]
    quickcheck!{
        fn new_invariant_qc(f: f64) -> TestResult {
            if f > 1. || f < 0. {
                TestResult::discard()
            } else {
                let a = Mf64::new(f, 1.);
                TestResult::from_bool(a.v == f)
            }
        }
    }

    #[ignore]
    quickcheck!{
        fn new_range_qc(f: f64) -> bool {
            let a = Mf64::new(f, 1.);
            0. <= a.v && a.v < 1.
        }
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

    #[ignore]
    quickcheck!{
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
    }

    #[test]
    fn subtraction() {
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

    #[test]
    fn subtraction_assign() {
        let lhs = [3.7, 0., 1., 1., 0.];
        let rhs = [6.7, 0., 1., 0., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);
            let b = Mf64::new(*r, 1.);

            a -= b;

            assert!(0. <= a.v && a.v < 1., "a = {:?}, b = {:?}", a, b);
            assert_eq!(a.v, 0.);
        }
    }

    #[test]
    fn subtraction_scalar_assign() {
        let lhs = [3.7, 0., 1., 1., 0.];
        let rhs = [6.7, 0., 1., 0., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);
            a -= *r;

            assert!(0. <= a.v && a.v < 1., "a = {:?}", a);
            assert_eq!(a.v, 0.);
        }
    }

    #[ignore]
    quickcheck!{
        fn subtraction_range_qc(lhs: f64, rhs: f64) -> bool {
            let a = Mf64::new(lhs, 1.);
            let b = Mf64::new(rhs, 1.);
            let c = a - b;
            0. <= c.v && c.v < 1.
        }
    }

    #[test]
    fn multiplication() {
        assert_eq!(*(Mf64::new(-1.125, 1.) * -22.0).as_ref(), 0.75);
    }

    #[test]
    fn multiplication_assign() {
        let lhs = [1., 0., 1.];
        let rhs = [2., 2., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);
            let b = Mf64::new(*r, 1.);

            a *= b;

            assert!(0. <= a.v && a.v < 1., "a = {:?}, b = {:?}", a, b);
            assert_eq!(a.v, 0.);
        }
    }

    #[test]
    fn multiplication_scalar_assign() {
        let lhs = [1., 0., 1.];
        let rhs = [2., 2., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);

            a *= *r;

            assert!(0. <= a.v && a.v < 1., "a = {:?}", a);
            assert_eq!(a.v, 0.);
        }
    }

    #[ignore]
    quickcheck!{
        fn multiplication_range_qc(lhs: f64, rhs: f64) -> bool {
            let a = Mf64::new(lhs, 1.);
            let b = a * rhs;
            0. <= b.v && b.v < 1.
        }
    }

    #[test]
    fn division_assign() {
        let lhs = [0.];
        let rhs = [0.5];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);
            let b = Mf64::new(*r, 1.);

            a /= b;

            assert!(0. <= a.v && a.v < 1., "a = {:?}, b = {:?}", a, b);
            assert_eq!(a.v, 0.);
        }
    }

    #[test]
    fn division_scalar_assign() {
        let lhs = [2., 0., 1.];
        let rhs = [2., 2., 1.];

        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            let mut a = Mf64::new(*l, 1.);

            a /= *r;

            assert!(0. <= a.v && a.v < 1., "a = {:?}", a);
            assert_eq!(a.v, 0.);
        }
    }
}
