pub mod vorticity;

use std::convert::From;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

pub struct Default();
pub type VectorD = Vector<Default>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vector<T> {
    pub v: [f64; 3],
    t: PhantomData<T>,
}

impl<T> Vector<T> {
    pub fn iter(&self) -> ::std::slice::Iter<f64> {
        self.v.iter()
    }

    pub fn to<D>(self) -> Vector<D> {
        Vector::<D> {
            v: self.v,
            t: PhantomData,
        }
    }

    pub fn dot<D>(&self, rhs: &Vector<D>) -> f64 {
        self.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
    }
}

impl<T, R> Add<Vector<R>> for Vector<T> {
    type Output = Self;

    fn add(self, other: Vector<R>) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]].into()
    }
}

impl<'a, T, R> Add<&'a Vector<R>> for Vector<T> {
    type Output = Self;

    fn add(self, other: &Vector<R>) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]].into()
    }
}

impl<T, R> AddAssign<Vector<R>> for Vector<T> {
    fn add_assign(&mut self, rhs: Vector<R>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s += o;
        }
    }
}

impl<T, R> Sub<Vector<R>> for Vector<T> {
    type Output = Self;

    fn sub(self, other: Vector<R>) -> Self {
        [self[0] - other[0], self[1] - other[1], self[2] - other[2]].into()
    }
}

impl<'a, T, R> Sub<&'a Vector<R>> for Vector<T> {
    type Output = Self;

    fn sub(self, other: &Vector<R>) -> Self {
        [self[0] - other[0], self[1] - other[1], self[2] - other[2]].into()
    }
}

impl<T, R> SubAssign<Vector<R>> for Vector<T> {
    fn sub_assign(&mut self, rhs: Vector<R>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s -= o;
        }
    }
}

impl<T, R> Mul<Vector<R>> for Vector<T> {
    type Output = Self;

    fn mul(self, rhs: Vector<R>) -> Self {
        [self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2]].into()
    }
}

impl<'a, T, R> Mul<&'a Vector<R>> for Vector<T> {
    type Output = Self;

    fn mul(self, rhs: &Vector<R>) -> Self {
        [self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2]].into()
    }
}

impl<T, R> MulAssign<Vector<R>> for Vector<T> {
    fn mul_assign(&mut self, rhs: Vector<R>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s *= o;
        }
    }
}

impl<T> Mul<f64> for Vector<T> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        [self[0] * rhs, self[1] * rhs, self[2] * rhs].into()
    }
}

impl<T> MulAssign<f64> for Vector<T> {
    fn mul_assign(&mut self, rhs: f64) {
        self.v.iter_mut().for_each(|v| *v *= rhs);
    }
}

impl<T, R> Div<Vector<R>> for Vector<T> {
    type Output = Self;

    fn div(self, rhs: Vector<R>) -> Self {
        [self[0] / rhs[0], self[1] / rhs[1], self[2] / rhs[2]].into()
    }
}

impl<'a, T, R> Div<&'a Vector<R>> for Vector<T> {
    type Output = Self;

    fn div(self, rhs: &Vector<R>) -> Self {
        [self[0] / rhs[0], self[1] / rhs[1], self[2] / rhs[2]].into()
    }
}

impl<T, R> DivAssign<Vector<R>> for Vector<T> {
    fn div_assign(&mut self, rhs: Vector<R>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s /= o;
        }
    }
}

impl<T> Div<f64> for Vector<T> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        [self[0] / rhs, self[1] / rhs, self[2] / rhs].into()
    }
}

impl<T> DivAssign<f64> for Vector<T> {
    fn div_assign(&mut self, rhs: f64) {
        self.v.iter_mut().for_each(|v| *v /= rhs);
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.v[index]
    }
}

impl<T> From<[f64; 3]> for Vector<T> {
    fn from(a: [f64; 3]) -> Self {
        Vector::<T> {
            v: a,
            t: PhantomData,
        }
    }
}

impl<T> From<Vector<T>> for [f64; 3] {
    fn from(a: Vector<T>) -> [f64; 3] {
        a.v
    }
}
