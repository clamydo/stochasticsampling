pub mod vorticity;

use std::convert::From;
use std::iter::Iterator;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};
use std::default::Default as StdDefault;

#[derive(Clone, Copy)]
pub struct Default();
pub type VectorD = Vector<Default>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vector<T> {
    pub v: [f32; 3],
    t: PhantomData<T>,
}

impl<T> Vector<T> {
    pub fn iter(&self) -> ::std::slice::Iter<f32> {
        self.v.iter()
    }

    pub fn to<D>(self) -> Vector<D> {
        Vector::<D> {
            v: self.v,
            t: PhantomData,
        }
    }

    pub fn convert<D>(&self) -> Vector<D> {
        Vector::<D> {
            v: self.v,
            t: PhantomData,
        }
    }

    pub fn dot<D>(&self, rhs: &Vector<D>) -> f32 {
        self.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn zero() -> Vector<T> {
        Vector::<T> {
            v: [0., 0., 0.],
            t: PhantomData,
        }
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

impl<T> Mul<f32> for Vector<T> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        [self[0] * rhs, self[1] * rhs, self[2] * rhs].into()
    }
}

impl<T> MulAssign<f32> for Vector<T> {
    fn mul_assign(&mut self, rhs: f32) {
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

impl<T> Div<f32> for Vector<T> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        [self[0] / rhs, self[1] / rhs, self[2] / rhs].into()
    }
}

impl<T> DivAssign<f32> for Vector<T> {
    fn div_assign(&mut self, rhs: f32) {
        self.v.iter_mut().for_each(|v| *v /= rhs);
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.v[index]
    }
}

impl<T> From<[f32; 3]> for Vector<T> {
    fn from(a: [f32; 3]) -> Self {
        Vector::<T> {
            v: a,
            t: PhantomData,
        }
    }
}

impl<T> From<Vector<T>> for [f32; 3] {
    fn from(a: Vector<T>) -> [f32; 3] {
        a.v
    }
}

impl<T> StdDefault for Vector<T> {
    fn default() -> Self {
        Vector::<T>::zero()
    }
}
