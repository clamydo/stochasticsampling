pub mod vorticity;

use crate::Float;
use num_traits::identities::Zero;
use num_traits::{NumAssignOps, NumOps};
use std::convert::From;
use std::default::Default as StdDefault;
use std::iter::Iterator;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy)]
pub struct Default();
pub type NumVectorD<N> = NumVector<Default, N>;
pub type Vector<T> = NumVector<T, Float>;
pub type VectorD = Vector<Default>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    pub v: [N; 3],
    t: PhantomData<T>,
}

impl<T, N> NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    pub fn iter(&self) -> ::std::slice::Iter<N> {
        self.v.iter()
    }

    pub fn to<D>(self) -> NumVector<D, N> {
        NumVector::<D, N> {
            v: self.v,
            t: PhantomData,
        }
    }

    pub fn convert<D>(&self) -> NumVector<D, N> {
        NumVector::<D, N> {
            v: self.v,
            t: PhantomData,
        }
    }

    pub fn dot<D>(&self, rhs: &NumVector<D, N>) -> N {
        self.iter().zip(rhs.iter()).map(|(a, b)| (*a) * (*b)).sum()
    }

    pub fn zero() -> NumVector<T, N> {
        NumVector::<T, N> {
            v: [N::zero(), N::zero(), N::zero()],
            t: PhantomData,
        }
    }
}

impl<T, R, N> Add<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn add(self, other: NumVector<R, N>) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]].into()
    }
}

impl<'a, T, R, N> Add<&'a NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn add(self, other: &NumVector<R, N>) -> Self {
        [self[0] + other[0], self[1] + other[1], self[2] + other[2]].into()
    }
}

impl<T, R, N> AddAssign<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn add_assign(&mut self, rhs: NumVector<R, N>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s += *o;
        }
    }
}

impl<T, N> Add<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn add(self, other: N) -> Self {
        [self[0] + other, self[1] + other, self[2] + other].into()
    }
}

impl<T, N> AddAssign<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn add_assign(&mut self, rhs: N) {
        for s in &mut self.v {
            *s += rhs;
        }
    }
}

impl<T, R, N> Sub<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn sub(self, other: NumVector<R, N>) -> Self {
        [self[0] - other[0], self[1] - other[1], self[2] - other[2]].into()
    }
}

impl<'a, T, R, N> Sub<&'a NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn sub(self, other: &NumVector<R, N>) -> Self {
        [self[0] - other[0], self[1] - other[1], self[2] - other[2]].into()
    }
}

impl<T, R, N> SubAssign<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn sub_assign(&mut self, rhs: NumVector<R, N>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s -= *o;
        }
    }
}

impl<T, N> Sub<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn sub(self, other: N) -> Self {
        [self[0] - other, self[1] - other, self[2] - other].into()
    }
}

impl<T, N> SubAssign<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn sub_assign(&mut self, rhs: N) {
        for s in &mut self.v {
            *s -= rhs;
        }
    }
}

impl<T, R, N> Mul<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn mul(self, rhs: NumVector<R, N>) -> Self {
        [self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2]].into()
    }
}

impl<'a, T, R, N> Mul<&'a NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn mul(self, rhs: &NumVector<R, N>) -> Self {
        [self[0] * rhs[0], self[1] * rhs[1], self[2] * rhs[2]].into()
    }
}

impl<T, R, N> MulAssign<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn mul_assign(&mut self, rhs: NumVector<R, N>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s *= *o;
        }
    }
}

impl<T, N> Mul<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn mul(self, rhs: N) -> Self {
        [self[0] * rhs, self[1] * rhs, self[2] * rhs].into()
    }
}

impl<T, N> MulAssign<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn mul_assign(&mut self, rhs: N) {
        self.v.iter_mut().for_each(|v| *v *= rhs);
    }
}

impl<T, R, N> Div<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn div(self, rhs: NumVector<R, N>) -> Self {
        [self[0] / rhs[0], self[1] / rhs[1], self[2] / rhs[2]].into()
    }
}

impl<'a, T, R, N> Div<&'a NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn div(self, rhs: &NumVector<R, N>) -> Self {
        [self[0] / rhs[0], self[1] / rhs[1], self[2] / rhs[2]].into()
    }
}

impl<T, R, N> DivAssign<NumVector<R, N>> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn div_assign(&mut self, rhs: NumVector<R, N>) {
        for (s, o) in izip!(&mut self.v, &rhs.v) {
            *s /= *o;
        }
    }
}

impl<T, N> Div<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = Self;

    fn div(self, rhs: N) -> Self {
        [self[0] / rhs, self[1] / rhs, self[2] / rhs].into()
    }
}

impl<T, N> DivAssign<N> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn div_assign(&mut self, rhs: N) {
        self.v.iter_mut().for_each(|v| *v /= rhs);
    }
}

impl<T, N> Index<usize> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        &self.v[index]
    }
}

impl<T, N> From<[N; 3]> for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn from(a: [N; 3]) -> Self {
        NumVector::<T, N> {
            v: a,
            t: PhantomData,
        }
    }
}

impl<T> From<NumVector<T, Float>> for [Float; 3] {
    fn from(a: NumVector<T, Float>) -> [Float; 3] {
        a.v
    }
}

impl<T> From<NumVector<T, usize>> for [usize; 3] {
    fn from(a: NumVector<T, usize>) -> [usize; 3] {
        a.v
    }
}

impl<T> From<NumVector<T, i32>> for [i32; 3] {
    fn from(a: NumVector<T, i32>) -> [i32; 3] {
        a.v
    }
}

impl<T, N> StdDefault for NumVector<T, N>
where
    N: Copy + NumOps + NumAssignOps + Zero + Sum,
{
    fn default() -> Self {
        NumVector::<T, N>::zero()
    }
}
