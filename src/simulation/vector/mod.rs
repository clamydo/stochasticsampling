pub mod vorticity;

use std::marker::PhantomData;
use std::ops::{Add, Mul};

#[derive(Debug, PartialEq)]
pub struct Vector<T> {
    x: f64,
    y: f64,
    z: f64,
    t: PhantomData<T>,
}

impl<T> Add for Vector<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vector::<T>{
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            t: PhantomData,
        }
    }
}

impl<T> Mul<Vector<T>> for Vector<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Vector::<T>{
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
            t: PhantomData,
        }
    }
}
