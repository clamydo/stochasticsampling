#[cfg(test)]
#[cfg(feature = "single")]
use std::f32::{EPSILON, MAX};

#[cfg(test)]
#[cfg(not(feature = "single"))]
use std::f64::{EPSILON, MAX};

#[cfg(test)]
use crate::Float;

#[cfg(test)]
pub fn equal_floats(a: Float, b: Float) -> bool {
    if a == 0. && b == 0. {
        return true;
    }

    let diff = (a - b).abs();

    if a == 0. || b == 0. {
        return diff < EPSILON;
    }

    diff / (a.abs() + b.abs()).min(MAX) < EPSILON
}

#[cfg(test)]
pub fn equal_floats_eps(a: Float, b: Float, eps: Float) -> bool {
    if a == 0. && b == 0. {
        return true;
    }

    let diff = (a - b).abs();

    if a == 0. || b == 0. {
        return diff < eps;
    }

    diff / (a.abs() + b.abs()).min(MAX) < eps
}
