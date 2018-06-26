#[cfg(test)]
use std::f64::{EPSILON, MAX};

#[cfg(test)]
pub fn equal_floats(a: f64, b: f64) -> bool {
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
pub fn equal_floats_eps(a: f64, b: f64, eps: f64) -> bool {
    if a == 0. && b == 0. {
        return true;
    }

    let diff = (a - b).abs();

    if a == 0. || b == 0. {
        return diff < eps;
    }

    diff / (a.abs() + b.abs()).min(MAX) < eps
}
