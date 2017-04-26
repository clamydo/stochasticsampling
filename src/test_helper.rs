#[cfg(test)]
use std::f64::{MAX, EPSILON};

#[cfg(test)]
pub fn equal_floats(a: f64, b: f64) -> bool {
    if a == 0. && b == 0. {
        return true;
    }

    let diff = (a - b).abs();

    if a == 0. || b == 0. {
        return diff < EPSILON
    }

    diff / (a.abs() + b.abs()).min(MAX) < EPSILON
}
