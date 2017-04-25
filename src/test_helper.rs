use std::f64::{MAX, EPSILON};

pub fn equal_floats(a: f64, b: f64) -> bool {
    if a == 0. && b == 0. {
        return true;
    }

    let diff = (a - b).abs();
    diff / (a.abs() + b.abs()).min(MAX) < EPSILON
}
