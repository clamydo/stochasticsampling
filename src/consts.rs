#[cfg(feature = "single")]
use std::f32::consts::PI;
#[cfg(not(feature = "single"))]
use std::f64::consts::PI;

use crate::Float;

pub const TWOPI: Float = 2. * PI;
