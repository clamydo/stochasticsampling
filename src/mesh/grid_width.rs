//! Data structure that holds the grid width's/

use crate::consts::TWOPI;
use crate::Float;
use crate::{BoxSize, GridSize};
use serde_derive::{Deserialize, Serialize};
#[cfg(feature = "single")]
use std::f32::consts::PI;
#[cfg(not(feature = "single"))]
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridWidth {
    pub x: Float,
    pub y: Float,
    pub z: Float,
    pub phi: Float,
    pub theta: Float,
}

impl GridWidth {
    /// Calculates width of a grid cell given the number of cells and box size.
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> GridWidth {
        GridWidth {
            x: box_size.x as Float / grid_size.x as Float,
            y: box_size.y as Float / grid_size.y as Float,
            z: box_size.z as Float / grid_size.z as Float,
            phi: TWOPI / grid_size.phi as Float,
            theta: PI / grid_size.theta as Float,
        }
    }
}
