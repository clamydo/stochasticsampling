//! Data structure that holds the grid width's/

use crate::consts::TWOPI;
use crate::{BoxSize, GridSize};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridWidth {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub phi: f32,
    pub theta: f32,
}

impl GridWidth {
    /// Calculates width of a grid cell given the number of cells and box size.
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> GridWidth {
        GridWidth {
            x: box_size.x as f32 / grid_size.x as f32,
            y: box_size.y as f32 / grid_size.y as f32,
            z: box_size.z as f32 / grid_size.z as f32,
            phi: TWOPI / grid_size.phi as f32,
            theta: PI / grid_size.theta as f32,
        }
    }
}
