//! Data structure that holds the grid width's/

use settings::{BoxSize, GridSize};
use consts::TWOPI;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridWidth {
    pub x: f64,
    pub y: f64,
    pub a: f64,
}

impl GridWidth {
    /// Calculates width of a grid cell given the number of cells and box size.
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> GridWidth {
        GridWidth {
            x: box_size[0] as f64 / grid_size[0] as f64,
            y: box_size[1] as f64 / grid_size[1] as f64,
            a: TWOPI / grid_size[2] as f64,
        }
    }
}
