//! Data structure that holds the grid width's/

use consts::TWOPI;
use simulation::settings::{BoxSize, GridSize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridWidth {
    pub x: f64,
    pub y: f64,
    pub phi: f64,
}

impl GridWidth {
    /// Calculates width of a grid cell given the number of cells and box size.
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> GridWidth {
        GridWidth {
            x: box_size.x as f64 / grid_size.x as f64,
            y: box_size.y as f64 / grid_size.y as f64,
            phi: TWOPI / grid_size.phi as f64,
        }
    }
}
