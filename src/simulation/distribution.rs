/// A representation for the probability distribution function.

use ndarray::{Array, Ix};

pub struct Distribution {
    dist: Array<f64, (Ix, Ix, Ix)>,
}

impl Distribution {
    pub fn new(grid: (Ix, Ix, Ix)) -> Distribution {
        Distribution { dist: Array::default(grid) }
    }
}
