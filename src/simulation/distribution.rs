/// A representation for the probability distribution function.

use ndarray::{Array, Ix};

pub struct Distribution {
    dist: Array<f64, (Ix, Ix)>,
}

impl Distribution {
    pub fn new() -> Distribution {
        unimplemented!()
    }
}
