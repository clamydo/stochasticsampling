use ndarray::{Array, Ix4};

pub type FlowField3D = Array<f64, Ix4>;

pub mod spectral_solver;
pub mod stress;
