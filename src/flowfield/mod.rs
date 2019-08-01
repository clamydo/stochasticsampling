use crate::Float;
use ndarray::{Array, Ix4};

pub type FlowField3D = Array<Float, Ix4>;

pub mod spectral_solver;
pub mod stress;
