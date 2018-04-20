use simulation::distribution::Distribution;
use simulation::flowfield::FlowField3D;
use simulation::particle::Particle;
use ndarray::{Array, Ix4};

/// Captures values that can be outputed during simulation.
/// Not all fields need to have values, which is reflected in the Option type.
#[derive(Debug, Default, Serialize)]
pub struct OutputEntry {
    pub distribution: Option<Distribution>,
    pub flowfield: Option<FlowField3D>,
    pub magneticfield: Option<Array<f64, Ix4>>,
    pub particles: Option<Vec<Particle>>,
    pub timestep: usize,
}
