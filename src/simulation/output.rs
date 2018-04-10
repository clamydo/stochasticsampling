use simulation::distribution::Distribution;
use simulation::flowfield::FlowField3D;
use simulation::particle::Particle;

/// Captures values that can be outputed during simulation.
/// Not all fields need to have values, which is reflected in the Option type.
#[derive(Debug, Default, Serialize)]
pub struct OutputEntry {
    pub distribution: Option<Distribution>,
    pub flowfield: Option<FlowField3D>,
    pub particles: Option<Vec<Particle>>,
    pub timestep: usize,
}
