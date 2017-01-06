use super::distribution::Distribution;
use super::integrator::FlowField;
use super::super::coordinates::particle::Particle;

/// Captures values that can be outputed during simulation.
/// Not all fields need to have values, which is reflected in the Option type.
#[derive(Debug, Default, Serialize)]
pub struct Output {
    pub distribution: Option<Distribution>,
    pub flow_field: Option<FlowField>,
    pub particles: Option<Vec<Particle>>,
    pub timestep: usize,
}
