use super::Simulation;
use super::integrator::FlowField;
use super::distribution::Distribution;
use super::super::coordinates::particle::Particle;


/// Captures values that can be outputed during simulation.
/// Not all fields need to have values, which is reflected in the Option type.
#[derive(Serialize)]
pub struct Output<'a> {
    #[serde(skip_serializing)]
    simulation: &'a Simulation,
    distribution: Option<Distribution>,
    flow_field: Option<FlowField>,
    particles: Option<Vec<Particle>>,
    timestep: usize,
}

impl<'a> Output<'a> {
    pub fn new(simulation: &Simulation) -> Output {
        Output {
            simulation: simulation,
            distribution: None,
            flow_field: None,
            particles: None,
            timestep: simulation.get_timestep(),
        }
    }

    // pub add_particles<'a>(&'a mut self, n: usize) -> &'a mut Output {
    //     self.particles = Some(self.simulation.get_particles_head(n));
    //     self
    // }
    //
    // pub add_distribution<'a>(&'a mut self, dist: Distribution) -> &'a mut Output {
    //     self.disttribution = Some(self.simulation.get_distribution());
    //     self
    // }
    //
    // pub add_flow_field<'a>(&'a mut self, flow_field: FlowField) -> &'a mut Output {
    //     self.flow_field = Some(self.simulation.get_flow_field());
    //     self
    // }
    //
    // pub build(&self) -> Output {
    //     self
    // }
}
