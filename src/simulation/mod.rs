//! Module that defines data structures and algorithms for the integration of
//! the simulation.

pub mod distribution;
pub mod integrators;
pub mod grid_width;
pub mod output;
pub mod particle;
pub mod settings;

use self::distribution::Distribution;
use self::grid_width::GridWidth;
use self::integrators::oseen_conv::{FlowField, IntegrationParameter, Integrator};
use self::particle::Particle;
use self::settings::{Settings, StressPrefactors};
use ndarray::Array;
use pcg_rand::Pcg64;
use rand::Rand;
use rand::SeedableRng;
use rand::distributions::normal::StandardNormal;
use std::f64;


/// Main data structure representing the simulation.
pub struct Simulation {
    integrator: Integrator,
    settings: Settings,
    state: SimulationState,
}


/// Holds the current state of the simulation.
struct SimulationState {
    distribution: Distribution,
    flow_field: FlowField,
    particles: Vec<Particle>,
    random_samples: Vec<[f64; 3]>,
    rng: Pcg64,
    /// count timesteps
    timestep: usize,
}


/// Seed of PCG PRNG
type Pcg64Seed = [u64; 4];

/// Captures the full state of the simulation
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    particles: Vec<Particle>,
    rng_seed: Pcg64Seed,
    /// current timestep number
    timestep: usize,
}


impl Simulation {
    /// Return a new simulation data structure, holding the state of the
    /// simulation.
    pub fn new(settings: Settings) -> Simulation {
        // helper bindings for brevity
        let sim = settings.simulation;
        let param = settings.parameters;

        let scaled_stress_prefactors = StressPrefactors {
            active: 0.5 * param.stress.active,
            magnetic: 0.5 * param.stress.magnetic,
        };

        let int_param = IntegrationParameter {
            timestep: sim.timestep,
            // see documentation of `integrator.evolve_particle_inplace` for a rational
            trans_diffusion: (2. * param.diffusion.translational * sim.timestep).sqrt(),
            rot_diffusion: (2. * param.diffusion.rotational * sim.timestep).sqrt(),
            stress: scaled_stress_prefactors,
            magnetic_reorientation: param.magnetic_reorientation,
        };

        let integrator = Integrator::new(sim.grid_size,
                                         GridWidth::new(sim.grid_size, sim.box_size),
                                         int_param);


        // normal distribution with variance timestep
        let seed = [sim.seed[0], sim.seed[1]];

        // initialize state with zeros
        let state = SimulationState {
            distribution: Distribution::new(sim.grid_size,
                                            GridWidth::new(sim.grid_size, sim.box_size)),
            flow_field: Array::zeros((2, sim.grid_size[0], sim.grid_size[1])),
            particles: Vec::with_capacity(sim.number_of_particles),
            random_samples: vec![[0f64; 3]; sim.number_of_particles],
            rng: SeedableRng::from_seed(seed),
            timestep: 0,
        };

        Simulation {
            integrator: integrator,
            settings: settings,
            state: state,
        }
    }

    /// Initialize the state of the simulation
    pub fn init(&mut self, mut particles: Vec<Particle>) {
        assert!(particles.len() == self.settings.simulation.number_of_particles,
                "Given initial condition has not the same number of particles ({}) as given in \
                 the parameter file ({}).",
                particles.len(),
                self.settings.simulation.number_of_particles);


        let bs = self.settings.simulation.box_size;

        // IMPORTANT: Set also the modulo quotiont for every particle, since it is not
        // provided for user given input.
        for p in &mut particles {
            // this makes sure, the input is sanitized
            *p = Particle::new(p.position.x.v, p.position.y.v, p.orientation.v, bs);
        }

        self.state.particles = particles;

        // Do a first sampling, so that the initial condition can also be obtained
        self.state.distribution.sample_from(&self.state.particles);

        self.state.distribution.dist *= self.settings.simulation.box_size[0] *
                                        self.settings.simulation.box_size[1];
    }


    /// Resumes from a given snapshot
    pub fn resume(&mut self, snapshot: Snapshot) {
        self.init(snapshot.particles);

        // Reset timestep
        self.state.timestep = snapshot.timestep;
        self.state.rng.reseed(snapshot.rng_seed);
    }

    /// Returns a fill Snapshot
    pub fn get_snapshot(&self) -> Snapshot {
        let seed = self.state.rng.extract_seed();

        Snapshot {
            particles: self.state.particles.clone(),
            // assuming little endianess
            rng_seed: [seed[0].lo, seed[0].hi, seed[1].lo, seed[1].hi],
            timestep: self.state.timestep,
        }
    }

    // Getter
    /// Returns all particles
    pub fn get_particles(&self) -> Vec<Particle> {
        self.state.particles.clone()
    }

    /// Returns the first `n` particles
    pub fn get_particles_head(&self, n: usize) -> Vec<Particle> {
        self.state.particles[..n].to_vec()
    }

    /// Returns sampled distribution field
    pub fn get_distribution(&self) -> Distribution {
        self.state.distribution.clone()
    }

    /// Returns sampled flow field
    pub fn get_flow_field(&self) -> FlowField {
        self.state.flow_field.clone()
    }

    /// Returns current timestep
    pub fn get_timestep(&self) -> usize {
        self.state.timestep
    }


    /// Do the actual simulation timestep
    pub fn do_timestep(&mut self) -> usize {
        // Sample probability distribution from ensemble.
        self.state.distribution.sample_from(&self.state.particles);
        // Renormalize distribution to keep number density constant.
        self.state.distribution.dist *= self.settings.simulation.box_size[0] *
                                        self.settings.simulation.box_size[1];

        // Calculate flow field from distribution.
        self.state.flow_field = self.integrator.calculate_flow_field(&self.state.distribution);

        // Generate all needed random numbers here. Makes parallelization easier.
        for r in &mut self.state.random_samples {
            *r = [StandardNormal::rand(&mut self.state.rng).0,
                  StandardNormal::rand(&mut self.state.rng).0,
                  StandardNormal::rand(&mut self.state.rng).0];
        }

        // Update particle positions
        self.integrator.evolve_particles_inplace(&mut self.state.particles,
                                                 &self.state.random_samples,
                                                 self.state.flow_field.view());

        // increment timestep counter to keep a continous identifier when resuming
        self.state.timestep += 1;
        self.state.timestep
    }
}

impl Iterator for Simulation {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        Some(self.do_timestep())
    }
}
