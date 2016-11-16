//! Module that defines data structures and algorithms for the integration of
//! the simulation.
mod distribution;
mod integrator;

use coordinates::TWOPI;
use coordinates::particle::Particle;
use mpi::topology::{SystemCommunicator, Universe};
use mpi::traits::*;
use pcg_rand::Pcg64;
use rand::SeedableRng;
use rand::distributions::{IndependentSample, Normal};
use self::distribution::Distribution;
use self::integrator::{IntegrationParameter, Integrator};
use settings::{BoxSize, GridSize, Settings};
use std::error::Error;
use std::f64;
use std::fmt;
use std::fmt::Display;


/// Error type that merges all errors that can happen during loading and parsing
/// of the settings file.
#[derive(Debug)]
pub enum SimulationError {
}

impl Display for SimulationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

impl Error for SimulationError {
    fn description(&self) -> &str {
        "SimulationError"
    }
}


/// Structure that holds state variables needed for MPI.
#[allow(dead_code)]
struct MPIState {
    universe: Universe,
    world: SystemCommunicator,
    size: i32,
    rank: i32,
}

/// Main data structure representing the simulation.
pub struct Simulation<'a> {
    integrator: Integrator,
    mpi: MPIState,
    normaldist: Normal,
    number_of_particles: usize,
    settings: &'a Settings,
    state: SimulationState,
}


/// Holds the current state of the simulation.
struct SimulationState {
    particles: Vec<Particle>,
    distribution: Distribution,
    rng: Pcg64,
}

macro_rules! zinfo {
    ($rank:expr, $($arg:tt)*) => {
        if $rank == 0 {
            info!($($arg)*);
        }
    }
}

macro_rules! zdebug {
    ($rank: expr, $($arg:tt)*) => {
        // only compile this when in debug mode
        if cfg!(debug_assertions) {
            if $rank == 0 {
                debug!($($arg)*);
            }
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct GridWidth {
    x: f64,
    y: f64,
    a: f64,
}

/// Calculates width of a grid cell given the number of cells and box size.
pub fn grid_width(grid_size: GridSize, box_size: BoxSize) -> GridWidth {
    GridWidth {
        x: box_size.0 as f64 / grid_size.0 as f64,
        y: box_size.1 as f64 / grid_size.1 as f64,
        a: TWOPI / grid_size.2 as f64,
    }
}

impl<'a> Simulation<'a> {
    /// Return a new simulation data structure, holding the state of the
    /// simulation.
    pub fn new(settings: &Settings) -> Simulation {
        let mpi_universe = ::mpi::initialize().unwrap();
        let mpi_world = mpi_universe.world();

        // helper bindings for brevity
        let sim = settings.simulation;
        let param = settings.parameters;


        // share particles evenly between all ranks
        let ranklocal_number_of_particles: usize = sim.number_of_particles /
                                                   (mpi_world.size() as usize);

        let mpi = MPIState {
            universe: mpi_universe,
            world: mpi_world,
            size: mpi_world.size(),
            rank: mpi_world.rank(),
        };

        // deterministically seed every mpi process (slightly) differently
        // normal distribution with variance timestep
        let seed = [sim.seed[0], sim.seed[1] + mpi.rank as u64];

        let state = SimulationState {
            particles: Vec::with_capacity(ranklocal_number_of_particles),
            distribution: Distribution::new(sim.grid_size, grid_width(sim.grid_size, sim.box_size)),
            rng: SeedableRng::from_seed(seed),
        };

        let int_param = IntegrationParameter {
            timestep: sim.timestep,
            trans_diffusion: param.diffusion.translational.sqrt() * 2.,
            rot_diffusion: param.diffusion.rotational.sqrt() * 2.,
            speed: param.self_propulsion_speed,
            stress: param.stress,
            magnetic_reoriantation: param.magnetic_reoriantation * 2.,
        };

        let integrator = Integrator::new(sim.grid_size,
                                         grid_width(sim.grid_size, sim.box_size),
                                         int_param);

        // initialize a normal distribution with variance sqrt(timestep)
        let normal = Normal::new(0.0, sim.timestep.sqrt());

        Simulation {
            integrator: integrator,
            mpi: mpi,
            normaldist: normal,
            number_of_particles: ranklocal_number_of_particles,
            settings: settings,
            state: state,
        }
    }


    /// Initialise the initial condition of the simulation. At the moment it is
    /// sampled from a uniform random distribution.
    pub fn init(&mut self) {
        zinfo!(self.mpi.rank,
               "Placing {} particles at their initial positions.",
               self.settings.simulation.number_of_particles);

        self.state.particles =
            Particle::randomly_placed_particles(self.number_of_particles,
                                                self.settings.simulation.box_size,
                                                self.settings.simulation.seed);

        assert_eq!(self.state.particles.len(), self.number_of_particles);
    }

    /// Run the simulation for the number of timesteps specified in the
    /// settings file.
    pub fn run(&mut self) -> Result<(), SimulationError> {
        for step in 0..self.settings.simulation.number_of_timesteps {
            // Sample probability distribution from ensemble
            self.state.distribution.sample_from(&self.state.particles);

            // Dirty hack, pretty inelegant!
            let random_samples = [self.normaldist.ind_sample(&mut self.state.rng),
                                  self.normaldist.ind_sample(&mut self.state.rng),
                                  self.normaldist.ind_sample(&mut self.state.rng)];

            // Update particle positions
            self.integrator.evolve_particles_inplace(&mut self.state.particles,
                                                     &random_samples,
                                                     &self.state.distribution);

            for (i, p) in self.state.particles.iter().enumerate() {
                zdebug!(self.mpi.rank, "{}, {}, {}, {}",
                    step,
                    i,
                    p.position.x.as_ref(),
                    p.position.y.as_ref(),
                );

            }
        }

        Ok(())
    }
}
