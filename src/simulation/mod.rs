//! Module that defines data structures and algorithms for the integration of
//! the simulation.
mod distribution;
mod integrator;

use coordinates::particle::Particle;
use mpi::topology::{SystemCommunicator, Universe};
use mpi::traits::*;
use rand::distributions::{IndependentSample, Normal};
use self::distribution::Distribution;
use settings::Settings;
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

/// Holds rotational and translational difussion parameters.
pub struct DiffusionParameter {
    dt: f64, // translational diffusion
    dr: f64, // rotational diffusion
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
    settings: &'a Settings,
    mpi: MPIState,
    state: SimulationState,
    number_of_particles: usize,
}


/// Holds the current state of the simulation.
struct SimulationState {
    particles: Vec<Particle>,
    distribution: Distribution,
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

impl<'a> Simulation<'a> {
    /// Return a new simulation data structure, holding the state of the
    /// simulation.
    pub fn new(settings: &Settings) -> Simulation {
        let mpi_universe = ::mpi::initialize().unwrap();
        let mpi_world = mpi_universe.world();

        // share particles evenly between all ranks
        let ranklocal_number_of_particles: usize = settings.simulation.number_of_particles /
                                                   (mpi_world.size() as usize);

        let mpi = MPIState {
            universe: mpi_universe,
            world: mpi_world,
            size: mpi_world.size(),
            rank: mpi_world.rank(),
        };

        let state = SimulationState {
            particles: Vec::with_capacity(ranklocal_number_of_particles),
            distribution: Distribution::new(settings.simulation.grid_size,
                                            settings.simulation.box_size),
        };

        Simulation {
            settings: settings,
            mpi: mpi,
            state: state,
            number_of_particles: ranklocal_number_of_particles,
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
                                                self.settings.simulation.box_size);

        assert_eq!(self.state.particles.len(), self.number_of_particles);
    }

    /// Run the simulation for the number of timesteps specified in the
    /// settings file.
    pub fn run(&mut self) -> Result<(), SimulationError> {

        let sqrt_timestep = f64::sqrt(self.settings.simulation.timestep);
        let mut rng = ::rand::thread_rng();
        let normal = Normal::new(0.0, sqrt_timestep);
        let mut normal_sample = move || normal.ind_sample(&mut rng);

        let diff = DiffusionParameter {
            dt: self.settings.simulation.translational_diffusion_constant,
            dr: self.settings.simulation.rotational_diffusion_constant,
        };

        for step in 1..self.settings.simulation.number_of_timesteps {
            for (i, mut p) in self.state.particles.iter_mut().enumerate() {
                integrator::evolve_inplace(&mut p, &diff, sqrt_timestep, &mut normal_sample);

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
