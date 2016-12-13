//! Module that defines data structures and algorithms for the integration of
//! the simulation.
mod distribution;
mod integrator;
pub mod output;

use coordinates::TWOPI;
use coordinates::particle::Particle;
use mpi::topology::{SystemCommunicator, Universe};
use mpi::traits::*;
use ndarray::Array;
use pcg_rand::Pcg64;
use rand::SeedableRng;
use rand::distributions::{IndependentSample, Normal};
use self::distribution::Distribution;
use self::integrator::{FlowField, IntegrationParameter, Integrator};
use settings::{BoxSize, GridSize, Settings};
use std::f64;



/// Structure that holds state variables needed for MPI.
#[allow(dead_code)]
struct MPIState {
    universe: Universe,
    world: SystemCommunicator,
    size: i32,
    rank: i32,
}

/// Main data structure representing the simulation.
pub struct Simulation {
    integrator: Integrator,
    mpi: MPIState,
    normaldist: Normal,
    number_of_particles: usize,
    settings: Settings,
    state: SimulationState,
}


/// Holds the current state of the simulation.
struct SimulationState {
    distribution: Distribution,
    flow_field: FlowField,
    particles: Vec<Particle>,
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


#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridWidth {
    x: f64,
    y: f64,
    a: f64,
}

/// Calculates width of a grid cell given the number of cells and box size.
pub fn grid_width(grid_size: GridSize, box_size: BoxSize) -> GridWidth {
    GridWidth {
        x: box_size[0] as f64 / grid_size[0] as f64,
        y: box_size[1] as f64 / grid_size[1] as f64,
        a: TWOPI / grid_size[2] as f64,
    }
}

impl Simulation {
    /// Return a new simulation data structure, holding the state of the
    /// simulation.
    pub fn new(settings: Settings) -> Simulation {
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

        let int_param = IntegrationParameter {
            timestep: sim.timestep,
            trans_diffusion: (2. * param.diffusion.translational * sim.timestep).sqrt(),
            rot_diffusion: (2. * param.diffusion.rotational * sim.timestep).sqrt(),
            stress: param.stress,
            magnetic_reorientation: param.magnetic_reorientation * 2.,
        };

        let integrator = Integrator::new(sim.grid_size,
                                         grid_width(sim.grid_size, sim.box_size),
                                         int_param);

        // initialize a normal distribution with variance sqrt(timestep)
        let normal = Normal::new(0.0, sim.timestep.sqrt());

        // deterministically seed every mpi process (slightly) differently
        // normal distribution with variance timestep
        let seed = [sim.seed[0], sim.seed[1] + mpi.rank as u64];

        // initialize state with zeros
        let state = SimulationState {
            distribution: Distribution::new(sim.grid_size, grid_width(sim.grid_size, sim.box_size)),
            flow_field: Array::zeros((2, sim.grid_size[0], sim.grid_size[1])),
            particles: Vec::with_capacity(ranklocal_number_of_particles),
            rng: SeedableRng::from_seed(seed),
            timestep: 0,
        };

        Simulation {
            integrator: integrator,
            mpi: mpi,
            normaldist: normal,
            number_of_particles: ranklocal_number_of_particles,
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
        for p in particles.iter_mut() {
            p.position.x.m = bs[0];
            p.position.y.m = bs[1];
            p.orientation.m = TWOPI;
        }

        self.state.particles = particles;

        // Do a first sampling, so that the initial condition can also be obtained
        self.state.distribution.sample_from(&self.state.particles);
    }


    /// Resumes from a given snapshot
    pub fn resume(&mut self, snapshot: Snapshot) {
        self.init(snapshot.particles);

        // Reset timestep
        self.state.timestep = snapshot.timestep;
        self.state.rng.reseed(snapshot.rng_seed);
    }


    /// Do the actual simulation timestep
    pub fn do_timestep(&mut self) -> usize {
        // Sample probability distribution from ensemble.
        self.state.distribution.sample_from(&self.state.particles);

        // Dirty hack, pretty inelegant! Problem is, that sampling will mutate self,
        // needs to
        // borrow mutably, can only be done once!
        let random_samples = [self.normaldist.ind_sample(&mut self.state.rng),
                              self.normaldist.ind_sample(&mut self.state.rng),
                              self.normaldist.ind_sample(&mut self.state.rng)];

        // Update particle positions
        self.state.flow_field = self.integrator.evolve_particles_inplace(&mut self.state.particles,
                                                                         &random_samples,
                                                                         &self.state.distribution);

        // increment timestep counter to keep a continous identifier when resuming
        self.state.timestep += 1;
        self.state.timestep
    }

    /// Returns a fill Snapshot
    pub fn get_snapshot(&self) -> Snapshot {
        let seed = self.state.rng.extract_seed();

        let snapshot = Snapshot {
            particles: self.state.particles.clone(),
            // assuming little endianess
            rng_seed: [seed[0].lo, seed[0].hi, seed[1].lo, seed[1].hi],
            timestep: self.state.timestep,
        };

        snapshot
    }

    // Getter

    /// Returns the first `n` particles
    pub fn get_particles_head(&self, n: usize) -> Vec<Particle> {
        self.state.particles[..n].to_vec()
    }

    /// Returns sampled distribution field
    pub fn get_distribution(&self) -> Distribution {
        self.state.distribution.clone()
    }

    /// Returns sampled flow field
    pub fn get_flow_field(&self) -> Distribution {
        self.state.distribution.clone()
    }

    /// Returns current timestep
    pub fn get_timestep(&self) -> usize {
        self.state.timestep
    }
}

impl Iterator for Simulation {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        Some(self.do_timestep())
    }
}
