extern crate mpi;

use rand::distributions::{IndependentSample, Range};
use stochasticsampling::random::NormalDistributionIterator;
use stochasticsampling::coordinates::Particle;
use stochasticsampling::coordinates::vector::ModVector64;
use std::f64;
use std::error::Error;
use std::fmt::Display;
use std::fmt;
use settings::Settings;
use self::mpi::traits::*;

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

fn init_positions() {
    unimplemented!()
}

pub fn simulate(settings: &Settings) -> Result<(), SimulationError> {
    // initialise mpi and recive world size and current rank
    let mpi_universe = mpi::initialize().unwrap();
    let mpi_world = mpi_universe.world();
    let mpi_size = mpi_world.size();
    let mpi_rank = mpi_world.rank();

    macro_rules! dlog {
        ($msg:expr) => {
            if mpi_rank == 0 {
                info!($msg);
            }
        }
    }

    // share particles evenly between all ranks
    let number_of_particles: usize = settings.simulation.number_of_particles / (mpi_size as usize);

    // Y(t) = sqrt(t) * X(t), if X is normally distributed with variance 1, then
    // Y is normally distributed with variance t.
    let sqrt_timestep = f64::sqrt(settings.simulation.timestep);
    let stepsize = sqrt_timestep * settings.simulation.diffusion_constant;
    let mut particles = Vec::with_capacity(settings.simulation.number_of_particles);


    // initialise random particle position
    let mut rng = ::rand::thread_rng();
    let between = Range::new(-1f64, 1.);
    for _ in 0..number_of_particles {
        particles.push(Particle {
            position: ModVector64::new(between.ind_sample(&mut rng),
                                       between.ind_sample(&mut rng),
                                       between.ind_sample(&mut rng)),
        })
    }

    assert_eq!(particles.len(), number_of_particles);

    // initialize normal distribution iterator, maybe not the most elegant way
    let mut ndi = NormalDistributionIterator::new();

    // for step in 1..settings.simulation.number_of_timesteps {
    //     for (i, p) in particles.iter_mut().enumerate() {
    //         let ModVector64{ref mut x, ref mut y, ref mut z} = p.position;
    //
    //         *x = *x + ndi.sample() * stepsize;
    //         *y = *y + ndi.sample() * stepsize;
    //         *z = *z + ndi.sample() * stepsize;
    //     }
    // }

    Ok(())
}
