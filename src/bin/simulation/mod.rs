//! Module that defines data structures and algorithms for the integration of
//! the simulation.

pub mod settings;

use self::settings::Settings;
use fftw3::fft;
use ndarray::{Array, ArrayView, Ix2, Ix4, Ix5};
use num_complex::Complex;

use rand::distributions::StandardNormal;
use rand::distributions::Uniform;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg32;
use rayon;
use rayon::prelude::*;
use std::env;
use std::str::FromStr;
use stochasticsampling::consts::TWOPI;
use stochasticsampling::distribution::density_gradient::DensityGradient;
use stochasticsampling::distribution::Distribution;
use stochasticsampling::flowfield::FlowField3D;
use stochasticsampling::integrators::langevin_builder::modifiers::*;
use stochasticsampling::integrators::langevin_builder::TimeStep;
use stochasticsampling::integrators::LangevinBuilder;
use stochasticsampling::magnetic_interaction::magnetic_solver::MagneticSolver;
use stochasticsampling::mesh::grid_width::GridWidth;
use stochasticsampling::particle::Particle;
use stochasticsampling::vector::VectorD;

struct ParamCache {
    trans_diff: f64,
    rot_diff: f64,
    grid_width: GridWidth,
}

#[derive(Clone, Copy)]
pub struct RandomVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub axis_angle: f64,
    pub rotate_angle: f64,
}

/// Main data structure representing the simulation.
pub struct Simulation {
    magnetic_solver: MagneticSolver,
    density_gradient: DensityGradient,
    settings: Settings,
    state: SimulationState,
    pcache: ParamCache,
}

/// Holds the current state of the simulation.
struct SimulationState {
    distribution: Distribution,
    particles: Vec<Particle>,
    random_samples: Vec<RandomVector>,
    rng: Vec<Pcg32>,
    /// count timesteps
    timestep: usize,
}

/// Captures the full state of the simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    particles: Vec<Particle>,
    rng_state: Vec<Pcg32>,
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


        let magnetic_solver = MagneticSolver::new(sim.grid_size, sim.box_size);
        let density_gradient = DensityGradient::new(sim.grid_size, sim.box_size);

        // normal distribution with variance timestep
        let seed = sim.seed;

        let num_threads = env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| usize::from_str(&s).ok())
            .expect("No environment variable 'RAYON_NUM_THREADS' set.");

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();

        // Initialze threads of FFTW
        fft::fftw_init(Some(num_threads)).unwrap();

        let rng = (0..num_threads)
            .map(|_| SeedableRng::seed_from_u64(seed))
            .collect();

        // initialize state with zeros
        let state = SimulationState {
            distribution: Distribution::new(sim.grid_size, sim.box_size),
            particles: Vec::with_capacity(sim.number_of_particles),
            random_samples: vec![
                RandomVector {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                    axis_angle: 0.,
                    rotate_angle: 0.,
                };
                sim.number_of_particles
            ],
            rng: rng,
            timestep: 0,
        };

        Simulation {
            magnetic_solver: magnetic_solver,
            density_gradient: density_gradient,
            settings: settings,
            state: state,
            pcache: ParamCache {
                trans_diff: (2. * param.diffusion.translational * sim.timestep).sqrt(),
                rot_diff: (2. * param.diffusion.rotational * sim.timestep).sqrt(),
                grid_width: GridWidth::new(sim.grid_size, sim.box_size),
            },
        }
    }

    /// Initialize the state of the simulation
    pub fn init(&mut self, mut particles: Vec<Particle>) {
        assert!(
            particles.len() == self.settings.simulation.number_of_particles,
            "Given initial condition has not the same number of particles ({}) as given in \
             the parameter file ({}).",
            particles.len(),
            self.settings.simulation.number_of_particles
        );

        let bs = self.settings.simulation.box_size;

        // IMPORTANT: Set also the modulo quotiont for every particle, since it is not
        // provided for user given input.
        for p in &mut particles {
            // this makes sure, the input is sanitized
            *p = Particle::new(
                p.position.x,
                p.position.y,
                p.position.z,
                p.orientation.phi,
                p.orientation.theta,
                &bs,
            );
        }

        self.state.particles = particles;

        // Do a first sampling, so that the initial condition can also be obtained
        self.state.distribution.sample_from(&self.state.particles);

        self.state.distribution.dist *= self.settings.simulation.box_size.x
            * self.settings.simulation.box_size.y
            * self.settings.simulation.box_size.z;
    }

    /// Resumes from a given snapshot
    pub fn resume(&mut self, snapshot: Snapshot) {
        self.init(snapshot.particles);

        // Reset timestep
        self.state.timestep = snapshot.timestep;
        for (r, s) in self.state.rng.iter_mut().zip(snapshot.rng_state) {
            *r = s.clone();
        }
    }

    /// Returns a fill Snapshot
    pub fn get_snapshot(&self) -> Snapshot {
        Snapshot {
            particles: self.state.particles.clone(),
            // assuming little endianess
            rng_state: self.state.rng.clone(),
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
    pub fn get_flow_field(&self) -> FlowField3D {
        unimplemented!()
    }

    /// Returns magnetic field
    pub fn get_magnetic_field(&self) -> Array<f64, Ix4> {
        self.magnetic_solver.get_real_magnet_field()
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
        self.state.distribution.dist *= self.settings.simulation.box_size.x
            * self.settings.simulation.box_size.y
            * self.settings.simulation.box_size.z;

        let range = Uniform::new(0f64, 1.);

        let chunksize = self.state.random_samples.len() / self.state.rng.len() + 1;

        let dt = self.pcache.trans_diff;
        let dr = self.pcache.rot_diff;

        self.state
            .random_samples
            .par_chunks_mut(chunksize)
            .zip(self.state.rng.par_iter_mut())
            .for_each(|(c, rng)| {
                for r in c.iter_mut() {
                    *r = RandomVector {
                        x: rng.sample(StandardNormal) * dt,
                        y: rng.sample(StandardNormal) * dt,
                        z: rng.sample(StandardNormal) * dt,
                        axis_angle: TWOPI * rng.sample(range),
                        rotate_angle: rayleigh_pdf(dr, rng.sample(range)),
                    };
                }
            });

        let sim = self.settings.simulation;
        let param = self.settings.parameters;
        let gw = self.pcache.grid_width;

        let (b, grad_b) = self
            .magnetic_solver
            .mean_magnetic_field(&self.state.distribution);

        let dens_grad = self.density_gradient.get_gradient(&self.state.distribution);

        self.state
            .particles
            .par_iter_mut()
            .zip(self.state.random_samples.par_iter())
            .for_each(|(p, r)| {
                let idx = get_cell_index(&p, &gw);

                let densg = vector_field_at_cell_c(&dens_grad, idx);

                let b =
                    vector_field_at_cell_c(&b, idx) * param.magnetic_dipole.magnetic_dipole_dipole;
                let grad_b = matrix_field_at_cell(&grad_b, idx);

                let dr = RotDiff {
                    axis_angle: r.axis_angle,
                    rotate_angle: r.rotate_angle,
                };

                *p = LangevinBuilder::new(&p)
                    .with(self_propulsion)
                    .with_param(
                        magnetic_dipole_dipole_force,
                        (param.magnetic_drag, grad_b.view()),
                    )
                    .with_param(volume_exclusion_force, (param.volume_exclusion, densg))
                    .with_param(external_field_alignment, param.magnetic_reorientation)
                    .with_param(magnetic_dipole_dipole_rotation, b)
                    .step(&TimeStep(sim.timestep))
                    .with_param(translational_diffusion, [r.x, r.y, r.z].into())
                    .with_param(rotational_diffusion, &dr)
                    .finalize(&sim.box_size);
            });

        // increment timestep counter to keep a continous identifier when resuming
        self.state.timestep += 1;
        self.state.timestep
    }
}

impl Drop for Simulation {
    fn drop(&mut self) {
        fft::fttw_finalize();
    }
}

fn rayleigh_pdf(sigma: f64, x: f64) -> f64 {
    sigma * f64::sqrt(-2. * f64::ln(1. - x))
}

impl Iterator for Simulation {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        Some(self.do_timestep())
    }
}

fn get_cell_index(p: &Particle, grid_width: &GridWidth) -> (usize, usize, usize) {
    let ix = (p.position.x / grid_width.x).floor() as usize;
    let iy = (p.position.y / grid_width.y).floor() as usize;
    let iz = (p.position.z / grid_width.z).floor() as usize;

    (ix, iy, iz)
}

fn vector_field_at_cell_c(
    field: &ArrayView<Complex<f64>, Ix4>,
    idx: (usize, usize, usize),
) -> VectorD {
    let f = unsafe {
        [
            (*field.uget((0, idx.0, idx.1, idx.2))).re,
            (*field.uget((1, idx.0, idx.1, idx.2))).re,
            (*field.uget((2, idx.0, idx.1, idx.2))).re,
        ]
    };
    f.into()
}

fn matrix_field_at_cell(
    field: &ArrayView<Complex<f64>, Ix5>,
    idx: (usize, usize, usize),
) -> Array<f64, Ix2> {
    field.slice(s![.., .., idx.0, idx.1, idx.2]).map(|v| v.re)
}
