//! Data structure representing the coordinates of a particle.

use consts::TWOPI;
use modulo::modulofloat::Mf64;
use modulo::vector::Mod64Vector2;
use pcg_rand::Pcg64;
use rand::SeedableRng;
use rand::distributions::{IndependentSample, Range};
use simulation::settings::BoxSize;

/// Coordinates (including the orientation) of a particle in 2D.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Particle {
    /// spatial position
    pub position: Mod64Vector2,
    /// orientation of particle as an angle
    pub orientation: Mf64,
}

impl Particle {
    /// Returns a `Particle` with given coordinates.
    pub fn new(x: f64, y: f64, a: f64, box_size: BoxSize) -> Particle {
        Particle {
            position: Mod64Vector2::new(x, y, [box_size.x, box_size.y]),
            orientation: Mf64::new(a, TWOPI),
        }
    }

    /// Places n particles at random positions
    pub fn randomly_placed_particles(n: usize, boxdim: BoxSize, seed: [u64; 2]) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(n);

        // initialise random particle position
        let mut rng: Pcg64 = SeedableRng::from_seed(seed);
        let between = Range::new(0f64, 1.);

        for _ in 0..n {
            particles.push(Particle {
                               position: Mod64Vector2::new(boxdim.x * between.ind_sample(&mut rng),
                                                           boxdim.y * between.ind_sample(&mut rng),
                                                           [boxdim.x, boxdim.y]),
                               orientation: Mf64::new(TWOPI * between.ind_sample(&mut rng), TWOPI),
                           })
        }

        particles
    }
}
