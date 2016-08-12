pub mod modulofloat;
pub mod vector;

use rand::distributions::{IndependentSample, Range};
use self::modulofloat::Mf64;
use self::vector::Mod64Vector2;
use settings::BoxSize;
use std::f64::consts::PI;

#[derive(Copy, Clone)]
pub struct Particle {
    pub position: Mod64Vector2,
    pub orientation: Mf64,
}


/// Places n particles at random positions
pub fn randomly_placed_particles(n: usize, boxdim: BoxSize) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(n);

    // initialise random particle position
    let mut rng = ::rand::thread_rng();
    let between = Range::new(0f64, 1.);

    for _ in 0..n {
        particles.push(Particle {
            position: Mod64Vector2::new(boxdim.0 * between.ind_sample(&mut rng),
                                        boxdim.1 * between.ind_sample(&mut rng),
                                        boxdim),
            orientation: Mf64::new(2. * PI * between.ind_sample(&mut rng), 2. * PI),
        })
    }

    particles
}
