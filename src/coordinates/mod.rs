pub mod modulofloat;
pub mod vector;

use rand::distributions::{IndependentSample, Range};
use self::vector::Mod64Vector2;
use std::f64;

#[derive(Copy, Clone)]
pub struct Particle {
    pub position: Mod64Vector2,
    pub orientation: f64,
}


/// Places n particles at random positions
pub fn randomly_placed_particles(n: usize) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(n);

    // initialise random particle position
    let mut rng = ::rand::thread_rng();
    let between1 = Range::new(0f64, 1.);
    let between2pi = Range::new(0f64, 2. * f64::consts::PI);
    for _ in 0..n {
        particles.push(Particle {
            position: Mod64Vector2::new(between1.ind_sample(&mut rng),
                                        between1.ind_sample(&mut rng)),
            orientation: between2pi.ind_sample(&mut rng),
        })
    }

    particles
}
