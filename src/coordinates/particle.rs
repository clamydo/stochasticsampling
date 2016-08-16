use rand::distributions::{IndependentSample, Range};
use settings::BoxSize;
use super::TWOPI;
use super::modulofloat::Mf64;
use super::vector::Mod64Vector2;

#[derive(Copy, Clone)]
pub struct Particle {
    pub position: Mod64Vector2,
    pub orientation: Mf64,
}

impl Particle {
    pub fn new(x: f64, y: f64, a: f64, box_size: BoxSize) -> Particle {
        Particle {
            position: Mod64Vector2::new(x, y, box_size),
            orientation: Mf64::new(a, TWOPI),
        }
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
                orientation: Mf64::new(TWOPI * between.ind_sample(&mut rng), TWOPI),
            })
        }

        particles
    }
}
