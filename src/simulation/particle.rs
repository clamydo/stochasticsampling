//! Data structure representing the coordinates of a particle.

use consts::TWOPI;
use pcg_rand::Pcg64;
use rand::SeedableRng;
use rand::distributions::{IndependentSample, Range};
use simulation::settings::BoxSize;
use std::f64::consts::PI;

pub fn modulo(f: f64, m: f64) -> f64 {
    ((f % m) + m) % m
}

pub fn ang_pbc(phi: f64, theta: f64) -> (f64, f64) {
    let theta = modulo(theta, TWOPI);
    if theta > PI {
        (modulo(phi + PI, TWOPI), TWOPI - theta)
    } else {
        (modulo(phi, TWOPI), theta)
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Position {
    pub fn new(x: f64, y: f64, z: f64, bs: BoxSize) -> Position {
        Position {
            x: modulo(x, bs.x),
            y: modulo(y, bs.y),
            z: modulo(z, bs.z),
        }
    }

    pub fn pbc(&mut self, bs: BoxSize) {
        self.x = modulo(self.x, bs.x);
        self.y = modulo(self.y, bs.y);
        self.z = modulo(self.z, bs.z);
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Orientation {
    pub phi: f64,
    pub theta: f64,
}

impl Orientation {
    pub fn new(phi: f64, theta: f64) -> Orientation {
        let (phi, theta) = ang_pbc(phi, theta);
        Orientation {
            phi: phi,
            theta: theta,
        }
    }

    pub fn pbc(&mut self) {
        let (phi, theta) = ang_pbc(self.phi, self.theta);
        self.phi = phi;
        self.theta = theta;
    }
}

/// Coordinates (including the orientation) of a particle in 2D.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Particle {
    /// spatial position
    pub position: Position,
    /// orientation of particle as an angle
    pub orientation: Orientation,
}

impl Particle {
    /// Returns a `Particle` with given coordinates.
    pub fn new(x: f64, y: f64, z: f64, phi: f64, theta: f64, box_size: BoxSize) -> Particle {
        Particle {
            position: Position::new(x, y, z, box_size),
            orientation: Orientation::new(phi, theta),
        }
    }

    pub fn pbc(&mut self, bs: BoxSize) {
        self.position.pbc(bs);
        self.orientation.pbc();
    }

    /// Places n particles at random positions
    pub fn randomly_placed_particles(n: usize, bs: BoxSize, seed: [u64; 2]) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(n);

        // initialise random particle position
        let mut rng: Pcg64 = SeedableRng::from_seed(seed);
        let between = Range::new(0f64, 1.);

        // WARNING: This is not isotrop! Needs to scale with sin(theta)
        for _ in 0..n {
            particles.push(
                Particle::new(
                    bs.x * between.ind_sample(&mut rng),
                    bs.y * between.ind_sample(&mut rng),
                    bs.z * between.ind_sample(&mut rng),
                    TWOPI * between.ind_sample(&mut rng),
                    PI * between.ind_sample(&mut rng),
                    bs,
                )
            )
        }

        particles
    }
}
