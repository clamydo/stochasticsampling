//! Data structure representing the coordinates of a particle.

use consts::TWOPI;
use pcg_rand::Pcg64;
use rand::SeedableRng;
use rand::distributions::{IndependentSample, Range};
use simulation::settings::BoxSize;
use std::f64::consts::PI;

const PIHALF: f64 = PI / 2.;

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

    pub fn from_vector_mut(&mut self, v: &[f64; 3]) {
        let rxy = (v[0] * v[0] + v[1] * v[1]).sqrt();

        // transform back to spherical coordinate
        self.phi = v[1].atan2(v[0]);
        self.theta = PIHALF - (v[2]).atan2(rxy);

        debug_assert!(self.theta.is_finite());
        debug_assert!(self.theta.is_finite());
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


        for _ in 0..n {
            particles.push(Particle::new(
                bs.x * between.ind_sample(&mut rng),
                bs.y * between.ind_sample(&mut rng),
                bs.z * between.ind_sample(&mut rng),
                TWOPI * between.ind_sample(&mut rng),
                // take care of the spherical geometry by drawing from sin
                pdf_sin(2. * between.ind_sample(&mut rng)),
                bs,
            ))
        }

        particles
    }
}


pub fn pdf_sin(x: f64) -> f64 {
    (1. - x).acos()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use test_helper::equal_floats;

    #[test]
    fn test_random_particles() {
        let bs = BoxSize {
            x: 1.,
            y: 2.,
            z: 3.,
        };

        let particles = Particle::randomly_placed_particles(1000, bs, [1, 1]);

        for p in &particles {
            let Position { x, y, z } = p.position;

            assert!(0. <= x && x < 1.);
            assert!(0. <= y && x < 2.);
            assert!(0. <= z && x < 3.);
        }
    }

    #[test]
    fn test_modulo() {

        let input = [
            [2. * ::std::f64::consts::PI, 2. * ::std::f64::consts::PI],
            [-4.440892098500626e-16, 2. * ::std::f64::consts::PI],
        ];
        let output = [0., 0.];

        for (i, o) in input.iter().zip(output.iter()) {
            let a = modulo(i[0], i[1]);
            assert!(
                a == *o,
                "in: {} mod {}, out: {}, expected: {}",
                i[0],
                i[1],
                a,
                *o
            );
        }
    }

    #[test]
    fn test_ang_pbc() {

        let input = [[1., 0.], [1., PI], [1., -0.1], [1., PI + 0.1]];
        let expect = [
            [1., 0.],
            [1., PI],
            [1. + PI, 0.09999999999999964],
            [1. + PI, PI - 0.1],
        ];

        for (i, e) in input.iter().zip(expect.iter()) {
            let (phi, theta) = ang_pbc(i[0], i[1]);

            assert!(
                equal_floats(phi, e[0]),
                "PHI; input: {:?}, expected: {}, output: {}",
                i,
                e[0],
                phi
            );
            assert!(
                equal_floats(theta, e[1]),
                "THETA; input: {:?}, expected: {}, output: {}",
                i,
                e[1],
                theta
            );
        }
    }

    #[test]
    fn test_pdf_sin() {
        // TODO: Check statstics
        use std::f64::consts::PI;
        let input = [0., 1., 2.];
        let expect = [0., PI / 2., PI];
        let output: Vec<_> = input.iter().map(|x| pdf_sin(*x)).collect();

        for ((i, o), e) in input.iter().zip(expect.iter()).zip(output.iter()) {
            assert!(equal_floats(*o, *e), "{} => {}, not {}", i, o, e)
        }
    }

    #[test]
    fn test_orientation_from_orientation_vector() {
        let input = [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 1., 0.],
            [1., 0., 1.],
            [-1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.],
            [-1., 1., 0.],
            [1., 0., -1.],
            [15.23456, 0., 0.],
        ];

        let expect = [
            [PI / 2., 0.],
            [PI / 2., PI / 2.],
            [0., 0.],
            [PI / 2., PI / 4.],
            [PI / 4., 0.],
            [PI / 2., PI],
            [PI / 2., -PI / 2.],
            [PI, 0.],
            [PI / 2., 3. * PI / 4.],
            [3. / 4. * PI, 0.],
            [PI / 2., 0.],
        ];

        let mut o = Orientation::new(0., 0.);

        for (i, e) in input.iter().zip(expect.iter()) {
            o.from_vector_mut(i);

            assert!(
                equal_floats(e[0], o.theta),
                "input: {:?} -> theta {} != {}",
                i,
                o.theta,
                e[0]
            );
            assert!(
                equal_floats(e[1], o.phi),
                "input: {:?} -> phi {} != {}",
                i,
                o.phi,
                e[1]
            );
        }
    }
}
