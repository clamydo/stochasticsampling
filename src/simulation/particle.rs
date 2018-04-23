//! Data structure representing the coordinates of a particle.

use consts::TWOPI;
use pcg_rand::Pcg64;
use rand::distributions::{IndependentSample, Range};
use rand::SeedableRng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use simulation::settings::BoxSize;
use simulation::vector::Vector;
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

    pub fn from_vector_mut(&mut self, v: &Vector<Position>) {
        self.x = v[0];
        self.y = v[1];
        self.z = v[2];
    }
}

pub type OrientationVector = Vector<Orientation>;

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

    pub fn from_vector_mut(&mut self, v: &OrientationVector) {
        let v = v.v;
        let rxy = (v[0] * v[0] + v[1] * v[1]).sqrt();

        // transform back to spherical coordinate
        self.phi = v[1].atan2(v[0]);
        self.theta = PIHALF - (v[2]).atan2(rxy);

        debug_assert!(self.theta.is_finite());
        debug_assert!(self.theta.is_finite());
    }

    pub fn to_vector(&self) -> OrientationVector {
        let cs = CosSinOrientation::from_orientation(self);
        cs.to_orientation_vecor()
    }
}

pub struct CosSinOrientation {
    pub cos_phi: f64,
    pub sin_phi: f64,
    pub cos_theta: f64,
    pub sin_theta: f64,
}

impl CosSinOrientation {
    pub fn from_orientation(o: &Orientation) -> CosSinOrientation {
        CosSinOrientation {
            cos_phi: o.phi.cos(),
            sin_phi: o.phi.sin(),
            cos_theta: o.theta.cos(),
            sin_theta: o.theta.sin(),
        }
    }

    pub fn to_orientation_vecor(&self) -> OrientationVector {
        [
            self.sin_theta * self.cos_phi,
            self.sin_theta * self.sin_phi,
            self.cos_theta,
        ].into()
    }
}

/// Coordinates (including the orientation) of a particle in 2D.
#[derive(Debug, Copy, Clone)]
pub struct Particle {
    /// spatial position
    pub position: Position,
    /// orientation of particle as an angle
    pub orientation: Orientation,
}

impl Particle {
    /// Returns a `Particle` with given coordinates.
    pub fn new(x: f64, y: f64, z: f64, phi: f64, theta: f64, box_size: BoxSize) -> Particle {
        let mut p = Particle {
            position: Position::new(x, y, z, box_size),
            orientation: Orientation::new(phi, theta),
        };

        p.pbc(box_size);
        p
    }

    pub fn pbc(&mut self, bs: BoxSize) {
        self.position.pbc(bs);
        self.orientation.pbc();
    }

    /// Places n particles at random positions following an isotropic
    /// distribution
    pub fn place_isotropic(n: usize, bs: BoxSize, seed: [u64; 2]) -> Vec<Particle> {
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

    /// Places n particles according the the spatial homogeneous distribution
    pub fn place_homogeneous(n: usize, kappa: f64, bs: BoxSize, seed: [u64; 2]) -> Vec<Particle> {
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
                pdf_homogeneous_fixpoint(kappa, between.ind_sample(&mut rng)),
                bs,
            ))
        }

        particles
    }
}

pub fn pdf_sin(x: f64) -> f64 {
    (1. - x).acos()
}

/// Samples the polar angle of the spatial homogeneous distribution, given by
/// $\sin(\theta) \psi(\kappa, \theta)$.
/// Including the measure of spherical coordinates $\sin(\theta)$ is crucial.
pub fn pdf_homogeneous_fixpoint(kappa: f64, x: f64) -> f64 {
    f64::acos(f64::ln(f64::exp(kappa) - 2. * x * f64::sinh(kappa)) / kappa)
}

/// Serialize particle as continuous array instead of struct
impl Serialize for Particle {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        [
            self.position.x,
            self.position.y,
            self.position.z,
            self.orientation.phi,
            self.orientation.theta,
        ].serialize(serializer)
    }
}

/// Deserialize particle from continuous array with [x, y, z, phi, theta]
/// entries
impl<'de> Deserialize<'de> for Particle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer).map(|(px, py, pz, op, ot)| Particle {
            position: Position {
                x: px,
                y: py,
                z: pz,
            },
            orientation: Orientation { phi: op, theta: ot },
        })
    }
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

        let particles = Particle::place_isotropic(1000, bs, [1, 1]);

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
            [-::std::f64::EPSILON, 2. * ::std::f64::consts::PI],
            [
                2. * ::std::f64::consts::PI + ::std::f64::EPSILON,
                2. * ::std::f64::consts::PI,
            ],
            [7., 4.],
            [7., -4.],
            [-7., 4.],
            [-7., -4.],
        ];
        let output = [
            0.,
            2. * ::std::f64::consts::PI - ::std::f64::EPSILON,
            0.,
            3.,
            3.,
            1.,
            1.,
        ];

        for (i, o) in input.iter().zip(output.iter()) {
            let a = modulo(i[0], i[1]);
            assert!(
                equal_floats(a, *o),
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
        let input = [
            [1., 0.],
            [1., PI],
            [1., -0.1],
            [1., PI + 0.1],
            [TWOPI, PI],
            [6.283185307179586, 1.5707963267948966],
            [PI, PI + 1.],
        ];
        let expect = [
            [1., 0.],
            [1., PI],
            [1. + PI, 0.09999999999999964],
            [1. + PI, PI - 0.1],
            [0., PI],
            [0., PI / 2.],
            [0., PI - 1.],
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
            o.from_vector_mut(&((*i).into()));

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
