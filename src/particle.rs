//! Data structure representing the coordinates of a particle.
// Move unit test into own file
#[cfg(test)]
#[path = "./particle_test.rs"]
mod particle_test;

use crate::consts::TWOPI;
use crate::vector::Vector;
use crate::BoxSize;
use crate::Float;
use derive_more::{Add, AddAssign, Div, Mul, Sub};
use quaternion;
use rand::distributions::Uniform;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::convert::From;
#[cfg(feature = "single")]
use std::f32::consts::PI;
#[cfg(not(feature = "single"))]
use std::f64::consts::PI;

const PIHALF: Float = PI / 2.;

pub fn modulo(f: Float, m: Float) -> Float {
    let r = f % m;
    if r < 0.0 {
        r + m.abs()
    } else {
        r
    }
    // when ready, replace with
    // f.rem_euclid(m)
}

pub fn ang_pbc(phi: Float, theta: Float) -> (Float, Float) {
    let theta = modulo(theta, TWOPI);
    if theta > PI {
        (modulo(phi + PI, TWOPI), TWOPI - theta)
    } else {
        (modulo(phi, TWOPI), theta)
    }
}

pub type PositionVector = Vector<Position>;

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Position {
    pub fn new(x: Float, y: Float, z: Float, bs: &BoxSize) -> Position {
        Position {
            x: modulo(x, bs.x),
            y: modulo(y, bs.y),
            z: modulo(z, bs.z),
        }
    }

    pub fn from_vector(v: &PositionVector) -> Position {
        Position {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }

    pub fn pbc(&mut self, bs: &BoxSize) {
        self.x = modulo(self.x, bs.x);
        self.y = modulo(self.y, bs.y);
        self.z = modulo(self.z, bs.z);
    }

    pub fn from_vector_mut(&mut self, v: &PositionVector) {
        self.x = v[0];
        self.y = v[1];
        self.z = v[2];
    }

    pub fn to_vector(&self) -> PositionVector {
        [self.x, self.y, self.z].into()
    }
}

pub type OrientationVector = Vector<Orientation>;

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct Orientation {
    pub phi: Float,
    pub theta: Float,
}

impl Orientation {
    pub fn new(phi: Float, theta: Float) -> Orientation {
        let (phi, theta) = ang_pbc(phi, theta);
        Orientation {
            phi: phi,
            theta: theta,
        }
    }

    pub fn from_vector(v: &OrientationVector) -> Orientation {
        let v = v.v;
        let rxy = (v[0] * v[0] + v[1] * v[1]).sqrt();

        // transform back to spherical coordinate
        let phi = v[1].atan2(v[0]);
        let theta = PIHALF - (v[2]).atan2(rxy);

        debug_assert!(phi.is_finite());
        debug_assert!(theta.is_finite());

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

        debug_assert!(self.phi.is_finite());
        debug_assert!(self.theta.is_finite());
    }

    pub fn to_vector(&self) -> OrientationVector {
        let cs = CosSinOrientation::from_orientation(self);
        cs.to_orientation_vector()
    }
}

#[derive(Clone, Copy)]
pub struct CosSinOrientation {
    pub cos_phi: Float,
    pub sin_phi: Float,
    pub cos_theta: Float,
    pub sin_theta: Float,
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

    pub fn to_orientation_vector(&self) -> OrientationVector {
        [
            self.sin_theta * self.cos_phi,
            self.sin_theta * self.sin_phi,
            self.cos_theta,
        ]
        .into()
    }
}

/// Coordinates (including the orientation) of a particle in 2D.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Particle {
    /// spatial position
    pub position: Position,
    /// orientation of particle as an angle
    pub orientation: Orientation,
}

impl Particle {
    /// Returns a `Particle` with given coordinates. Automatically applies pbc.
    pub fn new(
        x: Float,
        y: Float,
        z: Float,
        phi: Float,
        theta: Float,
        box_size: &BoxSize,
    ) -> Particle {
        let mut p = Particle {
            position: Position::new(x, y, z, box_size),
            orientation: Orientation::new(phi, theta),
        };

        p.pbc(box_size);
        p
    }

    /// Returns a `Particle` from a given position and orientation.
    /// Automatically applies pbc.
    pub fn from_position_orientation(
        pos: Position,
        o: Orientation,
        box_size: &BoxSize,
    ) -> Particle {
        let mut p = Particle {
            position: pos,
            orientation: o,
        };

        p.pbc(box_size);
        p
    }

    pub fn pbc(&mut self, bs: &BoxSize) {
        self.position.pbc(bs);
        self.orientation.pbc();
    }

    pub fn place_isotropic<F>(r: &mut F, bs: &BoxSize) -> Particle
    where
        F: FnMut() -> Float,
    {
        Particle::new(
            bs.x * r(),
            bs.y * r(),
            bs.z * r(),
            TWOPI * r(),
            // take care of the spherical geometry by drawing from sin
            pdf_sin(2. * r()),
            bs,
        )
    }

    /// Places n particles at random positions following an isotropic
    /// distribution
    pub fn create_isotropic(n: usize, bs: &BoxSize, seed: u64) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(n);

        // initialise random particle position
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let range: rand::distributions::Uniform<Float> = Uniform::new(0., 1.);

        let mut r = || rng.sample(range);

        for _ in 0..n {
            let p = Particle::place_isotropic(&mut r, bs);
            particles.push(p);
        }

        particles
    }

    pub fn place_homogeneous<F>(r: &mut F, kappa: Float, bs: &BoxSize) -> Particle
    where
        F: FnMut() -> Float,
    {
        let mut p = Particle::new(
            bs.x * r(),
            bs.y * r(),
            bs.z * r(),
            TWOPI * r(),
            pdf_homogeneous_fixpoint(kappa, r()),
            bs,
        );

        let ax = [1., 0., 0.];
        let q = quaternion::axis_angle(ax, -PIHALF);
        let mut o = p.orientation.to_vector();
        o = quaternion::rotate_vector(q, o.v).into();
        p.orientation.from_vector_mut(&o);
        p
    }

    /// Places n particles according the the spatial homogeneous distribution
    pub fn create_homogeneous(n: usize, kappa: Float, bs: &BoxSize, seed: u64) -> Vec<Particle> {
        let mut particles = Vec::with_capacity(n);

        // initialise random particle position
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let range: rand::distributions::Uniform<Float> = Uniform::new(0., 1.);

        let mut r = || rng.sample(range);

        for _ in 0..n {
            let p = Particle::place_homogeneous(&mut r, kappa, bs);
            particles.push(p);
        }

        particles
    }
}

#[derive(Debug, Clone, Copy, Add, Sub, Mul, Div, AddAssign)]
pub struct ParticleVector {
    pub position: PositionVector,
    pub orientation: OrientationVector,
}

impl ParticleVector {
    pub fn zero() -> ParticleVector {
        ParticleVector {
            position: PositionVector::zero(),
            orientation: OrientationVector::zero(),
        }
    }
}

impl From<Particle> for ParticleVector {
    fn from(p: Particle) -> ParticleVector {
        ParticleVector {
            position: p.position.to_vector(),
            orientation: p.orientation.to_vector(),
        }
    }
}

impl<'a> From<&'a Particle> for ParticleVector {
    fn from(p: &'a Particle) -> ParticleVector {
        ParticleVector {
            position: p.position.to_vector(),
            orientation: p.orientation.to_vector(),
        }
    }
}

impl From<ParticleVector> for Particle {
    fn from(p: ParticleVector) -> Particle {
        Particle {
            position: Position::from_vector(&p.position),
            orientation: Orientation::from_vector(&p.orientation),
        }
    }
}

pub fn pdf_sin(x: Float) -> Float {
    (1. - x).acos()
}

/// Samples the polar angle of the spatial homogeneous distribution, given by
/// $\sin(\theta) \psi(\kappa, \theta)$.
/// Including the measure of spherical coordinates $\sin(\theta)$ is crucial.
pub fn pdf_homogeneous_fixpoint(kappa: Float, x: Float) -> Float {
    assert!(
        kappa != 0.0,
        "Alignment of zero is the isotropic state. Please use that instead."
    );
    let r = Float::acos(Float::ln(Float::exp(kappa) - 2. * x * Float::sinh(kappa)) / kappa);

    assert!(
        !(r.is_nan()),
        "Caution, the alignment parameter is too high for the given precision."
    );

    r
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
        ]
        .serialize(serializer)
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
