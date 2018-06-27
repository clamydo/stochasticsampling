use particle::{Orientation, OrientationVector, Particle, Position, PositionVector, CosSinOrientation};
use std::ops::AddAssign;
use ndarray::{Array, ArrayView, Ix2, Ix4, Ix5};
use num_complex::Complex;
use quaternion;
use magnetic_interaction::mean_force;
use mesh::grid_width::GridWidth;
use {BoxSize, GridSize};
use vector::vorticity::vorticity3d_dispatch;
use vector::{Vector, VectorD};

#[derive(Clone, Copy)]
pub struct RandomVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub axis_angle: f64,
    pub rotate_angle: f64,
}

impl RandomVector {
    fn into_pos_vec(&self) -> PositionVector {
        [self.x, self.y, self.z].into()
    }
}

/// Holds parameter needed for time step
#[derive(Debug, Clone, Copy)]
pub struct IntegrationParameter {
    pub rot_diffusion: f64,
    pub timestep: f64,
    pub trans_diffusion: f64,
    pub magnetic_reorientation: f64,
    pub drag: f64,
    pub magnetic_dipole_dipole: f64,
}

pub struct FlowFieldValue {
    pub speed: Option<VectorD>,
    pub vorticity: Option<VectorD>,
}

pub struct MagneticDipoleFieldValue {
    pub field: Option<VectorD>,
    pub gradient_force: Option<VectorD>,
}

pub struct FieldValue {
    pub flow: FlowFieldValue,
    pub magnetic_dipole: MagneticDipoleFieldValue,
}

pub struct ParticleVector {
    position: PositionVector,
    orientation: OrientationVector,
}
pub struct ParticleDelta {
    position: Option<PositionVector>,
    orientation: Option<OrientationVector>,
}

impl ParticleVector {
    fn from_particle(p: &Particle) -> ParticleVector {
        ParticleVector {
            position: p.position.to_vector(),
            orientation: p.orientation.to_vector(),
        }
    }
}

impl AddAssign for ParticleVector {
    fn add_assign(&mut self, rhs: ParticleVector) {
        self.position += rhs.position;
        self.orientation += rhs.orientation;
    }
}

impl AddAssign<ParticleDelta> for ParticleVector {
    fn add_assign(&mut self, rhs: ParticleDelta) {
        match rhs.position {
            Some(p) => self.position += p,
            None => {}
        };
        match rhs.orientation {
            Some(o) => self.orientation += o,
            None => {}
        };
    }
}

pub struct LangevinBuilder {
    original: ParticleVector,
    cs: CosSinOrientation,
    fieldvalue: FieldValue,
    delta: ParticleVector,
}

impl LangevinBuilder {
    pub fn new(p: &Particle, f: FieldValue) -> LangevinBuilder {
        LangevinBuilder {
            original: ParticleVector::from_particle(p),
            cs: CosSinOrientation::from_orientation(&p.orientation),
            fieldvalue: f,
            delta: ParticleVector {
                position: PositionVector::zero(),
                orientation: OrientationVector::zero(),
            },
        }
    }

    pub fn with(mut self, f: fn(&LangevinBuilder) -> ParticleDelta) -> LangevinBuilder {
        self.delta += f(&self);
        self
    }

    pub fn finalize(mut self, box_size: BoxSize) -> Particle {
        self.original += self.delta;

        let pos = Position::from_vector(&self.original.position);
        let ori = Orientation::from_vector(&self.original.orientation);

        Particle::from_position_orientation(pos, ori, box_size)
    }
}

pub fn self_propulsion(l: &LangevinBuilder) -> ParticleDelta {
    ParticleDelta {
        position: Some(l.original.orientation.to()),
        orientation: None,
    }
}

pub fn convection(l: &LangevinBuilder) -> ParticleDelta {
    let u = l.fieldvalue.flow.speed.clone();
    ParticleDelta {
        position: u.and_then(|v| Some(v.convert())),
        orientation: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langevin_mod() {
        let bs = BoxSize {
            x: 10.,
            y: 10.,
            z: 10.,
        };
        let p = Particle::new(0.,0.,0.,0.,0., bs);

        let f = FieldValue {
            flow: FlowFieldValue {
                speed: None, vorticity: None
            },
            magnetic_dipole: MagneticDipoleFieldValue {
                field: None, gradient_force: None
            },
        };

        let l = LangevinBuilder::new(&p, f)
            .with(self_propulsion)
            .finalize(bs);

        println!("{:?}", l);

        assert!(false);
    }
}
