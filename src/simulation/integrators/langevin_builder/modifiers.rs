// Move unit test into own file
#[cfg(test)]
#[path = "./modifiers_test.rs"]
mod modifiers_test;

use simulation::particle::{OrientationVector, ParticleVector};
use simulation::vector::VectorD;

pub fn identity(_: ParticleVector, delta: ParticleVector) -> ParticleVector {
    delta
}

pub fn constant(_: ParticleVector, delta: ParticleVector, c: ParticleVector) -> ParticleVector {
    c
}

pub fn self_propulsion(orig: ParticleVector, delta: ParticleVector) -> ParticleVector {
    delta + ParticleVector {
        position: orig.orientation.to(),
        orientation: OrientationVector::zero(),
    }
}

pub fn convection(_: ParticleVector, delta: ParticleVector, flow: VectorD) -> ParticleVector {
    delta + ParticleVector {
        position: flow.to(),
        orientation: OrientationVector::zero(),
    }
}
