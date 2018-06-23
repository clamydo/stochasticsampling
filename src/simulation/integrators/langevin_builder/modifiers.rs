// Move unit test into own file
#[cfg(test)]
#[path = "./modifiers_test.rs"]
mod modifiers_test;

use simulation::particle::{OrientationVector, PositionVector, ParticleVector};
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

pub fn jeffrey_vorticity(
    p: ParticleVector,
    delta: ParticleVector,
    vort: VectorD,
) -> ParticleVector {
    // (1-nn) . (-W[u] . n) == 0.5 * Curl[u] x n

    let mut r: VectorD = [
        vort[1] * p.orientation[2] - vort[2] * p.orientation[1],
        vort[2] * p.orientation[0] - vort[0] * p.orientation[2],
        vort[0] * p.orientation[1] - vort[1] * p.orientation[0],
    ].into();
    r *= 0.5;

    delta + ParticleVector {
        position: PositionVector::zero(),
        orientation: r.to(),
    }
}
