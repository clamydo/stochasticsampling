//! Implementation of different terms in the Langevin-equation as stream
//! modifiers.
//!
//! Every modifier has the signature
//! `fn foobar(p: ParticleVector, delta: ParticleVector) -> ParticleVector {}`.
//! `p` is the current particle state, `delta` is the accumulating delta that
//! needs to be added to propagate through time. The modifier should modify the
//! delta (`new = old + modification`) given as a parameter, to build a chain
//! of modifications,

// Move unit test into own file
#[cfg(test)]
#[path = "./modifiers_test.rs"]
mod modifiers_test;

use super::OriginalParticle;
use quaternion;
use simulation::particle::{OrientationVector, ParticleVector, PositionVector};
use simulation::vector::VectorD;

/// Does not change anything, just returns the given `delta`.
pub fn identity(_p: OriginalParticle, delta: ParticleVector) -> ParticleVector {
    delta
}

/// Returns the delta given as a parameter `c`.
pub fn constant(_p: OriginalParticle, _delta: ParticleVector, c: ParticleVector) -> ParticleVector {
    c
}

/// Translates particle due to self propulsion in the direction of its
/// orentation.
pub fn self_propulsion(p: OriginalParticle, delta: ParticleVector) -> ParticleVector {
    delta + ParticleVector {
        position: p.vector.orientation.to(),
        orientation: OrientationVector::zero(),
    }
}

/// Translates the particle in the convective local flow field.
pub fn convection(_p: OriginalParticle, delta: ParticleVector, flow: VectorD) -> ParticleVector {
    delta + ParticleVector {
        position: flow.to(),
        orientation: OrientationVector::zero(),
    }
}

/// Translates according to translational diffusion. CAUTION: Must be called only after `.step`. The random vector should already include the timestep and translatinal diffusion constant.
pub fn translational_diffusion(_p: OriginalParticle, delta: ParticleVector, random_vector: VectorD) -> ParticleVector {
    delta + ParticleVector {
        position: random_vector.to(),
        orientation: OrientationVector::zero(),
    }

}


/// Rotates the particle due to coupling to the flow's vorticiyt. Symmetric
/// Jeffrey's term.
pub fn jeffrey_vorticity(
    p: OriginalParticle,
    delta: ParticleVector,
    vort: VectorD,
) -> ParticleVector {
    // (1-nn) . (-W[u] . n) == 0.5 * Curl[u] x n

    let n = p.vector.orientation;

    let mut r: VectorD = [
        vort[1] * n[2] - vort[2] * n[1],
        vort[2] * n[0] - vort[0] * n[2],
        vort[0] * n[1] - vort[1] * n[0],
    ].into();
    r *= 0.5;

    delta + ParticleVector {
        position: PositionVector::zero(),
        orientation: r.to(),
    }
}

/// Rotates the particle in the mean magnetic field of all other particles.
pub fn magnetic_dipole_dipole_rotation(
    p: OriginalParticle,
    delta: ParticleVector,
    b: VectorD,
) -> ParticleVector {
    let mut b = b;
    b -= p.vector.orientation * p.vector.orientation.dot(&b);

    delta + ParticleVector {
        position: PositionVector::zero(),
        orientation: b.to(),
    }
}

/// Parameters for rotational diffusion
pub struct RotDiff {
    /// Angle of rotation axis perpendicular to the current orientation with
    /// respect to the x axis.
    pub axis_angle: f64,
    /// Angle of rotation around rotation axis.
    pub rotate_angle: f64,
}

/// Rotates particle according to rotational diffusion. Needs to come after
/// `.step`! Timestep must already be included in the rotation angle.
pub fn rotational_diffusion(
    p: OriginalParticle,
    delta: ParticleVector,
    r: &RotDiff,
) -> ParticleVector {
    let rotational_axis = |alpha: f64| {
        let cs = p.orientation_angles;
        let cos_ax = alpha.cos();
        let sin_ax = alpha.sin();
        // axis perpendicular to orientation vector
        [
            cs.cos_phi * cs.cos_theta * sin_ax - cos_ax * cs.sin_phi,
            cos_ax * cs.cos_phi + cs.cos_theta * sin_ax * cs.sin_phi,
            -sin_ax * cs.sin_theta,
        ]
    };

    let ax = rotational_axis(r.axis_angle);

    // quaternion encoding a rotation around `rotational_axis` with
    // angle drawn from Rayleigh-distribution
    let q = quaternion::axis_angle(ax, r.rotate_angle);
    let v = p.vector.orientation.v;
    let new: OrientationVector = quaternion::rotate_vector(q, v).into();

    delta + ParticleVector {
        position: PositionVector::zero(),
        // required to make order of modifiers independent
        orientation: new - p.vector.orientation,
    }
}
