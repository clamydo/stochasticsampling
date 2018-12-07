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
use ndarray::{Array, ArrayView, Ix2};
use quaternion;
use crate::magnetic_interaction;
use crate::particle::{OrientationVector, ParticleVector, PositionVector};
use crate::vector::VectorD;

/// Does not change anything, just returns the given `delta`.
#[inline(always)]
pub fn identity(_p: OriginalParticle, delta: ParticleVector) -> ParticleVector {
    delta
}

/// Returns the delta given as a parameter `c`.
#[inline(always)]
pub fn constant(_p: OriginalParticle, _delta: ParticleVector, c: ParticleVector) -> ParticleVector {
    c
}

/// Translates particle due to self propulsion in the direction of its
/// orentation.
#[inline(always)]
pub fn self_propulsion(p: OriginalParticle, delta: ParticleVector) -> ParticleVector {
    delta + ParticleVector {
        position: p.vector.orientation.to(),
        orientation: OrientationVector::zero(),
    }
}

/// Translates the particle in the convective local flow field.
#[inline(always)]
pub fn convection(_p: OriginalParticle, delta: ParticleVector, flow: VectorD) -> ParticleVector {
    delta + ParticleVector {
        position: flow.to(),
        orientation: OrientationVector::zero(),
    }
}

/// Translates the particle due to magnetic dipole forces.
#[inline(always)]
pub fn magnetic_dipole_dipole_force(
    p: OriginalParticle,
    delta: ParticleVector,
    (drag, grad_b): (f64, ArrayView<f64, Ix2>),
) -> ParticleVector {
    delta + ParticleVector {
        position: magnetic_interaction::mean_force(grad_b, &p.vector.orientation).to() * drag,
        orientation: OrientationVector::zero(),
    }
}

/// Translates according to translational diffusion. CAUTION: Must be called
/// only after `.step`. The random vector should already include the timestep
/// and translatinal diffusion constant.
#[inline(always)]
pub fn translational_diffusion(
    _p: OriginalParticle,
    delta: ParticleVector,
    random_vector: VectorD,
) -> ParticleVector {
    delta + ParticleVector {
        position: random_vector.to(),
        orientation: OrientationVector::zero(),
    }
}

/// Rotates particle to align with external magnetic field.
#[inline(always)]
pub fn external_field_alignment(
    p: OriginalParticle,
    delta: ParticleVector,
    realignment: f64,
) -> ParticleVector {
    // magnetic field points in y direction
    // CAUTION, changing this requires a change of magnetic stress as well
    let mut b: VectorD = [0., realignment, 0.].into();
    b -= p.vector.orientation * p.vector.orientation.dot(&b);

    delta + ParticleVector {
        position: PositionVector::zero(),
        orientation: b.to(),
    }
}

/// Rotates the particle due to coupling to the flow's vorticity. Anti-symmetric
/// Jeffrey's term.
#[inline(always)]
pub fn jeffrey_vorticity(
    p: OriginalParticle,
    delta: ParticleVector,
    vortm: ArrayView<f64, Ix2>,
) -> ParticleVector {
    // (1-nn) . (-W[u] . n) == -W[u] . n == 0.5 * Curl[u] x n

    let n = p.vector.orientation;

    let n = Array::from_vec(n.v.to_vec());
    let f = vortm.dot(&n) * (-1.);

    let r = unsafe{[
        *f.uget(0),
        *f.uget(1),
        *f.uget(2),
    ]}.into();

    delta + ParticleVector {
        position: PositionVector::zero(),
        orientation: r,
    }
}

/// Rotates the particle due to coupling to the flow's strain. Symmetric
/// Jeffrey's term.
#[inline(always)]
pub fn jeffrey_strain(
    p: OriginalParticle,
    delta: ParticleVector,
    (shape, strainm): (f64, ArrayView<f64, Ix2>),
) -> ParticleVector {
    // (1-nn) . (g E[u] . n)

    let n = p.vector.orientation;

    let n = Array::from_vec(n.v.to_vec());
    let mut f = strainm.dot(&n);
    f *= shape;
    let nne = &n * f.dot(&n);
    f -= &nne;

    let r = unsafe{[
        *f.uget(0),
        *f.uget(1),
        *f.uget(2),
    ]}.into();

    delta + ParticleVector {
        position: PositionVector::zero(),
        orientation: r,
    }
}

/// Rotates the particle in the mean magnetic field of all other particles.
#[inline(always)]
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
#[inline(always)]
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
