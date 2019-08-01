// Move unit test into own file
#[cfg(test)]
#[path = "./stress_test.rs"]
mod langevin_test;

use crate::consts::TWOPI;
use crate::distribution::Distribution;
use crate::mesh::grid_width::GridWidth;
use crate::Float;
use crate::GridSize;
use ndarray::{Array, ArrayView, ArrayViewMut, Axis, Ix2, Ix4, Ix5};
use num_complex::Complex;
use serde_derive::{Deserialize, Serialize};
#[cfg(feature = "single")]
use std::f32::consts::PI;
#[cfg(not(feature = "single"))]
use std::f64::consts::PI;

/// Holds prefactors for active and magnetic stress
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StressPrefactors {
    pub active: Float,
    pub magnetic: Float,
}

/// Calculates approximation of discretized stress kernel, to be used in ///
/// the expectation value to obtain the stress tensor.
pub fn stress_kernel<F>(grid_size: GridSize, grid_width: GridWidth, stress: F) -> Array<Float, Ix4>
where
    F: Fn(Float, Float) -> Array<Float, Ix2>,
{
    let mut s = Array::<Float, _>::zeros((3, 3, grid_size.phi, grid_size.theta));
    // Calculate discrete angles, considering the cell centered sample points of
    // the distribution
    let gw_half_phi = grid_width.phi / 2.;
    let gw_half_theta = grid_width.theta / 2.;
    let angles_phi = Array::linspace(0. + gw_half_phi, TWOPI - gw_half_phi, grid_size.phi);
    let angles_theta = Array::linspace(0. + gw_half_theta, PI - gw_half_theta, grid_size.theta);

    // TODO implement stress due to magnetic dipole interaction

    for (mut ax1, phi) in s.axis_iter_mut(Axis(2)).zip(&angles_phi) {
        for (mut e, theta) in ax1.axis_iter_mut(Axis(2)).zip(&angles_theta) {
            // let s = stress_active(*phi, *theta) * a + stress_magnetic(*phi, *theta) * b;
            let s = stress(*phi, *theta);

            e.assign(&s);

            // Already taken care of by the modified cell average in the distribution code
            // e *= theta.sin();
        }
    }

    s
}

/// It consumes `stress_field` and updates it given a stress kernel `kernel`
/// and a distribution `dist`. It returns the updated stress field.
pub fn average_stress<'a>(
    stress_field: ArrayViewMut<'a, Complex<Float>, Ix5>,
    kernel: &ArrayView<Float, Ix4>,
    dist: &Distribution,
) -> ArrayViewMut<'a, Complex<Float>, Ix5> {
    let dist_sh = dist.dim();
    let stress_sh = kernel.dim();

    let n_angle = dist_sh.3 * dist_sh.4;
    let n_stress = stress_sh.0 * stress_sh.1;
    let n_dist = dist_sh.0 * dist_sh.1 * dist_sh.2;

    let gw = dist.get_grid_width();

    // Put axis in order, so that components fields are continuous in memory,
    // so it can be passed to FFTW easily
    let stress = kernel.into_shape([n_stress, n_angle]).unwrap();

    let dist = dist.dist.view().into_shape([n_dist, n_angle]).unwrap();

    let mut stress_field = stress_field.into_shape((n_stress, n_dist)).unwrap();

    // integration measures and FFT normalization
    let measure = gw.phi * gw.theta;

    // Calculating the integral over the orientation. `norm` includes weights for
    // integration and normalisation of DFT
    for (s, mut o1) in stress.outer_iter().zip(stress_field.outer_iter_mut()) {
        for (d, o2) in dist.outer_iter().zip(o1.iter_mut()) {
            *o2 = Complex::from(s.dot(&d) * measure)
        }
    }

    stress_field
        .into_shape([stress_sh.0, stress_sh.1, dist_sh.0, dist_sh.1, dist_sh.2])
        .unwrap()
}

pub mod stresses {
    use crate::Float;
    use ndarray::{Array, Ix2};

    /// Calculate active stress tensor for polar angles `phi` and `theta`.
    pub fn stress_active(phi: Float, theta: Float) -> Array<Float, Ix2> {
        let mut s = Array::zeros((3, 3));

        s[[0, 0]] = -(1. / 3.) + phi.cos() * phi.cos() * theta.sin() * theta.sin();
        s[[0, 1]] = phi.cos() * theta.sin() * theta.sin() * phi.sin();
        s[[0, 2]] = theta.cos() * phi.cos() * theta.sin();

        s[[1, 0]] = phi.cos() * theta.sin() * theta.sin() * phi.sin();
        s[[1, 1]] = -(1. / 3.) + theta.sin() * theta.sin() * phi.sin() * phi.sin();
        s[[1, 2]] = theta.cos() * theta.sin() * phi.sin();

        s[[2, 0]] = theta.cos() * phi.cos() * theta.sin();
        s[[2, 1]] = theta.cos() * theta.sin() * phi.sin();
        s[[2, 2]] = -(1. / 3.) + theta.cos() * theta.cos();

        s
    }

    /// Calculate magnetic stress tensor for polar angles `phi` and `theta`.
    /// Magnetic field points into y-direction to avoid binning instability in
    /// spherical coordinates at the poles.
    /// Calculates `0.5 (nb-bn)`, with `b = [0, 1, 0]` and orientation `n`.
    pub fn stress_magnetic(phi: Float, theta: Float) -> Array<Float, Ix2> {
        let mut s = Array::zeros((3, 3));

        s[[0, 0]] = 0.;
        s[[0, 1]] = phi.cos() * theta.sin();
        s[[0, 2]] = 0.;

        s[[1, 0]] = -phi.cos() * theta.sin();
        s[[1, 1]] = 0.;
        s[[1, 2]] = -theta.cos();

        s[[2, 0]] = 0.;
        s[[2, 1]] = theta.cos();
        s[[2, 2]] = 0.;

        s * 0.5
    }

    /// Calculate magnetic stress tensor for polar angles `phi` and `theta`.
    pub fn stress_magnetic_rods(phi: Float, theta: Float) -> Array<Float, Ix2> {
        let mut s = Array::zeros((3, 3));

        s[[0, 0]] = 2. * theta.cos() * phi.cos().powi(2) * theta.sin().powi(2);
        s[[0, 1]] = theta.cos() * theta.sin().powi(2) * (2. * phi).cos();
        s[[0, 2]] = (2. * theta).cos() * phi.cos() * theta.sin();
        s[[1, 0]] = theta.cos() * theta.sin().powi(2) * (2. * phi).cos();
        s[[1, 1]] = theta.sin() * (2. * theta).sin() * phi.sin().powi(2);
        s[[1, 2]] = (2. * theta).cos() * theta.sin() * phi.sin();

        s[[2, 0]] = (2. * theta).cos() * phi.cos() * theta.sin();
        s[[2, 1]] = (2. * theta).cos() * theta.sin() * phi.sin();
        s[[2, 2]] = -2. * theta.cos() * theta.sin().powi(2);

        s * 0.5
    }
}
