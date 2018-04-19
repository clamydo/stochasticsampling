// Move unit test into own file
#[cfg(test)]
#[path = "./stress_test.rs"]
mod langevin_test;

use consts::TWOPI;
use ndarray::{Array, ArrayView, ArrayViewMut, Axis, Ix2, Ix4, Ix5};
use num_complex::Complex;
use simulation::distribution::Distribution;
use simulation::mesh::grid_width::GridWidth;
use simulation::settings::{GridSize, StressPrefactors};
use std::f64::consts::PI;

/// Calculates approximation of discretized stress kernel, to be used in ///
/// the expectation value to obtain the stress tensor.
pub fn stress_kernel(
    grid_size: GridSize,
    grid_width: GridWidth,
    stress: StressPrefactors,
) -> Array<f64, Ix4> {
    let mut s = Array::<f64, _>::zeros((3, 3, grid_size.phi, grid_size.theta));
    // Calculate discrete angles, considering the cell centered sample points of
    // the distribution
    let gw_half_phi = grid_width.phi / 2.;
    let gw_half_theta = grid_width.theta / 2.;
    let angles_phi = Array::linspace(0. + gw_half_phi, TWOPI - gw_half_phi, grid_size.phi);
    let angles_theta = Array::linspace(0. + gw_half_theta, PI - gw_half_theta, grid_size.theta);

    let a = stress.active;
    let b = stress.magnetic;

    for (mut ax1, phi) in s.axis_iter_mut(Axis(2)).zip(&angles_phi) {
        for (mut e, theta) in ax1.axis_iter_mut(Axis(2)).zip(&angles_theta) {
            let s = stress_active(*phi, *theta) * a + stress_magnetic(*phi, *theta) * b;

            e.assign(&s);

            // Already taken care of by the modified cell average in the distribution code
            // e *= theta.sin();
        }
    }

    s
}

/// Calculate active stress tensor for polar angles `phi` and `theta`.
fn stress_active(phi: f64, theta: f64) -> Array<f64, Ix2> {
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
fn stress_magnetic(phi: f64, theta: f64) -> Array<f64, Ix2> {
    let mut s = Array::zeros((3, 3));

    s[[0, 0]] = 0.;
    s[[0, 1]] = phi.cos() * theta.sin();
    s[[0, 2]] = 0.;

    s[[1, 0]] = (-1.) * phi.cos() * theta.sin();
    s[[1, 1]] = 0.;
    s[[1, 2]] = (-1.) * theta.cos();

    s[[2, 0]] = 0.;
    s[[2, 1]] = theta.cos();
    s[[2, 2]] = 0.;

    s
}

/// It consumes `stress_field` and updates it given a stress kernel `kernel`
/// and a distribution `dist`. It returns the updated stress field.
pub fn average_stress<'a>(
    stress_field: ArrayViewMut<'a, Complex<f64>, Ix5>,
    kernel: &ArrayView<f64, Ix4>,
    dist: &Distribution,
) -> ArrayViewMut<'a, Complex<f64>, Ix5> {
    let dist_sh = dist.dim();
    let stress_sh = kernel.dim();

    let n_angle = dist_sh.3 * dist_sh.4;
    let n_stress = stress_sh.0 * stress_sh.1;
    let n_dist = dist_sh.0 * dist_sh.1 * dist_sh.2;

    let gs = dist.get_grid_size();
    let gw = dist.get_grid_width();

    // Put axis in order, so that components fields are continuous in memory,
    // so it can be passed to FFTW easily
    let stress = kernel.into_shape([n_stress, n_angle]).unwrap();

    let dist = dist.dist.view().into_shape([n_dist, n_angle]).unwrap();

    // let len = n_stress * n_dist;
    // let mut uninit = Vec::with_capacity(len);
    // unsafe {
    //     uninit.set_len(len);
    // }
    //
    // let mut stress_field = Array::from_vec(uninit)
    //     .into_shape((n_stress, n_dist))
    //     .unwrap();
    let mut stress_field = stress_field.into_shape((n_stress, n_dist)).unwrap();

    // integration measures and FFT normalization
    let norm = gw.phi * gw.theta / (gs.x as f64 * gs.y as f64 * gs.z as f64);

    // Calculating the integral over the orientation. `norm` includes weights for
    // integration and normalisation of DFT
    for (s, mut o1) in stress.outer_iter().zip(stress_field.outer_iter_mut()) {
        for (d, o2) in dist.outer_iter().zip(o1.iter_mut()) {
            *o2 = Complex::from(s.dot(&d) * norm)
        }
    }

    stress_field
        .into_shape([stress_sh.0, stress_sh.1, dist_sh.0, dist_sh.1, dist_sh.2])
        .unwrap()
}
