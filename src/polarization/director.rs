// Move unit test into own file
#[cfg(test)]
#[path = "./director_test.rs"]
mod director_test;

use crate::consts::TWOPI;
use crate::distribution::Distribution;
use crate::mesh::grid_width::GridWidth;
use crate::particle::Orientation;
use crate::GridSize;
use ndarray::{Array, Axis, Ix3, Ix4, Zip};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;

pub struct DirectorField {
    pub field: Array<Complex<f64>, Ix4>,
    pub grid_width: GridWidth,
    /// precomputed kernel for angular expectation value
    kernel: Array<f64, Ix3>,
}

impl DirectorField {
    pub fn new(grid_size: GridSize, grid_width: GridWidth) -> DirectorField {
        DirectorField {
            field: Array::default([3, grid_size.x, grid_size.y, grid_size.z]),
            grid_width: grid_width,
            kernel: orientation_kernel(grid_size, grid_width),
        }
    }

    pub fn from_distribution(&mut self, dist: &Distribution, threshold: Option<f64>) {
        let dist_sh = dist.dim();
        let n_angle = dist_sh.3 * dist_sh.4;
        let n_dist = dist_sh.0 * dist_sh.1 * dist_sh.2;

        let gw = dist.get_grid_width();

        // collapse dimension to ease calculations
        let kernel = self.kernel.view_mut().into_shape([3, n_angle]).unwrap();

        let dist = dist.dist.view().into_shape([n_dist, n_angle]).unwrap();
        let mut field = self.field.view_mut().into_shape([3, n_dist]).unwrap();

        // Integration measure. sin(theta) is already included.
        let measure = gw.phi * gw.theta;

        // Calculating the integral over the orientation. `norm` includes weights for
        // integration and normalisation of DFT
        Zip::from(field.axis_iter_mut(Axis(1)))
            .and(dist.outer_iter())
            .par_apply(|mut f, d| {
                for (f, kern) in f.iter_mut().zip(kernel.outer_iter()) {
                    *f = match threshold {
                        Some(t) => {
                            let c = d.sum() * measure;
                            if c > t {
                                Complex::from(kern.dot(&(&d * (t / c))) * measure)
                            } else {
                                Complex::from(kern.dot(&d) * measure)
                            }
                        }
                        None => Complex::from(kern.dot(&d) * measure),
                    }
                }
            });
    }
}

/// Calculates discrete orientation kernel, to be used in
/// the expectation value to obtain the stress tensor.
///
/// It returns n(theta, sin) = [sin(theta) cos(phi), sin(theta) sin(phi),
/// cos(theta)] as a discrete field over angles.
fn orientation_kernel(grid_size: GridSize, grid_width: GridWidth) -> Array<f64, Ix3> {
    let mut s = Array::<f64, _>::zeros((3, grid_size.phi, grid_size.theta));
    // Calculate discrete angles, considering the cell centered sample points of
    // the distribution
    let gw_half_phi = grid_width.phi / 2.;
    let gw_half_theta = grid_width.theta / 2.;
    let angles_phi = Array::linspace(0. + gw_half_phi, TWOPI - gw_half_phi, grid_size.phi);
    let angles_theta = Array::linspace(0. + gw_half_theta, PI - gw_half_theta, grid_size.theta);

    for (mut ax1, phi) in s.axis_iter_mut(Axis(1)).zip(&angles_phi) {
        for (mut e, theta) in ax1.axis_iter_mut(Axis(1)).zip(&angles_theta) {
            let o = Orientation::new(*phi, *theta).to_vector().v;
            // Create ndarray::Array from array
            let o = Array::from_vec(o.to_vec());

            e.assign(&o);

            // Gram's determinant already taken care of by the modified cell average in the
            // distribution code e *= theta.sin();
        }
    }

    s
}
