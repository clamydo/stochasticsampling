use consts::TWOPI;
use ndarray::{Array, Axis, Ix3, Ix4};
use num_complex::Complex;
use simulation::distribution::Distribution;
use simulation::mesh::grid_width::GridWidth;
use simulation::particle::Orientation;
use simulation::settings::GridSize;
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

    pub fn from_distribution(&mut self, dist: &Distribution) {
        let dist_sh = dist.dim();
        // let kern_sh = self.kernel.dim();

        let n_angle = dist_sh.3 * dist_sh.4;
        let n_dist = dist_sh.0 * dist_sh.1 * dist_sh.2;

        let gs = dist.get_grid_size();
        let gw = dist.get_grid_width();

        // collapse dimension to ease calculations
        let director = self.kernel.view_mut().into_shape([3, n_angle]).unwrap();

        let dist = dist.dist.view().into_shape([n_dist, n_angle]).unwrap();
        let mut field = self.field.view_mut().into_shape([3, n_dist]).unwrap();

        // integration measures and FFT normalization
        let norm = gw.phi * gw.theta / (gs.x as f64 * gs.y as f64 * gs.z as f64);

        // Calculating the integral over the orientation. `norm` includes weights for
        // integration and normalisation of DFT
        for (s, mut o1) in director.outer_iter().zip(field.outer_iter_mut()) {
            for (d, o2) in dist.outer_iter().zip(o1.iter_mut()) {
                *o2 = Complex::from(s.dot(&d) * norm)
            }
        }

        // Does not work, because Zip requires that all producers have exactly the same
        // shape. Zip::from(field.lanes_mut(Axis(1)))
        // .and(dist.lanes(Axis(0)))
        // .apply(|mut f, d| {
        //     let o = director.dot(&d).map(|v| Complex::new(v * norm, 0.));
        //     f.assign(&o);
        // });
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
