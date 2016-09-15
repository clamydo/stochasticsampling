use coordinates::TWOPI;
use coordinates::particle::Particle;
use ndarray::{Array, Ix};
use settings::{DiffusionConstants, GridSize, StressPrefactors};
use super::DiffusionParameter;
use super::distribution::GridWidth;
use super::distribution::Distribution;

/// Holds precomuted values
#[derive(Debug)]
pub struct Integrator {
    /// First axis holds submatrices for different discrete angles.
    stress_kernel: Array<f64, (Ix, Ix, Ix)>,
}

impl Integrator {
    /// Returns a new instance of the mc_sampling integrator
    pub fn new(grid_size: GridSize,
               grid_width: GridWidth,
               stresses: &StressPrefactors)
               -> Integrator {
        let mut s = Array::<f64, _>::zeros((grid_size.2, 2, 2));
        // Calculate discrete angles, considering the cell centered sample points of
        // the distribution
        let angles = Array::linspace(0. + grid_width.a / 2.,
                                     TWOPI - grid_width.a / 2.,
                                     grid_size.2);

        for (mut e, a) in s.outer_iter_mut().zip(&angles) {
            e[[0, 0]] = stresses.active * 0.5 * (2. * a).cos();
            e[[0, 1]] = stresses.active * a.sin() * a.cos() - stresses.magnetic * a.sin();
            e[[1, 0]] = stresses.active * a.sin() * a.cos() + stresses.magnetic * a.sin();
            e[[1, 1]] = -e[[0, 0]];
        }

        Integrator { stress_kernel: s }
    }

    /// Calculates force on the flow field because of the sress contributions.
    /// The first axis represents the direction of the derivative, the other
    /// correspond to the spatial dimension.
    /// 
    /// self.stress is a 2 x 2 matrix sampled for different angles,
    /// resulting in 2 x 2 x a
    /// dist is a 2 x l x t x a, with l x t spatial samples
    /// Want to calculate the matrix product of the transpose of the first to
    /// axies of self.stress and the first axis of dist.
    ///
    /// In Python's numpy, I'd do
    /// ´´´
    /// t[0, nx, ny, :] = s[:, 0, 0] * d[0, nx, ny, :] + s[:, 1, 0] * d[1, nx, ny, :] 
    /// t[1, nx, ny, :] = s[:, 0, 1] * d[0, nx, ny, :] + s[:, 1, 1] * d[1, nx, ny, :] 
    /// ´´´
    ///
    /// Followed by an simpson rule integration along the angular axis for 
    /// every component of the flow field u and for every point in 
    /// space (nx, ny).
    /// 
    /// Example suggestive implementation in Python for first component of u
    /// and position (nx, ny).
    /// ´´´
    /// l = t[0, nx, ny, 0] + t[0, nx, ny, -1]
    ///
    /// for i in range(1, na, 2):
    ///     l += 4 * t[0, nx, ny, i]
    ///
    /// for i in range(2, na - 1, 2):
    ///     l += 2 * t[0, nx, ny, i]
    ///
    /// f[0, nx, ny] = l * grid_width_angle / 3 
    /// ´´´
    fn calc_stress_gradient(&self, dist: &Distribution) -> Array<f64, (Ix, Ix, Ix)> {
        let h = dist.get_grid_width();

        let shape = dist.shape();

        let mut res = Array::zeros((shape.0, shape.1, 2));

        // Calculate just first component

        // Calculates (grad Psi)_i * stress_kernel_(i, j) for every point on the
        // grid and j = 0.
        // The subview returns only the first colum of every submatrix of the
        // stress kernel. This makes use of broadcasting the stress kernel to
        // every point on the grid. Finally the last axis ist contracted.
        //
        // Two calculate both components at the same time, maybe
        // `dist.spatgrad()` could be broadcasted along the last axis? So
        // basically adding an axis of size one at the end and broadcast to 2. 
        let x_comp = (dist.spatgrad() * self.stress_kernel.subview(Axis(2), 0)).sum(Axis(3)); 
        let y_comp = (dist.spatgrad() * self.stress_kernel.subview(Axis(2), 0)).sum(Axis(3)); 

    }
}


pub fn evolve_inplace<F>(p: &mut Particle, diffusion: &DiffusionConstants, timestep: f64, mut c: F)
    where F: FnMut() -> f64
{
    // Y(t) = sqrt(t) * X(t), if X is normally distributed with variance 1, then
    // Y is normally distributed with variance t.
    let trans_diff_step = timestep * diffusion.translational;
    let rot_diff_step = timestep * diffusion.rotational;

    p.position += c() * trans_diff_step;
    p.orientation += c() * rot_diff_step;
}


#[cfg(test)]
mod tests {
    use coordinates::TWOPI;
    use coordinates::particle::Particle;
    use settings::{DiffusionConstants, GridSize, StressPrefactors};
    use super::*;
    use super::super::distribution::GridWidth;

    #[test]
    fn new() {
        let gs = (10, 10, 3);
        let gw = GridWidth {
            x: 1.,
            y: 1.,
            a: TWOPI / gs.2 as f64,
        };
        let s = StressPrefactors {
            active: 1.,
            magnetic: 1.,
        };

        let i = Integrator::new(gs, gw, &s);

        unimplemented!();
    }

    #[test]
    fn test_evolve() {
        let mut p = Particle::new(0.4, 0.5, 1., (1., 1.));
        let d = DiffusionConstants {
            translational: 1.,
            rotational: 2.,
        };

        let t = 1.;
        let c = || 0.1;

        evolve_inplace(&mut p, &d, t, c);

        assert_eq!(p.position.x.v, 0.5);
        assert_eq!(p.position.y.v, 0.6);
        assert_eq!(p.orientation.v, 1.2);
    }
}
