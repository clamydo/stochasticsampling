use coordinates::TWOPI;
use coordinates::particle::Particle;
use ndarray::{Array, ArrayView, Axis, Ix};
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

        // Calculate just first component

        // Calculates (grad Psi)_i * stress_kernel_(i, j) for every point on the grid and j = 0.
        // This makes implicit and explicit use of broadcasting. Implicetly the stress kernel ´sk´
        // is broadcasted for all points in space. Then, explicetly, the gradient ´g´ is
        // broadcasted along the last axis. This effectively repeats the gradient along the second
        // index of the stress kernel ´sk´. Multiplying it elementwise results in
        //
        //  [[g_1, g_1],  *  [[sk_11, sk_12],  =  [[g_1 * sk_11, g_1 * sk_12],
        //   [g_2, g_2]]      [sk_21, sk_22]]      [g_2 * sk_21, g_2 * sk_22]
        //
        // Now buy summing along the first index of the matrix, it results in
        // [ g_1 * sk_11 + g_2 * sk_21, g_1 * sk_12 + g_2 * sk_22]
        //
        // This is done for every point (x, y, alpha).

        let g = dist.spatgrad();
        let ref sk = self.stress_kernel;

        let sh = g.dim();
        let sh_a = (sh.0, sh.1, sh.2, sh.3, 1);
        let sh_b = (sh.0, sh.1, sh.2, sh.3, 2);

        let int = (g.into_shape(sh).unwrap().broadcast(sh_b).unwrap().to_owned() * sk).sum(Axis(3));

        // Implement Simpson's rule integration as a fold. Maybe do this in loops.
        int.map_axis(Axis(2), |v| periodic_simpson_integrate(v, h.a))
    }
}

/// Implements Simpon's Rule integration on an array, representing sampled points of a periodic
/// function.
fn periodic_simpson_integrate(samples: ArrayView<f64, Ix>, h: f64) -> f64 {
    let len = samples.dim();

    assert!(len % 2 == 0,
            "Periodic Simpson's rule only works for even number of sample points, since the \
             first point in the integration interval is also the last.");

    unsafe {
        let mut s = samples.uget(0) + samples.uget(0);

        for i in 1..(len / 2) {
            s += 2. * samples.uget(2 * i);
            s += 4. * samples.uget(2 * i - 1);
        }

        s += 4. * samples.uget(len - 1);
        s * h / 3.
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
    use ndarray::{Array, Axis};
    use settings::{DiffusionConstants, GridSize, StressPrefactors};
    use super::*;
    use super::super::distribution::GridWidth;
    use std::f64::consts::PI;
    use std::f64::EPSILON;

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

    #[test]
    fn test_simpson() {
        let h = PI / 100.;
        let f = Array::range(0., PI, h).map(|x| x.sin());
        let integral = super::periodic_simpson_integrate(f.view(), h);

        assert!((integral - 2.000000010824505).abs() < EPSILON,
                "h: {}, result: {}",
                h,
                integral);


        let h = 4. / 100.;
        let f = Array::range(0., 4., h).map(|x| x * x);
        let integral = super::periodic_simpson_integrate(f.view(), h);
        assert!((integral - 21.120000000000001).abs() < EPSILON,
                "h: {}, result: {}",
                h,
                integral);
    }

    #[test]
    fn test_simpson_map_axis() {
        let points = 100;
        let h = PI / points as f64;
        let f = Array::range(0., PI, h).map(|x| x.sin())
            .into_shape((1, 1, points)).unwrap()
            .broadcast((10, 10, points)).unwrap().to_owned();

        let integral = f.map_axis(Axis(2), |v| super::periodic_simpson_integrate(v, h));

        for e in integral.iter() {
            assert!((e - 2.000000010824505).abs() < EPSILON);
        }
    }
}
