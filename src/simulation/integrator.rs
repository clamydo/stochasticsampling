
use coordinates::TWOPI;
use coordinates::particle::Particle;
use ndarray::{Array, Ix};
use settings::{DiffusionConstants, GridSize, StressPrefactors};
use super::DiffusionParameter;
use super::distribution::GridWidth;

/// Holds precomuted values
#[derive(Debug)]
pub struct Integrator {
    /// First axis holds submatrices for different discrete angles.
    stress: Array<f64, (Ix, Ix, Ix)>,
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

        // hi, that should be relatively easy to do with subviews. A bit verbose maybe.
        // .subview(Axis(1), 1).subview(Axis(1), 1) is a one-dimensional array of all
        // the ten A_11 entries. You can also do that with slicing. .slice(s![.., 1..,
        // 1..]) is a 3D array with dimensions 10x1x1

        for (mut e, a) in s.outer_iter_mut().zip(&angles) {
            e[[0, 0]] = stresses.active * 0.5 * (2. * a).cos();
            e[[0, 1]] = stresses.active * a.sin() * a.cos() - stresses.magnetic * a.sin();
            e[[1, 0]] = stresses.active * a.sin() * a.cos() + stresses.magnetic * a.sin();
            e[[1, 1]] = -e[[0, 0]];
        }

        Integrator { stress: s }
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
