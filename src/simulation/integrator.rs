
use coordinates::TWOPI;
use coordinates::particle::Particle;
use ndarray::{Array, Ix};
use settings::GridSize;
use super::DiffusionParameter;

/// Holds precomuted values
pub struct Integrator {
    /// First axis holds submatrices for different discrete angles.
    stress: Array<f64, (Ix, Ix, Ix)>,
}

impl Integrator {
    pub fn init(grid_size: GridSize) {
        let mut s = Array::<f64, _>::zeros((grid_size.2, 2, 2));
        let angles = Array::linspace(0., TWOPI, grid_size.2);

        // hi, that should be relatively easy to do with subviews. A bit verbose maybe.
        // .subview(Axis(1), 1).subview(Axis(1), 1) is a one-dimensional array of all
        // the ten A_11 entries. You can also do that with slicing. .slice(s![.., 1..,
        // 1..]) is a 3D array with dimensions 10x1x1

        for (mut e, a) in s.outer_iter_mut().zip(&angles) {
            e[[0, 0]] = 0.5 * (2. * a).cos();
            e[[0, 1]] = -a.sin() * a.sin() * a.cos();
            e[[1, 0]] = -e[[0, 1]];
            e[[1, 1]] = -e[[0, 0]];
        }
    }
}

pub fn evolve_inplace<F>(p: &mut Particle, diffusion: &DiffusionParameter, timestep: f64, mut c: F)
    where F: FnMut() -> f64
{
    // Y(t) = sqrt(t) * X(t), if X is normally distributed with variance 1, then
    // Y is normally distributed with variance t.
    let trans_diff_step = timestep * diffusion.dt;
    let rot_diff_step = timestep * diffusion.dr;

    p.position += c() * trans_diff_step;
    p.orientation += c() * rot_diff_step;
}





#[cfg(test)]
mod tests {
    use coordinates::particle::Particle;
    use super::*;
    use super::super::DiffusionParameter;

    #[test]
    fn test_evolve() {
        let mut p = Particle::new(0.4, 0.5, 1., (1., 1.));
        let d = DiffusionParameter { dt: 1., dr: 2. };

        let t = 1.;
        let c = || 0.1;

        evolve_inplace(&mut p, &d, t, c);

        assert_eq!(p.position.x.v, 0.5);
        assert_eq!(p.position.y.v, 0.6);
        assert_eq!(p.orientation.v, 1.2);
    }
}
