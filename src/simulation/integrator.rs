use coordinates::Particle;
use super::DiffusionParameter;


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
    use coordinates::Particle;
    use coordinates::modulofloat::Mf64;
    use coordinates::vector::Mod64Vector2;
    use super::*;
    use super::super::DiffusionParameter;

    #[test]
    fn test_evolve() {
        let mut p = Particle {
            position: Mod64Vector2::new(0.4, 0.5, (1., 1.)),
            orientation: Mf64::new(1., 2. * ::std::f64::consts::PI),
        };

        let d = DiffusionParameter { dt: 1., dr: 2. };

        let t = 1.;
        let c = || 0.1;

        evolve_inplace(&mut p, &d, t, c);

        assert_eq!(p.position.x.v, 0.5);
        assert_eq!(p.position.y.v, 0.6);
        assert_eq!(p.orientation.v, 1.2);
    }
}
