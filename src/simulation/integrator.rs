use coordinates::Particle;
use coordinates::vector::Mod64Vector2;
use super::DiffusionParameter;


pub fn evolve<F>(pos: &Particle,
                 diffusion: &DiffusionParameter,
                 timestep: f64,
                 mut c: F)
                 -> Particle
    where F: FnMut() -> f64
{

    // Y(t) = sqrt(t) * X(t), if X is normally distributed with variance 1, then
    // Y is normally distributed with variance t.
    let trans_diff_step = timestep * diffusion.dt;
    let rot_diff_step = timestep * diffusion.dr;
    let Mod64Vector2 { ref x, ref y } = pos.position;

    Particle {
        position: Mod64Vector2 {
            x: *x + c() * trans_diff_step,
            y: *y + c() * trans_diff_step,
        },
        orientation: pos.orientation + c() * rot_diff_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coordinates::Particle;
    use coordinates::vector::Mod64Vector2;
    use super::super::DiffusionParameter;

    #[test]
    fn test_evolve() {
        let mut p = Particle {
            position: Mod64Vector2::new(0.4, 0.5),
            orientation: 1.,
        };

        let d = DiffusionParameter { dt: 1., dr: 2. };

        let t = 1.;
        let c = || 0.1;

        p = evolve(&p, &d, t, c);

        assert_eq!(*p.position.x.as_ref(), 0.5);
        assert_eq!(*p.position.y.as_ref(), 0.6);
        assert_eq!(p.orientation, 1.2);
    }
}
