use super::super::*;
use super::*;
use simulation::settings::BoxSize;

const BS: BoxSize = BoxSize {
    x: 10.,
    y: 10.,
    z: 10.,
};

fn prepare() -> LangevinBuilder {
    let p = Particle::new(0., 0., 0., 0., 0., BS);
    LangevinBuilder::new(&p)
}

macro_rules! quicktest_modifier {
    ($name:ident; $x:expr; ($($y:expr),+)) => (
        let l = prepare();
        let p = l.with_param(super::$name, $x).finalize(BS);
        let expect = Particle::new($($y),*, BS);

        assert_eq!(p, expect);
    );
    ($name:ident; ($($y:expr),+)) => (
        let l = prepare();
        let p = l.with(super::$name).finalize(BS);
        let expect = Particle::new($($y),*, BS);

        assert_eq!(p, expect);
    );
}

#[test]
fn self_propulsion() {
    quicktest_modifier!(self_propulsion; (0., 0., 1., 0., 0.));
}

#[test]
fn convection() {
    quicktest_modifier!(convection; [1., 1., 0.].into(); (1., 1., 0., 0., 0.));
}
