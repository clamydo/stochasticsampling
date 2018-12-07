#![allow(clippy::unreadable_literal,clippy::excessive_precision)]
use super::super::*;
use super::*;
use ndarray::arr2;
use crate::BoxSize;

const BS: BoxSize = BoxSize {
    x: 10.,
    y: 10.,
    z: 10.,
};

fn prepare() -> LangevinBuilder {
    let p = Particle::new(0., 0., 0., 0., 0., &BS);
    LangevinBuilder::new(&p)
}

macro_rules! quicktest_modifier {
    ($name:ident; $x:expr; ($($y:expr),+)) => (
        let l = prepare();
        let p = l.with_param(super::$name, $x).finalize(&BS);
        let expect = Particle::new($($y),*, &BS);

        assert_eq!(p, expect);
    );
    ($name:ident; ($($y:expr),+)) => (
        let l = prepare();
        let p = l.with(super::$name).finalize(&BS);
        let expect = Particle::new($($y),*, &BS);

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

#[test]
fn magnetic_dipole_dipole_force() {
    let grad_b = arr2(&[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]]);

    quicktest_modifier!(magnetic_dipole_dipole_force; (0.1, grad_b.view()); (0.1, 0.1, 0.1, 0., 0.));
}

#[test]
fn translational_diffusion() {
    quicktest_modifier!(translational_diffusion; [1., 1., 0.].into(); (1., 1., 0., 0., 0.));
}

#[test]
fn external_field_alignment() {
    let p = Particle::new(0., 0., 0., 0., ::std::f64::consts::PI / 2., &BS);
    let l = LangevinBuilder::new(&p);
    let p = l
        .with_param(super::external_field_alignment, 0.1)
        .finalize(&BS);
    let expect = Particle::new(0., 0., 0., 0.09966865249116204, ::std::f64::consts::PI / 2., &BS);

    assert_eq!(p, expect);
}

#[test]
fn jeffrey_vorticity() {
    let vortm = arr2(&[[0.0, 0.0, -0.01], [0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]);
    quicktest_modifier!(jeffrey_vorticity; vortm.view(); (0., 0., 0., 0., 0.009999666686665076));
}

#[test]
fn jeffrey_strain() {
    let strainm = arr2(&[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02], [0.01, 0.02, 0.0]]);
    quicktest_modifier!(jeffrey_vorticity; strainm.view(); (0., 0., 0., 4.2487413713838835, 0.022356954112670246));
}

#[test]
fn magnetic_dipole_dipole_rotation() {
    quicktest_modifier!(magnetic_dipole_dipole_rotation; [0.01, 0., 0.].into(); (0., 0., 0., 0., 0.009999666686665076));
}

#[test]
fn rotational_diffusion() {
    let r = RotDiff {
        axis_angle: 0.,
        rotate_angle: 0.1,
    };
    quicktest_modifier!(rotational_diffusion; &r; (0., 0., 0., 0., 0.09999999999999987));
}
