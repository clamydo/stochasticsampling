use super::modifiers::*;
use super::*;
use crate::test_helper::{equal_floats, equal_floats_eps};
use crate::Float;
#[cfg(feature = "single")]
use std::f32::consts::PI;
#[cfg(feature = "single")]
use std::f32::EPSILON;
#[cfg(not(feature = "single"))]
use std::f64::consts::PI;
#[cfg(not(feature = "single"))]
use std::f64::EPSILON;

#[test]
fn test_langevin_builder_conversion() {
    let bs = BoxSize {
        x: 10.,
        y: 10.,
        z: 10.,
    };
    let coord = [
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., PI / 2.],
        [0., 0., 0., 0., PI],
        [1., 2., 3., 0., 0.],
        [0., 5., 0., PI / 2., PI / 2.],
    ];

    let exp = [
        [[0., 0., 0.], [0., 0., 1.]],
        [[0., 0., 0.], [1., 0., 0.]],
        [[0., 0., 0.], [0., 0., -1.]],
        [[1., 2., 3.], [0., 0., 1.]],
        [[0., 5., 0.], [0., 1., 0.]],
    ];

    for (p, e) in coord.iter().zip(exp.iter()) {
        let p = Particle::new(p[0], p[1], p[2], p[3], p[4], &bs);

        let m = LangevinBuilder::new(&p).with(identity);
        let Modification { old: v, .. } = m;
        let pp = m.finalize(&bs);

        println!("IN: {:?}", p);
        println!("VEC: {:?}", v.vector);
        println!("OUT: {:?}", pp);

        assert_eq!(p, pp);

        for (a, b) in e[0].iter().zip(v.vector.position.iter()) {
            assert!(equal_floats(*a, *b), "left: {}, right: {}", *a, *b);
        }
        for (a, b) in e[1].iter().zip(v.vector.orientation.iter()) {
            assert!(equal_floats(*a, *b), "left: {}, right: {}", *a, *b);
        }
    }
}

#[test]
fn test_langevin_builder_step() {
    let bs = BoxSize {
        x: 10.,
        y: 10.,
        z: 10.,
    };
    let p = Particle::new(0., 0., 0., 0., 0., &bs);
    let c: ParticleVector = Particle::new(1., 2., 3., 3. / 2. * PI, PI / 2., &bs).into();
    let e = [[2., 4., 6.], [0., -2., 0.]];

    let m = LangevinBuilder::new(&p)
        .with_param(constant, c)
        .step(&TimeStep(2.));
    let Modification { old: _, delta: v } = m;

    println!("IN: {:?}", p);
    println!("VEC: {:?}", v);

    for (a, b) in e[0].iter().zip(v.position.iter()) {
        assert!(equal_floats(*a, *b), "left: {}, right: {}", *a, *b);
    }
    for (a, b) in e[1].iter().zip(v.orientation.iter()) {
        assert!(
            equal_floats_eps(*a, *b, 2. * EPSILON),
            "left: {}, right: {}",
            *a,
            *b
        );
    }
}
