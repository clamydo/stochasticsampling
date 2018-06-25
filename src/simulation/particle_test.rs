use super::*;
use std::f64::consts::PI;
use test_helper::equal_floats;

#[test]
fn test_random_particles() {
    let bs = BoxSize {
        x: 1.,
        y: 2.,
        z: 3.,
    };

    let particles = Particle::place_isotropic(1000, bs, [1, 1]);

    for p in &particles {
        let Position { x, y, z } = p.position;

        assert!(0. <= x && x < 1.);
        assert!(0. <= y && x < 2.);
        assert!(0. <= z && x < 3.);
    }
}

#[test]
fn test_modulo() {
    let input = [
        [2. * ::std::f64::consts::PI, 2. * ::std::f64::consts::PI],
        [
            2. * ::std::f64::consts::PI + ::std::f64::EPSILON,
            2. * ::std::f64::consts::PI,
        ],
        [7., 4.],
        [7., -4.],
        [-7., 4.],
        [-7., -4.],
    ];
    let output = [0., 0., 3., 3., 1., 1.];

    for (i, o) in input.iter().zip(output.iter()) {
        let a = modulo(i[0], i[1]);
        assert!(
            equal_floats(a, *o),
            "in: {} mod {}, out: {}, expected: {}",
            i[0],
            i[1],
            a,
            *o
        );
    }

    // CAUTION: This is due floating point roundoff error
    assert!(modulo(-::std::f64::EPSILON, 2. * ::std::f64::consts::PI) != 0.9);
}

#[test]
fn test_ang_pbc() {
    let input = [
        [1., 0.],
        [1., PI],
        [1., -0.1],
        [1., PI + 0.1],
        [TWOPI, PI],
        [6.283185307179586, 1.5707963267948966],
        [PI, PI + 1.],
    ];
    let expect = [
        [1., 0.],
        [1., PI],
        [1. + PI, 0.09999999999999964],
        [1. + PI, PI - 0.1],
        [0., PI],
        [0., PI / 2.],
        [0., PI - 1.],
    ];

    for (i, e) in input.iter().zip(expect.iter()) {
        let (phi, theta) = ang_pbc(i[0], i[1]);

        assert!(
            equal_floats(phi, e[0]),
            "PHI; input: {:?}, expected: {}, output: {}",
            i,
            e[0],
            phi
        );
        assert!(
            equal_floats(theta, e[1]),
            "THETA; input: {:?}, expected: {}, output: {}",
            i,
            e[1],
            theta
        );
    }
}

#[test]
fn test_pdf_sin() {
    // TODO: Check statstics
    use std::f64::consts::PI;
    let input = [0., 1., 2.];
    let expect = [0., PI / 2., PI];
    let output: Vec<_> = input.iter().map(|x| pdf_sin(*x)).collect();

    for ((i, o), e) in input.iter().zip(expect.iter()).zip(output.iter()) {
        assert!(equal_floats(*o, *e), "{} => {}, not {}", i, o, e)
    }
}

#[test]
fn test_orientation_from_orientation_vector() {
    let input = [
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 0.],
        [1., 0., 1.],
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
        [-1., 1., 0.],
        [1., 0., -1.],
        [15.23456, 0., 0.],
    ];

    let expect = [
        [PI / 2., 0.],
        [PI / 2., PI / 2.],
        [0., 0.],
        [PI / 2., PI / 4.],
        [PI / 4., 0.],
        [PI / 2., PI],
        [PI / 2., -PI / 2.],
        [PI, 0.],
        [PI / 2., 3. * PI / 4.],
        [3. / 4. * PI, 0.],
        [PI / 2., 0.],
    ];

    let mut o = Orientation::new(0., 0.);

    for (i, e) in input.iter().zip(expect.iter()) {
        o.from_vector_mut(&((*i).into()));

        assert!(
            equal_floats(e[0], o.theta),
            "input: {:?} -> theta {} != {}",
            i,
            o.theta,
            e[0]
        );
        assert!(
            equal_floats(e[1], o.phi),
            "input: {:?} -> phi {} != {}",
            i,
            o.phi,
            e[1]
        );
    }
}