#![allow(clippy::float_cmp, clippy::unreadable_literal)]
use super::*;
use crate::distribution::Distribution;
use crate::particle::Particle;
use crate::test_helper::equal_floats;
use ndarray::s;
use ndarray::{Array, Ix1, Ix4};
#[cfg(feature = "single")]
use std::f32::consts::PI;
#[cfg(not(feature = "single"))]
use std::f64::consts::PI;

#[test]
#[ignore]
fn test_magnetic() {
    let bs = BoxSize {
        x: 3.,
        y: 3.,
        z: 1.,
    };
    let gs = GridSize {
        x: 3,
        y: 3,
        z: 1,
        phi: 35,
        theta: 5,
    };

    let p = Particle::new(0.0, 0.0, 0.0, PI, PI / 2., &bs);
    let mut d = Distribution::new(gs, bs);

    let p = vec![p];
    d.sample_from(&p);

    let mut solver = MagneticSolver::new(gs, bs);
    let (_b, _gb) = solver.mean_magnetic_field(&d);

    // println!("{:?}", b);
    // println!("{:?}", gb);
    unimplemented!();
}

#[test]
#[ignore]
// Is for some reason not deterministic. Probably FFTW3 does select different
// algorithms from time to time.
fn test_magnetic_field_against_cache() {
    use bincode;
    use std::fs::File;

    let mut f = File::open("test/magneticfield/b_test.bincode").unwrap();
    let cache_b: Array<Float, Ix4> = bincode::deserialize_from(&mut f).unwrap();

    let bs = BoxSize {
        x: 11.,
        y: 11.,
        z: 11.,
    };
    let gs = GridSize {
        x: 11,
        y: 11,
        z: 11,
        phi: 11,
        theta: 11,
    };

    let p = Particle::new(0.0, 0.0, 0.0, PI, PI / 2., &bs);
    let mut d = Distribution::new(gs, bs);

    let p = vec![p];
    d.sample_from(&p);
    d.dist *= bs.x * bs.y * bs.z;

    let mut solver = MagneticSolver::new(gs, bs);
    let (b, _) = solver.mean_magnetic_field(&d);

    let b = b.map(|v| v.re);

    // let mut f = File::create("test/magneticfield/b_test.bincode").unwrap(); bincode::serialize_into(&mut f, &b).unwrap();

    for (idx, value) in b.indexed_iter().zip(cache_b.indexed_iter()) {
        let (ia, va) = idx;
        let (_, vb) = value;

        let f: Float = (2.0 as Float).powi(51);

        let va = (va * f).round() / f;
        let vb = (vb * f).round() / f;

        assert!(equal_floats(va, vb), "{} != {} at {:?}", va, vb, ia);
    }
}

#[test]
fn test_magnetic_vector_gradient() {
    let bs = BoxSize {
        x: 2. * PI,
        y: 2. * PI,
        z: 2. * PI,
    };
    let gs = GridSize {
        x: 10,
        y: 10,
        z: 10,
        phi: 1,
        theta: 1,
    };

    // Construct fourier transformation of
    let mut b: Array<Complex<Float>, Ix4> = Array::zeros([3, 10, 10, 10]);

    b[[0, 1, 0, 0]] = Complex::new(0., -0.5);
    b[[0, 9, 0, 0]] = Complex::new(0., 0.5);

    let mut solver = MagneticSolver::new(gs, bs);

    solver.director_field.field.assign(&b);
    solver.update_gradient();

    let expected: Vec<Complex<Float>> = [
        1.,
        0.8090169943749475,
        0.30901699437494745,
        -0.30901699437494745,
        -0.8090169943749475,
        -1.,
        -0.8090169943749475,
        -0.30901699437494745,
        0.30901699437494745,
        0.8090169943749475,
    ]
    .iter()
    .map(|v| v.into())
    .collect();

    let expected = Array::from_vec(expected);

    assert_eq!(solver.gradient_meanb.slice(s![0, 0, .., 0, 0]), expected);

    let zero = Array::<Complex<Float>, Ix1>::zeros([10]);
    let one = Array::from_elem([10], Complex::new(1., 0.));

    assert_eq!(solver.gradient_meanb.slice(s![1, 0, .., 0, 0]), zero);
    assert_eq!(solver.gradient_meanb.slice(s![0, 1, .., 0, 0]), zero);
    assert_eq!(solver.gradient_meanb.slice(s![1, 1, .., 0, 0]), zero);

    assert_eq!(solver.gradient_meanb.slice(s![0, 0, 0, .., 0]), one);
    assert_eq!(solver.gradient_meanb.slice(s![1, 0, 0, .., 0]), zero);
    assert_eq!(solver.gradient_meanb.slice(s![0, 1, 0, .., 0]), zero);
    assert_eq!(solver.gradient_meanb.slice(s![1, 1, 0, .., 0]), zero);

    assert_eq!(solver.gradient_meanb.slice(s![0, 0, 0, 0, ..]), one);
    assert_eq!(solver.gradient_meanb.slice(s![1, 0, 0, 0, ..]), zero);
    assert_eq!(solver.gradient_meanb.slice(s![0, 1, 0, 0, ..]), zero);
    assert_eq!(solver.gradient_meanb.slice(s![1, 1, 0, 0, ..]), zero);
}
