use super::*;
// use ndarray::Array;
use simulation::distribution::Distribution;
use simulation::particle::Particle;
use simulation::settings::MagneticDipolePrefactors;
use std::f64::consts::PI;
use test::Bencher;

#[test]
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
    let s = MagneticDipolePrefactors {
        magnetic_moment: 1.0,
    };

    let p = Particle::new(0.0, 0.0, 0.0, PI, PI / 2., bs);
    let mut d = Distribution::new(gs, bs);

    let p = vec![p];
    d.sample_from(&p);

    let mut solver = MagneticSolver::new(gs, bs, s);
    let (b, gb) = solver.mean_magnetic_field(&d);

    println!("{:?}", b);
    // TODO: Do some meaningfull test
    assert_eq!(b[[0, 0, 0, 0]], Complex::new(0., 0.));
    assert_eq!(b[[1, 0, 0, 0]], Complex::new(0., 0.));
    assert_eq!(b[[2, 0, 0, 0]], Complex::new(0., 0.));
}
