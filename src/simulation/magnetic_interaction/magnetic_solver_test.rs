use super::*;
// use ndarray::Array;
use simulation::distribution::Distribution;
use simulation::particle::Particle;
use simulation::settings::MagneticDipolePrefactors;
use test::Bencher;

#[test]
fn test_magnetic() {
    let bs = BoxSize {
        x: 2.,
        y: 2.,
        z: 1.,
    };
    let gs = GridSize {
        x: 2,
        y: 2,
        z: 1,
        phi: 6,
        theta: 6,
    };
    let s = MagneticDipolePrefactors {
        magnetic_moment: 1.0,
    };

    let p = Particle::new(0.6, 0.3, 0., 0., 0., bs);
    let mut d = Distribution::new(gs, bs);

    let p = vec![p];
    d.sample_from(&p);

    let mut solver = MagneticSolver::new(gs, bs, s);
    let b = solver.mean_magnetic_field(&d);

    println!("{:?}", b);
    // TODO: Do some meaningfull test
    assert!(false);
}
