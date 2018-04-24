use super::*;
use simulation::settings::{BoxSize, GridSize};
use simulation::particle::Particle;

#[test]
fn test_polarization_from_distribution() {
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

    let gw = GridWidth::new(gs, bs);

    let p = Particle::new(0.0, 0.0, 0.0, PI, PI / 2., bs);
    let mut d = Distribution::new(gs, bs);
    let p = vec![p];
    d.sample_from(&p);

    let mut p = DirectorField::new(gs, gw);

    p.from_distribution(&d);

    // assert!(false, "{:?}", p.field)
    assert_eq!(p.field[[0, 0, 0, 0]].re, -1.0_f64);
    assert_eq!((p.field[[1, 0, 0, 0]].re * 10e14).round(), 0.0_f64);
    assert_eq!((p.field[[2, 0, 0, 0]].re * 10e14).round(), 0.0_f64);


    p.field.slice_mut(s![.., 0, 0, 0])
        .map_inplace(|v| *v = 0.0_f64.into());

    for (i, v) in p.field.iter().enumerate() {
        assert!(*v == 0.0f64.into(), "Value at index {} should be zero, but is {}.", i, v);
    }
}
