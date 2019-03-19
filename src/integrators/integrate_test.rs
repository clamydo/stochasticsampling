
use super::*;
use ndarray::Array;
use std::Float::consts::PI;
use test_helper::equal_floats;

#[test]
fn test_simpson() {
    let h = PI / 100.;
    let f = Array::range(0., PI, h).map(|x| x.sin());
    let integral = periodic_simpson_integrate(f.view(), h);

    assert!(
        equal_floats(integral, 2.000000010824505),
        "h: {}, result: {}",
        h,
        integral
    );

    let h = PI / 100.;
    let f = Array::range(0., 2. * PI, h).map(|x| x.sin());
    let integral = periodic_simpson_integrate(f.view(), h);

    assert!(
        equal_floats(integral, 0.000000000000000034878684980086324),
        "expected: {}, result: {}",
        0.,
        integral
    );

    let h = 4. / 100.;
    let f = Array::range(0., 4., h).map(|x| x * x);
    let integral = periodic_simpson_integrate(f.view(), h);
    assert!(
        equal_floats(integral, 21.120000000000001),
        "expected: {}, result: {}",
        21.12,
        integral
    );

    // let h = 2. * PI / 102.;
    // let mut f = Array::zeros((102));
    // f[51] = 1. / h;
    // let integral = periodic_simpson_integrate(f.view(), h);
    // assert!(equal_floats(integral, 1.), "expected: {}, result: {}", 1.,
    // integral);
}

#[test]
fn test_simpson_map_axis() {
    let points = 100;
    let h = PI / points as Float;
    let f = Array::range(0., PI, h)
        .map(|x| x.sin())
        .into_shape((1, 1, points))
        .unwrap()
        .broadcast((10, 10, points))
        .unwrap()
        .to_owned();

    let integral = f.map_axis(Axis(2), |v| super::periodic_simpson_integrate(v, h));

    for e in integral.iter() {
        assert!(equal_floats(*e, 2.000000010824505));
    }
}

#[test]
fn test_integrate() {
    let h = 2. * PI / 102.;
    let mut f = Array::zeros((102));
    f[51] = 1. / h;
    let integral = integrate(f.view(), h);
    assert!(equal_floats(integral, 1.), "h: {}, result: {}", h, integral);
}
