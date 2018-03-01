use ndarray::{ArrayView, Axis, Ix1};

/// Implements Simpon's Rule integration on an array, representing sampled
/// points of a periodic function.
#[allow(dead_code)]
pub fn periodic_simpson_integrate(samples: ArrayView<f64, Ix1>, h: f64) -> f64 {
    let len = samples.dim();

    assert!(
        len % 2 == 0,
        "Periodic Simpson's rule only works for even number of sample points, since the \
         first point in the integration interval is also the last. Please specify an even \
         number of grid cells."
    );

    unsafe {
        let mut s = samples.uget(0) + samples.uget(0);

        for i in 1..(len / 2) {
            s += 2. * samples.uget(2 * i);
            s += 4. * samples.uget(2 * i - 1);
        }

        s += 4. * samples.uget(len - 1);
        s * h / 3.
    }
}

/// Implements most straight forward step-sum integration method
pub fn integrate(samples: ArrayView<f64, Ix1>, h: f64) -> f64 {
    (&samples * h).sum(Axis(0))[()]
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array;
    use std::f64::consts::PI;
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
        let h = PI / points as f64;
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
}
