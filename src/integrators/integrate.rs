use ndarray::{ArrayView, Axis, Ix1};

/// Implements Simpon's Rule integration on an array, representing sampled
/// points of a periodic function.
pub fn periodic_simpson_integrate(samples: ArrayView<Float, Ix1>, h: Float) -> Float {
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
pub fn integrate(samples: ArrayView<Float, Ix1>, h: Float) -> Float {
    (&samples * h).sum(Axis(0))[()]
}
