use super::mesh3d;
use consts::TWOPI;
use ndarray::{Array, ArrayView, Axis, Ix1, Ix3, Ix4};
use num_complex::Complex;
use simulation::settings::{BoxSize, GridSize};

/// Returns a sampling of k values along all grid axes in FFTW standard form.
/// In this case 3D.
///
/// For a grid size of `n`, the 0th-mode is at index `0`. For even n the index
/// `n/2` represents both the largest positive and negative frequency. For odd
/// `n` index `(n-1)/2` is the largest positive frequency and `(n+1)/2` the
/// largest negative frequency. For the values at index `i`, `-i = n-k` holds
/// true.
///
/// Example:
///     n = 10 => k = [0, 1, 2, 3, 4, (5, -5), -4, -3, -2, -1]
///     n = 11 => k = [0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1]
///
fn get_k_sampling(grid_size: GridSize, box_size: BoxSize) -> Vec<Array<Complex<f64>, Ix1>> {
    let ks: Vec<Array<Complex<f64>, Ix1>> = [grid_size.x, grid_size.y, grid_size.z]
        .iter()
        .zip([box_size.x, box_size.y, box_size.z].iter())
        .map(|(gs, bs)| {
            let a = (gs / 2) as isize;
            let b = if gs % 2 == 0 { gs / 2 } else { gs / 2 + 1 } as isize;
            let step = TWOPI / bs;

            let values: Array<Complex<f64>, Ix1> = Array::from_vec(
                (-(a as i64)..(b as i64))
                    .into_iter()
                    .map(|i| Complex::new((i as f64) * step, 0.))
                    .collect(),
            );

            let mut k = Array::from_elem(*gs, Complex::new(0., 0.));

            k.slice_mut(s![..b]).assign(&values.slice(s![a..]));
            k.slice_mut(s![b..]).assign(&values.slice(s![..a]));

            k
        })
        .collect();

    ks
}

/// Returns a meshgrid of k values for FFT.
///
/// The first axis denotes the components of the k-vector:
///     `res[c, i, j, m] -> k_c[i, j, m]`
pub fn get_k_mesh(grid_size: GridSize, box_size: BoxSize) -> Array<Complex<f64>, Ix4> {
    let ks = get_k_sampling(grid_size, box_size);
    mesh3d::<Complex<f64>>(&ks)
}

/// Returns a normalized meshgrid of k values for FFT, except for zero which is zero.
///
/// The first axis denotes the components of the k-vector:
///     `res[c, i, j, m] -> k_c[i, j, m]`
pub fn get_norm_k_mesh(grid_size: GridSize, box_size: BoxSize) -> Array<Complex<f64>, Ix4> {
    let ks = get_k_sampling(grid_size, box_size);
    let mesh = mesh3d::<Complex<f64>>(&ks);

    let kinv = get_inverse_norm(mesh.view());

    &mesh * &kinv
}

/// Returns scalar field of inversed norm squared of k-vector-values.
///
/// The inverse norm of k=0 is set to zero, i.e. 1/(k=0)^2 == 0
pub fn get_inverse_norm_squared(k_mesh: ArrayView<Complex<f64>, Ix4>) -> Array<Complex<f64>, Ix3> {
    let squared = &k_mesh * &k_mesh;

    let mut inorm = squared.sum_axis(Axis(0)).map(|v| 1. / v);
    inorm[[0, 0, 0]] = Complex::new(0., 0.);

    inorm
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;
    use test_helper::equal_floats;

    #[test]
    fn test_get_k_sampling() {
        let bs = BoxSize {
            x: 6.,
            y: 7.,
            z: TWOPI,
        };
        let gs = GridSize {
            x: 6,
            y: 7,
            z: 3,
            phi: 1,
            theta: 1,
        };

        let k = get_k_sampling(gs, bs);

        let expect0 = [
            0.,
            1.0471975511965976,
            2.0943951023931953,
            -3.1415926535897931,
            -2.0943951023931953,
            -1.0471975511965976,
        ];
        let expect1 = [
            0.,
            0.8975979010256552,
            1.7951958020513104,
            2.6927937030769655,
            -2.6927937030769655,
            -1.7951958020513104,
            -0.8975979010256552,
        ];

        let expect2 = [0., 1., -1.];

        for (v, e) in k[0].iter().zip(&expect0) {
            assert!(equal_floats(v.re, *e), "{} != {}", v.re, *e);
        }

        for (v, e) in k[1].iter().zip(&expect1) {
            assert!(equal_floats(v.re, *e), "{} != {}", v.re, *e);
        }

        for (v, e) in k[2].iter().zip(&expect2) {
            assert!(equal_floats(v.re, *e), "{} != {}", v.re, *e);
        }
    }

    #[test]
    fn test_get_k_mesh() {
        let bs = BoxSize {
            x: TWOPI,
            y: TWOPI,
            z: TWOPI,
        };
        let gs = GridSize {
            x: 4,
            y: 3,
            z: 2,
            phi: 1,
            theta: 1,
        };

        let mesh = get_k_mesh(gs, bs);

        let expect = [
            [
                [[0., 0.], [0., 0.], [0., 0.]],
                [[1., 1.], [1., 1.], [1., 1.]],
                [[-2., -2.], [-2., -2.], [-2., -2.]],
                [[-1., -1.], [-1., -1.], [-1., -1.]],
            ],
            [
                [[0., 0.], [1., 1.], [-1., -1.]],
                [[0., 0.], [1., 1.], [-1., -1.]],
                [[0., 0.], [1., 1.], [-1., -1.]],
                [[0., 0.], [1., 1.], [-1., -1.]],
            ],
            [
                [[0., -1.], [0., -1.], [0., -1.]],
                [[0., -1.], [0., -1.], [0., -1.]],
                [[0., -1.], [0., -1.], [0., -1.]],
                [[0., -1.], [0., -1.], [0., -1.]],
            ],
        ];

        let expect: Vec<f64> = expect
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().cloned())
            .collect();

        let expect = Array::from_vec(expect).into_shape([3, 4, 3, 2]).unwrap();
        assert_eq!(expect.shape(), [3, 4, 3, 2]);

        println!("{}", mesh);

        for (v, e) in mesh.iter().zip(expect.iter()) {
            assert!(equal_floats(v.re, *e), "{} != {:?}", v.re, *e);
        }
    }

    #[test]
    fn test_get_norm_k_mesh() {
        let bs = BoxSize {
            x: TWOPI,
            y: TWOPI,
            z: TWOPI,
        };
        let gs = GridSize {
            x: 4,
            y: 3,
            z: 2,
            phi: 1,
            theta: 1,
        };

        let mesh = get_norm_k_mesh(gs, bs);

        let expect = [
            [
                [[0., 0.], [0., 0.], [0., 0.]],
                [[1., 0.5], [1., 1.], [1., 1.]],
                [[-2., -2.], [-2., -2.], [-2., -2.]],
                [[-1., -1.], [-1., -1.], [-1., -1.]],
            ],
            [
                [[0., 0.], [1., 1.], [-1., -1.]],
                [[0., 0.], [1., 1.], [-1., -1.]],
                [[0., 0.], [1., 1.], [-1., -1.]],
                [[0., 0.], [1., 1.], [-1., -1.]],
            ],
            [
                [[0., -1.], [0., -1.], [0., -1.]],
                [[0., -1.], [0., -1.], [0., -1.]],
                [[0., -1.], [0., -1.], [0., -1.]],
                [[0., -1.], [0., -1.], [0., -1.]],
            ],
        ];

        let expect: Vec<f64> = expect
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().cloned())
            .collect();

        let expect = Array::from_vec(expect).into_shape([3, 4, 3, 2]).unwrap();
        assert_eq!(expect.shape(), [3, 4, 3, 2]);


        println!("{}", mesh);

        for (v, (i, e)) in mesh.iter().zip(expect.iter().enumerate()) {
            assert!(equal_floats(v.re, *e), "{} != {:?} at index {}", v.re, *e, i);
        }
    }

    #[test]
    fn test_get_inverse_norm_squared() {
        let bs = BoxSize {
            x: TWOPI,
            y: TWOPI,
            z: TWOPI,
        };
        let gs = GridSize {
            x: 4,
            y: 3,
            z: 2,
            phi: 1,
            theta: 1,
        };

        let mesh = get_k_mesh(gs, bs);

        let inorm = get_inverse_norm_squared(mesh.view());

        let expect = arr3(&[
            [
                [0.0, 1.000000000000000],
                [1.000000000000000, 0.5000000000000000],
                [1.000000000000000, 0.5000000000000000],
            ],
            [
                [1.000000000000000, 0.5000000000000000],
                [0.5000000000000000, 0.3333333333333333],
                [0.5000000000000000, 0.3333333333333333],
            ],
            [
                [0.2500000000000000, 0.2000000000000000],
                [0.2000000000000000, 0.1666666666666667],
                [0.2000000000000000, 0.1666666666666667],
            ],
            [
                [1.000000000000000, 0.5000000000000000],
                [0.5000000000000000, 0.3333333333333333],
                [0.5000000000000000, 0.3333333333333333],
            ],
        ]);

        println!("{}", inorm);

        for (v, e) in inorm.iter().zip(expect.iter()) {
            assert!(equal_floats(v.re, *e), "{} != {}", v.re, *e);
        }
    }

}
