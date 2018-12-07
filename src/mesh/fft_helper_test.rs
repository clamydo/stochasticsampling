#![allow(clippy::unreadable_literal, clippy::excessive_precision)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;
    use test_helper::equal_floats;

    #[test]
    #[allow(clippy::approx_constant)]
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
            y: 4,
            z: 4,
            phi: 1,
            theta: 1,
        };

        let mesh = get_norm_k_mesh(gs, bs);

        let expect = [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [
                        1.000000000000000,
                        0.7071067811865475,
                        0.4472135954999579,
                        0.7071067811865475,
                    ],
                    [
                        0.7071067811865475,
                        0.5773502691896258,
                        0.4082482904638630,
                        0.5773502691896258,
                    ],
                    [
                        0.4472135954999579,
                        0.4082482904638630,
                        0.3333333333333333,
                        0.4082482904638630,
                    ],
                    [
                        0.7071067811865475,
                        0.5773502691896258,
                        0.4082482904638630,
                        0.5773502691896258,
                    ],
                ],
                [
                    [
                        -1.000000000000000,
                        -0.8944271909999159,
                        -0.7071067811865475,
                        -0.8944271909999159,
                    ],
                    [
                        -0.8944271909999159,
                        -0.8164965809277260,
                        -0.6666666666666667,
                        -0.8164965809277260,
                    ],
                    [
                        -0.7071067811865475,
                        -0.6666666666666667,
                        -0.5773502691896258,
                        -0.6666666666666667,
                    ],
                    [
                        -0.8944271909999159,
                        -0.8164965809277260,
                        -0.6666666666666667,
                        -0.8164965809277260,
                    ],
                ],
                [
                    [
                        -1.000000000000000,
                        -0.7071067811865475,
                        -0.4472135954999579,
                        -0.7071067811865475,
                    ],
                    [
                        -0.7071067811865475,
                        -0.5773502691896258,
                        -0.4082482904638630,
                        -0.5773502691896258,
                    ],
                    [
                        -0.4472135954999579,
                        -0.4082482904638630,
                        -0.3333333333333333,
                        -0.4082482904638630,
                    ],
                    [
                        -0.7071067811865475,
                        -0.5773502691896258,
                        -0.4082482904638630,
                        -0.5773502691896258,
                    ],
                ],
            ],
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [
                        1.000000000000000,
                        0.7071067811865475,
                        0.4472135954999579,
                        0.7071067811865475,
                    ],
                    [
                        -1.000000000000000,
                        -0.8944271909999159,
                        -0.7071067811865475,
                        -0.8944271909999159,
                    ],
                    [
                        -1.000000000000000,
                        -0.7071067811865475,
                        -0.4472135954999579,
                        -0.7071067811865475,
                    ],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [
                        0.7071067811865475,
                        0.5773502691896258,
                        0.4082482904638630,
                        0.5773502691896258,
                    ],
                    [
                        -0.8944271909999159,
                        -0.8164965809277260,
                        -0.6666666666666667,
                        -0.8164965809277260,
                    ],
                    [
                        -0.7071067811865475,
                        -0.5773502691896258,
                        -0.4082482904638630,
                        -0.5773502691896258,
                    ],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [
                        0.4472135954999579,
                        0.4082482904638630,
                        0.3333333333333333,
                        0.4082482904638630,
                    ],
                    [
                        -0.7071067811865475,
                        -0.6666666666666667,
                        -0.5773502691896258,
                        -0.6666666666666667,
                    ],
                    [
                        -0.4472135954999579,
                        -0.4082482904638630,
                        -0.3333333333333333,
                        -0.4082482904638630,
                    ],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [
                        0.7071067811865475,
                        0.5773502691896258,
                        0.4082482904638630,
                        0.5773502691896258,
                    ],
                    [
                        -0.8944271909999159,
                        -0.8164965809277260,
                        -0.6666666666666667,
                        -0.8164965809277260,
                    ],
                    [
                        -0.7071067811865475,
                        -0.5773502691896258,
                        -0.4082482904638630,
                        -0.5773502691896258,
                    ],
                ],
            ],
            [
                [
                    [
                        0.0,
                        1.000000000000000,
                        -1.000000000000000,
                        -1.000000000000000,
                    ],
                    [
                        0.0,
                        0.7071067811865475,
                        -0.8944271909999159,
                        -0.7071067811865475,
                    ],
                    [
                        0.0,
                        0.4472135954999579,
                        -0.7071067811865475,
                        -0.4472135954999579,
                    ],
                    [
                        0.0,
                        0.7071067811865475,
                        -0.8944271909999159,
                        -0.7071067811865475,
                    ],
                ],
                [
                    [
                        0.0,
                        0.7071067811865475,
                        -0.8944271909999159,
                        -0.7071067811865475,
                    ],
                    [
                        0.0,
                        0.5773502691896258,
                        -0.8164965809277260,
                        -0.5773502691896258,
                    ],
                    [
                        0.0,
                        0.4082482904638630,
                        -0.6666666666666667,
                        -0.4082482904638630,
                    ],
                    [
                        0.0,
                        0.5773502691896258,
                        -0.8164965809277260,
                        -0.5773502691896258,
                    ],
                ],
                [
                    [
                        0.0,
                        0.4472135954999579,
                        -0.7071067811865475,
                        -0.4472135954999579,
                    ],
                    [
                        0.0,
                        0.4082482904638630,
                        -0.6666666666666667,
                        -0.4082482904638630,
                    ],
                    [
                        0.0,
                        0.3333333333333333,
                        -0.5773502691896258,
                        -0.3333333333333333,
                    ],
                    [
                        0.0,
                        0.4082482904638630,
                        -0.6666666666666667,
                        -0.4082482904638630,
                    ],
                ],
                [
                    [
                        0.0,
                        0.7071067811865475,
                        -0.8944271909999159,
                        -0.7071067811865475,
                    ],
                    [
                        0.0,
                        0.5773502691896258,
                        -0.8164965809277260,
                        -0.5773502691896258,
                    ],
                    [
                        0.0,
                        0.4082482904638630,
                        -0.6666666666666667,
                        -0.4082482904638630,
                    ],
                    [
                        0.0,
                        0.5773502691896258,
                        -0.8164965809277260,
                        -0.5773502691896258,
                    ],
                ],
            ],
        ];

        let expect: Vec<f64> = expect
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter())
            .flat_map(|v| v.iter().cloned())
            .collect();

        let expect = Array::from_vec(expect).into_shape([3, 4, 4, 4]).unwrap();
        assert_eq!(expect.shape(), [3, 4, 4, 4]);

        println!("{}", mesh);

        for (v, (i, e)) in mesh.iter().zip(expect.iter().enumerate()) {
            assert!(
                equal_floats(v.re, *e),
                "{} != {:?} at index {}",
                v.re,
                *e,
                i
            );
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
