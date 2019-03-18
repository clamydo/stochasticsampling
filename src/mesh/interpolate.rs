use super::grid_width::GridWidth;
use crate::particle::Position;
use crate::vector::{NumVectorD, VectorD};
use crate::Float;
use lerp::Lerp;
use ndarray::{arr3, Array3, ArrayView, Axis, Ix3, Ix4};

pub fn interpolate_vector_field(
    position: &Position,
    v: &ArrayView<Float, Ix4>,
    gw: &GridWidth,
) -> VectorD {
    let p = Position {
        x: (position.x / gw.x + 0.5) % 1.0,
        y: (position.y / gw.y + 0.5) % 1.0,
        z: (position.y / gw.z + 0.5) % 1.0,
    };

    [
        trilinear_interpolation(
            p,
            get_neighbouring_cell_values(&v.index_axis(Axis(0), 0), position, gw),
        ),
        trilinear_interpolation(
            p,
            get_neighbouring_cell_values(&v.index_axis(Axis(0), 1), position, gw),
        ),
        trilinear_interpolation(
            p,
            get_neighbouring_cell_values(&v.index_axis(Axis(0), 2), position, gw),
        ),
    ]
    .into()
}

fn wrap_idx(i: [i32; 3], dim: (usize, usize, usize)) -> [usize; 3] {
    fn wrap(i: i32, d: i32) -> usize {
        ((i + d) % d) as usize
    }

    [
        wrap(i[0], dim.0 as i32),
        wrap(i[1], dim.1 as i32),
        wrap(i[2], dim.2 as i32),
    ]
}

fn relidx(lhs: NumVectorD<i32>, rhs: [i32; 3], dim: (usize, usize, usize)) -> [usize; 3] {
    let rhs: NumVectorD<i32> = rhs.into();
    let idx: [i32; 3] = (lhs + rhs).into();
    wrap_idx(idx, dim)
}

fn get_neighbouring_cell_values(
    v: &ArrayView<Float, Ix3>,
    position: &Position,
    gw: &GridWidth,
) -> Array3<Float> {
    let mut p = position.to_vector();
    let gw: VectorD = [gw.x, gw.y, gw.z].into();
    p /= gw;
    p -= 0.5;

    let idx: NumVectorD<i32> = [
        p[0].floor() as i32,
        p[1].floor() as i32,
        p[2].floor() as i32,
    ]
    .into();

    arr3(&[
        [
            [
                v[relidx(idx, [0, 0, 0], v.dim())],
                v[relidx(idx, [0, 0, 1], v.dim())],
            ],
            [
                v[relidx(idx, [0, 1, 0], v.dim())],
                v[relidx(idx, [0, 1, 1], v.dim())],
            ],
        ],
        [
            [
                v[relidx(idx, [1, 0, 0], v.dim())],
                v[relidx(idx, [1, 0, 1], v.dim())],
            ],
            [
                v[relidx(idx, [1, 1, 0], v.dim())],
                v[relidx(idx, [1, 1, 1], v.dim())],
            ],
        ],
    ])
}

fn trilinear_interpolation(relative_position: Position, values: Array3<Float>) -> Float {
    let c00 = values[[0, 0, 0]].lerp(values[[1, 0, 0]], relative_position.x);
    let c01 = values[[0, 0, 1]].lerp(values[[1, 0, 1]], relative_position.x);
    let c10 = values[[0, 1, 0]].lerp(values[[1, 1, 0]], relative_position.x);
    let c11 = values[[0, 1, 1]].lerp(values[[1, 1, 1]], relative_position.x);

    let c0 = c00.lerp(c10, relative_position.y);
    let c1 = c01.lerp(c11, relative_position.y);

    let c = c0.lerp(c1, relative_position.z);

    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::grid_width::GridWidth;
    use crate::test_helper::equal_floats;
    use crate::Float;
    use ndarray::{arr3, Array, Array3};

    #[test]
    fn trilinear_test() {
        let test = |x: Float, y: Float, z: Float, v: Float| {
            let p = Position { x, y, z };
            let a: Array3<Float> = arr3(&[[[0.0, -1.0], [1.0, 0.0]], [[1.0, 0.0], [2.0, 1.0]]]);

            assert!(equal_floats(trilinear_interpolation(p, a), v));
        };

        test(0.1, 0.4, 0.3, 0.2);
        test(0.4, 0.1, 0.9, -0.4);
        test(0.2, 0.1, 0.7, -0.4);
        test(0.5, 0.3, 0.3, 0.5);
        test(0.5, 0.3, 0.3, 0.5);
        test(0.9, 0.9, 0.9, 0.9);
        test(0.0, 0.0, 0.0, 0.0);
        test(0.0, 1.0, 0.0, 1.0);
        test(0.0, 0.0, 1.0, -1.0);
    }

    #[test]
    fn get_neighbouring_cell_values_test() {
        let test = |x: Float, y: Float, z: Float, v: Array3<Float>| {
            let gw = GridWidth {
                x: 2.0,
                y: 2.0,
                z: 2.0,
                phi: 0.0,
                theta: 0.0,
            };

            let p = Position { x, y, z };
            let a = Array::linspace(0., 999., 1000)
                .into_shape((10, 10, 10))
                .unwrap();

            let r = get_neighbouring_cell_values(&a.view(), &p, &gw);

            println!("{}", r);

            for (a, b) in v.iter().zip(r.iter()) {
                assert_eq!(a, b)
            }
        };

        let c = arr3(&[[[999., 990.], [909., 900.]], [[99., 90.], [9., 0.]]]);
        test(0.1, 0.2, 0.3, c);

        let c = arr3(&[[[999., 990.], [909., 900.]], [[99., 90.], [9., 0.]]]);
        test(19.9, 19.9, 19.9, c);

        let c = arr3(&[[[0., 1.], [10., 11.]], [[100., 101.], [110., 111.]]]);
        test(1.1, 1.1, 1.1, c);

        let c = arr3(&[[[990., 991.], [900., 901.]], [[90., 91.], [0., 1.]]]);
        test(0.1, 0.1, 1.1, c);

        let c = arr3(&[[[0., 1.], [10., 11.]], [[100., 101.], [110., 111.]]]);
        test(1.0, 1.0, 1.0, c);
    }

    #[test]
    fn interpolate_vector_field_test() {
        let test = |x: Float, y: Float, z: Float, v: [Float; 3]| {
            let p = Position { x, y, z };
            let a = Array::linspace(0., 2999., 3000)
                .into_shape((3, 10, 10, 10))
                .unwrap();
            let gw = GridWidth {
                x: 2.0,
                y: 2.0,
                z: 2.0,
                phi: 0.0,
                theta: 0.0,
            };
            let res = interpolate_vector_field(&p, &a.view(), &gw);

            for (a, b) in v.iter().zip(res.iter()) {
                assert!(
                    equal_floats(*a, *b),
                    "({}, {}, {}): {:.16} != {:.16}",
                    x,
                    y,
                    z,
                    a,
                    b
                );
            }
        };

        test(1.0, 1.0, 1.0, [0.0, 1000.0, 2000.0]);
        test(1.0, 1.0, 3.0, [1.0, 1001.0, 2001.0]);
        test(1.5, 1.5, 1.5, [27.75, 1027.75, 2027.75]);
    }
}
