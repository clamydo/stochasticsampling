use super::get_cell_index;
use super::grid_width::GridWidth;
use crate::particle::Position;
use crate::vector::VectorD;
use crate::GridSize;
use lerp::Lerp;
use ndarray::{ArrayView, Ix3, Ix4};

pub fn interpolate_vector_field(
    position: &Position,
    v: &ArrayView<f32, Ix4>,
    gw: &GridWidth,
    gs: &GridSize,
) -> VectorD {
    let idx = get_cell_index(position, gw, gs);
    unimplemented!()
}

fn trilinear_interpolation(
    relative_position: Position,
    gw: &GridWidth,
    values: &ArrayView<f32, Ix3>,
) -> f32 {
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
    use ndarray::arr3;

    #[test]
    fn trilinear_test() {
        let a = arr3(&[[[0.0f32, -1.0], [1.0, 0.0]], [[1.0, 0.0], [2.0, 1.0]]]);

        let gw = GridWidth {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            phi: 0.0,
            theta: 0.0,
        };

        let test = |x: f32, y: f32, z: f32, v: f32| {
            let p = Position { x, y, z };

            assert!(equal_floats(trilinear_interpolation(p, &gw, &a.view()), v));
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
}
