use ndarray::{Array, ArrayView, Axis, Ix3, Ix4};
use simulation::mesh::grid_width::GridWidth;

pub type VectorField3D = Array<f64, Ix4>;
pub type ScalarField3D = Array<f64, Ix3>;

/// Calculates the vorticity on a given discretized flow field
/// `u=(ux, uy, uz)`, curl of u.
pub fn vorticity3d(grid_width: GridWidth, u: ArrayView<f64, Ix4>) -> VectorField3D {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];
    let sz = sh[3];

    // allocate uninitialized memory, is assigned later
    let len = 3 * sx * sy * sz;
    let mut uninit = Vec::with_capacity(len);
    unsafe {
        uninit.set_len(len);
    }

    let mut res = Array::from_vec(uninit).into_shape((3, sx, sy, sz)).unwrap();

    let hx = 2. * grid_width.x;
    let hy = 2. * grid_width.y;
    let hz = 2. * grid_width.z;

    let hyx = hy / hx;
    let hzy = hz / hy;
    let hxz = hx / hz;

    let ux = u.subview(Axis(0), 0);
    let uy = u.subview(Axis(0), 1);
    let uz = u.subview(Axis(0), 2);

    {
        // calculate dy uz
        let mut vx = res.subview_mut(Axis(0), 0);
        // bulk
        {
            let mut s = vx.slice_mut(s![.., 1..-1, ..]);
            s.assign(&uz.slice(s![.., 2.., ..]));
            s -= &uz.slice(s![.., ..-2, ..]);
            s *= hzy;
        }
        // borders
        {
            let mut s = vx.slice_mut(s![.., ..1, ..]);
            s.assign(&uz.slice(s![.., 1..2, ..]));
            s -= &uz.slice(s![.., -1.., ..]);
            s *= hzy;
        }
        {
            let mut s = vx.slice_mut(s![.., -1.., ..]);
            s.assign(&uz.slice(s![.., ..1, ..]));
            s -= &uz.slice(s![.., -2..-1, ..]);
            s *= hzy;
        }

        // calculate -dz uy, mind the switched signes
        // bulk
        {
            let mut s = vx.slice_mut(s![.., .., 1..-1]);
            s -= &uy.slice(s![.., .., 2..]);
            s += &uy.slice(s![.., .., ..-2]);
            s /= hz;
        }
        // borders
        {
            let mut s = vx.slice_mut(s![.., .., ..1]);
            s -= &uy.slice(s![.., .., 1..2]);
            s += &uy.slice(s![.., .., -1..]);
            s /= hz;
        }
        {
            let mut s = vx.slice_mut(s![.., .., -1..]);
            s -= &uy.slice(s![.., .., ..1]);
            s += &uy.slice(s![.., .., -2..-1]);
            s /= hz;
        }
    }
    {
        let mut vy = res.subview_mut(Axis(0), 1);
        // calculate dz ux
        // bulk
        {
            let mut s = vy.slice_mut(s![.., .., 1..-1]);
            s.assign(&ux.slice(s![.., .., 2..]));
            s -= &ux.slice(s![.., .., ..-2]);
            s *= hxz;
        }
        // borders
        {
            let mut s = vy.slice_mut(s![.., .., ..1]);
            s.assign(&ux.slice(s![.., .., 1..2]));
            s -= &ux.slice(s![.., .., -1..]);
            s *= hxz;
        }
        {
            let mut s = vy.slice_mut(s![.., .., -1..]);
            s.assign(&ux.slice(s![.., .., ..1]));
            s -= &ux.slice(s![.., .., -2..-1]);
            s *= hxz;
        }

        // calculate -dx uz, mind the switched signes
        // bulk
        {
            let mut s = vy.slice_mut(s![1..-1, .., ..]);
            s -= &uz.slice(s![2.., .., ..]);
            s += &uz.slice(s![..-2, .., ..]);
            s /= hx;
        }
        // borders
        {
            let mut s = vy.slice_mut(s![..1, .., ..]);
            s -= &uz.slice(s![1..2, .., ..]);
            s += &uz.slice(s![-1.., .., ..]);
            s /= hx;
        }
        {
            let mut s = vy.slice_mut(s![-1.., .., ..]);
            s -= &uz.slice(s![..1, .., ..]);
            s += &uz.slice(s![-2..-1, .., ..]);
            s /= hx;
        }
    }
    {
        let mut vz = res.subview_mut(Axis(0), 2);
        // calculate dx uy
        // bulk
        {
            let mut s = vz.slice_mut(s![1..-1, .., ..]);
            s.assign(&uy.slice(s![2.., .., ..]));
            s -= &uy.slice(s![..-2, .., ..]);
            s *= hyx;
        }
        // borders
        {
            let mut s = vz.slice_mut(s![..1, .., ..]);
            s.assign(&uy.slice(s![1..2, .., ..]));
            s -= &uy.slice(s![-1.., .., ..]);
            s *= hyx;
        }
        {
            let mut s = vz.slice_mut(s![-1.., .., ..]);
            s.assign(&uy.slice(s![..1, .., ..]));
            s -= &uy.slice(s![-2..-1, .., ..]);
            s *= hyx;
        }

        // calculate -dy ux, mind the switched signes
        // bulk
        {
            let mut s = vz.slice_mut(s![.., 1..-1, ..]);
            s -= &ux.slice(s![.., 2.., ..]);
            s += &ux.slice(s![.., ..-2, ..]);
            s /= hy;
        }
        // borders
        {
            let mut s = vz.slice_mut(s![.., ..1, ..]);
            s -= &ux.slice(s![.., 1..2, ..]);
            s += &ux.slice(s![.., -1.., ..]);
            s /= hy;
        }
        {
            let mut s = vz.slice_mut(s![.., -1.., ..]);
            s -= &ux.slice(s![.., ..1, ..]);
            s += &ux.slice(s![.., -2..-1, ..]);
            s /= hy;
        }
    }
    res
}

/// Calculates cross product for (1, 1, nz) grid.
///
/// $v_x = \partial_y u_z - \partial_z u_y$
/// $v_y = \partial_z u_x - \partial_x u_z$
/// $v_z = \partial_x u_y - \partial_y u_x$
///
/// where $\partial_{x,y} = 0$. So
///
/// $v_x = - \partial_z u_y$
/// $v_y = \partial_z u_x$
/// $v_z = 0$
///
pub fn vorticity3d_quasi1d(grid_width: GridWidth, u: ArrayView<f64, Ix4>) -> VectorField3D {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];
    let sz = sh[3];

    assert_eq!(sx, 1);
    assert_eq!(sy, 1);

    let mut res = Array::zeros((3, sx, sy, sz));

    let hz = 2. * grid_width.z;

    let ux = u.subview(Axis(0), 0);
    let uy = u.subview(Axis(0), 1);

    {
        let mut vx = res.subview_mut(Axis(0), 0);
        // calculate -dz uy, mind the switched signes
        // bulk
        // dz uy(z) = (uy(z + h) - u(z - h)) / 2h
        //          [., 1, 2, 3, 4, .]
        //      +[0, 1, 2, 3, 4, 5]
        //            -[0, 1, 2, 3, 4, 5]
        {
            // res[0, 1, 2, 3, 4, 5] -> res[1, 2, 3, 4]
            let mut s = vx.slice_mut(s![.., .., 1..-1]);
            //   u[0, 1, 2, 3]
            s.assign(&uy.slice(s![.., .., ..-2]));
            // - u[2, 3, 4, 5]
            s -= &uy.slice(s![.., .., 2..]);
            s /= hz;
        }
        // borders
        //          [0, ., ., ., ., 5]
        //      +[0, 1, 2, 3, 4, 5, 0]
        //         -[5, 0, 1, 2, 3, 4, 5]
        {
            let mut s = vx.slice_mut(s![.., .., ..1]);
            s.assign(&uy.slice(s![.., .., -1..]));
            s -= &uy.slice(s![.., .., 1..2]);
            s /= hz;
        }
        {
            let mut s = vx.slice_mut(s![.., .., -1..]);
            s.assign(&uy.slice(s![.., .., -2..-1]));
            s -= &uy.slice(s![.., .., ..1]);
            s /= hz;
        }
    }

    {
        let mut vy = res.subview_mut(Axis(0), 1);
        // calculate dz ux
        // bulk
        {
            let mut s = vy.slice_mut(s![.., .., 1..-1]);
            s.assign(&ux.slice(s![.., .., 2..]));
            s -= &ux.slice(s![.., .., ..-2]);
            s /= hz;
        }
        // borders
        {
            let mut s = vy.slice_mut(s![.., .., ..1]);
            s.assign(&ux.slice(s![.., .., 1..2]));
            s -= &ux.slice(s![.., .., -1..]);
            s /= hz;
        }
        {
            let mut s = vy.slice_mut(s![.., .., -1..]);
            s.assign(&ux.slice(s![.., .., ..1]));
            s -= &ux.slice(s![.., .., -2..-1]);
            s /= hz;
        }
    }

    res
}

pub fn vorticity3d_dispatch(grid_width: GridWidth, u: ArrayView<f64, Ix4>) -> VectorField3D {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];
    let sz = sh[3];

    if sx == 1 && sy == 1 && sz > 1 {
        vorticity3d_quasi1d(grid_width, u)
    } else {
        vorticity3d(grid_width, u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, arr2};
    use simulation::mesh::grid_width::GridWidth;
    use simulation::settings::{BoxSize, GridSize};
    use test::Bencher;
    use test_helper::equal_floats;

    #[test]
    fn test_vorticity3d() {
        use ndarray::arr3;

        let bs = BoxSize {
            x: 50.,
            y: 50.,
            z: 50.,
        };
        let gs = GridSize {
            x: 50,
            y: 50,
            z: 50,
            phi: 1,
            theta: 1,
        };
        let gw = GridWidth::new(gs, bs);

        let mut u: Array<f64, Ix4> = Array::linspace(1., 81., 81)
            .into_shape((3, 3, 3, 3))
            .unwrap();
        u[[0, 1, 2, 2]] = 42.;

        let v = vorticity3d(gw, u.view());

        let mut should = Array::zeros((3, 3, 3, 3));

        {
            let mut sx = should.subview_mut(Axis(0), 0);
            let x = arr3(&[
                [[-1., -2.5, -1.], [3.5, 2., 3.5], [-1., -2.5, -1.]],
                [[-1., -2.5, -1.], [3.5, 2., 3.5], [-1., -2.5, -1.]],
                [[-1., -2.5, -1.], [3.5, 2., 3.5], [-1., -2.5, -1.]],
            ]);
            sx.assign(&x);
        }

        {
            let mut sy = should.subview_mut(Axis(0), 1);
            let y = arr3(&[
                [[4., 5.5, 4.], [4., 5.5, 4.], [4., 5.5, 4.]],
                [[-9.5, -8., -9.5], [-9.5, -8., -9.5], [-21.5, 4., -9.5]],
                [[4., 5.5, 4.], [4., 5.5, 4.], [4., 5.5, 4.]],
            ]);
            sy.assign(&y);
        }

        {
            let mut sz = should.subview_mut(Axis(0), 2);
            let z = arr3(&[
                [[-3., -3., -3.], [-7.5, -7.5, -7.5], [-3., -3., -3.]],
                [[10.5, 10.5, 22.5], [6., 6., -6.], [10.5, 10.5, 10.5]],
                [[-3., -3., -3.], [-7.5, -7.5, -7.5], [-3., -3., -3.]],
            ]);
            sz.assign(&z);
        }

        println!("result: {}", v);
        println!("expected: {}", should);

        for ((i, a), b) in v.indexed_iter().zip(should.iter()) {
            assert!(
                equal_floats(*a, *b),
                "expected {}, got {} at index {:?}",
                b,
                a,
                i
            );
        }
    }

}
