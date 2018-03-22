use ndarray::{Array, ArrayView, Axis, Ix2, Ix3, Ix4};
use simulation::grid_width::GridWidth;
use simulation::settings::{BoxSize, GridSize};

pub type FlowField2D = Array<f64, Ix3>;
pub type VectorField2D = Array<f64, Ix3>;
pub type ScalarField2D = Array<f64, Ix2>;

pub type FlowField3D = Array<f64, Ix4>;
pub type VectorField3D = Array<f64, Ix4>;
pub type ScalarField3D = Array<f64, Ix3>;

/// Implements the operation `dx uy - dy ux` on a given discretized flow field
/// `u=(ux, uy)`.
pub fn vorticity2d(grid_width: GridWidth, u: ArrayView<f64, Ix3>) -> ScalarField2D {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];

    // allocate uninitialized memory, is assigned later
    let len = sx * sy;
    let mut uninit = Vec::with_capacity(len);
    unsafe {
        uninit.set_len(len);
    }

    let mut res = Array::from_vec(uninit).into_shape((sx, sy)).unwrap();

    let hx = 2. * grid_width.x;
    let hy = 2. * grid_width.y;

    let ux = u.subview(Axis(0), 0);
    let uy = u.subview(Axis(0), 1);

    // trick to calculate dx dy / hx - dy dx / hy more easily
    let hyx = hy / hx;

    // calculate dx uy
    // bulk
    {
        let mut s = res.slice_mut(s![1..-1, ..]);
        s.assign(&uy.slice(s![2.., ..]));
        s -= &uy.slice(s![..-2, ..]);
        s *= hyx;
    }
    // borders
    {
        let mut s = res.slice_mut(s![..1, ..]);
        s.assign(&uy.slice(s![1..2, ..]));
        s -= &uy.slice(s![-1.., ..]);
        s *= hyx;
    }
    {
        let mut s = res.slice_mut(s![-1.., ..]);
        s.assign(&uy.slice(s![..1, ..]));
        s -= &uy.slice(s![-2..-1, ..]);
        s *= hyx;
    }

    // calculate -dy dx, mind the switched signes
    // bulk
    {
        let mut s = res.slice_mut(s![.., 1..-1]);
        s -= &ux.slice(s![.., 2..]);
        s += &ux.slice(s![.., ..-2]);
        s /= hy;
    }
    // borders
    {
        let mut s = res.slice_mut(s![.., ..1]);
        s -= &ux.slice(s![.., 1..2]);
        s += &ux.slice(s![.., -1..]);
        s /= hy;
    }
    {
        let mut s = res.slice_mut(s![.., -1..]);
        s -= &ux.slice(s![.., ..1]);
        s += &ux.slice(s![.., -2..-1]);
        s /= hy;
    }

    res
}

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
/// where $\partial_y = 0$. Consequently,
///
/// $v_x = - \partial_z u_y$
/// $v_y = \partial_z u_x - \partial_x u_z$
/// $v_z = \partial_x u_y$
///
pub fn vorticity3d_quasi2d(grid_width: GridWidth, u: ArrayView<f64, Ix4>) -> VectorField3D {
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
    let hz = 2. * grid_width.z;

    let hxz = hx / hz;

    let ux = u.subview(Axis(0), 0);
    let uy = u.subview(Axis(0), 1);
    let uz = u.subview(Axis(0), 2);

    {
        // calculate dy uz
        let mut vx = res.subview_mut(Axis(0), 0);
        // calculate -dz uy, mind the switched signes
        // bulk
        {
            let mut s = vx.slice_mut(s![.., .., 1..-1]);
            s.assign(&uy.slice(s![.., .., ..-2]));
            s -= &uy.slice(s![.., .., 2..]);
            s /= hz;
        }
        // borders
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
            s /= hx;
        }
        // borders
        {
            let mut s = vz.slice_mut(s![..1, .., ..]);
            s.assign(&uy.slice(s![1..2, .., ..]));
            s -= &uy.slice(s![-1.., .., ..]);
            s /= hx;
        }
        {
            let mut s = vz.slice_mut(s![-1.., .., ..]);
            s.assign(&uy.slice(s![..1, .., ..]));
            s -= &uy.slice(s![-2..-1, .., ..]);
            s /= hx;
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

pub fn vorticity3d_dispatch(grid_width: GridWidth, u: ArrayView<f64, Ix4>) -> FlowField3D {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];
    let sz = sh[3];

    if sx > 1 && sy > 1 && sz > 1 {
        vorticity3d(grid_width, u)
    } else if sx > 1 && sy == 1 && sz > 1 {
        vorticity3d_quasi2d(grid_width, u)
    } else if sx == 1 && sy == 1 && sz > 1 {
        vorticity3d_quasi1d(grid_width, u)
    } else {
        panic!("Dimensions not supported.")
    }
}

/// Returns a quasi 2D Poiseuille-flow with period boundaries give a grid size and
/// width. Flow is in z-direction, as is the magnetic field. Poiseuille-flow gradient is in y
/// direction.
pub fn create_poiseuille_flow_quasi2d(
    grid_size: GridSize,
    grid_width: GridWidth,
    box_size: BoxSize,
    strength: f64,
) -> VectorField3D {
    assert!(grid_size.y == 1, "A full 3D Poiseuille is not supported at the moment. Please set y-dimension to 1.");
    let l = box_size.x;

    let mut res = Array::zeros((3, grid_size.x, grid_size.y, grid_size.z));

    let xs = Array::linspace(
        grid_width.x / 2.,
        box_size.x - grid_width.x / 2.,
        grid_size.x,
    );

    for (mut ax0, x) in res.axis_iter_mut(Axis(1)).zip(&xs) {
        for mut ax1 in ax0.axis_iter_mut(Axis(1)) {
            for mut e in ax1.axis_iter_mut(Axis(1)) {
                e[[2]] = strength
                    * (-l * l / 8. + l * x / 4. + l / 4. * (x - l / 2.).abs()
                        - x / 2. * (x - l / 2.).abs());
            }
        }
    }

    res
}

#[cfg(test)]
mod tests;
