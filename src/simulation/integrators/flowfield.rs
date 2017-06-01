use ndarray::{Array, ArrayView, Axis, Ix2, Ix3, Ix4};
use simulation::grid_width::GridWidth;

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


/// Implements the operation `dx uy - dy ux` on a given discretized flow field
/// `u=(ux, uy)`.
pub fn vorticity3d(grid_width: GridWidth, u: ArrayView<f64, Ix4>) -> VectorField3D {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];
    let sz = sh[3];

    // allocate uninitialized memory, is assigned later
    let len = sx * sy * sz;
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


    /// /////////////////////
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
            s += &uy.slice(s![.., -1.., ..]);
            s /= hz;
        }
        {
            let mut s = vx.slice_mut(s![.., .., -1..]);
            s -= &uy.slice(s![.., .., ..1]);
            s += &uy.slice(s![.., .., -2..-1]);
            s /= hz;
        }
    }
    /// ////////////////
    /// /////////////////////
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

        // calculate -dx dz, mind the switched signes
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
    /// ////////////////
    /// /////////////////////
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
    /// ////////////////

    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, arr2};
    use simulation::grid_width::GridWidth;
    use simulation::settings::{BoxSize, GridSize};
    use test::Bencher;
    use test_helper::equal_floats;


    #[test]
    fn test_vorticity() {
        let bs = BoxSize {
            x: 50.,
            y: 50.,
            z: 0.,
        };
        let gs = GridSize {
            x: 50,
            y: 50,
            z: 0,
            phi: 1,
        };
        let gw = GridWidth::new(gs, bs);

        let mut u: Array<f64, Ix3> = Array::linspace(1., 50., 50).into_shape((2, 5, 5)).unwrap();
        u[[0, 1, 4]] = 42.;

        let v = vorticity(gw, u.view());

        let should = arr2(
            &[
                [-6., -8.5, -8.5, -8.5, -6.],
                [22.5, 4., 4., -12., 6.5],
                [6.5, 4., 4., 4., 6.5],
                [6.5, 4., 4., 4., 6.5],
                [-6., -8.5, -8.5, -8.5, -6.],
            ]
        );

        println!("result: {}", v);
        println!("expected: {}", should);

        for (a, b) in v.iter().zip(should.iter()) {
            assert!(equal_floats(*a, *b), "expected {}, got {}", b, a);
        }
    }

    #[bench]
    fn bench_vorticity(b: &mut Bencher) {
        let bs = BoxSize {
            x: 400.,
            y: 400.,
            z: 0.,
        };
        let gs = GridSize {
            x: 400,
            y: 400,
            z: 0,
            phi: 1,
        };
        let gw = GridWidth::new(gs, bs);

        let u: Array<f64, Ix3> = Array::linspace(1., 80000., 80000)
            .into_shape((2, 200, 200))
            .unwrap();

        b.iter(|| vorticity(gw, u.view()));
    }

}
