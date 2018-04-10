use ndarray::{Array, Axis, Ix1, Ix4};

pub mod fft_helper;
pub mod grid_width;

pub fn mesh3d<T: Clone + Default>(k: &[Array<T, Ix1>]) -> Array<T, Ix4> {
    let sh_x = k[0].len();
    let sh_y = k[1].len();
    let sh_z = k[2].len();

    let mut res = Array::from_elem([3, sh_x, sh_y, sh_z], T::default());

    // first component varies along first axis of field
    for (kx, mut x) in k[0].iter()
        .zip(res.subview_mut(Axis(0), 0).axis_iter_mut(Axis(0)))
    {
        x.fill(kx.clone());
    }

    // second component varies along second axis of field
    for (ky, mut y) in k[1].iter()
        .zip(res.subview_mut(Axis(0), 1).axis_iter_mut(Axis(1)))
    {
        y.fill(ky.clone());
    }

    // third component varies along third axis of field
    for (kz, mut z) in k[2].iter()
        .zip(res.subview_mut(Axis(0), 2).axis_iter_mut(Axis(2)))
    {
        z.fill(kz.clone());
    }

    res
}