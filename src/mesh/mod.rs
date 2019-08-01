use crate::particle::Position;
use crate::GridSize;
use grid_width::GridWidth;
use ndarray::{Array, Axis, Ix1, Ix4};

pub mod fft_helper;
pub mod grid_width;
pub mod interpolate;

pub fn mesh3d<T: Clone + Default>(k: &[Array<T, Ix1>]) -> Array<T, Ix4> {
    let sh_x = k[0].len();
    let sh_y = k[1].len();
    let sh_z = k[2].len();

    let mut res = Array::from_elem([3, sh_x, sh_y, sh_z], T::default());

    // first component varies along first axis of field
    for (kx, mut x) in k[0]
        .iter()
        .zip(res.index_axis_mut(Axis(0), 0).axis_iter_mut(Axis(0)))
    {
        x.fill(kx.clone());
    }

    // second component varies along second axis of field
    for (ky, mut y) in k[1]
        .iter()
        .zip(res.index_axis_mut(Axis(0), 1).axis_iter_mut(Axis(1)))
    {
        y.fill(ky.clone());
    }

    // third component varies along third axis of field
    for (kz, mut z) in k[2]
        .iter()
        .zip(res.index_axis_mut(Axis(0), 2).axis_iter_mut(Axis(2)))
    {
        z.fill(kz.clone());
    }

    res
}

pub fn get_cell_index(
    p: &Position,
    grid_width: &GridWidth,
    grid_size: &GridSize,
) -> (usize, usize, usize) {
    let mut ix = (p.x / grid_width.x).floor() as usize;
    let mut iy = (p.y / grid_width.y).floor() as usize;
    let mut iz = (p.z / grid_width.z).floor() as usize;

    if ix == grid_size.x {
        ix -= 1;
    }

    if iy == grid_size.y {
        iy -= 1;
    }

    if iz == grid_size.z {
        iz -= 1;
    }

    (ix, iy, iz)
}
