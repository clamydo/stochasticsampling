use super::Distribution;
use crate::mesh::fft_helper::get_k_mesh;
use crate::mesh::grid_width::GridWidth;
use crate::{BoxSize, GridSize};
use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, ArrayView, Axis, Ix3, Ix4, Zip};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use std::sync::Arc;

pub struct DensityGradient {
    fft_plan_forward: Arc<FFTPlan>,
    fft_plan_backward: Arc<FFTPlan>,
    k_mesh: Array<Complex<f64>, Ix4>,
    gradient: Array<Complex<f64>, Ix4>,
    grid_width: GridWidth,
    density: Array<Complex<f64>, Ix3>,
}

impl DensityGradient {
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> DensityGradient {
        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);

        let mut dummy: Array<Complex<f64>, Ix3> =
            Array::default([grid_size.x, grid_size.y, grid_size.z]);
        let plan_forward = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Forward,
            fft::FFTFlags::Patient,
        )
        .unwrap();

        let plan_backward = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Backward,
            fft::FFTFlags::Patient,
        )
        .unwrap();

        DensityGradient {
            k_mesh: mesh,
            fft_plan_forward: Arc::new(plan_forward),
            fft_plan_backward: Arc::new(plan_backward),
            gradient: Array::default([3, grid_size.x, grid_size.y, grid_size.z]),
            grid_width: grid_width,
            density: Array::default([grid_size.x, grid_size.y, grid_size.z]),
        }
    }

    /// Returns vector gradient of magnetic field.
    fn update_gradient(&mut self, dist: &Distribution) {
        let sh = dist.dim();
        let n = sh.0 * sh.1 * sh.2;

        let dist = dist.dist.view();
        let dist = dist.into_shape([n, sh.3 * sh.4]).unwrap();

        // use first component of gradient to safe fourier transformation of density
        let density = self.density.view_mut();
        let mut density = density.into_shape([n]).unwrap();

        let dthph = self.grid_width.theta * self.grid_width.phi;
        density
            .iter_mut()
            .zip(dist.axis_iter(Axis(0)))
            .for_each(|(dens, dist)| *dens = Complex::from(dist.sum()) * dthph);

        let mut density = density.into_shape([sh.0, sh.1, sh.2]).unwrap();
        let fft = &self.fft_plan_forward;
        fft.reexecute3d(&mut density.view_mut());

        let k = self.k_mesh.view();
        let k = k.into_shape([3, n]).unwrap();

        let g = self.gradient.view_mut();
        let mut g = g.into_shape([3, n]).unwrap();

        let density = density.into_shape([n]).unwrap();

        // FFT normalization
        let norm = n as f64;
        let norm = Complex::new(0., 1.) / norm;

        Zip::from(g.axis_iter_mut(Axis(1)))
            .and(k.axis_iter(Axis(1)))
            .and(density.axis_iter(Axis(0)))
            .par_apply(|mut g, k, d| {
                let d = unsafe { *d.as_ptr() };
                let e = (&k * d) * norm;
                g.assign(&e)
            });

        let mut g = g.into_shape([3, sh.0, sh.1, sh.2]).unwrap();

        let fft = &self.fft_plan_backward;
        g.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));
    }

    pub fn get_gradient(&mut self, dist: &Distribution) -> ArrayView<Complex<f64>, Ix4> {
        self.update_gradient(dist);
        self.gradient.view()
    }

    pub fn get_real_gradient(&self) -> Array<f64, Ix4> {
        self.gradient.map(|v| v.re)
    }
}
