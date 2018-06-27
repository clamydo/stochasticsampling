// Move unit test into own file
#[cfg(test)]
#[path = "./spectral_solver_test.rs"]
mod spectral_solver_test;

use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, Axis, Ix2, Ix3, Ix4, Ix5, Zip};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use distribution::Distribution;
use flowfield::stress::{average_stress, stress_kernel};
use flowfield::FlowField3D;
use mesh::fft_helper::{
    get_inverse_norm, get_inverse_norm_squared, get_k_mesh, get_norm_k_mesh,
};
use mesh::grid_width::GridWidth;
use {BoxSize, GridSize};
use std::sync::Arc;

pub struct SpectralSolver {
    flow_field: Array<Complex<f64>, Ix4>,
    fft_plan_forward: Arc<FFTPlan>,
    fft_plan_backward: Arc<FFTPlan>,
    k_invnormsquared: Array<Complex<f64>, Ix3>,
    k_invnorm: Array<Complex<f64>, Ix3>,
    k_mesh: Array<Complex<f64>, Ix4>,
    k_normed_mesh: Array<Complex<f64>, Ix4>,
    stress_kernel: Array<f64, Ix4>,
    stress_field: Array<Complex<f64>, Ix5>,
}

impl SpectralSolver {
    pub fn new<F>(
        grid_size: GridSize,
        box_size: BoxSize,
        stress: F,
    ) -> SpectralSolver
    where
        F: Fn(f64, f64) -> Array<f64, Ix2>,
    {
        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);

        let mut dummy: Array<Complex<f64>, Ix3> =
            Array::default([grid_size.x, grid_size.y, grid_size.z]);
        let plan_stress = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Forward,
            fft::FFTFlags::Patient,
        ).unwrap();

        let plan_ff = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Backward,
            fft::FFTFlags::Patient,
        ).unwrap();

        SpectralSolver {
            k_invnorm: get_inverse_norm(mesh.view()),
            k_invnormsquared: get_inverse_norm_squared(mesh.view()),
            k_mesh: mesh,
            k_normed_mesh: get_norm_k_mesh(grid_size, box_size),
            flow_field: Array::zeros((3, grid_size.x, grid_size.y, grid_size.z)),
            stress_kernel: stress_kernel(grid_size, grid_width, stress),
            fft_plan_forward: Arc::new(plan_stress),
            fft_plan_backward: Arc::new(plan_ff),
            stress_field: Array::zeros((3, 3, grid_size.x, grid_size.y, grid_size.z)),
        }
    }

    /// Calculate flow field by convolving the Green's function of the stokes
    /// equation (Oseen tensor) with the stress field divergence (force
    /// density).
    ///
    /// Given the continuous Fourier coefficient `F[f][k]`` of a function `f`,
    /// a periodicity `T` and a sampling `f_n = f(dx n)` with step width `dx`,
    /// the DFT of `f_n` is given by
    /// ```latex
    ///     DFT[f_n] = N 2 pi / T F[f][2 pi / T k]
    /// ```
    /// Which means,
    /// ```latex
    ///     f_n = IDFT[DFT[f_n]]
    ///     = N/N 2 pi /T \sum_n^{N-1} F[f][2 pi / T * k] exp(i 2 pi k n / N)
    /// ```
    #[deprecated(since="1.3.0", note="please use `update_flow_field` instead")]
    pub fn solve_flow_field(&mut self, dist: &Distribution) -> FlowField3D {
        let dist_sh = dist.dim();
        let stress_sh = self.stress_kernel.dim();
        let n_stress = stress_sh.0 * stress_sh.1;

        let mut stress_field = self.stress_field.view_mut();
        stress_field = average_stress(stress_field, &self.stress_kernel.view(), dist);

        // calculate FFT of averaged stress field
        let mut stress_field = stress_field
            .into_shape((n_stress, dist_sh.0, dist_sh.1, dist_sh.2))
            .unwrap();

        let fft = &self.fft_plan_forward;
        stress_field
            .outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        let stress_field = stress_field
            .into_shape([stress_sh.0, stress_sh.1, dist_sh.0, dist_sh.1, dist_sh.2])
            .unwrap();

        // calculate divergence of average stress field in Fourier space
        let sigmak = ((&stress_field * &self.k_mesh.view()).sum_axis(Axis(1))
            * &self.k_invnormsquared.view()) * Complex::new(0., 1.);

        // convolve with free Green's function (Oseen tensor)
        let ksigmak = (&self.k_mesh.view() * &sigmak.view()).sum_axis(Axis(0))
            * &self.k_invnormsquared.view();
        let kksigmak = &self.k_mesh.view() * &ksigmak.view();

        let mut u = sigmak - &kksigmak.view();

        // transform back to real space
        let fft = &self.fft_plan_backward;
        u.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        let norm = (dist_sh.0 * dist_sh.1 * dist_sh.2) as f64;

        // convert to real vector field
        u.map(|v| v.re / norm)
    }

    pub fn update_flow_field(&mut self, dist: &Distribution) {
        let dist_sh = dist.dim();
        let stress_sh = self.stress_kernel.dim();
        let n_stress = stress_sh.0 * stress_sh.1;
        let n = dist_sh.0 * dist_sh.1 * dist_sh.2;

        let mut stress_field = self.stress_field.view_mut();
        stress_field = average_stress(stress_field, &self.stress_kernel.view(), dist);

        // calculate FFT of averaged stress field
        let mut stress_field = stress_field
            .into_shape((n_stress, dist_sh.0, dist_sh.1, dist_sh.2))
            .unwrap();

        let fft = &self.fft_plan_forward;
        stress_field
            .outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        let stress_field = stress_field
            .into_shape([stress_sh.0, stress_sh.1, n])
            .unwrap();

        let ff = self.flow_field.view_mut();
        let mut ff = ff.into_shape([3, n]).unwrap();

        let k = self.k_normed_mesh.view();
        let k = k.into_shape([3, n]).unwrap();

        let ik = self.k_invnorm.view();
        let ik = ik.into_shape([n]).unwrap();

        let norm = n as f64;

        Zip::from(ff.axis_iter_mut(Axis(1)))
            .and(stress_field.axis_iter(Axis(2)))
            .and(k.axis_iter(Axis(1)))
            .and(ik.outer_iter())
            .par_apply(|mut ff, s, k, ik| {
                // trick needed, because Zip cannot iterate over scalar array yet
                let ik = unsafe { *ik.as_ptr() };

                let mut sigmak = s.dot(&k);
                let ksigmak = k.dot(&sigmak);

                let kksigmak = &k * ksigmak;
                sigmak -= &kksigmak;
                sigmak *= ik / norm;
                sigmak *= Complex::new(0., 1.);
                ff.assign(&sigmak)
            });

        let mut ff = ff.into_shape([3, dist_sh.0, dist_sh.1, dist_sh.2]).unwrap();

        // transform back to real space
        let fft = &self.fft_plan_backward;
        ff.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));
    }

    pub fn get_real_flow_field(&self) -> Array<f64, Ix4> {
        self.flow_field.map(|v| v.re)
    }
}
