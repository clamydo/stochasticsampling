use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, Axis, Ix3, Ix4, Ix5};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use simulation::distribution::Distribution;
use simulation::flowfield::stress::{average_stress, stress_kernel};
use simulation::flowfield::FlowField3D;
use simulation::mesh::fft_helper::{get_inverse_norm_squared, get_k_mesh};
use simulation::mesh::grid_width::GridWidth;
use simulation::settings::{BoxSize, GridSize, StressPrefactors};
use std::sync::Arc;

pub struct SpectralSolver {
    fft_plan_forward: Arc<FFTPlan>,
    fft_plan_backward: Arc<FFTPlan>,
    k_inorm: Array<Complex<f64>, Ix3>,
    k_mesh: Array<Complex<f64>, Ix4>,
    stress_kernel: Array<f64, Ix4>,
    stress_field: Array<Complex<f64>, Ix5>,
}

impl SpectralSolver {
    pub fn new(
        grid_size: GridSize,
        box_size: BoxSize,
        parameter: StressPrefactors,
    ) -> SpectralSolver {
        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);

        let mut dummy: Array<Complex<f64>, Ix3> =
            Array::default([grid_size.x, grid_size.y, grid_size.z]);
        let plan_stress = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Forward,
            fft::FFTFlags::Measure,
        ).unwrap();

        let plan_ff = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Backward,
            fft::FFTFlags::Measure,
        ).unwrap();

        SpectralSolver {
            k_inorm: get_inverse_norm_squared(mesh.view()),
            k_mesh: mesh,
            stress_kernel: stress_kernel(grid_size, grid_width, parameter),
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

        // calculate gradient of average stress field in Fourier space
        let sigmak = ((&stress_field * &self.k_mesh.view()).sum_axis(Axis(1))
            * &self.k_inorm.view()) * Complex::new(0., 1.);

        // convolve with free Green's function (Oseen tensor)
        let ksigmak =
            (&self.k_mesh.view() * &sigmak.view()).sum_axis(Axis(0)) * &self.k_inorm.view();
        let kksigmak = &self.k_mesh.view() * &ksigmak.view();

        let mut u = sigmak - &kksigmak.view();

        // transform back to real space
        let fft = &self.fft_plan_backward;
        u.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        // convert to real vector field
        u.map(|v| v.re)
    }
}
