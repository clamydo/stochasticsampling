// Move unit test into own file
#[cfg(test)]
#[path = "./magnetic_solver_test.rs"]
mod magnetic_solver_test;

use distribution::Distribution;
use fftw3::fft;
use fftw3::fft::FFTPlan;
use mesh::fft_helper::{get_k_mesh, get_norm_k_mesh};
use mesh::grid_width::GridWidth;
use ndarray::{Array, ArrayView, Axis, Ix3, Ix4, Ix5, Zip};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use polarization::director::DirectorField;
use std::sync::Arc;
use {BoxSize, GridSize};

pub type MagneticField = Array<Complex<f64>, Ix4>;

pub struct MagneticSolver {
    fft_plan_forward: Arc<FFTPlan>,
    fft_plan_backward: Arc<FFTPlan>,
    k_mesh: Array<Complex<f64>, Ix4>,
    k_norm_mesh: Array<Complex<f64>, Ix4>,
    director_field: DirectorField,
    gradient_meanb: Array<Complex<f64>, Ix5>,
    // magnetic_field: Array<Complex<f64>, Ix4>,
}

impl MagneticSolver {
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> MagneticSolver {
        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);
        let norm_mesh = get_norm_k_mesh(grid_size, box_size);

        let mut dummy: Array<Complex<f64>, Ix3> =
            Array::default([grid_size.x, grid_size.y, grid_size.z]);
        let plan_forward = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Forward,
            fft::FFTFlags::Patient,
        ).unwrap();

        let plan_backward = FFTPlan::new_c2c_inplace_3d(
            &mut dummy.view_mut(),
            fft::FFTDirection::Backward,
            fft::FFTFlags::Patient,
        ).unwrap();

        MagneticSolver {
            k_mesh: mesh,
            k_norm_mesh: norm_mesh,
            fft_plan_forward: Arc::new(plan_forward),
            fft_plan_backward: Arc::new(plan_backward),
            director_field: DirectorField::new(grid_size, grid_width),
            gradient_meanb: Array::default([3, 3, grid_size.x, grid_size.y, grid_size.z]),
        }
    }

    /// Calculates the fourier transform of the mean magnetic field.
    /// CAUTION: In order to prevent reallocation the magnetic field is saved
    /// in the DirectorField. Which is complete and utter non-sense. But
    /// more efficient. Sorry for that. There is surely a better way to
    /// deal with that.
    ///
    /// Set k=0 mode to zero, since an constant offset field is unphysical.
    fn fft_mean_magnetic_field(&mut self, dist: &Distribution) {
        // let dist_sh = dist.dim();

        // calculate FFT of averaged stress field
        self.director_field.from_distribution(dist);
        let mut p = self.director_field.field.view_mut();

        let sh = p.dim();

        let fft = &self.fft_plan_forward;
        p.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        // FFT normalization
        let norm = (sh.1 * sh.2 * sh.3) as f64;

        Zip::from(p.lanes_mut(Axis(0)))
            .and(self.k_norm_mesh.lanes(Axis(0)))
            .par_apply(|mut p, k| {
                let kdotp = k.dot(&p);
                let mag = (&p - &(&k * kdotp)) / norm;
                p.assign(&mag);
            });
    }

    /// Returns vector gradient of magnetic field.
    fn update_gradient(&mut self) {
        let sh = self.director_field.field.dim();
        let n = sh.1 * sh.2 * sh.3;

        // Construct an outer product by making use of broadcasting
        // a = [a1, a2, a3], b = [b1, b2, b3]
        // ab =
        // [[a1, a1, a1],    [[b1, b2, b3]
        //  [a2, a2, a2], .*  [b1, b2, b3]
        //  [a3, a3, a3]]     [b1, b2, b3]]

        let k = self.k_mesh.view();
        let k = k.into_shape([3, n]).unwrap();

        // Yep. Magnetic field is in there. Very stupid.
        let b = self.director_field.field.view();
        let b = b.into_shape([3, n]).unwrap();

        let g = self.gradient_meanb.view_mut();
        let mut g = g.into_shape([3, 3, n]).unwrap();

        Zip::from(g.axis_iter_mut(Axis(2)))
            .and(k.axis_iter(Axis(1)))
            .and(b.axis_iter(Axis(1)))
            .par_apply(|mut g, k, b| {
                let k = k.broadcast([3, 3]).unwrap();
                let k = k.t();
                let e = (&k * &b) * Complex::new(0., 1.);
                g.assign(&e)
            });

        let mut g = g.into_shape([9, sh.1, sh.2, sh.3]).unwrap();

        let fft = &self.fft_plan_backward;
        g.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));
    }

    /// Given a distribution `d`, it returns a view into the mean magnetic
    /// field and the (flattened) vector gradient field of it.
    pub fn mean_magnetic_field(
        &mut self,
        d: &Distribution,
    ) -> (ArrayView<Complex<f64>, Ix4>, ArrayView<Complex<f64>, Ix5>) {
        // TODO refactor to make it more explicit and readable
        // calculate FFT of magnetic field and store result in self.director_field.field
        self.fft_mean_magnetic_field(d);
        // use stored value of FFT of magnetic field to calculate vector gradient and
        // store it in self.gradient_meanb
        self.update_gradient();

        let fft = &self.fft_plan_backward;
        self.director_field
            .field
            .outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        (self.director_field.field.view(), self.gradient_meanb.view())
    }

    pub fn get_real_magnet_field(&self) -> Array<f64, Ix4> {
        self.director_field.field.map(|v| v.re)
    }
}
