// Move unit test into own file
#[cfg(test)]
#[path = "./magnetic_solver_test.rs"]
mod magnetic_solver_test;

use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, ArrayView, Axis, Ix3, Ix4, Ix5, Zip};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use simulation::distribution::Distribution;
use simulation::mesh::fft_helper::{get_k_mesh, get_norm_k_mesh};
use simulation::mesh::grid_width::GridWidth;
use simulation::polarization::director::DirectorField;
use simulation::settings::{BoxSize, GridSize, MagneticDipolePrefactors};
use std::sync::Arc;

pub struct MagneticSolver {
    fft_plan_forward: Arc<FFTPlan>,
    fft_plan_backward: Arc<FFTPlan>,
    k_mesh: Array<Complex<f64>, Ix4>,
    k_norm_mesh: Array<Complex<f64>, Ix4>,
    director_field: DirectorField,
    gradient_meanb: Array<Complex<f64>, Ix5>,
    parameter: MagneticDipolePrefactors,
    // magnetic_field: Array<Complex<f64>, Ix4>,
}

impl MagneticSolver {
    pub fn new(
        grid_size: GridSize,
        box_size: BoxSize,
        parameter: MagneticDipolePrefactors,
    ) -> MagneticSolver {
        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);
        let norm_mesh = get_norm_k_mesh(grid_size, box_size);

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

        MagneticSolver {
            k_mesh: mesh,
            k_norm_mesh: norm_mesh,
            fft_plan_forward: Arc::new(plan_stress),
            fft_plan_backward: Arc::new(plan_ff),
            director_field: DirectorField::new(grid_size, grid_width),
            gradient_meanb: Array::default([3, 3, grid_size.x, grid_size.y, grid_size.z]),
            parameter: parameter,
        }
    }

    /// Calculates the fourier transform of the mean magnetic field.
    /// CAUTION: In order to prevent reallocation the magnetiv field is saved
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

        let fft = &self.fft_plan_forward;
        p.outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        let m = self.parameter.magnetic_moment;

        Zip::from(p.lanes_mut(Axis(0)))
            .and(self.k_norm_mesh.lanes(Axis(0)))
            .apply(|mut p, k| {
                let kdotp = k.dot(&p);
                let mag = (&p / 3. - &k * kdotp) * m;
                p.assign(&mag);
            });
    }

    /// Returns vector gradient of magnetic field.
    /// TODO avoid allocation, return only view
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

        for ((mut g, k), b) in g.axis_iter_mut(Axis(2))
            .zip(k.axis_iter(Axis(1)))
            .zip(b.axis_iter(Axis(1)))
        {
            let b = b.broadcast([3, 3]).unwrap();
            let e = k.dot(&b);
            g.assign(&e)
        }

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
        self.fft_mean_magnetic_field(d);
        self.update_gradient();

        let fft = &self.fft_plan_backward;
        self.director_field
            .field
            .outer_iter_mut()
            .into_par_iter()
            .for_each(|mut v| fft.reexecute3d(&mut v));

        (self.director_field.field.view(), self.gradient_meanb.view())
    }

    // // TODO define own type for force vector to make typesafe
    // fn mean_force_on_magnetic_dipole(&self, orientation_vector:
    // &OrientationVector) -> [f64; 3] {     let i = Complex::new(0., 1.);
    //     let o = Array::from_iter(orientation_vector.iter()).br
    //
    //
    // }
}
