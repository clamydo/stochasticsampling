//! Implements an hybrid integration scheme for a Fokker-Planck
//! (Smoulochkowski) equation coupled to a continous Stokes flow field. The
//! corresponding stochastic Langevin equation is used to evolve test particle
//! positions and orientations. Considered as a probabilistic sample of the
//! probability distribution function (PDF), which is described by the
//! Fokker-Planck equation, the particle configuration is used to sample the
//! PDF. To close the integration scheme, the flow-field is calculated in terms
//! of probabilstic moments (i.e. expectation values) of the PDF on a grid.
//!
//! The integrator is implemented in dimensionless units, scaling out the
//! self-propulsion speed of the particle and the average volue taken by a
//! particle (i.e. the particle number density). In this units a particle needs
//! one unit of time to cross a volume per particle, meaning a unit of length,
//! due to self propulsion. Since the model describes an ensemble of point
//! particles, the flow-field at the position of such a particle is undefined
//! (die to the divergence of the Oseen-tensor). As  a consequence it is
//! necessary to define a minimal radius around a point particle, on which the
//! flow-field is calculated. A natural choice in the above mentioned units, is
//! to choose the radius of the volume per particle. This means, that a grid
//! cell on which the flow-field is calculated, should be a unit cell! The flow
//! field is now calculated using contributions of every other cell but the
//! cell itself.

use super::fft_helper::{get_inverse_norm_squared, get_k_mesh};
use super::flowfield::FlowField3D;
use super::flowfield::vorticity3d;
use consts::TWOPI;
use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, ArrayView, Axis,  Ix1, Ix3, Ix4, Ix5, Ix6, IxDyn};
use num::Complex;
use rayon::prelude::*;
use simulation::distribution::Distribution;
use simulation::grid_width::GridWidth;
use simulation::particle::Particle3D;
use simulation::settings::{BoxSize, GridSize, StressPrefactors};
// use std::f64::consts::PI;


/// Holds parameter needed for time step
#[derive(Debug, Clone, Copy)]
pub struct IntegrationParameter {
    pub rot_diffusion: f64,
    pub stress: StressPrefactors,
    pub timestep: f64,
    pub trans_diffusion: f64,
    pub magnetic_reorientation: f64,
}

/// Holds precomuted values
#[derive(Debug)]
pub struct Integrator {
    /// First axis holds submatrices for different discrete angles.
    box_size: BoxSize,
    grid_size: GridSize,
    grid_width: GridWidth,
    k_inorm: Array<Complex<f64>, Ix3>,
    k_mesh: Array<Complex<f64>, Ix4>,
    stress_kernel: Array<f64, Ix4>,
    parameter: IntegrationParameter,
    sin_theta_grid: Array<f64, Ix1>,
}

impl Integrator {
    /// Returns a new instance of the mc_sampling integrator.
    pub fn new(grid_size: GridSize,
               box_size: BoxSize,
               parameter: IntegrationParameter)
               -> Integrator {

        if (grid_size.phi - 2) % 4 != 0 {
            warn!(
                "To have an orientation grid point in the direction of magnetic field, use a \
                   grid size of 2 + 4 * n, with n an integer."
            );
        }

        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);
        let sin_theta_grid = Array::linspace(
            0. + grid_width.theta / 2.,
            TWOPI - grid_width.theta / 2.,
            grid_size.theta,
        )
                .map(|v| v.sin());


        Integrator {
            box_size: box_size,
            grid_size: grid_size,
            grid_width: grid_width,
            k_inorm: get_inverse_norm_squared(mesh.view()),
            k_mesh: mesh,
            parameter: parameter,
            stress_kernel: Integrator::calc_stress_kernel(grid_size, grid_width, parameter.stress),
            sin_theta_grid: sin_theta_grid,
        }

    }


    /// Calculates approximation of discretized stress kernel, to be used in
    /// the expectation value to obtain the stress tensor.
    fn calc_stress_kernel(grid_size: GridSize,
                          grid_width: GridWidth,
                          stress: StressPrefactors)
                          -> Array<f64, Ix4> {

        let mut s = Array::<f64, _>::zeros((3, 3, grid_size.phi, grid_size.theta));
        // Calculate discrete angles, considering the cell centered sample points of
        // the distribution
        let gw_half_phi = grid_width.phi / 2.;
        let gw_half_theta = grid_width.theta / 2.;
        let angles_phi = Array::linspace(0. + gw_half_phi, TWOPI - gw_half_phi, grid_size.phi);
        let angles_theta =
            Array::linspace(0. + gw_half_theta, TWOPI - gw_half_theta, grid_size.theta);

        let a = stress.active;
        let b = stress.magnetic;


        for (mut ax1, phi) in s.axis_iter_mut(Axis(2)).zip(&angles_phi) {
            for (mut e, theta) in ax1.axis_iter_mut(Axis(2)).zip(&angles_theta) {
                e[[0, 0]] = a * (-(1. / 3.) + phi.cos() * phi.cos() * theta.sin() * theta.sin());
                e[[0, 1]] = phi.cos() * theta.sin() * (b + a * theta.sin() * phi.sin());
                e[[0, 2]] = a * theta.cos() * phi.cos() * theta.sin();

                e[[1, 0]] = phi.cos() * theta.sin() * (-b + a * theta.sin() * phi.sin());
                e[[1, 1]] = a * (-(1. / 3.) + theta.sin() * theta.sin() * phi.sin() * phi.sin());
                e[[1, 2]] = theta.cos() * (-b + a * theta.sin() * phi.sin());

                e[[2, 0]] = a * theta.cos() * phi.cos() * theta.sin();
                e[[2, 1]] = theta.cos() * (b + a * theta.sin() * phi.sin());
                e[[2, 2]] = a * (-(1. / 3.) + theta.cos() * theta.cos());
            }
        }

        s
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
    pub fn calculate_flow_field(&self, dist: &Distribution) -> FlowField3D {
        fn fft_stress(kernel: &ArrayView<f64, Ix4>,
                      dist: &ArrayView<f64, Ix5>,
                      sin_theta: &ArrayView<f64, Ix1>,
                      gs: &GridSize,
                      gw: &GridWidth)
                      -> Array<Complex<f64>, Ix6> {

            let dist_sh = dist.dim();
            let stress_sh = kernel.dim();

            // Put axis in order, so that components fields are continuous in memory,
            // so it can be passed to FFTW easily
            let stress = kernel
                .into_shape(
                    IxDyn(
                        &[
                            stress_sh.0,
                            stress_sh.1,
                            1usize,
                            1usize,
                            1usize,
                            dist_sh.3,
                            dist_sh.4,
                        ]
                    )
                )
                .unwrap();
            let stress = stress
                .broadcast(
                    IxDyn(
                        &[
                            stress_sh.0,
                            stress_sh.1,
                            dist_sh.0,
                            dist_sh.1,
                            dist_sh.2,
                            dist_sh.3,
                            dist_sh.4,
                        ]
                    )
                )
                .unwrap();

            let dist = dist * sin_theta;

            let stress_field = ((&stress * &dist).sum(Axis(6)).sum(Axis(5))) * gw.phi * gw.theta;

            let mut stress_field = stress_field.map(|v| Complex::from(v));

            let mut plans = Vec::with_capacity(9);
            // calculate FFT of stress tensor component wise
            for mut row in stress_field.outer_iter_mut() {
                for mut elem in row.outer_iter_mut() {
                    let plan = FFTPlan::new_c2c_inplace_3d_dyn(
                        &mut elem,
                        fft::FFTDirection::Forward,
                        fft::FFTFlags::Estimate,
                    )
                            .unwrap();
                    plans.push(plan);
                }
            }

            plans.par_iter_mut().for_each(|p| p.execute());


            (stress_field / Complex::new(gs.x as f64 * gs.y as f64 * gs.z as f64, 0.))
                .into_shape([stress_sh.0, stress_sh.1, dist_sh.0, dist_sh.1, dist_sh.2, 1])
                .unwrap()
        }

        let d = dist.dist.view();

        let stress_field = fft_stress(
            &self.stress_kernel.view(),
            &d,
            &self.sin_theta_grid.view(),
            &self.grid_size,
            &self.grid_width,
        );
        let stress_field = stress_field
            .broadcast([3, 3, self.grid_size.x, self.grid_size.y, self.grid_size.z])
            .unwrap();


        let sigmak = ((&stress_field * &self.k_mesh.view()).sum(Axis(1)) * &self.k_inorm.view()) *
            Complex::new(0., 1.);


        let ksigmak = (&self.k_mesh.view() * &sigmak.view()).sum(Axis(0)) * &self.k_inorm.view();
        let kksigmak = &self.k_mesh.view() * &ksigmak.view();

        // let mut u = (sigmak - &kksigmak.view()) * &self.pre_phase.view();
        let mut u = sigmak - &kksigmak.view();

        // Fourier transform flow velocity component-wise
        let mut plans: Vec<FFTPlan> = u.outer_iter_mut()
            .map(
                |mut a| {
                    FFTPlan::new_c2c_inplace_3d(
                        &mut a,
                        fft::FFTDirection::Backward,
                        fft::FFTFlags::Estimate,
                    )
                            .unwrap()
                }
            )
            .collect();

        plans.par_iter_mut().for_each(|p| p.execute());

        u.map(|v| v.re)
    }


    /// Updates a test particle configuration according to the given parameters.
    ///
    /// Y(t) = sqrt(t) * X(t), if X is normally distributed with variance 1,
    /// then Y is normally distributed with variance t.
    /// A diffusion coefficient `d` translates to a normal distribuion with
    /// variance `s^2` as `d = s^2 / 2`.
    /// Together this leads to an update of the position due to the diffusion of
    /// `x_d(t + dt) = sqrt(2 d dt) N(0, 1)``.
    ///
    /// Assumes the magnetic field to be oriented along Y-axis.
    ///
    /// *IMPORTANT*: This function expects `sqrt(2 d dt)` as a precomputed
    /// effective diffusion constant.
    fn evolve_particle_inplace(&self,
                               p: &mut Particle3D,
                               random_samples: &[f64; 5],
                               flow_field: &ArrayView<f64, Ix4>,
                               vort: &ArrayView<f64, Ix4>) {

        let ix = (p.position.x.as_ref() / self.grid_width.x).floor() as isize;
        let iy = (p.position.y.as_ref() / self.grid_width.y).floor() as isize;
        let iz = (p.position.z.as_ref() / self.grid_width.z).floor() as isize;


        let flow_x = flow_field[[0, ix as usize, iy as usize, iz as usize]];
        let flow_y = flow_field[[1, ix as usize, iy as usize, iz as usize]];
        let flow_z = flow_field[[2, ix as usize, iy as usize, iz as usize]];

        let param = &self.parameter;

        let cos_phi = p.orientation.phi.as_ref().cos();
        let sin_phi = p.orientation.phi.as_ref().sin();
        let cos_theta = p.orientation.theta.as_ref().cos();
        let sin_theta = p.orientation.theta.as_ref().sin();
        let cot_theta = cos_theta / sin_theta;

        // Evolve particle position.
        p.position.x += (flow_x + sin_theta * cos_phi) * param.timestep +
            param.trans_diffusion * random_samples[0];
        p.position.y += (flow_y + sin_theta * sin_phi) * param.timestep +
            param.trans_diffusion * random_samples[1];
        p.position.z += (flow_z + cos_theta) * param.timestep +
            param.trans_diffusion * random_samples[2];


        // Get vorticity d/dx uy - d/dy ux
        let vort = vort.slice(s![.., (ix - 1)..ix, (iy - 1)..iy, (iz - 1)..iz]).into_shape(3).unwrap();

        // TODO check rotational diffusion!

        p.orientation.phi += param.rot_diffusion * random_samples[3] +
            (0.5 * cot_theta * cos_phi * vort[0] + 0.5 * cot_theta * sin_phi * vort[1] -
                 0.5 * vort[3]) * param.timestep;


        p.orientation.theta += param.rot_diffusion * random_samples[4] +
            (-param.magnetic_reorientation * sin_theta + 0.5 * cos_theta * cos_phi * vort[0] +
                 0.5 * cos_theta * sin_phi * vort[1] -
                 0.5 * sin_theta * vort[3]) * param.timestep;
    }


    pub fn evolve_particles_inplace<'a>(&self,
                                        particles: &mut Vec<Particle3D>,
                                        random_samples: &[[f64; 5]],
                                        flow_field: ArrayView<'a, f64, Ix4>) {
        // Calculate vorticity dx uy - dy ux
        let vort = vorticity3d(self.grid_width, flow_field);

        particles
            .par_iter_mut()
            .zip(random_samples.par_iter())
            .for_each(
                |(ref mut p, r)| self.evolve_particle_inplace(p, r, &flow_field, &vort.view()),
            );
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::fft_helper::get_k_mesh;
    use ndarray::{Array, Axis, arr2};
    use simulation::distribution::Distribution;
    use simulation::grid_width::GridWidth;
    use simulation::particle::Particle3D;
    use simulation::settings::StressPrefactors;
    use std::f64::consts::PI;
    use test::Bencher;
    use test_helper::equal_floats;

    /// WARNING: Since fftw3 is not thread safe by default, DO NOT test this
    /// function in parallel. Instead test with RUST_TEST_THREADS=1.
    #[test]
    fn new() {
        let bs = BoxSize {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let gs = GridSize {
            x: 10,
            y: 10,
            z: 10,
            phi: 3,
        };
        let s = StressPrefactors {
            active: 1.,
            magnetic: 1.,
        };

        let int_param = IntegrationParameter {
            timestep: 1.,
            trans_diffusion: 1.,
            rot_diffusion: 1.,
            stress: s,
            magnetic_reorientation: 1.,
        };

        let i = Integrator::new(gs, bs, int_param);

        let should0 = arr2(
            &[
                [-0.2499999999999999, 0.9330127018922195, 0.],
                [-0.0669872981077807, 0.2499999999999999, 0.],
                [0., 0., 0.],
            ]
        );
        let should1 = arr2(
            &[
                [0.5, -1.0000000000000002, 0.],
                [0.9999999999999999, -0.5, 0.],
                [0., 0., 0.],
            ]
        );
        let should2 = arr2(
            &[
                [-0.2499999999999999, 0.0669872981077807, 0.],
                [-0.9330127018922195, 0.2499999999999999, 0.],
                [0., 0., 0.],
            ]
        );

        let check = |should: Array<f64, Ix2>, stress: ArrayView<f64, Ix2>| for (a, b) in
            should.iter().zip(stress.iter())
        {
            assert!(equal_floats(*a, *b), "{} != {}", should, stress);
        };

        check(should0, i.stress_kernel.subview(Axis(2), 0));
        check(should1, i.stress_kernel.subview(Axis(2), 1));
        check(should2, i.stress_kernel.subview(Axis(2), 2));

        assert_eq!(i.stress_kernel.dim(), (3, 3, 3));
    }

    #[test]
    fn test_stress_expectation_value() {

        let bs = BoxSize {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let gs = GridSize {
            x: 1,
            y: 1,
            z: 1,
            phi: 6,
        };

        let gw = GridWidth::new(gs, bs);

        let s = StressPrefactors {
            active: 1.,
            magnetic: 0.,
        };

        let int_param = IntegrationParameter {
            timestep: 0.0,
            trans_diffusion: 0.0,
            rot_diffusion: 0.0,
            stress: s,
            magnetic_reorientation: 0.0,
        };

        let i = Integrator::new(gs, bs, int_param);

        let test_it = |d: &Distribution, expect: ArrayView<f64, Ix2>, case| {
            println!("Distribution {}", d.dist);

            let dist_sh = d.dim();
            let stress_sh = i.stress_kernel.dim();

            // Put axis in order, so that components fields are continuous in memory,
            // so it can be passed to FFTW easily
            let stress = &i.stress_kernel
                              .view()
                              .into_shape((stress_sh.0, stress_sh.1, 1, 1, dist_sh.2))
                              .unwrap();
            let stress = stress
                .broadcast((stress_sh.0, stress_sh.1, dist_sh.0, dist_sh.1, dist_sh.2))
                .unwrap();

            let stress_field = (&stress * &d.dist.view()).map_axis(
                Axis(4), |v| {
                    // Complex::from(periodic_simpson_integrate(v, h))
                    Complex::from(integrate(v, gw.phi))
                }
            );


            let is = stress_field.slice(s![.., .., ..1, ..1]);
            println!("stress {}", is.into_shape([3, 3]).unwrap());

            for (i, e) in is.iter().zip(expect.iter()) {
                assert!(equal_floats(i.re, *e), "{}: {} != {}", case, i.re, *e);
            }
        };

        let p = vec![Particle3D::new(0.0, 0.0, 1.5707963267948966, bs)];
        let mut d1 = Distribution::new(gs, gw);
        d1.sample_from(&p);

        let mut d2 = Distribution::new(gs, gw);
        d2.dist = Array::from_elem([gs.x, gs.y, gs.phi], 1. / gw.phi / gs.phi as f64);


        let expect1 = arr2(
            &[
                [-0.49999999999999994, 0., 0.],
                [0., 0.5, 0.],
                [0., 0., -0.49999999999999994],
            ]
        );
        let expect2 = arr2(&[[0., 0., 0.], [0., 0., 0.], [0., 0., -0.49999999999999994]]);

        test_it(&d1, expect1.view(), "1");
        test_it(&d2, expect2.view(), "2");
    }

    #[test]
    fn test_calculate_flow_field() {
        let bs = BoxSize {
            x: 21.,
            y: 21.,
            z: 21.,
        };
        let gs = GridSize {
            x: 21,
            y: 21,
            z: 21,
            phi: 50,
        };
        let s = StressPrefactors {
            active: 1.,
            magnetic: 0.,
        };

        let int_param = IntegrationParameter {
            timestep: 0.0,
            trans_diffusion: 0.0,
            rot_diffusion: 0.0,
            stress: s,
            magnetic_reorientation: 0.0,
        };

        let i = Integrator::new(gs, bs, int_param);

        let p = vec![Particle3D::new(0.0, 0.0, 1.5707963267948966, bs)];
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&p);

        let u = i.calculate_flow_field(&d);

        let theory = |x1: f64, x2: f64| {
            [
                (x1 * (x1 * x1 - 2. * x2 * x2)) / (8. * PI * (x1 * x1 + x2 * x2).powf(5. / 2.)),
                (x2 * (x1 * x1 - 2. * x2 * x2)) / (8. * PI * (x1 * x1 + x2 * x2).powf(5. / 2.)),
            ]
        };

        let mut grid = get_k_mesh(
            gs,
            BoxSize {
                x: TWOPI,
                y: TWOPI,
                z: 1.,
            },
        )
                .remove_axis(Axis(3));


        // bring components to the back
        grid.swap_axes(0, 1);
        grid.swap_axes(1, 2);

        let mut th_u_x = grid.map_axis(Axis(2), |v| theory(v[0].re, v[1].re)[0]);

        let mut th_u_y = grid.map_axis(Axis(2), |v| theory(v[0].re, v[1].re)[1]);

        th_u_x[[0, 0]] = 0.;
        th_u_y[[0, 0]] = 0.;

        // println!("th_u_x: {}", th_u_x);
        // println!("sim_u_x: {}", u.subview(Axis(0), 0));

        let diff = (th_u_x - u.subview(Axis(0), 0))
            .map(|v| v.abs())
            .scalar_sum();

        assert!(diff <= 0.14, "diff: {}", diff);

        let diff = (th_u_y - u.subview(Axis(0), 1))
            .map(|v| v.abs())
            .scalar_sum();

        assert!(diff <= 0.22, "diff: {}", diff);

        assert!(equal_floats(u.scalar_sum(), 0.), "{} != 0", u.scalar_sum());
    }

    #[test]
    fn test_evolve_particles_inplace() {
        let bs = BoxSize {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let gs = GridSize {
            x: 10,
            y: 10,
            z: 10,
            phi: 6,
        };
        let s = StressPrefactors {
            active: 1.,
            magnetic: 1.,
        };

        let int_param = IntegrationParameter {
            timestep: 0.1,
            trans_diffusion: 0.1,
            rot_diffusion: 0.1,
            stress: s,
            magnetic_reorientation: 0.1,
        };

        let i = Integrator::new(gs, bs, int_param);

        let mut p = vec![
            Particle3D::new(0.6, 0.3, 0., bs),
            Particle3D::new(0.6, 0.3, 1.5707963267948966, bs),
            Particle3D::new(0.6, 0.3, 2.0943951023931953, bs),
            Particle3D::new(0.6, 0.3, 4.71238898038469, bs),
            Particle3D::new(0.6, 0.3, 6., bs),
        ];
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&p);

        let u = i.calculate_flow_field(&d);

        i.evolve_particles_inplace(
            &mut p,
            &vec![
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1],
            ],
            u.view(),
        );

        // TODO Check these values!
        assert!(
            equal_floats(p[0].position.x.v, 0.71),
            "got {}, expected {}",
            p[0].position.x.v,
            0.71
        );
        assert!(
            equal_floats(p[0].position.y.v, 0.31),
            "got {}, expected {}",
            p[0].position.y.v,
            0.31
        );


        let orientations = [
            0.24447719561927586,
            1.8052735224141725,
            2.3238722980124713,
            4.946866176003965,
            6.244078898485779,
        ];

        for (p, o) in p.iter().zip(&orientations) {
            assert!(
                equal_floats(p.orientation.v, *o),
                "got {} = {}",
                p.orientation.v,
                *o
            );
        }
    }

    #[bench]
    fn bench_evolve_particle_inplace(b: &mut Bencher) {
        let bs = BoxSize {
            x: 10.,
            y: 10.,
            z: 10.,
        };
        let gs = GridSize {
            x: 10,
            y: 10,
            z: 10,
            phi: 6,
        };
        let gw = GridWidth::new(gs, bs);
        let s = StressPrefactors {
            active: 1.,
            magnetic: 1.,
        };

        let int_param = IntegrationParameter {
            timestep: 0.1,
            trans_diffusion: 0.1,
            rot_diffusion: 0.1,
            stress: s,
            magnetic_reorientation: 0.1,
        };

        let i = Integrator::new(gs, bs, int_param);

        let mut p = Particle3D::new(0.6, 0.3, 0., bs);
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&vec![p]);

        let u = i.calculate_flow_field(&d);

        let vort = vorticity(gw, u.view());

        b.iter(|| i.evolve_particle_inplace(&mut p, &[0.1, 0.1, 0.1], &u.view(), &vort.view()));
    }
}
