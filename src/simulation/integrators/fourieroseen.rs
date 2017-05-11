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
use super::flowfield::FlowField;
use super::flowfield::vorticity;
use super::integrate::integrate;
use consts::TWOPI;
use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, ArrayView, Axis, Ix, Ix2, Ix3, Ix4};
use num::Complex;
use rayon::prelude::*;
use simulation::distribution::Distribution;
use simulation::grid_width::GridWidth;
use simulation::particle::Particle;
use simulation::settings::{BoxSize, GridSize, StressPrefactors};
use std::f64::consts::PI;


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
    grid_width: GridWidth,
    k_inorm: Array<Complex<f64>, Ix2>,
    k_mesh: Array<Complex<f64>, Ix3>,
    pre_phase: Array<Complex<f64>, Ix2>,
    stress_kernel: Array<f64, Ix3>,
    parameter: IntegrationParameter,
}

impl Integrator {
    /// Returns a new instance of the mc_sampling integrator.
    pub fn new(grid_size: GridSize,
               box_size: BoxSize,
               parameter: IntegrationParameter)
               -> Integrator {

        if (grid_size[2] - 2) % 4 != 0 {
            warn!("To have an orientation grid point in the direction of magnetic field, use a \
                   grid size of 2 + 4 * n, with n an integer.");
        }

        let grid_width = GridWidth::new(grid_size, box_size);

        let mesh = get_k_mesh(grid_size, box_size);

        let gs = Array::from_vec(grid_size[..2]
                                     .iter()
                                     .map(|x| Complex::new(*x as f64, 0.))
                                     .collect())
                .into_shape([2, 1, 1])
                .unwrap();

        let phase = (&mesh / &gs)
            .sum(Axis(0))
            .map(|k| Complex::new(0., PI * k.re).exp());

        Integrator {
            box_size: box_size,
            grid_width: grid_width,
            k_inorm: get_inverse_norm_squared(mesh.view()),
            k_mesh: mesh,
            pre_phase: phase,
            parameter: parameter,
            stress_kernel: Integrator::calc_stress_kernel(grid_size, grid_width, parameter.stress),
        }

    }


    /// Calculates approximation of discretized stress kernel, to be used in
    /// the expectation value to obtain the stress tensor.
    fn calc_stress_kernel(grid_size: GridSize,
                          grid_width: GridWidth,
                          stress: StressPrefactors)
                          -> Array<f64, Ix3> {

        let mut s = Array::<f64, _>::zeros((2, 2, grid_size[2]));
        // Calculate discrete angles, considering the cell centered sample points of
        // the distribution
        let gw_half = grid_width.a / 2.;
        let angles = Array::linspace(0. + gw_half, TWOPI - gw_half, grid_size[2]);

        for (mut e, a) in s.axis_iter_mut(Axis(2)).zip(&angles) {
            e[[0, 0]] = stress.active * (a.cos() * a.cos() - 1. / 3.);
            e[[0, 1]] = stress.active * a.sin() * a.cos() + stress.magnetic * a.cos();
            e[[1, 0]] = stress.active * a.sin() * a.cos() - stress.magnetic * a.cos();
            e[[1, 1]] = stress.active * (a.sin() * a.sin() - 1. / 3.);
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
    ///
    ///     DFT[f_n] = N 2 pi / T F[f][2 pi / T k]
    ///
    /// Which means,
    ///
    ///     f_n = IDFT[DFT[f_n]]
    /// = N/N 2 pi /T \sum_n^{N-1} F[f][2 pi / T * k] exp(i 2 pi k n /
    /// N)
    ///
    /// The normalisation `1/N` cancels. Because FFTW does not use
    /// normalisation,
    /// we do not need to provide it in this case.
    pub fn calculate_flow_field(&self, dist: &Distribution) -> FlowField {

        fn fft_stress(kernel: &ArrayView<f64, Ix3>,
                      dist: &ArrayView<f64, Ix3>,
                      h: f64)
                      -> Array<Complex<f64>, Ix4> {

            let dist_sh = dist.dim();
            let stress_sh = kernel.shape();

            // Put axis in order, so that components fields are continuous in memory,
            // so it can be passed to FFTW easily
            let stress_tmp = kernel
                .into_shape((stress_sh[0], stress_sh[1], 1, 1, dist_sh.2))
                .unwrap();
            let stress = stress_tmp
                .broadcast((stress_sh[0], stress_sh[1], dist_sh.0, dist_sh.1, dist_sh.2))
                .unwrap();

            let mut stress_field = (&stress * dist).map_axis(Axis(4), |v| {
                // Complex::from(periodic_simpson_integrate(v, h))
                Complex::from(integrate(v, h))
            });

            // calculate FFT of stress tensor component wise
            for mut row in stress_field.outer_iter_mut() {
                for mut elem in row.outer_iter_mut() {
                    let plan = FFTPlan::new_c2c_inplace(&mut elem,
                                                        fft::FFTDirection::Forward,
                                                        fft::FFTFlags::Estimate)
                            .unwrap();
                    plan.execute()
                }
            }

            stress_field
        }

        let d = dist.dist.view();

        let stress_field = fft_stress(&self.stress_kernel.view(), &d, self.grid_width.a);

        let sigmak = ((stress_field * self.k_mesh.view()).sum(Axis(1)) * &self.k_inorm.view()) *
                     Complex::new(0., 1.);


        let ksigmak = (&self.k_mesh.view() * &sigmak.view()).sum(Axis(0)) * &self.k_inorm.view();
        let kksigmak = &self.k_mesh.view() * &ksigmak.view();

        let mut u = (sigmak - &kksigmak.view()) * &self.pre_phase.view();

        for mut component in u.outer_iter_mut() {
            let plan = FFTPlan::new_c2c_inplace(&mut component,
                                                fft::FFTDirection::Backward,
                                                fft::FFTFlags::Estimate)
                    .unwrap();
            plan.execute()

        }

        // standard normalisation `1/N` cancels
        u.map(|v| v.re / self.box_size[0] / self.box_size[1])
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
                               p: &mut Particle,
                               random_samples: &[f64; 3],
                               flow_field: &ArrayView<f64, Ix3>,
                               vort: &ArrayView<f64, Ix2>) {

        let nearest_grid_point_index = [(p.position.x.as_ref() / self.grid_width.x).floor() as Ix,
                                        (p.position.y.as_ref() / self.grid_width.y).floor() as Ix];

        let flow_x = flow_field[[0, nearest_grid_point_index[0], nearest_grid_point_index[1]]];
        let flow_y = flow_field[[1, nearest_grid_point_index[0], nearest_grid_point_index[1]]];

        let param = &self.parameter;

        let cos_orient = p.orientation.as_ref().cos();
        let sin_orient = p.orientation.as_ref().sin();
        // Calculating sqrt() is more efficient than sin().
        // let sin_orient = if p.orientation.v <= PI {
        //     (1. - cos_orient * cos_orient).sqrt()
        // } else {
        //     -(1. - cos_orient * cos_orient).sqrt()
        // };

        // Evolve particle position.
        p.position.x += (flow_x + cos_orient) * param.timestep +
                        param.trans_diffusion * random_samples[0];
        p.position.y += (flow_y + sin_orient) * param.timestep +
                        param.trans_diffusion * random_samples[1];


        // Get vorticity d/dx uy - d/dy ux
        let vort = vort[nearest_grid_point_index];

        // Evolve particle orientation. Assumes magnetic field to be oriented along
        // Y-axis.
        p.orientation += param.rot_diffusion * random_samples[2] +
                         (param.magnetic_reorientation * cos_orient + 0.5 * vort) * param.timestep;
    }


    pub fn evolve_particles_inplace<'a>(&self,
                                        particles: &mut Vec<Particle>,
                                        random_samples: &[[f64; 3]],
                                        flow_field: ArrayView<'a, f64, Ix3>) {
        // Calculate vorticity dx uy - dy ux
        let vort = vorticity(self.grid_width, flow_field);

        particles
            .par_iter_mut()
            .zip(random_samples.par_iter())
            .for_each(|(ref mut p, r)| {
                          self.evolve_particle_inplace(p, r, &flow_field, &vort.view())
                      });
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Axis, arr2, arr3};
    use simulation::distribution::Distribution;
    use simulation::grid_width::GridWidth;
    use simulation::particle::Particle;
    use simulation::settings::StressPrefactors;
    use test::Bencher;
    use test_helper::equal_floats;

    /// WARNING: Since fftw3 is not thread safe by default, DO NOT test this
    /// function in parallel. Instead test with RUST_TEST_THREADS=1.
    #[test]
    fn new() {
        let bs = [1., 1.];
        let gs = [10, 10, 3];
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

        let should0 = arr2(&[[-0.0833333333333332, 0.9330127018922195],
                             [-0.0669872981077807, 0.4166666666666666]]);
        let should1 = arr2(&[[0.6666666666666666, -1.0], [1.0, -0.3333333333333333]]);
        let should2 = arr2(&[[-0.0833333333333332, 0.0669872981077807],
                             [-0.9330127018922195, 0.4166666666666666]]);

        let check = |should: Array<f64, Ix2>, stress: ArrayView<f64, Ix2>| for (a, b) in
            should.iter().zip(stress.iter()) {
            assert!(equal_floats(*a, *b), "{} != {}", should, stress);
        };

        check(should0, i.stress_kernel.subview(Axis(2), 0));
        check(should1, i.stress_kernel.subview(Axis(2), 1));
        check(should2, i.stress_kernel.subview(Axis(2), 2));

        assert_eq!(i.stress_kernel.dim(), (2, 2, 3));
    }

    #[test]
    fn test_calculate_flow_field() {
        let bs = [3., 4.];
        let gs = [3, 4, 6];
        let s = StressPrefactors {
            active: 1.,
            magnetic: 0.,
        };

        let int_param = IntegrationParameter {
            timestep: 0.1,
            trans_diffusion: 0.1,
            rot_diffusion: 0.1,
            stress: s,
            magnetic_reorientation: 0.1,
        };

        let i = Integrator::new(gs, bs, int_param);

        let p = vec![Particle::new(0.0, 0.0, 1.5707963267948966, bs)];
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&p);

        let u = i.calculate_flow_field(&d);

        println!("u {}", u);

        let expect = arr3(&[[[0.,
                              -0.000000000000000004183988419563217,
                              0.,
                              0.000000000000000004183988419563217],
                             [0.046436899579834066,
                              -0.01468035520353885,
                              -0.01707618917275638,
                              -0.014680355203538836],
                             [-0.046436899579834066,
                              0.014680355203538836,
                              0.01707618917275638,
                              0.01468035520353885]],
                            [[0., -0.048892398517830254, 0., 0.048892398517830254],
                             [-0.000000000000000003004629197474319,
                              0.024446199258915124,
                              -0.000000000000000003004629197474319,
                              -0.02444619925891513],
                             [0.000000000000000003004629197474319,
                              0.02444619925891513,
                              0.000000000000000003004629197474319,
                              -0.024446199258915124]]]);


        for (a, b) in u.iter().zip(expect.iter()) {
            assert!(equal_floats(*a, *b), "{} != {}", *a, *b);
        }
    }

    #[test]
    fn test_evolve_particles_inplace() {
        let bs = [1., 1.];
        let gs = [10, 10, 6];
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

        let mut p = vec![Particle::new(0.6, 0.3, 0., bs),
                         Particle::new(0.6, 0.3, 1.5707963267948966, bs),
                         Particle::new(0.6, 0.3, 2.0943951023931953, bs),
                         Particle::new(0.6, 0.3, 4.71238898038469, bs),
                         Particle::new(0.6, 0.3, 6., bs)];
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&p);

        let u = i.calculate_flow_field(&d);

        i.evolve_particles_inplace(&mut p,
                                   &vec![[0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1]],
                                   u.view());

        // TODO Check these values!
        assert!(equal_floats(p[0].position.x.v, 0.71),
                "got {}, expected {}",
                p[0].position.x.v,
                0.71);
        assert!(equal_floats(p[0].position.y.v, 0.31),
                "got {}, expected {}",
                p[0].position.y.v,
                0.31);


        let orientations = [0.3997098418658896,
                            1.960506168660786,
                            2.4791049442590847,
                            5.102098822250579,
                            0.11612623755280715];

        for (p, o) in p.iter().zip(&orientations) {
            assert!(equal_floats(p.orientation.v, *o),
                    "got {}={}",
                    p.orientation.v,
                    *o);
        }
    }

    #[bench]
    fn bench_evolve_particle_inplace(b: &mut Bencher) {
        let bs = [1., 1.];
        let gs = [6, 6, 6];
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

        let mut p = Particle::new(0.6, 0.3, 0., bs);
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&vec![p]);

        let u = i.calculate_flow_field(&d);

        let vort = vorticity(gw, u.view());

        b.iter(|| i.evolve_particle_inplace(&mut p, &[0.1, 0.1, 0.1], &u.view(), &vort.view()));
    }
}
