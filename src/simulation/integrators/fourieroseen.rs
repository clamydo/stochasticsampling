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
use consts::TWOPI;
use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, ArrayView, Axis, Ix, Ix1, Ix2, Ix3, Ix4};
use num::Complex;
use rayon::prelude::*;
use simulation::grid_width::GridWidth;
use simulation::particle::Particle;
use simulation::settings::{BoxSize, GridSize, StressPrefactors};


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
    grid_width: GridWidth,
    k_inorm: Array<Complex<f64>, Ix2>,
    k_mesh: Array<Complex<f64>, Ix3>,
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

        Integrator {
            grid_width: grid_width,
            k_inorm: get_inverse_norm_squared(mesh.view()),
            k_mesh: mesh,
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
    /// equation (Oseen tensor) with the stress field divergence (force density)
    pub fn calculate_flow_field(&self, dist: ArrayView<f64, Ix3>) -> FlowField {

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
                Complex::from(periodic_simpson_integrate(v, h))
            });

            println!("{}", dist);
            println!("{}", stress_field);

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

        let stress_field = fft_stress(&self.stress_kernel.view(), &dist, self.grid_width.a);

        let k_mesh_sh = self.k_mesh.shape();
        // let sigmak = (&stress_field * self.k_mesh.broadcast((2, 2, k_mesh_sh[1],
        // k_mesh_sh[2])).unwrap());

        let sigmak = ((stress_field * self.k_mesh.view()).sum(Axis(1)) * &self.k_inorm.view()) *
                     Complex::new(0., 1.);

        let ksigmak = (&self.k_mesh.view() * &sigmak.view()).sum(Axis(0)) * &self.k_inorm.view();
        let kksigmak = &self.k_mesh.view() * &ksigmak.view();

        let mut u = sigmak - &kksigmak.view();


        for mut component in u.outer_iter_mut() {
            let plan = FFTPlan::new_c2c_inplace(&mut component,
                                                fft::FFTDirection::Backward,
                                                fft::FFTFlags::Estimate)
                    .unwrap();
            plan.execute()

        }

        u.map(|v| v.re / (k_mesh_sh[1] * k_mesh_sh[2]) as f64)
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




/// Implements Simpon's Rule integration on an array, representing sampled
/// points of a periodic function.
fn periodic_simpson_integrate(samples: ArrayView<f64, Ix1>, h: f64) -> f64 {
    let len = samples.dim();

    assert!(len % 2 == 0,
            "Periodic Simpson's rule only works for even number of sample points, since the \
             first point in the integration interval is also the last. Please specify an even \
             number of grid cells.");

    unsafe {
        let mut s = samples.uget(0) + samples.uget(0);

        for i in 1..(len / 2) {
            s += 2. * samples.uget(2 * i);
            s += 4. * samples.uget(2 * i - 1);
        }

        s += 4. * samples.uget(len - 1);
        s * h / 3.
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Axis, arr1, arr2};
    use num::Complex;
    use simulation::distribution::Distribution;
    use simulation::grid_width::GridWidth;
    use simulation::particle::Particle;
    use simulation::settings::StressPrefactors;
    use std::f64::EPSILON;
    use std::f64::consts::PI;
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
        let bs = [10., 10.];
        let gs = [10, 10, 6];
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

        let u = i.calculate_flow_field(d.dist.view());

        println!("{}", u);

        assert!(false);
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

        let u = i.calculate_flow_field(d.dist.view());

        i.evolve_particles_inplace(&mut p,
                                   &vec![[0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1],
                                         [0.1, 0.1, 0.1]],
                                   u.view());

        // TODO Check these values!
        assert!(equal_floats(p[0].position.x.v, 0.7100000000000016),
                "got {}",
                p[0].position.x.v);
        assert!(equal_floats(p[0].position.y.v, 0.3100000000000015),
                "got {}",
                p[0].position.y.v);


        let orientations = [1.9880564727277061,
                            3.5488527995226065,
                            4.067451575120913,
                            0.4072601459328098,
                            1.7044728684146264];

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

        let u = i.calculate_flow_field(d.dist.view());

        let vort = vorticity(gw, u.view());

        b.iter(|| i.evolve_particle_inplace(&mut p, &[0.1, 0.1, 0.1], &u.view(), &vort.view()));
    }


    #[test]
    fn test_simpson() {
        let h = PI / 100.;
        let f = Array::range(0., PI, h).map(|x| x.sin());
        let integral = periodic_simpson_integrate(f.view(), h);

        assert!(equal_floats(integral, 2.000000010824505),
                "h: {}, result: {}",
                h,
                integral);


        let h = PI / 100.;
        let f = Array::range(0., TWOPI, h).map(|x| x.sin());
        let integral = periodic_simpson_integrate(f.view(), h);

        assert!(equal_floats(integral, 0.000000000000000034878684980086324),
                "h: {}, result: {}",
                h,
                integral);


        let h = 4. / 100.;
        let f = Array::range(0., 4., h).map(|x| x * x);
        let integral = periodic_simpson_integrate(f.view(), h);
        assert!(equal_floats(integral, 21.120000000000001),
                "h: {}, result: {}",
                h,
                integral);


        let h = TWOPI / 102.;
        let mut f = Array::zeros((102));
        f[51] = 1. / h;
        let integral = periodic_simpson_integrate(f.view(), h);
        assert!(equal_floats(integral, 1.),
                "h: {}, result: {}",
                h,
                integral);
    }

    #[test]
    fn test_simpson_map_axis() {
        let points = 100;
        let h = PI / points as f64;
        let f = Array::range(0., PI, h)
            .map(|x| x.sin())
            .into_shape((1, 1, points))
            .unwrap()
            .broadcast((10, 10, points))
            .unwrap()
            .to_owned();

        let integral = f.map_axis(Axis(2), |v| super::periodic_simpson_integrate(v, h));

        for e in integral.iter() {
            assert!((e - 2.000000010824505).abs() < EPSILON);
        }
    }


}
