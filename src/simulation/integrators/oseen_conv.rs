use consts::TWOPI;
use fftw3::complex::Complex;
use fftw3::fft;
use fftw3::fft::FFTPlan;
use ndarray::{Array, ArrayView, Axis, Ix, Ix1, Ix2, Ix3, Ix4};
use rayon::prelude::*;
use ::simulation::distribution::Distribution;
use ::simulation::grid_width::GridWidth;
use simulation::particle::Particle;
use simulation::settings::{GridSize, StressPrefactors};
use std::f64::consts::PI;

pub type FlowField = Array<f64, Ix3>;

/// Holds parameter needed for time step
#[derive(Debug)]
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
    stress_kernel: Array<f64, Ix3>,
    oseen_kernel_fft: Array<Complex<f64>, Ix4>,
    parameter: IntegrationParameter,
    grid_width: GridWidth,
}

impl Integrator {
    /// Returns a new instance of the mc_sampling integrator.
    pub fn new(grid_size: GridSize,
               grid_width: GridWidth,
               parameter: IntegrationParameter)
               -> Integrator {

        if (grid_size[2] - 2) % 4 != 0 {
            warn!("To have an orientation grid point in the direction of magnetic field, use a \
                   grid size of 2 + 4 * n, with n an integer.");
        }

        Integrator {
            stress_kernel: Integrator::calc_stress_kernel(grid_size, grid_width, parameter.stress),
            oseen_kernel_fft: Integrator::calc_oseen_kernel(grid_size, grid_width),
            parameter: parameter,
            grid_width: grid_width,
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
            e[[0, 0]] = stress.active * 0.5 * (2. * a).cos();
            e[[0, 1]] = stress.active * a.sin() * a.cos() + stress.magnetic * a.cos();
            e[[1, 0]] = stress.active * a.sin() * a.cos() - stress.magnetic * a.cos();
            e[[1, 1]] = -e[[0, 0]];
        }

        s
    }


    /// The Oseen tensor diverges at the origin and is not defined. Thus, it
    /// can't be sampled in the origin. To work around this, a Oseen kernel at
    /// an even number of grid points is used. Since this also means, that the
    /// Oseen kernel has no identical center it cannot be centered on the
    /// 'image'. When just using an even sampled Oseen tensor as a filter
    /// kernel, the value of the filter at a grid point is the value at a
    /// corner of the grid cell. To get an interpolated value at the center of
    /// the cell an average of all cell corners is calculated.
    fn calc_oseen_kernel(grid_size: GridSize, grid_width: GridWidth) -> Array<Complex<f64>, Ix4> {

        /*
        // Grid size should be odd. Origin of the kernel must be in cell [0, 0]. To
        // have symmetric kernel, an odd number of grid cells is needed. To fix this,
        // insert zeros at the border of the kernel, when having an even grid.
        assert!(grid_size[0] % 2 == 1 && grid_size[1] % 2 == 1,
                "Even sized grids are not supported yet for calculating the flow field. Found a \
                 grid-size of ({}, {})",
                grid_size[0],
                grid_size[1]);
        */


        // Grid size should be even. Origin of the kernel must be in cell [0, 0].
        // Needs even kernel, to skip origin and do a sub-cell average sampling.
        // To fix this,/ insert zeros at the border of the kernel, when having
        // an odd sized grid.
        assert!(grid_size[0] % 2 == 0 && grid_size[1] % 2 == 0,
                "Odd sized grids are not supported yet for calculating the flow field. Found a \
                 grid-size of ({}, {})",
                grid_size[0],
                grid_size[1]);

        // Define Oseen-Tensor
        let oseen = |x: f64, y: f64| {
            let norm: f64 = (x * x + y * y).sqrt();
            // Normalization due to forth and back Fourier transformation. FFTW3 does not
            // do this! Calculate it here once for further use in
            //     u = 1/n IFFT( FFT(oseen) * FFT(forcedensity))
            //     == IFFT(FFT(1/n oseen) * FFT(forcedensity))
            let fft_norm = (grid_size[0] * grid_size[1]) as f64;
            let p = 1. / 8. / PI / norm / norm / norm / fft_norm;

            [[Complex::new(2. * x * x + y * y, 0.) * p, Complex::new(x * y, 0.) * p],
             [Complex::new(y * x, 0.) * p, Complex::new(x * x + 2. * y * y, 0.) * p]]
        };

        // Allocate array to prepare FFT
        // Consider to align memory for SIMD
        let mut res = Array::<Complex<f64>, _>::from_elem((2, 2, grid_size[0], grid_size[1]),
                                                          Complex::new(0., 0.));

        let gw_x = grid_width.x;
        let gw_y = grid_width.y;

        let gs_x = grid_size[0] as i64;
        let gs_y = grid_size[1] as i64;

        for (i, v) in res.indexed_iter_mut() {
            // sample Oseen tensor, so that the origin lies on the [0, 0]
            let xi = ((i.2 as i64 + gs_x as i64 / 2) % gs_x) - gs_x as i64 / 2;
            let yi = ((i.3 as i64 + gs_y as i64 / 2) % gs_y) - gs_y as i64 / 2;
            // let x = gw_x * xi as f64;
            // let y = gw_y * yi as f64;
            let x = gw_x * xi as f64 + gw_x / 2.;
            let y = gw_y * yi as f64 + gw_y / 2.;

            // // because the oseen tensor diverges at the origin, set this to zero, since
            // // also the force of a singular particles onto the fluid is zero at the
            // // particle's position
            // if xi == 0 && yi == 0 {
            //     *v = Complex::new(0., 0.);
            // } else {
            //     *v = oseen(x, y)[i.0][i.1];
            // }

            // Calcualte the average of a shifted kernel, where all four points next to the
            // origin are shifted once into the center. This is done, to get an estimate of
            // the correct value in the center of the cell. It is necessary, since we're
            // using an even dimensioned kernel.
            // Because of the linearity of the fourier transform, it does not matter if the
            // average is calculated before or after the transformation.
            *v = (oseen(x, y)[i.0][i.1] + oseen(x - gw_x, y)[i.0][i.1] +
                  oseen(x, y - gw_y)[i.0][i.1] +
                  oseen(x - gw_x, y - gw_y)[i.0][i.1]) / 4.;
        }


        for mut row in res.outer_iter_mut() {
            for mut elem in row.outer_iter_mut() {

                let plan = FFTPlan::new_c2c_inplace(&mut elem,
                                                    fft::FFTDirection::Forward,
                                                    fft::FFTFlags::Estimate)
                    .unwrap();
                plan.execute()

            }
        }

        res
    }


    /// Calculates force on the flow field because of the sress contributions.
    /// The first axis represents the direction of the derivative, the other
    /// correspond to the spatial dimension.
    ///
    /// self.stress is a 2 x 2 matrix sampled for different angles,
    /// resulting in 2 x 2 x a
    /// dist is a 2 x l x t x a, with l x t spatial samples
    /// Want to calculate the matrix product of the transpose of the first to
    /// axies of self.stress and the first axis of dist.
    ///
    /// In Python's numpy, I could do
    /// ´´´
    /// t[0, nx, ny, :] =
    ///     s[:, 0, 0] * d[0, nx, ny, :] + s[:, 1, 0] * d[1, nx, ny, :]
    /// t[1, nx, ny, :] =
    ///     s[:, 0, 1] * d[0, nx, ny, :] + s[:, 1, 1] * d[1, nx, ny, :]
    /// ´´´
    ///
    /// Followed by an simpson rule integration along the angular axis for
    /// every component of the flow field u and for every point in
    /// space (nx, ny).
    ///
    /// Example suggestive implementation in Python for first component of u
    /// and position (nx, ny).
    /// ´´´
    /// l = t[0, nx, ny, 0] + t[0, nx, ny, -1]
    ///
    /// for i in range(1, na, 2):
    ///     l += 4 * t[0, nx, ny, i]
    ///
    /// for i in range(2, na - 1, 2):
    ///     l += 2 * t[0, nx, ny, i]
    ///
    /// f[0, nx, ny] = l * grid_width_angle / 3
    /// ´´´
    ///
    /// The result as dimensions (compontent, x, y).
    pub fn calc_stress_divergence(&self, dist: &Distribution) -> Array<Complex<f64>, Ix3> {
        // Calculates (grad Psi)_i * stress_kernel_(i, j) for every point on the
        // grid and j = 0.
        // This makes implicit and explicit use of broadcasting. Implicitly the
        // stress_kernel ´sk´ is broadcasted for all points in space. Then,
        // explicitly, the gradient ´g´ is broadcasted along the last axis. This
        // effectively repeats the gradient along the second index of the stress
        // kernel ´sk´. Multiplying it elementwise results in
        //
        //  [[g_1, g_1],  *  [[sk_11, sk_12],  =  [[g_1 * sk_11, g_1 * sk_12],
        //   [g_2, g_2]]      [sk_21, sk_22]]      [g_2 * sk_21, g_2 * sk_22]
        //
        // Now buy summing along the first index of the matrix, it results in
        // [ g_1 * sk_11 + g_2 * sk_21, g_1 * sk_12 + g_2 * sk_22]
        //
        // This is done for every point (x, y, alpha).

        let h = dist.get_grid_width();

        let g = dist.spatgrad();
        let sk = self.stress_kernel.view();

        let sh_g = g.dim();
        let sh_sk = sk.dim();
        assert_eq!(sh_g.3,
                   sh_sk.2,
                   "Distribution gradient and stress_kernel should have same number of angles!");

        // Haven't found a better way to do this, since ndarray uses tuples for
        // encoding shapes.
        let shape_g_newaxis = (1, sh_g.0, sh_g.1, sh_g.2, sh_g.3);
        let shape_sk_newaxis = (sh_sk.0, sh_sk.1, 1, 1, sh_sk.2);
        let shape_broadcast = (2, sh_g.0, sh_g.1, sh_g.2, sh_g.3);

        // TODO: Error handling
        let g = g.into_shape(shape_g_newaxis).unwrap();
        let g = g.broadcast(shape_broadcast).unwrap();
        let sk = sk.into_shape(shape_sk_newaxis).unwrap();
        let sk = sk.broadcast(shape_broadcast).unwrap();

        // TODO: Test if this actually works, as expected. Should produce a
        // matrix-vector-product for every (x, y, alpha) coordinate.
        // .to_owned() creates a uniquly owned array that will contain the result and
        // `int` is then bound to.
        let int = (g.to_owned() * sk).sum(Axis(1));

        // Integrate along angle
        int.map_axis(Axis(3),
                     |v| Complex::from(periodic_simpson_integrate(v, h.a)))
    }


    /// Calculate flow field by convolving the Green's function of the stokes
    /// equation (Oseen tensor) with the stress field divergence (force density)
    pub fn calculate_flow_field(&self, dist: &Distribution) -> Array<f64, Ix3> {
        let mut f = self.calc_stress_divergence(dist);

        // // Just for testing, if memory is continuous.
        // f.subview(Axis(0), 0).to_owned().as_slice().unwrap();

        // Fourier transform force density component-wise
        // WARNING: Not thread safe!
        for mut a in f.outer_iter_mut() {
            let plan = FFTPlan::new_c2c_inplace(&mut a,
                                                fft::FFTDirection::Forward,
                                                fft::FFTFlags::Estimate)
                .unwrap();
            plan.execute();
        }

        // Make use of auto-broadcasting of lhs
        let mut u = (&self.oseen_kernel_fft * &f).sum(Axis(1));

        // Inverse Fourier transform flow field component-wise
        for mut a in u.outer_iter_mut() {
            let plan = FFTPlan::new_c2c_inplace(&mut a,
                                                fft::FFTDirection::Backward,
                                                fft::FFTFlags::Estimate)
                .unwrap();
            plan.execute();
        }

        u.map(|x| x.re())
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
        // let sin_orient = p.orientation.as_ref().sin();
        // Calculating sqrt() is more efficient than sin().
        let sin_orient = if (0. <= p.orientation.v && p.orientation.v <= 1.5707963267948966) ||
                            p.orientation.v >= 4.71238898038469 {
            (1. - cos_orient * cos_orient).sqrt()
        } else {
            -(1. - cos_orient * cos_orient).sqrt()
        };

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
        let vort = vorticity(self.grid_width, &flow_field);


        particles.par_iter_mut()
            .zip(random_samples.par_iter())
            .for_each(|(ref mut p, r)| {
                self.evolve_particle_inplace(p, &r, &flow_field, &vort.view())
            });
    }
}


/// Implements the operation `dx uy - dy ux` on a given discretized flow field
/// `u=(ux, uy)`.
fn vorticity(grid_width: GridWidth, u: &ArrayView<f64, Ix3>) -> Array<f64, Ix2> {
    let sh = u.shape();
    let sx = sh[1];
    let sy = sh[2];
    let mut res = Array::zeros((sx, sy));

    let hx = 2. * grid_width.x;
    let hy = 2. * grid_width.y;

    for (i, _) in u.indexed_iter() {
        match i {
            (0, ix, iy) => {
                let ym = (iy + sy - 1) % sy;
                let yp = (iy + 1) % sy;
                unsafe {
                    *res.uget_mut((ix, iy)) -= (u.uget((0, ix, yp)) - u.uget((0, ix, ym))) / hy
                }
            }
            (1, ix, iy) => {
                let xm = (ix + sx - 1) % sx;
                let xp = (ix + 1) % sx;
                unsafe {
                    *res.uget_mut((ix, iy)) += (u.uget((1, xp, iy)) - u.uget((1, xm, iy))) / hx
                }
            }
            (_, _, _) => {}
        }
    }

    res
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
    use fftw3::complex::Complex;
    use ndarray::{Array, Axis, arr2};
    use simulation::distribution::Distribution;
    use simulation::grid_width::GridWidth;
    use simulation::particle::Particle;
    use simulation::settings::StressPrefactors;
    use std::f64::{EPSILON, MAX};
    use std::f64::consts::PI;
    use super::*;
    use test::Bencher;

    fn equal_floats(a: f64, b: f64) -> bool {
        let diff = (a - b).abs();
        diff / (a.abs() + b.abs()).min(MAX) < EPSILON
    }


    /// WARNING: Since fftw3 is not thread safe by default, DO NOT test this
    /// function in parallel. Instead test with RUST_TEST_THREADS=1.
    #[test]
    fn new() {
        let bs = [1., 1.];
        let gs = [10, 10, 3];
        let gw = GridWidth::new(gs, bs);
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

        let i = Integrator::new(gs, gw, int_param);

        let should0 = arr2(&[[-0.2499999999999999, 0.9330127018922195],
                             [-0.0669872981077807, 0.2499999999999999]]);
        let should1 = arr2(&[[0.5, -1.0], [1.0, -0.5]]);
        let should2 = arr2(&[[-0.25, 0.066987298107780712], [-0.9330127018922195, 0.25]]);

        let check = |should: Array<f64, Ix2>, stress: ArrayView<f64, Ix2>| for (a, b) in
            should.iter().zip(stress.iter()) {
            assert!(equal_floats(*a, *b), "{} != {}", should, stress);
        };

        check(should0, i.stress_kernel.subview(Axis(2), 0));
        check(should1, i.stress_kernel.subview(Axis(2), 1));
        check(should2, i.stress_kernel.subview(Axis(2), 2));

        assert_eq!(i.stress_kernel.dim(), (2, 2, 3));
        assert_eq!(i.oseen_kernel_fft.dim(), (2, 2, gs[0], gs[1]));

        // TODO check if average oseen tensor is reasonable
    }

    #[test]
    fn test_evolve_particles_inplace() {
        let bs = [1., 1.];
        let gs = [10, 10, 6];
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

        let i = Integrator::new(gs, gw, int_param);

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
        assert!(equal_floats(p[0].position.x.v, 0.7100000000000002),
                "got {}",
                p[0].position.x.v);
        assert!(equal_floats(p[0].position.y.v, 0.30999999999999867),
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

        let i = Integrator::new(gs, gw, int_param);

        let mut p = Particle::new(0.6, 0.3, 0., bs);
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
        d.sample_from(&vec![p]);

        let u = i.calculate_flow_field(&d);

        let vort = vorticity(gw, &u.view());

        b.iter(|| i.evolve_particle_inplace(&mut p, &[0.1, 0.1, 0.1], &u.view(), &vort.view()));
    }

    #[test]
    fn test_simpson() {
        let h = PI / 100.;
        let f = Array::range(0., PI, h).map(|x| x.sin());
        let integral = super::periodic_simpson_integrate(f.view(), h);

        assert!(equal_floats(integral, 2.000000010824505),
                "h: {}, result: {}",
                h,
                integral);


        let h = 4. / 100.;
        let f = Array::range(0., 4., h).map(|x| x * x);
        let integral = super::periodic_simpson_integrate(f.view(), h);
        assert!(equal_floats(integral, 21.120000000000001),
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

    #[test]
    fn test_calc_stress_divergence() {
        let bs = [1., 1.];
        let gs = [10, 10, 10];
        let gw = GridWidth::new(gs, bs);
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

        let i = Integrator::new(gs, gw, int_param);
        let mut d = Distribution::new(gs, GridWidth::new(gs, bs));

        d.dist = Array::zeros(gs);

        let res = i.calc_stress_divergence(&d);

        assert_eq!(Array::from_elem((2, gs[0], gs[1]), Complex::new(0., 0.)),
                   res);
    }
}
