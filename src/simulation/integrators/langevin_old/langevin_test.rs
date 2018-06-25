use super::*;
// use ndarray::Array;
use simulation::distribution::Distribution;
// use simulation::mesh::grid_width::GridWidth;
use simulation::flowfield::spectral_solver::SpectralSolver;
use simulation::magnetic_interaction::magnetic_solver::MagneticSolver;
use simulation::particle::Particle;
use simulation::settings::{MagneticDipolePrefactors, StressPrefactors};
use test::Bencher;

// /// WARNING: Since fftw3 is not thread safe by default, DO NOT test this
// /// function in parallel. Instead test with RUST_TEST_THREADS=1.
// #[test]
// fn new() {
//     use ndarray::{Ix2, arr2};
//
//     let bs = BoxSize {
//         x: 1.,
//         y: 1.,
//         z: 1.,
//     };
//     let gs = GridSize {
//         x: 10,
//         y: 10,
//         z: 10,
//         phi: 2,
//         theta: 2,
//     };
//     let s = StressPrefactors {
//         active: 1.,
//         magnetic: 1.,
//     };
//
//     let int_param = IntegrationParameter {
//         timestep: 1.,
//         trans_diffusion: 1.,
//         rot_diffusion: 1.,
//         stress: s,
//         magnetic_reorientation: 1.,
//     };
//
//     let i = Integrator::new(gs, bs, int_param);
//
//     let should00 = arr2(&[
//         [-0.3333333333333333, 0., 0.],
//         [0., 0.1666666666666667, -0.2071067811865475],
//         [0., 1.207106781186547, 0.1666666666666667],
//     ]);
//     let should01 = arr2(&[
//         [-0.3333333333333333, 0., 0.],
//         [0., 0.1666666666666667, 0.2071067811865475],
//         [0., -1.207106781186547, 0.1666666666666667],
//     ]);
//     let should10 = arr2(&[
//         [-0.3333333333333333, 0., 0.],
//         [0., 0.1666666666666667, -1.207106781186547],
//         [0., 0.2071067811865475, 0.1666666666666667],
//     ]);
//
//     let should11 = arr2(&[
//         [-0.3333333333333333, 0., 0.],
//         [0., 0.1666666666666667, 1.207106781186547],
//         [0., -0.2071067811865475, 0.1666666666666667],
//     ]);
//
//     fn round(a: f64, digit: i32) -> f64 {
//         (a * 2f64.powi(digit)).round() * 2f64.powi(-digit)
//     }
//
//     fn check(should: Array<f64, Ix2>, stress: Array<f64, Ix2>) {
//         for (a, b) in should.iter().zip(stress.iter()) {
//             assert!(
//                 // round to neglect numerical noise
//                 equal_floats(round(*a, 48), round(*b, 48)),
//                 "{} != {}",
//                 should,
//                 stress
//             );
//         }
//     }
//
//     assert_eq!(i.stress_kernel.dim(), (3, 3, 2, 2));
//
//     println!("00");
//     check(
//         should00,
//         i.stress_kernel
//             .slice(s![.., .., ..1, ..1])
//             .to_owned()
//             .into_shape((3, 3))
//             .unwrap(),
//     );
//     println!("01");
//     check(
//         should01,
//         i.stress_kernel
//             .slice(s![.., .., ..1, 1..2])
//             .to_owned()
//             .into_shape((3, 3))
//             .unwrap(),
//     );
//     println!("10");
//     check(
//         should10,
//         i.stress_kernel
//             .slice(s![.., .., 1..2, ..1])
//             .to_owned()
//             .into_shape((3, 3))
//             .unwrap(),
//     );
//     println!("11");
//     check(
//         should11,
//         i.stress_kernel
//             .slice(s![.., .., 1..2, 1..2])
//             .to_owned()
//             .into_shape((3, 3))
//             .unwrap(),
//     );
// }

// #[test]
// fn test_evolve_particles_inplace() {
//     let bs = BoxSize {
//         x: 1.,
//         y: 1.,
//         z: 1.,
//     };
//     let gs = GridSize {
//         x: 10,
//         y: 10,
//         z: 10,
//         phi: 6,
//     };
//     let s = StressPrefactors {
//         active: 1.,
//         magnetic: 1.,
//     };
//
//     let int_param = IntegrationParameter {
//         timestep: 0.1,
//         trans_diffusion: 0.1,
//         rot_diffusion: 0.1,
//         stress: s,
//         magnetic_reorientation: 0.1,
//     };
//
//     let i = Integrator::new(gs, bs, int_param);
//
//     let mut p = vec![
//         Particle::new(0.6, 0.3, 0., bs),
//         Particle::new(0.6, 0.3, 1.5707963267948966, bs),
//         Particle::new(0.6, 0.3, 2.0943951023931953, bs),
//         Particle::new(0.6, 0.3, 4.71238898038469, bs),
//         Particle::new(0.6, 0.3, 6., bs),
//     ];
//     let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
//     d.sample_from(&p);
//
//     let u = i.calculate_flow_field(&d);
//
//     i.evolve_particles_inplace(
//         &mut p,
//         &vec![
//             [0.1, 0.1, 0.1],
//             [0.1, 0.1, 0.1],
//             [0.1, 0.1, 0.1],
//             [0.1, 0.1, 0.1],
//             [0.1, 0.1, 0.1],
//         ],
//         u.view(),
//     );
//
//     // TODO Check these values!
//     assert!(
//         equal_floats(p[0].position.x.v, 0.71),
//         "got {}, expected {}",
//         p[0].position.x.v,
//         0.71
//     );
//     assert!(
//         equal_floats(p[0].position.y.v, 0.31),
//         "got {}, expected {}",
//         p[0].position.y.v,
//         0.31
//     );
//
//
//     let orientations = [
//         0.24447719561927586,
//         1.8052735224141725,
//         2.3238722980124713,
//         4.946866176003965,
//         6.244078898485779,
//     ];
//
//     for (p, o) in p.iter().zip(&orientations) {
//         assert!(
//             equal_floats(p.orientation.v, *o),
//             "got {} = {}",
//             p.orientation.v,
//             *o
//         );
//     }
// }
//

// #[bench]
// fn bench_evolve_particle_inplace(b: &mut Bencher) {
//     let bs = BoxSize {
//         x: 10.,
//         y: 10.,
//         z: 10.,
//     };
//     let gs = GridSize {
//         x: 10,
//         y: 10,
//         z: 10,
//         phi: 6,
//         theta: 6,
//     };
//     let gw = GridWidth::new(gs, bs);
//     let s = StressPrefactors {
//         active: 1.,
//         magnetic: 1.,
//     };
//
//     let int_param = IntegrationParameter {
//         timestep: 0.1,
//         trans_diffusion: 0.1,
//         rot_diffusion: 0.1,
//         magnetic_reorientation: 0.1,
//         drag: 0.1,
//     };
//
//     let m_param = MagneticDipolePrefactors {
//         magnetic_moment: 1.23,
//     };
//
//     let i = Integrator::new(gs, bs, int_param);
//
//     let mut p = Particle::new(0.6, 0.3, 0., 0., 0., bs);
//     let mut d = Distribution::new(gs, bs);
//     d.sample_from(&vec![p]);
//
//     let mut spectral_solver = SpectralSolver::new(gs, bs, s);
//     let u = spectral_solver.solve_flow_field(&d);
//
//     let m = MagneticSolver::new(gs, bs, m_param);
//
//     let (b, gb) = m.mean_magnetic_field(&d);
//
//     let vort = vorticity3d_dispatch(gw, u.view());
//
//     let r = RandomVector {
//         x: 0.1,
//         y: 0.1,
//         z: 0.1,
//         axis_angle: 0.1,
//         rotate_angle: 0.1,
//     };
//
//     b.iter(|| {
//         for _ in 0..10000 {
//             i.evolve_particle_inplace(&mut p, &r, &u.view(), &vort.view(), &b, &gb)
//         }
//     });
// }
