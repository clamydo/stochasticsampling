use super::*;
// use ndarray::Array;
// use simulation::mesh::grid_width::GridWidth;
use bincode;
use ndarray::{Array, Ix4};
// use num_complex::Complex;
// use simulation::distribution::Distribution;
// use simulation::flowfield::spectral_solver::SpectralSolver;
// use simulation::integrators::langevin::{IntegrationParameter, Integrator};
use simulation::mesh::grid_width::GridWidth;
// use simulation::particle::Particle;
use simulation::settings::{BoxSize, GridSize, StressPrefactors};
// use std::f64::consts::PI;
use test_helper::equal_floats;

// #[test]
// fn test_stress_expectation_value() {
//     let bs = BoxSize {
//         x: 1.,
//         y: 1.,
//         z: 1.,
//     };
//     let gs = GridSize {
//         x: 1,
//         y: 1,
//         z: 1,
//         phi: 6,
//         theta: 5,
//     };
//
//     let gw = GridWidth::new(gs, bs);
//
//     let s = StressPrefactors {
//         active: 1.,
//         magnetic: 0.,
//     };
//
//     let int_param = IntegrationParameter {
//         timestep: 0.0,
//         trans_diffusion: 0.0,
//         rot_diffusion: 0.0,
//         stress: s,
//         magnetic_reorientation: 0.0,
//     };
//
//     let i = Integrator::new(gs, bs, int_param);
//     let spectral_solver = SpectralSolver::new(gs, bs, s);
//
//     let test_it = |d: &Distribution, expect: ArrayView<f64, Ix2>, case| {
//         println!("Distribution {}", d.dist);
//
//         let
//         let is = stress_field.slice(s![.., .., ..1, ..1]);
//         println!("stress {}", is.into_shape([3, 3]).unwrap());
//
//         for (i, e) in is.iter().zip(expect.iter()) {
//             assert!(equal_floats(i.re, *e), "{}: {} != {}", case, i.re, *e);
//         }
//     };
//
//     let p = vec![Particle::new(0.0, 0.0, 1.5707963267948966, bs)];
//     let mut d1 = Distribution::new(gs, gw);
//     d1.sample_from(&p);
//
//     let mut d2 = Distribution::new(gs, gw);
// d2.dist = Array::from_elem([gs.x, gs.y, gs.phi], 1. / gw.phi / gs.phi as
// f64);
//
//     let expect1 = arr2(&[
//         [-0.49999999999999994, 0., 0.],
//         [0., 0.5, 0.],
//         [0., 0., -0.49999999999999994],
//     ]);
// let expect2 = arr2(&[[0., 0., 0.], [0., 0., 0.], [0., 0.,
// -0.49999999999999994]]);
//
//     test_it(&d1, expect1.view(), "1");
//     test_it(&d2, expect2.view(), "2");
// }

#[test]
fn stress_kernel_test() {
    let bs = BoxSize {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let gs = GridSize {
        x: 1,
        y: 1,
        z: 1,
        phi: 9,
        theta: 9,
    };

    let gw = GridWidth::new(gs, bs);

    let s = StressPrefactors {
        active: 1.,
        magnetic: 1.,
    };

    let sk = stress_kernel(gs, gw, s);

    let cf = "test/control/stress_kernel.bincode";

    // Serialize into control file
    // bincode::serialize_into(::std::fs::File::create(cf).unwrap(), &sk).unwrap();

    // Deserialize from previously saved control file
    let sk_control: Array<f64, Ix4> =
        bincode::deserialize_from(::std::fs::File::open(cf).unwrap()).unwrap();

    // TODO, WARNING this is prone to heavy numerical noise
    fn round(a: f64) -> f64 {
        (a * 1e12).round()
    }

    for (a, b) in sk.iter().zip(sk_control.iter()) {
        let a = round(*a);
        let b = round(*b);
        assert!(equal_floats(a, b), "left: {} != right: {}", a, b);
    }

}
