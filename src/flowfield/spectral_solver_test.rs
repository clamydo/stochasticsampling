#![allow(clippy::float_cmp,clippy::unreadable_literal)]
use super::super::stress::stresses::*;
use super::*;

use crate::distribution::Distribution;
use crate::particle::Particle;
use test::Bencher;
use crate::test_helper::equal_floats;

#[test]
#[ignore]
// Is for some reason not deterministic. Probably FFTW3 does select different
// algorithms from time to time.
fn test_calculate_flow_field_against_cache() {
    use bincode;
    use std::fs::File;

    let mut f = File::open("test/flowfield/ff_test.bincode").unwrap();
    let cache_ff: FlowField3D = bincode::deserialize_from(&mut f).unwrap();

    let bs = BoxSize {
        x: 11.,
        y: 11.,
        z: 11.,
    };
    let gs = GridSize {
        x: 11,
        y: 11,
        z: 11,
        phi: 11,
        theta: 11,
    };

    let s = |phi, theta| 1. * stress_active(phi, theta) + 0. * stress_magnetic(phi, theta);

    let mut ff_s = SpectralSolver::new(gs, bs, s);

    let p = vec![Particle::new(
        0.0,
        0.0,
        0.0,
        ::std::f32::consts::PI / 2.,
        ::std::f32::consts::PI / 2.,
        &bs,
    )];
    let mut d = Distribution::new(gs, bs);
    d.sample_from(&p);
    d.dist *= bs.x * bs.y * bs.z;

    let (ff, _) = ff_s.mean_flow_field(1., &d);
    let ff = ff.map(|v| v.re);

    // let mut f = File::create("test/flowfield/ff_test.bincode").unwrap();
    // bincode::serialize_into(&mut f, &ff).unwrap();

    for (a, b) in ff.indexed_iter().zip(cache_ff.indexed_iter()) {
        let (ia, va) = a;
        let (_, vb) = b;

        let f = 2.0f32.powi(51);

        let va = (va * f).round() / f;
        let vb = (vb * f).round() / f;

        assert!(equal_floats(va, vb), "{} != {} at {:?}", va, vb, ia);
    }
}

// #[test]
// fn test_calculate_flow_field() {
//     let bs = BoxSize {
//         x: 21.,
//         y: 21.,
//         z: 21.,
//     };
//     let gs = GridSize {
//         x: 21,
//         y: 21,
//         z: 21,
//         phi: 50,
//         theta: 50,
//     };
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
//
// let p = vec![Particle::new(0.0, 0.0, 0.0, 0.0, ::std::f32::consts::PI /
// 2., 1.5707963267948966, bs)];
//     let mut d = Distribution::new(gs, GridWidth::new(gs, bs));
//     d.sample_from(&p);
//
//     let u = i.calculate_flow_field(&d);
//
//     let theory = |x1: f32, x2: f32| {
//         [
// (x1 * (x1 * x1 - 2. * x2 * x2)) / (8. * PI * (x1 * x1 + x2 *
// x2).powf(5. / 2.)),
// (x2 * (x1 * x1 - 2. * x2 * x2)) / (8. * PI * (x1 * x1 + x2 *
// x2).powf(5. / 2.)),
//         ]
//     };
//
//     let mut grid = get_k_mesh(
//         gs,
//         BoxSize {
//             x: TWOPI,
//             y: TWOPI,
//             z: 1.,
//         },
//     ).remove_axis(Axis(3));
//
//
//     // bring components to the back
//     grid.swap_axes(0, 1);
//     grid.swap_axes(1, 2);
//
//     let mut th_u_x = grid.map_axis(Axis(2), |v| theory(v[0].re, v[1].re)[0]);
//
//     let mut th_u_y = grid.map_axis(Axis(2), |v| theory(v[0].re, v[1].re)[1]);
//
//     th_u_x[[0, 0]] = 0.;
//     th_u_y[[0, 0]] = 0.;
//
//     // println!("th_u_x: {}", th_u_x);
//     // println!("sim_u_x: {}", u.subview(Axis(0), 0));
//
//     let diff = (th_u_x - u.subview(Axis(0), 0))
//         .map(|v| v.abs())
//         .scalar_sum();
//
//     assert!(diff <= 0.14, "diff: {}", diff);
//
//     let diff = (th_u_y - u.subview(Axis(0), 1))
//         .map(|v| v.abs())
//         .scalar_sum();
//
//     assert!(diff <= 0.22, "diff: {}", diff);
//
//     assert!(equal_floats(u.scalar_sum(), 0.), "{} != 0", u.scalar_sum());
// }

#[test]
fn test_compare_implementations() {
    let bs = BoxSize {
        x: 10.,
        y: 10.,
        z: 10.,
    };
    let gs = GridSize {
        x: 20,
        y: 20,
        z: 20,
        phi: 15,
        theta: 15,
    };

    let s = |phi, theta| 1. * stress_active(phi, theta) + 0. * stress_magnetic(phi, theta);

    let mut ff_s = SpectralSolver::new(gs, bs, s);

    let p = Particle::create_isotropic(100000, &bs, 1);

    let mut d = Distribution::new(gs, bs);
    d.sample_from(&p);

    let ff = ff_s.solve_flow_field(&d);
    let (ff_new, _) = ff_s.mean_flow_field(1., &d);
    let ff_new = ff_new.map(|v| v.re);

    for (a, b) in ff.indexed_iter().zip(ff_new.indexed_iter()) {
        let (ia, va) = a;
        let (_, vb) = b;

        let f = 2.0f32.powi(51);

        let va = (va * f).round() / f;
        let vb = (vb * f).round() / f;

        assert!(equal_floats(va, vb), "{} != {} at {:?}", va, vb, ia);
    }
}

#[bench]
fn bench_calculate_flow(b: &mut Bencher) {
    let bs = BoxSize {
        x: 30.,
        y: 30.,
        z: 30.,
    };
    let gs = GridSize {
        x: 30,
        y: 30,
        z: 30,
        phi: 15,
        theta: 15,
    };

    let s = |phi, theta| 1. * stress_active(phi, theta) + 0. * stress_magnetic(phi, theta);

    let mut ff_s = SpectralSolver::new(gs, bs, s);

    let p = Particle::create_isotropic(10000, &bs, 1);

    let mut d = Distribution::new(gs, bs);
    d.sample_from(&p);
    d.dist *= bs.x * bs.y * bs.z;

    b.iter(|| {
        ::test::black_box(ff_s.solve_flow_field(&d));
    })
}

// #[bench]
// fn bench_calculate_flow_new(b: &mut Bencher) {
//     let bs = BoxSize {
//         x: 30.,
//         y: 30.,
//         z: 30.,
//     };
//     let gs = GridSize {
//         x: 30,
//         y: 30,
//         z: 30,
//         phi: 15,
//         theta: 15,
//     };
//     let s = |phi, theta| 1. * stress_active(phi, theta) + 0. * stress_magnetic(phi, theta);
//
//     let mut ff_s = SpectralSolver::new(gs, bs, s);
//
//     let p = Particle::place_isotropic(10000, bs, [1, 1]);
//
//     let mut d = Distribution::new(gs, bs);
//     d.sample_from(&p);
//     d.dist *= bs.x * bs.y * bs.z;
//
//     b.iter(|| {
//         ::test::black_box(ff_s.update_flow_field(&d));
//     })
// }
