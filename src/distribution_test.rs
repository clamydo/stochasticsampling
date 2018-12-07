use super::*;
use mesh::grid_width::GridWidth;
use particle::Particle;
use {BoxSize, GridSize};
use test_helper::equal_floats;

#[test]
fn new() {
    let gs = GridSize {
        x: 10,
        y: 11,
        z: 12,
        phi: 13,
        theta: 14,
    };
    let bs = BoxSize {
        x: 1.,
        y: 2.,
        z: 3.,
    };
    let dist = Distribution::new(gs, bs);
    assert_eq!(dist.dim(), (10, 11, 12, 13, 14));
}

#[test]
fn histogram() {
    let grid_size = GridSize {
        x: 5,
        y: 5,
        z: 1,
        phi: 2,
        theta: 2,
    };
    let box_size = BoxSize {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let n = 1000;
    let p = Particle::create_isotropic(n, &box_size, 1);
    let mut d = Distribution::new(grid_size, box_size);

    d.histogram_from(&p);

    let sum = d.dist.fold(0., |s, x| s + x);
    // Sum over all bins should be particle number
    assert_eq!(sum, n as f64);
    let mut p2 = Particle::new(0.6, 0.3, 0., 4., 1., &box_size);
    p2.pbc(&box_size);
    let p2v = vec![p2];

    d.histogram_from(&p2v);
    println!("{:?}", d.grid_width);
    println!("{:?}", d.coord_to_grid(&p2));
    println!("{}", d.dist);

    assert_eq!(d.dist[[2, 1, 0, 1, 0]], 1.0);
}

#[test]
fn sample_from() {
    let box_size = BoxSize {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let grid_size = GridSize {
        x: 5,
        y: 5,
        z: 1,
        phi: 2,
        theta: 2,
    };
    let n = 1000;
    let p = Particle::create_isotropic(n, &box_size, 1);
    let mut d = Distribution::new(grid_size, box_size);

    d.sample_from(&p);

    // calculate approximate integral over the distribution function//
    // interpreted as a step function
    let GridWidth {
        x: gx,
        y: gy,
        z: gz,
        phi: gphi,
        theta: gtheta,
    } = d.grid_width;
    let vol = gx * gy * gz * gphi * gtheta;
    // Naive integration, sin(theta) is already included
    let sum = vol * d.dist.scalar_sum();
    assert!(
        equal_floats(sum, 1.),
        "Step function sum is: {}, but expected: {}. Should be normalised.",
        sum,
        1.
    );

    let p2 = vec![Particle::new(0.6, 0.3, 0., 0., 0., &box_size)];

    d.sample_from(&p2);
    println!("{}", d.dist);

    // Check if properly normalised to 1 (N = 1)
    assert!(
        equal_floats(d.dist[[2, 1, 0, 0, 0]] * vol, 1.),
        "Value is {:?}, but expected: {:?}.",
        d.dist[[2, 1, 0, 0, 0]] * vol,
        1.0
    );
}
#[test]
fn coord_to_grid() {
    let box_size = BoxSize {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let grid_size = GridSize {
        x: 50,
        y: 50,
        z: 1,
        phi: 10,
        theta: 1,
    };

    fn check(i: &[f64; 5], o: &[usize; 5], p: Particle, gs: GridSize, bs: BoxSize) {
        let dist = Distribution::new(gs, bs);

        let g = dist.coord_to_grid(&p);

        for ((a, b), c) in i.iter().zip(o.iter()).zip(g.iter()) {
            assert!(
                b == c,
                "For input {:?}. Expected coordinate to be '{}', got '{}'.",
                a,
                b,
                c
            );
        }
    };

    let input = [
        // [0., 0., 2. * ::std::f64::consts::PI - ::std::f64::EPSILON],
        [0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 0.5, 0.],
        [0., 0., 0., 7., 0.],
        [0., 0., 0., -1., 0.],
        [0.96, 0., 0., 0., 0.],
        [0.5, 0.5, 0., -1., 0.],
        [0.5, 0.5, 0., 0., 0.],
        [0., 0., 0., 2. * ::std::f64::consts::PI, 0.],
        [0.51000000000000005, 0.5, 0., 6.283185307179584, 0.],
    ];

    let result = [
        // [0usize, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 8, 0],
        [48, 0, 0, 0, 0],
        [25, 25, 0, 8, 0],
        [25, 25, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [25, 25, 0, 9, 0],
    ];

    for (i, o) in input.iter().zip(result.iter()) {
        let p = Particle::new(i[0], i[1], i[2], i[3], i[4], &box_size);

        check(i, o, p, grid_size, box_size);
    }
}

#[test]
fn index() {
    let box_size = BoxSize {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let grid_size = GridSize {
        x: 2,
        y: 3,
        z: 1,
        phi: 2,
        theta: 1,
    };
    let mut d = Distribution::new(grid_size, box_size);

    d.dist[[1, 2, 0, 1, 0]] = 42.;

    assert_eq!(d[[1, 2, 0, 1, 0]], 42.);
    assert_eq!(d[[-1, 2, 0, 1, 0]], 42.);
    assert_eq!(d[[1, -1, 0, 1, 0]], 42.);
    assert_eq!(d[[3, -1, 0, 1, 0]], 42.);
    assert_eq!(d[[3, 5, 0, 1, 0]], 42.);
}
