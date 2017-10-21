//! A representation for the probability distribution function.

use super::grid_width::GridWidth;
use super::particle::Particle;
use super::settings::GridSize;
use ndarray::{Array, Ix, Ix5};
use std::ops::Index;


/// Holds a normalised sampled distribution function on a grid, assuming the
/// sampling points to be centered in a grid cell. This means, that the value
/// at position `x_j` (for `j=0,...,N-1`, on a grid with `N` cells and `x_0 =
/// w/2`) is the average of particles in the interval `[x_j - w/2, x_j +
/// w/2]`, with the grid width `w`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    /// `dist` contains the probability for a particle in the box at position of
    /// first two axis and the direction of the last axis.
    pub dist: Array<f64, Ix5>,
    /// `grid_width` contains the size of a unit cell of the grid.
    grid_width: GridWidth,
}

type GridCoordinate = [Ix; 5];

impl Distribution {
    /// Returns a zero initialised instance of Distribution.
    pub fn new(grid: GridSize, grid_width: GridWidth) -> Distribution {

        let grid = [grid.x, grid.y, grid.z, grid.phi, grid.theta];

        Distribution {
            dist: Array::default(grid),
            grid_width: grid_width,
        }
    }

    /// Returns the width of on cell for every axis
    pub fn get_grid_width(&self) -> GridWidth {
        self.grid_width
    }

    pub fn dim(&self) -> (Ix, Ix, Ix, Ix, Ix) {
        self.dist.dim()
    }

    /// Transforms a continous particle coordinate into a discrete grid
    /// coordinate. Maps particle inside an volume *centered* around the grid
    /// point to that grid point.
    /// The first grid point does not lie on the box border, but a half cell
    /// width from it.
    /// WARNING: Expects coordinates to be in interval `[0, box_size)],
    /// excluding the right border.
    pub fn coord_to_grid(&self, p: &Particle) -> GridCoordinate {
        debug_assert!(
            p.position.x >= 0. && p.position.y >= 0. && p.position.z >= 0. &&
                p.orientation.phi >= 0. && p.orientation.theta >= 0.,
            "Got negative position or orientation {:?}",
            p
        );

        debug_assert!(
            p.orientation.theta <= ::std::f64::consts::PI,
            "Theta is not in range> {:?}",
            p
        );

        let gx = (p.position.x / self.grid_width.x).floor() as Ix;
        let gy = (p.position.y / self.grid_width.y).floor() as Ix;
        let gz = (p.position.z / self.grid_width.z).floor() as Ix;
        let gphi = (p.orientation.phi / self.grid_width.phi).floor() as Ix;
        let gtheta = (p.orientation.theta / self.grid_width.theta).floor() as Ix %
            self.dist.dim().4;

        // trust in positions are in bound of PBC
        [gx, gy, gz, gphi, gtheta]
    }

    /// Initialises the distribution with a number histogram. It counts the
    /// `particles` inside a bin of the grid. Returns the overall number of
    /// particles counted.
    fn histogram_from(&mut self, particles: &[Particle]) -> usize {
        // zero out distribution
        for i in self.dist.iter_mut() {
            *i = 0.0;
        }

        // build histogram
        for p in particles {
            let c = self.coord_to_grid(p);
            // WARNING: Does not check boundaries at compile time!
            self.dist[c] += 1.;
        }

        particles.len()
    }

    /// Estimates the approximate values for the distribution function at the
    /// grid points using grid cell averages.
    pub fn sample_from(&mut self, particles: &[Particle]) {
        let n = self.histogram_from(particles) as f64;

        // Scale by grid cell volume, in order to arrive at a sampled function,
        // averaged over a grid cell. Missing this would result into the
        // integral over/ the grid cell volume at a given grid coordinate and
        // not the approximate value of the distribution function.
        // Also normalise to one.
        let GridWidth {
            x: gx,
            y: gy,
            z: gz,
            phi: gphi,
            theta: gtheta,
        } = self.grid_width;

        // WARNING: In principle the scaling goes with sin(theta), since the volume on
        // the sphere surface shrinks. But this is canceld by integration over the
        // orientation in the flow-field calculation, so skipped here
        self.dist /= gx * gy * gz * gphi * gtheta * n;
    }
}


/// Implement index operator that wraps around for periodic boundaries.
impl Index<[i32; 5]> for Distribution {
    type Output = f64;

    fn index(&self, index: [i32; 5]) -> &f64 {
        fn wrap(i: i32, b: i32) -> usize {
            (((i % b) + b) % b) as usize
        }

        let (sx, sy, sz, sphi, stheta) = self.dim();
        unsafe {
            self.dist.uget((
                wrap(index[0], sx as i32),
                wrap(index[1], sy as i32),
                wrap(index[2], sz as i32),
                wrap(index[3], sphi as i32),
                wrap(index[4], stheta as i32),
            ))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use simulation::grid_width::GridWidth;
    use simulation::particle::Particle;
    use simulation::settings::{BoxSize, GridSize};
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
        let dist = Distribution::new(gs, GridWidth::new(gs, bs));
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
        let gw = GridWidth::new(grid_size, box_size);
        let n = 1000;
        let p = Particle::place_isotropic(n, box_size, [1, 1]);
        let mut d = Distribution::new(grid_size, gw);

        d.histogram_from(&p);

        let sum = d.dist.fold(0., |s, x| s + x);
        // Sum over all bins should be particle number
        assert_eq!(sum, n as f64);

        let p2 = vec![Particle::new(0.6, 0.3, 0., 4., 1., box_size)];

        d.histogram_from(&p2);
        println!("{}", d.dist);

        assert_eq!(d.dist[[3, 1, 0, 1, 0]], 1.0);
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
        let p = Particle::place_isotropic(n, box_size, [1, 1]);
        let mut d = Distribution::new(grid_size, GridWidth::new(grid_size, box_size));

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

        let p2 = vec![Particle::new(0.6, 0.3, 0., 0., 0., box_size)];

        d.sample_from(&p2);
        println!("{}", d.dist);

        // Check if properly normalised to 1 (N = 1)
        assert!(
            equal_floats(d.dist[[3, 1, 0, 0, 0]] * vol, 1.),
            "Value is {}, but expected: {}.",
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
            let dist = Distribution::new(gs, GridWidth::new(gs, bs));

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
            let p = Particle::new(i[0], i[1], i[2], i[3], i[4], box_size);

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
        let mut d = Distribution::new(grid_size, GridWidth::new(grid_size, box_size));

        d.dist[[1, 2, 0, 1, 0]] = 42.;


        assert_eq!(d[[1, 2, 0, 1, 0]], 42.);
        assert_eq!(d[[-1, 2, 0, 1, 0]], 42.);
        assert_eq!(d[[1, -1, 0, 1, 0]], 42.);
        assert_eq!(d[[3, -1, 0, 1, 0]], 42.);
        assert_eq!(d[[3, 5, 0, 1, 0]], 42.);
    }
}
