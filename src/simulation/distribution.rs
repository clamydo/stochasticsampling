//! A representation for the probability distribution function.

use coordinates::particle::Particle;
use ndarray::{Array, Ix};
use settings::GridSize;
use std::ops::Index;
use super::GridWidth;

/// Array type, that holds the sampled function values.
pub type Bins = Array<f64, (Ix, Ix, Ix)>;

/// Holds a normalised sampled distribution function on a grid, assuming the
/// sampling points to be centered in a grid cell. This means, that the value
/// at position `x_j` (for `j=0,...,N-1`, on a grid with `N` cells and `x_0 =
/// w/2`) is the average of particles in the interval `[x_j - w/2, x_j +
/// w/2]`, with the grid width `w`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    /// `dist` contains the probability for a particle in the box at position of
    /// first two axis and the direction of the last axis.
    pub dist: Bins,
    /// `grid_width` contains the size of a unit cell of the grid.
    grid_width: GridWidth,
}

type GridCoordinate = [Ix; 3];

impl Distribution {
    /// Returns a zero initialised instance of Distribution.
    pub fn new(grid: GridSize, grid_width: GridWidth) -> Distribution {

        Distribution {
            dist: Array::default(grid),
            grid_width: grid_width,
        }
    }

    /// Returns the width of on cell for every axis
    pub fn get_grid_width(&self) -> GridWidth {
        self.grid_width
    }

    pub fn shape(&self) -> (Ix, Ix, Ix) {
        self.dist.dim()
    }

    /// Transforms a continous particle coordinate into a discrete grid
    /// coordinate. Maps particle inside an volume *centered* around the grid
    /// point to that grid point.
    /// The first grid point does not lie on the box border, but a half cell
    /// width from it.
    pub fn coord_to_grid(&self, p: &Particle) -> GridCoordinate {
        let gx = (p.position.x.as_ref() / self.grid_width.x).floor() as Ix;
        let gy = (p.position.y.as_ref() / self.grid_width.y).floor() as Ix;
        let ga = (p.orientation.as_ref() / self.grid_width.a).floor() as Ix;

        [gx, gy, ga]
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
        let GridWidth { x: gx, y: gy, a: ga } = self.grid_width;
        self.dist /= gx * gy * ga * n;
    }

    /// Returns spatial gradient as an array.
    /// The first axis specifies the direction of the derivative, the component
    /// of the gradient, whereas the other axises are the same as for the
    /// distribution. So effectively the result is an array of dimension ´n+1´
    /// if the input was of dimension ´n´.
    ///
    /// The derivative is implemented as a symmetric finite differential
    /// quotient with wrap around coordinates.
    pub fn spatgrad(&self) -> Array<f64, (Ix, Ix, Ix, Ix)> {
        let (sx, sy, sa) = self.shape();
        let mut res = Array::zeros((2, sx, sy, sa));

        let h = &self.grid_width;
        let hx = 2. * h.x;
        let hy = 2. * h.y;

        for (i, _) in self.dist.indexed_iter() {
            let (ix, iy, ia) = i;

            // Make index wrap around because of periodic boundary conditions.
            // Does not use index operation of Distribution, because this is
            // cheaper as it only has to wrap around one dimension at a time.
            let xm = (ix + sx - 1) % sx;
            let xp = (ix + 1) % sx;
            res[(0, ix, iy, ia)] =
                unsafe { (self.dist.uget((xp, iy, ia)) - self.dist.uget((xm, iy, ia))) / hx };

            let ym = (iy + sy - 1) % sy;
            let yp = (iy + 1) % sy;
            res[(1, ix, iy, ia)] =
                unsafe { (self.dist.uget((ix, yp, ia)) - self.dist.uget((ix, ym, ia))) / hy };
        }

        res
    }
}

/// Implement index operator that wraps around for periodic boundaries.
impl Index<[i32; 3]> for Distribution {
    type Output = f64;

    fn index(&self, index: [i32; 3]) -> &f64 {

        fn wrap(i: i32, b: i32) -> usize {
            (((i % b) + b) % b) as usize
        }

        let (sx, sy, sa) = self.shape();
        unsafe {
            self.dist.uget((wrap(index[0], sx as i32),
                            wrap(index[1], sy as i32),
                            wrap(index[2], sa as i32)))
        }
    }
}

#[cfg(test)]
mod tests {
    use coordinates::particle::Particle;
    use ndarray::{Array, Axis, arr3};
    use std::f64::EPSILON;
    use super::*;
    use super::super::{GridWidth, grid_width};

    #[test]
    fn new() {
        let gs = (10, 10, 6);
        let bs = (1., 1.);
        let dist = Distribution::new(gs, grid_width(gs, bs));
        assert_eq!(dist.shape(), (10, 10, 6));
    }

    #[test]
    fn histogram() {
        let grid_size = (5, 5, 2);
        let box_size = (1., 1.);
        let gw = grid_width(grid_size, box_size);
        let n = 1000;
        let p = Particle::randomly_placed_particles(n, box_size, [1, 1]);
        let mut d = Distribution::new(grid_size, gw);

        d.histogram_from(&p);

        let sum = d.dist.fold(0., |s, x| s + x);
        assert_eq!(sum, n as f64);

        let p2 = vec![Particle::new(0.6, 0.3, 0., box_size)];

        d.histogram_from(&p2);
        println!("{}", d.dist);

        assert_eq!(d.dist[[2, 1, 0]], 1.0);
    }

    #[test]
    fn sample_from() {
        let box_size = (1., 1.);
        let grid_size = (5, 5, 2);
        let n = 1000;
        let p = Particle::randomly_placed_particles(n, box_size, [1, 1]);
        let mut d = Distribution::new(grid_size, grid_width(grid_size, box_size));

        d.sample_from(&p);

        // calculate approximate integral over the distribution function//
        // interpreted as a step function
        let GridWidth { x: gx, y: gy, a: ga } = d.grid_width;
        let vol = gx * gy * ga;
        // Naive integration
        let sum = vol * d.dist.fold(0., |s, x| s + x);
        assert!((sum - 1.).abs() <= EPSILON * n as f64,
                "Step function sum is: {}, but expected: {}. Should be normalised.",
                sum,
                n);

        let p2 = vec![Particle::new(0.6, 0.3, 0., box_size)];

        d.sample_from(&p2);
        println!("{}", d.dist);

        // Check if properly normalised to 1 (N = 1)
        assert!((d.dist[[2, 1, 0]] * vol - 1.0).abs() <= EPSILON,
                "Sum is: {}, but expected: {}.",
                d.dist[[2, 1, 0]] * vol,
                1.0);
    }

    #[test]
    fn coord_to_grid() {
        let box_size = (1., 1.);
        let grid_size = (10, 10, 6);

        let input = [[0., 0., 0.],
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 0.5],
                     [0., 0., 7.],
                     [0., 0., -1.],
                     [0.96, 0., 0.]];

        let result = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 5], [9, 0, 0]];

        for (i, o) in input.iter().zip(result.iter()) {
            let p = Particle::new(i[0], i[1], i[2], box_size);
            let dist = Distribution::new(grid_size, grid_width(grid_size, box_size));

            let g = dist.coord_to_grid(&p);

            assert!(g[0] == o[0],
                    "For input {:?}. Expected '{}', got '{}'.",
                    i,
                    o[0],
                    g[0]);
            assert!(g[1] == o[1],
                    "For input {:?}. Expected '{}', got '{}'.",
                    i,
                    o[1],
                    g[1]);
            assert!(g[2] == o[2],
                    "For input {:?}. Expected '{}', got '{}'.",
                    i,
                    o[2],
                    g[2]);
        }
    }

    #[test]
    fn spatgrad() {
        let box_size = (1., 1.);
        let grid_size = (5, 5, 1);
        let mut d = Distribution::new(grid_size, grid_width(grid_size, box_size));

        d.dist = arr3(&[[[1.], [2.], [3.], [4.], [5.]],
                        [[2.], [3.], [4.], [5.], [6.]],
                        [[3.], [4.], [5.], [6.], [7.]],
                        [[4.], [5.], [6.], [7.], [8.]],
                        [[5.], [6.], [7.], [8.], [9.]]]);

        let res_x = arr3(&[[[-7.5], [-7.5], [-7.5], [-7.5], [-7.5]],
                           [[5.0], [5.0], [5.0], [5.0], [5.0]],
                           [[5.0], [5.0], [5.0], [5.0], [5.0]],
                           [[5.0], [5.0], [5.0], [5.0], [5.0]],
                           [[-7.5], [-7.5], [-7.5], [-7.5], [-7.5]]]);

        let res_y = arr3(&[[[-7.5], [5.0], [5.0], [5.0], [-7.5]],
                           [[-7.5], [5.0], [5.0], [5.0], [-7.5]],
                           [[-7.5], [5.0], [5.0], [5.0], [-7.5]],
                           [[-7.5], [5.0], [5.0], [5.0], [-7.5]],
                           [[-7.5], [5.0], [5.0], [5.0], [-7.5]]]);

        let grad = d.spatgrad();


        assert_eq!(grad.subview(Axis(0), 0), res_x);
        assert_eq!(grad.subview(Axis(0), 1), res_y);

        d.dist = Array::zeros(grid_size);
        assert!(d.spatgrad() == Array::<f64, _>::zeros((2, grid_size.0, grid_size.1, grid_size.2)));

        d.dist = Array::zeros(grid_size) + 1.;
        assert!(d.spatgrad() == Array::<f64, _>::zeros((2, grid_size.0, grid_size.1, grid_size.2)));
    }

    #[test]
    fn index() {
        let box_size = (1., 1.);
        let grid_size = (2, 3, 2);
        let mut d = Distribution::new(grid_size, grid_width(grid_size, box_size));

        d.dist = arr3(&[[[1., 1.5], [2., 2.5], [3., 3.5]], [[4., 4.5], [5., 5.5], [6., 6.5]]]);

        assert_eq!(d[[0, 0, 0]], 1.0);
        assert_eq!(d[[-1, 0, 0]], 4.0);
        assert_eq!(d[[-9, 0, 0]], 4.0);
        assert_eq!(d[[-9, -3, 0]], 4.0);
        assert_eq!(d[[0, -3, 0]], 1.0);
        assert_eq!(d[[21, -3, 0]], 4.0);
        assert_eq!(d[[21, -3, 3]], 4.5);
        assert_eq!(d[[21, -3, 4]], 4.0);
    }
}