//! A representation for the probability distribution function.

use coordinates::TWOPI;
use coordinates::particle::Particle;
use ndarray::{Array, ArrayBase, Ix};
use settings::{BoxSize, GridSize};
use std::ops::Index;

#[derive(Debug)]
struct GridWidth {
    x: f64,
    y: f64,
    a: f64,
}

/// Array type, that holds the bins. Since the distribution can be normalised,
/// and thus contain non-integer values, the type f64 was choses for ergnomic
/// reasons.
pub type Bins = Array<f64, (Ix, Ix, Ix)>;

/// Discrete distribution.
///
/// `dist` contains the probability for a particle in the box at position of
/// first two axis and the direction of the last axis.
/// `grid_width` contains the size of a unit cell of the grid.
#[derive(Debug)]
pub struct Distribution {
    pub dist: Bins,
    grid_width: GridWidth,
}

type GridCoordinate = [usize; 3];

impl Distribution {
    /// Returns a zero initialised instance of Distribution.
    pub fn new(grid: GridSize, boxdim: BoxSize) -> Distribution {
        let grid_width = GridWidth {
            x: boxdim.0 as f64 / grid.0 as f64,
            y: boxdim.1 as f64 / grid.1 as f64,
            a: TWOPI / grid.2 as f64,
        };

        Distribution {
            dist: Array::default(grid),
            grid_width: grid_width,
        }
    }

    pub fn shape(&self) -> (Ix, Ix, Ix) {
        self.dist.dim()
    }

    /// Maps particle coordinate onto grid coordinate/index (starting from
    /// zero).
    /// Caution: It's a bit quiry, because of floating point arithmetics.
    fn coord_to_grid(&self, p: &Particle) -> GridCoordinate {

        let gx = (*p.position.x.as_ref() / self.grid_width.x).floor() as usize;
        let gy = (*p.position.y.as_ref() / self.grid_width.y).floor() as usize;
        let ga = (*p.orientation.as_ref() / self.grid_width.a).floor() as usize;

        [gx, gy, ga]
    }

    /// Builts up a histogram in space and orientation for a give particle
    /// configuration `particles`.
    pub fn sample_from(&mut self, particles: &Vec<Particle>) {
        // zero out distribution
        self.dist = ArrayBase::zeros(self.shape());

        for p in particles {
            let c = self.coord_to_grid(p);
            self.dist[c] += 1.;
        }
    }

    /// Returns a normalised distribution array
    pub fn normalized(&self) -> Bins {
        // number of particles
        let n = self.dist.fold(0., |sum, x| sum + x);
        self.dist.clone() / n
    }

    /// Returns spatial gradient as an array.
    /// The first axis specifies the direction of the derivative, the component
    /// of the gradient, whereas the other axises are the same as for the
    /// distribution.
    ///
    /// The derivative is implemented as a symmetric finite differential
    /// quotient with wrap around coordinates.
    pub fn spatgrad(&self) -> Array<f64, (Ix, Ix, Ix, Ix)> {
        let (sx, sy, sa) = self.shape();
        let mut res = ArrayBase::zeros((2, sx, sy, sa));

        let ref h = self.grid_width;
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

    fn index<'a>(&'a self, index: [i32; 3]) -> &'a f64 {

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
    use coordinates::modulofloat::Mf64;
    use coordinates::particle::Particle;
    use coordinates::vector::Mod64Vector2;
    use ndarray::{Array, Axis, arr3};
    use super::*;

    #[test]
    fn new() {
        let dist = Distribution::new((10, 10, 6), (1., 1.));
        assert_eq!(dist.shape(), (10, 10, 6));
    }

    #[test]
    fn sample_from() {
        let boxsize = (1., 1.);
        let p = Particle::randomly_placed_particles(1000, boxsize);
        let mut d = Distribution::new((5, 5, 2), boxsize);

        d.sample_from(&p);

        let sum = d.dist.fold(0., |s, x| s + x);
        assert_eq!(sum, 1000.);

        let p2 = vec![Particle::new(0.6, 0.3, 0., boxsize)];

        d.sample_from(&p2);
        println!("{}", d.dist);

        assert_eq!(d.dist[[2, 1, 0]], 1.);
    }

    #[test]
    fn normalized() {
        let boxsize = (1., 1.);
        let mut d = Distribution::new((5, 5, 2), boxsize);
        let p2 = vec!{
            // Caution, 0.6 / 0.2 = 0.29999 in floating point arithmetic
            Particle::new(0.55, 0.3, 0., boxsize),
            Particle::new(0.61, 0.3, 2., boxsize),
        };
        assert_eq!(p2.len(), 2);

        d.sample_from(&p2);

        println!("{}", d.normalized());
        assert_eq!(d.normalized()[[2, 1, 0]], 0.5);
        assert_eq!(d.normalized()[[3, 1, 0]], 0.5);
    }

    #[test]
    fn coord_to_grid() {
        let boxsize = (1., 1.);

        let input =
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 7.], [0., 0., -1.]];

        let result = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 5]];

        for (i, o) in input.iter().zip(result.iter()) {
            let p = Particle {
                position: Mod64Vector2::new(i[0], i[1], boxsize),
                orientation: Mf64::new(i[2], 2. * ::std::f64::consts::PI),
            };

            let dist = Distribution::new((10, 10, 6), boxsize);

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
        let boxsize = (1., 1.);
        let shape = (5, 5, 1);
        let mut d = Distribution::new(shape, boxsize);

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

        assert!(grad.subview(Axis(0), 0) == res_x);
        assert!(grad.subview(Axis(0), 1) == res_y);

        d.dist = Array::zeros(shape);
        assert!(d.spatgrad() == Array::<f64, _>::zeros((2, shape.0, shape.1, shape.2)));

        d.dist = Array::zeros(shape) + 1.;
        assert!(d.spatgrad() == Array::<f64, _>::zeros((2, shape.0, shape.1, shape.2)));
    }

    #[test]
    fn index() {
        let boxsize = (1., 1.);
        let shape = (2, 3, 2);
        let mut d = Distribution::new(shape, boxsize);

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
