/// A representation for the probability distribution function.

use coordinates::Particle;
use ndarray::{Array, Ix};
use settings::{BoxSize, GridSize};

#[derive(Debug)]
struct GridWidth {
    x: f64,
    y: f64,
    a: f64,
}

/// _Normalised_ discrete distribution. *dist* contains the probability for a
/// particle in the box at position of first two axis and the direction of the
/// last axis.
#[derive(Debug)]
pub struct Distribution {
    pub dist: Array<f64, (Ix, Ix, Ix)>,
    grid_width: GridWidth,
}

type GridCoordinate = (usize, usize, usize);

const TWOPI: f64 = 2. * ::std::f64::consts::PI;

impl Distribution {
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

    pub fn grid_size(&self) -> (Ix, Ix, Ix) {
        self.dist.dim()
    }
    // map particle coordinate onto grid coordinate
    fn coord_to_grid(&self, p: Particle) -> GridCoordinate {

        let gx = (p.position.x.as_ref() / self.grid_width.x).floor() as usize;
        let gy = (p.position.y.as_ref() / self.grid_width.y).floor() as usize;
        let ga = (p.orientation.as_ref() / self.grid_width.a).floor() as usize;

        (gx, gy, ga)
    }

    /// Naive implementation of a binning and counting algorithm.
    pub fn sample(&mut self, particles: Vec<Particle>) {

        // split into subtasks



    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coordinates::{Particle, randomly_placed_particles};
    use coordinates::modulofloat::Mf64;
    use coordinates::vector::Mod64Vector2;

    #[test]
    fn sample() {
        let boxsize = (1., 1.);
        let p = randomly_placed_particles(100, boxsize);
        let mut dist = Distribution::new((10, 10, 6), boxsize);

        dist.sample(p);

        for f in dist.dist.iter() {
            println!("{}", f);
        }
    }

    #[test]
    fn coord_to_grid() {
        let boxsize = (1., 1.);

        let input =
            [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0., 0., 7.), (0., 0., -1.)];

        let result = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 5)];

        for (i, o) in input.iter().zip(result.iter()) {
            let p = Particle {
                position: Mod64Vector2::new(i.0, i.1, boxsize),
                orientation: Mf64::new(i.2, 2. * ::std::f64::consts::PI),
            };

            let mut dist = Distribution::new((10, 10, 6), boxsize);

            let g = dist.coord_to_grid(p);

            assert!(g.0 == o.0,
                    "For input {:?}. Expected '{}', got '{}'.",
                    i,
                    o.0,
                    g.0);
            assert!(g.1 == o.1,
                    "For input {:?}. Expected '{}', got '{}'.",
                    i,
                    o.1,
                    g.1);
            assert!(g.2 == o.2,
                    "For input {:?}. Expected '{}', got '{}'.",
                    i,
                    o.2,
                    g.2);
        }
    }
}
