/// A representation for the probability distribution function.

use coordinates::particle::Particle;
use ndarray::{Array, Ix, ArrayBase, Axis};
use settings::{BoxSize, GridSize};

#[derive(Debug)]
struct GridWidth {
    x: f64,
    y: f64,
    a: f64,
}

pub type Bins = Array<f64, (Ix, Ix, Ix)>;

/// _Normalised_ discrete distribution. *dist* contains the probability for a
/// particle in the box at position of first two axis and the direction of the
/// last axis.
#[derive(Debug)]
pub struct Distribution {
    pub dist: Bins,
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

    pub fn shape(&self) -> (Ix, Ix, Ix) {
        self.dist.dim()
    }

    /// Maps particle coordinate onto grid coordinate/index (starting from zero).
    fn coord_to_grid(&self, p: &Particle) -> GridCoordinate {

        let gx = (p.position.x.as_ref() / self.grid_width.x).floor() as usize;
        let gy = (p.position.y.as_ref() / self.grid_width.y).floor() as usize;
        let ga = (p.orientation.as_ref() / self.grid_width.a).floor() as usize;

        (gx, gy, ga)
    }

    /// Naive implementation of a binning and counting algorithm.
    pub fn sample_from(&mut self, particles: &Vec<Particle>) {
        // zero out distribution
        self.dist = ArrayBase::zeros(self.shape());

        for p in particles {
            let c = self.coord_to_grid(p);
            self.dist[[c.0, c.1, c.2]] += 1.;
        }
    }

    pub fn normalized(&self) -> Bins {
        let n = self.dist.sum(Axis(0)).sum(Axis(0)).sum(Axis(0));

        self.dist.clone() / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coordinates::particle::Particle;
    use coordinates::modulofloat::Mf64;
    use coordinates::vector::Mod64Vector2;
    use ndarray::{aview0, Axis};

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

        let sum = d.dist.sum(Axis(0)).sum(Axis(0)).sum(Axis(0));
        assert_eq!(sum, aview0(&1000.));

        let p2 = vec!{Particle::new(0.6, 0.3, 0., boxsize)};

        d.sample_from(&p2);
        println!("{}", d.dist);

        assert_eq!(d.dist[[2, 1, 0]], 1.);
    }

    #[test]
    fn normalized() {
        let boxsize = (1., 1.);
        let mut d = Distribution::new((5, 5, 2), boxsize);
        let p2 = vec!{
            Particle::new(0.6, 0.3, 0., boxsize),
            Particle::new(0.6, 0.3, 0., boxsize),
        };
        assert_eq!(p2.len(), 2);

        d.sample_from(&p2);

        println!("{}", d.normalized());
        assert_eq!(d.normalized()[[2, 1, 0]], 1.);
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

            let dist = Distribution::new((10, 10, 6), boxsize);

            let g = dist.coord_to_grid(&p);

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
