//! A representation for the probability distribution function.

// Move unit test into own file
#[cfg(test)]
#[path = "./distribution_test.rs"]
mod distribution_test;

use ndarray::{Array, Ix, Ix5};
use simulation::mesh::grid_width::GridWidth;
use simulation::particle::Particle;
use simulation::settings::{BoxSize, GridSize};
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
    box_size: BoxSize,
    grid_size: GridSize,
}

type GridCoordinate = [Ix; 5];

impl Distribution {
    /// Returns a zero initialised instance of Distribution.
    pub fn new(grid_size: GridSize, box_size: BoxSize) -> Distribution {
        let grid = [
            grid_size.x,
            grid_size.y,
            grid_size.z,
            grid_size.phi,
            grid_size.theta,
        ];
        let grid_width = GridWidth::new(grid_size, box_size);

        Distribution {
            dist: Array::default(grid),
            grid_width: grid_width,
            box_size: box_size,
            grid_size: grid_size,
        }
    }

    /// Returns the width of on cell for every axis
    pub fn get_grid_width(&self) -> GridWidth {
        self.grid_width
    }

    /// Returns the box dimensions
    pub fn get_box_size(&self) -> BoxSize {
        self.box_size
    }

    /// Returns grid size
    pub fn get_grid_size(&self) -> GridSize {
        self.grid_size
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
            p.position.x >= 0. && p.position.y >= 0. && p.position.z >= 0.
                && p.orientation.phi >= 0. && p.orientation.theta >= 0.,
            "Got negative position or orientation {:?}",
            p
        );

        debug_assert!(
            p.position.x < self.box_size.x && p.position.y < self.box_size.y
                && p.position.z < self.box_size.z,
            "Position out of range {:?}",
            p
        );

        debug_assert!(
            p.orientation.phi <= 2. * ::std::f64::consts::PI,
            "Theta is not in range> {:?}",
            p
        );
        debug_assert!(
            p.orientation.theta <= ::std::f64::consts::PI,
            "Theta is not in range> {:?}",
            p
        );

        let mut gx = (p.position.x / self.grid_width.x).floor() as Ix;
        let mut gy = (p.position.y / self.grid_width.y).floor() as Ix;
        let mut gz = (p.position.z / self.grid_width.z).floor() as Ix;
        let mut gphi = (p.orientation.phi / self.grid_width.phi).floor() as Ix;
        let mut gtheta = (p.orientation.theta / self.grid_width.theta).floor() as Ix;

        // In some case positions at the right border are possible due to floating
        // point roundoff-errors in the modulo calculation.
        // It happens for very small negative values after the modulo operation.
        if gx == self.grid_size.x {
            gx -= 1
        };
        if gy == self.grid_size.y {
            gy -= 1
        };
        if gz == self.grid_size.z {
            gz -= 1
        };
        if gphi == self.grid_size.phi {
            gphi -= 1
        };

        // treat theta = PI as a null set and include it in the last cell
        if gtheta == self.grid_size.theta {
            gtheta -= 1
        };

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
