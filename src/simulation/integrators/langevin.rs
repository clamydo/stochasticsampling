//! Implements an hybrid integration scheme for a Fokker-PlanckZ//!
//! (Smoulochkowski) equation coupled to a continous Stokes flow field. The
//! corresponding stochastic Langevin equation is used to evolve test particle
//! positions and orientations. Considered as a probabilistic sample of the
//! probability distribution function (PDF), which is described by the
//! Fokker-Planck equation, the particle configuration is used to sample the
//! PDF. To close the integration scheme, the flow-field is calculated in terms
//! of probabilstic moments (i.e. expectation values) of the PDF on a grid.
//!
//! The integrator is implemented in dimensionless units, scaling out the
//! self-propulsion speed of the particle and the average volue taken by a
//! particle (i.e. the particle number density). In this units a particle needs
//! one unit of time to cross a volume per particle, meaning a unit of length,
//! due to self propulsion. Since the model describes an ensemble of point
//! particles, the flow-field at the position of such a particle is undefined
//! (die to the divergence of the Oseen-tensor). As  a consequence it is
//! necessary to define a minimal radius around a point particle, on which the
//! flow-field is calculated. A natural choice in the above mentioned units, is
//! to choose the radius of the volume per particle. This means, that a grid
//! cell on which the flow-field is calculated, should be a unit cell! The flow
//! field is now calculated using contributions of every other cell but the
//! cell itself.

// Move unit test into own file
#[cfg(test)]
#[path = "./langevin_test.rs"]
mod langevin_test;

use ndarray::{Array, ArrayView, Ix2, Ix4, Ix5};
use ndarray_parallel::prelude::*;
use num_complex::Complex;
use quaternion;
use rayon::prelude::*;
use simulation::magnetic_interaction::mean_force;
use simulation::mesh::grid_width::GridWidth;
use simulation::particle::{CosSinOrientation, OrientationVector, Particle, Position};
use simulation::settings::{BoxSize, GridSize};
use simulation::vector::vorticity::vorticity3d_dispatch;
use simulation::vector::{Vector, VectorD};

#[derive(Clone, Copy)]
pub struct RandomVector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub axis_angle: f64,
    pub rotate_angle: f64,
}

impl RandomVector {
    fn into_pos_vec(&self) -> Vector<Position> {
        [self.x, self.y, self.z].into()
    }
}

/// Holds parameter needed for time step
#[derive(Debug, Clone, Copy)]
pub struct IntegrationParameter {
    pub rot_diffusion: f64,
    pub timestep: f64,
    pub trans_diffusion: f64,
    pub magnetic_reorientation: f64,
    pub drag: f64,
    pub magnetic_dipole_dipole: f64,
}

/// Holds precomuted values
pub struct Integrator {
    box_size: BoxSize,
    grid_width: GridWidth,
    parameter: IntegrationParameter,
}

impl Integrator {
    /// Returns a new instance of the mc_sampling integrator.
    pub fn new(
        grid_size: GridSize,
        box_size: BoxSize,
        parameter: IntegrationParameter,
    ) -> Integrator {
        let grid_width = GridWidth::new(grid_size, box_size);

        Integrator {
            box_size: box_size,
            grid_width: grid_width,
            parameter: parameter,
        }
    }

    /// Updates a test particle configuration according to the given parameters.
    ///
    /// Y(t) = sqrt(t) * X(t), if X is normally distributed with variance 1,
    /// then Y is normally distributed with variance t.
    /// A diffusion coefficient `d` translates to a normal distribuion with
    /// variance `s^2` as `d = s^2 / 2`.
    /// Together this leads to an update of the position due to the diffusion of
    /// `x_d(t + dt) = sqrt(2 d dt) N(0, 1)``.
    ///
    /// Assumes the magnetic field to be oriented along Y-axis.
    ///
    /// *IMPORTANT*: This function expects `sqrt(2 d dt)` as a precomputed
    /// effective diffusion constant.
    fn evolve_particle_inplace(
        &self,
        p: &mut Particle,
        rv: &RandomVector,
        flow_field: &ArrayView<f64, Ix4>,
        vort: &ArrayView<f64, Ix4>,
        magnetic_field: &ArrayView<Complex<f64>, Ix4>,
        grad_magnetic_field: &ArrayView<Complex<f64>, Ix5>,
    ) {
        let param = &self.parameter;

        // retreive flow in cell which contains the particle
        let idx = get_cell_index(&p, &self.grid_width);
        let flow = field_at_cell(flow_field, idx);
        let b = field_at_cell_c(magnetic_field, idx);
        let gradb = vector_gradient_at_cell(grad_magnetic_field, idx);

        // precompute trigonometric functions
        let cs = CosSinOrientation::from_orientation(&p.orientation);

        // orientation vector of `n`, switch to cartesian coordinates to ease some
        // computations
        let vector = cs.to_orientation_vecor();

        // Get force in magnetic field
        let fb = mean_force(gradb.view(), &vector);

        // Evolve particle position.
        // convection + self-propulsion + diffusion

        let mut new_position: Vector<Position> =
            (flow + &vector + fb * param.drag).to() * param.timestep;
        new_position += rv.into_pos_vec() * param.trans_diffusion;

        p.position.from_vector_mut(&new_position);

        // rotational diffusion
        let mut new_vector = rotational_diffusion_quat_mut(&vector, &cs, rv);

        // rotational coupling to the flow field
        let jef = jeffrey(&vector, idx, vort, param.timestep);
        new_vector += jef;

        let mag = magnetic_dipole_rotation(&vector, b);

        new_vector += mag * param.timestep * param.magnetic_dipole_dipole;

        // update particles orientation
        p.orientation.from_vector_mut(&new_vector);

        // influence of magnetic field pointing in z-direction
        p.orientation.theta -= param.magnetic_reorientation * cs.sin_theta * param.timestep;

        // IMPORTANT: apply periodic boundary condition
        p.pbc(self.box_size);
    }

    pub fn evolve_particles_inplace<'a>(
        &self,
        particles: &mut Vec<Particle>,
        random_samples: &[RandomVector],
        flow_field: ArrayView<'a, f64, Ix4>,
        magnetic_field: ArrayView<'a, Complex<f64>, Ix4>,
        grad_magnetic_field: ArrayView<'a, Complex<f64>, Ix5>,
    ) {
        // Calculate vorticity dx uy - dy ux
        let vort = vorticity3d_dispatch(self.grid_width, flow_field);

        particles
            .par_iter_mut()
            .zip(random_samples.par_iter())
            .for_each(|(ref mut p, r)| {
                self.evolve_particle_inplace(
                    p,
                    r,
                    &flow_field,
                    &vort.view(),
                    &magnetic_field,
                    &grad_magnetic_field,
                )
            });
    }
}

fn get_cell_index(p: &Particle, grid_width: &GridWidth) -> (usize, usize, usize) {
    let ix = (p.position.x / grid_width.x).floor() as usize;
    let iy = (p.position.y / grid_width.y).floor() as usize;
    let iz = (p.position.z / grid_width.z).floor() as usize;

    // debug_assert!(
    //     0. <= p.position.x && p.position.x < self.box_size.x,
    //     "x: {}",
    //     p.position.x
    // );
    // debug_assert!(
    //     0. <= p.position.y && p.position.y < self.box_size.y,
    //     "y: {}",
    //     p.position.y
    // );
    // debug_assert!(
    //     0. <= p.position.z && p.position.z < self.box_size.z,
    //     "z: {}",
    //     p.position.z
    // );
    // debug_assert!(0 <= ix && ix < self.grid_size.x as isize, "ix: {}", ix);
    // debug_assert!(0 <= iy && iy < self.grid_size.y as isize, "iy: {}", iy);
    // debug_assert!(0 <= iz && iz < self.grid_size.z as isize, "iz: {}", iz);

    (ix, iy, iz)
}

fn field_at_cell(field: &ArrayView<f64, Ix4>, idx: (usize, usize, usize)) -> VectorD {
    let f = unsafe {
        [
            *field.uget((0, idx.0, idx.1, idx.2)),
            *field.uget((1, idx.0, idx.1, idx.2)),
            *field.uget((2, idx.0, idx.1, idx.2)),
        ]
    };
    f.into()
}

fn field_at_cell_c(field: &ArrayView<Complex<f64>, Ix4>, idx: (usize, usize, usize)) -> VectorD {
    let f = unsafe {
        [
            (*field.uget((0, idx.0, idx.1, idx.2))).re,
            (*field.uget((1, idx.0, idx.1, idx.2))).re,
            (*field.uget((2, idx.0, idx.1, idx.2))).re,
        ]
    };
    f.into()
}

fn vector_gradient_at_cell(
    field: &ArrayView<Complex<f64>, Ix5>,
    idx: (usize, usize, usize),
) -> Array<f64, Ix2> {
    field.slice(s![.., .., idx.0, idx.1, idx.2]).map(|v| v.re)
}

fn rotational_diffusion_quat_mut(
    vector: &OrientationVector,
    cs: &CosSinOrientation,
    r: &RandomVector,
) -> OrientationVector {
    let rotational_axis = |alpha: f64| {
        let cos_ax = alpha.cos();
        let sin_ax = alpha.sin();
        // axis perpendicular to orientation vector
        [
            cs.cos_phi * cs.cos_theta * sin_ax - cos_ax * cs.sin_phi,
            cos_ax * cs.cos_phi + cs.cos_theta * sin_ax * cs.sin_phi,
            -sin_ax * cs.sin_theta,
        ]
    };

    let ax = rotational_axis(r.axis_angle);

    // quaternion encoding a rotation around `rotational_axis` with
    // angle drawn from Rayleigh-distribution
    let q = quaternion::axis_angle(ax, r.rotate_angle);

    // return rot
    quaternion::rotate_vector(q, vector.v).into()
}

fn jeffrey(
    vector: &OrientationVector,
    idx: (usize, usize, usize),
    vort: &ArrayView<f64, Ix4>,
    timestep: f64,
) -> VectorD {
    let half_timestep = 0.5 * timestep;
    // Get vorticity
    let vort_x = unsafe { vort.uget((0, idx.0, idx.1, idx.2)) };
    let vort_y = unsafe { vort.uget((1, idx.0, idx.1, idx.2)) };
    let vort_z = unsafe { vort.uget((2, idx.0, idx.1, idx.2)) };

    // (1-nn) . (-W[u] . n) == 0.5 * Curl[u] x n
    [
        half_timestep * (vort_y * vector[2] - vort_z * vector[1]),
        half_timestep * (vort_z * vector[0] - vort_x * vector[2]),
        half_timestep * (vort_x * vector[1] - vort_y * vector[0]),
    ].into()
}

fn magnetic_dipole_rotation(
    vector: &OrientationVector,
    mut b: VectorD,
) -> VectorD {
    b -= (*vector) * vector.dot(&b);
    b
}
