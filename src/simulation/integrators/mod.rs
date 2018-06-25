pub mod langevin_builder;
pub mod langevin_old;

pub use self::langevin_builder::LangevinBuilder;
pub use self::langevin_old as langevin;


pub fn evolve_particles<'a>(
    &self,
    particles: &mut Vec<Particle>,
    random_samples: &[RandomVector],
    flow_field: ArrayView<'a, f64, Ix4>,
    magnetic_field: Option<(ArrayView<'a, Complex<f64>, Ix4>, ArrayView<'a, Complex<f64>, Ix5>)>
) {
    // TODO move into caller
    // Calculate vorticity
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
                magnetic_field,
            )
        });
}
