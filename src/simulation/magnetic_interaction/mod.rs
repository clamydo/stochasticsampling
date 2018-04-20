pub mod magnetic_solver;

use ndarray::{Array, ArrayView, Ix2};
use simulation::particle::OrientationVector;
use simulation::vector::Vector;

pub struct Force();

/// Returns force on unit magnetic moment with orientation `o` in a given field
/// gradient `grad_b`.
///
pub fn mean_force(grad_b: ArrayView<f64, Ix2>, o: &OrientationVector) -> Vector<Force> {
    let o = Array::from_vec(o.v.to_vec());
    let f = grad_b.dot(&o);
    [
        f[0],
        f[1],
        f[2],
    ].into()
}
