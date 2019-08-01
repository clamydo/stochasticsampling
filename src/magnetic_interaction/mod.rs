pub mod magnetic_solver;

use crate::particle::OrientationVector;
use crate::vector::Vector;
use crate::Float;
use ndarray::{Array, ArrayView, Ix2};

pub struct Force();

/// Returns force on unit magnetic moment with orientation `o` in a given field
/// gradient `grad_b`.
///
pub fn mean_force(grad_b: ArrayView<Float, Ix2>, o: &OrientationVector) -> Vector<Force> {
    let o = Array::from_vec(o.v.to_vec());
    let f = grad_b.dot(&o);
    unsafe { [*f.uget(0), *f.uget(1), *f.uget(2)] }.into()
}
