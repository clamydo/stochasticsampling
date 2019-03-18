//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#![crate_type = "staticlib"]
// #![feature(euclidean_division)]
#![recursion_limit = "1024"]

#[cfg(test)]
extern crate bincode;
#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate quickcheck;

#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate error_chain;
extern crate fftw3;
#[macro_use]
extern crate itertools;
extern crate lerp;
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate num_complex;
extern crate num_traits;
extern crate quaternion;
extern crate rand;
extern crate rand_pcg;
extern crate rayon;
// extern crate rustfft;
extern crate serde;
#[macro_use]
extern crate serde_derive;

pub mod consts;
pub mod distribution;
pub mod flowfield;
pub mod integrators;
pub mod magnetic_interaction;
pub mod mesh;
pub mod output;
pub mod particle;
pub mod polarization;
mod test_helper;
pub mod vector;

#[cfg(feature = "single")]
type Float = f32;

#[cfg(not(feature = "single"))]
type Float = f64;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain! {}
}

/// Size of the simulation box an arbitary physical dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BoxSize {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
/// Size of the discrete grid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GridSize {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub phi: usize,
    pub theta: usize,
}
