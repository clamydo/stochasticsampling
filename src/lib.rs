//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#![crate_type = "staticlib"]
#![feature(slice_patterns)]
#![feature(euclidean_division)]
#![recursion_limit = "1024"]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate bincode;
#[cfg(test)]
extern crate ndarray_rand;
#[cfg(test)]
extern crate quickcheck;

#[macro_use]
extern crate derive_more;
#[cfg(test)]
extern crate test;
#[macro_use]
extern crate error_chain;
extern crate extprim;
extern crate fftw3;
#[macro_use]
extern crate itertools;
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate num_complex;
extern crate pcg_rand;
extern crate quaternion;
extern crate rand;
extern crate rayon;
// extern crate rustfft;
extern crate serde;
#[macro_use]
extern crate serde_derive;

pub mod consts;
mod test_helper;
pub mod distribution;
pub mod flowfield;
pub mod integrators;
pub mod magnetic_interaction;
pub mod mesh;
pub mod output;
pub mod particle;
pub mod polarization;
pub mod vector;


mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
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
