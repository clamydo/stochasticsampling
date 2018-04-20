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

// #[macro_use]
// extern crate derive_more;
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
extern crate toml;

pub mod consts;
pub mod simulation;
mod test_helper;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
}
