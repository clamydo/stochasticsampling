//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#![crate_type = "staticlib"]
#![feature(slice_patterns)]
#![recursion_limit = "1024"]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
extern crate ndarray_rand;

#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate fftw3;
#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate num;
extern crate pcg_rand;
extern crate rand;
extern crate rayon;
// extern crate rustfft;
extern crate serde;
extern crate serde_cbor;
#[macro_use]
extern crate serde_derive;
extern crate toml;


pub mod consts;
#[macro_use]
pub mod serialization_helper;
pub mod simulation;
mod test_helper;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
}
