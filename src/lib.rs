//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#![crate_type = "staticlib"]
#![feature(proc_macro)]
#![feature(slice_patterns)]
#![recursion_limit = "1024"]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;
#[cfg(test)]
extern crate ieee754;

#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate fftw3;
extern crate ndarray;
extern crate pcg_rand;
extern crate rand;
extern crate serde;
extern crate serde_cbor;
#[macro_use]
extern crate serde_derive;
extern crate toml;


pub mod coordinates;
#[macro_use]
pub mod serialization_helper;
pub mod settings;
pub mod simulation;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain! { }
}
