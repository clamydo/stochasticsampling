//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#![crate_type = "staticlib"]
#![feature(proc_macro)]

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[macro_use]
extern crate log;

extern crate mpi;
#[macro_use(s)]
extern crate ndarray;
extern crate rand;
extern crate pcg_rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_cbor;
extern crate toml;
extern crate fftw3;
#[macro_use]
extern crate quick_error;

pub mod coordinates;
#[macro_use]
pub mod serialization_helper;
pub mod settings;
pub mod simulation;
