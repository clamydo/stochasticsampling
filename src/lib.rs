#![crate_type = "staticlib"]
#![feature(slice_patterns)]

//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

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
extern crate rustc_serialize;
extern crate toml;
extern crate fftw3;

pub mod coordinates;
pub mod settings;
pub mod simulation;
