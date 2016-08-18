#![crate_type = "staticlib"]
#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(quickcheck_macros))]

//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#[cfg(test)]
extern crate quickcheck;

#[macro_use]
extern crate log;

extern crate mpi;
#[macro_use(s)]
extern crate ndarray;
extern crate rand;
extern crate rustc_serialize;

pub mod coordinates;
pub mod settings;
pub mod simulation;
