#![crate_type = "staticlib"]
#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(quickcheck_macros))]

#[cfg(test)]
extern crate quickcheck;

#[macro_use]
extern crate log;

extern crate mpi;
extern crate ndarray;
extern crate rand;
extern crate rustc_serialize;

pub mod coordinates;
pub mod settings;
pub mod simulation;
