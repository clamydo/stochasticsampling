//! This crate implements a Monte-Carlo method for solving a (specific)
//! Fokker-Planck type equation. Most of the implementation is rather general.
//! The parts specific to the problem can be found mainly in  the
//! `simulation::integrator` module. TODO Missing docs.

#![crate_type = "staticlib"]
// #![feature(euclidean_division)]
#![recursion_limit = "1024"]

#[macro_use]
extern crate error_chain;
use serde_derive::{Deserialize, Serialize};

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
pub type Float = f32;

#[cfg(not(feature = "single"))]
pub type Float = f64;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain! {}
}

/// Size of the simulation box an arbitary physical dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BoxSize {
    pub x: Float,
    pub y: Float,
    pub z: Float,
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
