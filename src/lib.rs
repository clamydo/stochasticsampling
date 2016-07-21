#![crate_type = "staticlib"]
#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(quickcheck_macros))]

#[cfg(test)]
extern crate quickcheck;

extern crate rand;

pub mod coordinates;
