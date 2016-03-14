#![crate_type = "bin"]

extern crate stochasticsampling;
extern crate rand;
extern crate toml;
extern crate rustc_serialize;
extern crate mpi;


mod settings;
mod simulation;

use std::env;

#[cfg_attr(test, allow(dead_code))]
fn main() {
    // parse command line arguments
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            println!("Please pass a parameter file.");
            std::process::exit(1)
        },
        2 => {
            let settings = match settings::read_parameter_file(&args[1]) {
                Ok(s) => s,
                Err(e) => {
                    println!("Error reading parameter file: {}", e);
                    std::process::exit(1)
                }
            };

            match simulation::simulate(&settings) {
                Ok(_) => {},
                Err(e) => {
                    println!("Error during simulation: {}", e);
                    std::process::exit(1)
                },
            }
        },
        _ => {
            println!("You've passed too many arguments. Please don't do that.");
            std::process::exit(1)
        }
    }

    std::process::exit(0);
}
