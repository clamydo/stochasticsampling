#![crate_type = "bin"]

extern crate stochasticsampling;
#[macro_use]
extern crate log;
extern crate env_logger;

use std::env;
use stochasticsampling::settings;
use stochasticsampling::simulation::Simulation;


fn main() {

    // initialize the env_logger implementation
    env_logger::init().unwrap();


    // parse command line arguments
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            error!("Please pass a parameter file.");
            std::process::exit(1)
        }
        2 => {
            let settings = match settings::read_parameter_file(&args[1]) {
                Ok(s) => s,
                Err(e) => {
                    error!("Error reading parameter file: {}", e);
                    std::process::exit(1)
                }
            };

            let mut simulation = Simulation::new(settings);

            simulation.init();

            match simulation.run() {
                Ok(_) => {}
                Err(e) => {
                    error!("Error during simulation: {}", e);
                    std::process::exit(1)
                }
            }
        }
        _ => {
            error!("You've passed too many arguments. Please don't do that.");
            std::process::exit(1)
        }
    }

    std::process::exit(0);
}
