#![crate_type = "bin"]

extern crate stochasticsampling;
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate serde_cbor;
extern crate time;

use serde_cbor::ser;
use std::env;
use std::fs::File;
use std::path::Path;
use stochasticsampling::settings;
use stochasticsampling::simulation::Simulation;
use stochasticsampling::simulation::Snapshot;

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

            let mut simulation = Simulation::new(settings.clone());
            simulation.init();

            // run the actual simulation
            let data: Vec<Snapshot> = simulation.take(settings.simulation.number_of_timesteps)
                .collect();


            // TODO Add option to add prefix to output path
            // TODO Save version of simulation
            let filename = settings.environment.prefix +
                           &time::now().strftime("%Y-%m-%d_%H%M").unwrap().to_string() +
                           ".cbor";
            let filepath = Path::new(&settings.environment.output_dir).join(filename);

            let mut file = match File::create(&filepath) {
                Err(why) => panic!("couldn't create {}: {}", filepath.display(), why),
                Ok(file) => file,
            };

            // write all snapshots into one cbor file
            match ser::to_writer(&mut file, &data) {
                Err(e) => panic!("The world has stopped turnung, because: {}", e),
                _ => {}
            }

            // match simulation.run() {
            //     Ok(_) => {}
            //     Err(e) => {
            //         error!("Error during simulation: {}", e);
            //         std::process::exit(1)
            //     }
            // }
        }
        _ => {
            error!("You've passed too many arguments. Please don't do that.");
            std::process::exit(1)
        }
    }

    std::process::exit(0);
}
