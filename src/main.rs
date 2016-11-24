#![crate_type = "bin"]
#![recursion_limit = "1024"]

extern crate stochasticsampling;
#[macro_use]
extern crate log;
extern crate env_logger;
extern crate serde_cbor;
extern crate bincode;
extern crate time;
#[macro_use]
extern crate quick_error;

use bincode::serde::serialize_into;
use serde_cbor::ser;
use std::env;
use std::fs::File;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use stochasticsampling::settings;
use stochasticsampling::settings::OutputFormat;
use stochasticsampling::simulation::Simulation;
use stochasticsampling::simulation::Snapshot;

const VERSION: &'static str = env!("CARGO_PKG_VERSION");
// TODO: Maybe replace this arbitrary hardcoded number with somehing different
const COLLECT_TIMESTEPS: usize = 100;
const IOWORKER_BUFFER_SIZE: usize = 100;

// Implement Error type for IOWorker thread
quick_error! {
    #[derive(Debug)]
    pub enum IOWorkerError {
        BincodeError(err: bincode::serde::SerializeError) {
            from()
        }
        CBORError(err: serde_cbor::Error) {
            from()
        }
    }
}

enum IOWorkerMsg {
    Quit,
    Data(Vec<Snapshot>),
}

// TODO: Add Result return type
fn run(settings_file_name: &str) {
    let settings = match settings::read_parameter_file(settings_file_name) {
        Ok(s) => s,
        Err(e) => {
            error!("Error reading parameter file: {}", e);
            std::process::exit(1)
        }
    };

    // Create and initialize output file
    let filename = format!("{prefix}-{time}_v{version}.cbor",
                           prefix = settings.environment.prefix,
                           time = &time::now().strftime("%Y-%m-%d_%H%M").unwrap().to_string(),
                           version = VERSION);

    let filepath = Path::new(&settings.environment.output_dir).join(filename);

    let mut file = match File::create(&filepath) {
        Err(why) => {
            panic!("couldn't create output file '{}': {}",
                   filepath.display(),
                   why)
        }
        Ok(file) => file,
    };

    // Setup simulation
    let mut simulation = Simulation::new(settings.clone());
    simulation.init();

    // Serialize parameter as first object in file

    match settings.environment.output_format {
        OutputFormat::CBOR => ser::to_writer_sd(&mut file, &settings).unwrap(),
        OutputFormat::Bincode => {
            serialize_into(&mut file, &settings, bincode::SizeLimit::Infinite).unwrap()
        }
    }

    let n = settings.simulation.number_of_timesteps / COLLECT_TIMESTEPS;

    // Create commuication channel for thread
    let (tx, rx) = mpsc::sync_channel::<IOWorkerMsg>(IOWORKER_BUFFER_SIZE);

    // Copy output_format, so it can be captured by the thread closure.
    let output_format = settings.environment.output_format;

    // Spawn worker thread, that periodically flushes collections of simultaions states to disk.
    let io_worker = thread::spawn(move || -> Result<(), IOWorkerError> {
        loop {
            match rx.recv().unwrap() {
                IOWorkerMsg::Quit => break,
                IOWorkerMsg::Data(v) => {
                    // write all snapshots into one cbor file
                    match output_format {
                        OutputFormat::CBOR => {
                            ser::to_writer_sd(&mut file, &v)?
                        }
                        OutputFormat::Bincode => {
                            serialize_into(&mut file, &v, bincode::SizeLimit::Infinite)?
                        }
                    }
                }
            }
        }

        Ok(())
    });

    // Run the simulation. Split timesteps into bundles of COLLECT_TIMESTEP and
    // flush them
    // periodically to disk.
    for _ in 0..n {
        let data: Vec<Snapshot> = (&mut simulation).take(COLLECT_TIMESTEPS).collect();
        tx.send(IOWorkerMsg::Data(data)).unwrap();
    }

    let remaining_data: Vec<Snapshot> =
        simulation.take(settings.simulation.number_of_timesteps - n * COLLECT_TIMESTEPS)
            .collect();
    tx.send(IOWorkerMsg::Data(remaining_data)).unwrap();

    // Stop worker
    tx.send(IOWorkerMsg::Quit).unwrap();

    // Wait for worker to quit
    match io_worker.join() {
        Ok(_) => println!("Simulation finished successful."),
        Err(e) => error!("Error during flushing to disk: {:?}", e),
    }
}


fn main() {
    // initialize the env_logger implementation
    env_logger::init().unwrap();

    // parse command line arguments
    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            error!("Please pass a parameter file.");
            std::process::exit(1);
        }
        2 => {
            run(&args[1]);
        }
        _ => {
            error!("You've passed too many arguments. Please don't do that.");
            std::process::exit(1);
        }
    }

    std::process::exit(0);
}
