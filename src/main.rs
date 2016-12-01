#![crate_type = "bin"]
#![recursion_limit = "1024"]

extern crate stochasticsampling;
extern crate bincode;
#[macro_use]
extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate serde_cbor;
extern crate time;
#[macro_use]
extern crate quick_error;

use bincode::serde::serialize_into;
use clap::App;
use serde_cbor::{de, ser};
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use stochasticsampling::settings;
use stochasticsampling::settings::OutputFormat;
use stochasticsampling::simulation::Simulation;
use stochasticsampling::simulation::Snapshot;

const VERSION: &'static str = env!("CARGO_PKG_VERSION");
// TODO: Maybe replace this arbitrary hardcoded number with somehing different

// Implement Error type for IOWorker thread
quick_error! {
    #[derive(Debug)]
    pub enum IOWorkerError {
        BincodeError(err: bincode::serde::SerializeError) {
            cause(err)
            description(err.description())
            from()
        }
        CBORError(err: serde_cbor::Error) {
            cause(err)
            description(err.description())
            from()
        }
    }
}

enum IOWorkerMsg {
    Quit,
    Data(Snapshot),
}

// TODO: Add Result return type
fn run(settings_file_name: &str) {
    let settings = match settings::read_parameter_file(settings_file_name) {
        Ok(s) => s,
        Err(e) => {
            error!("Error reading parameter file. {}", e);
            std::process::exit(1)
        }
    };

    let fileext = match settings.environment.output_format {
        OutputFormat::CBOR => "cbor",
        OutputFormat::Bincode => "bincode",
    };

    // Create and initialize output file
    let filename = format!("{prefix}-{time}_v{version}.{fileext}",
                           prefix = settings.environment.prefix,
                           time = &time::now().strftime("%Y-%m-%d_%H%M%S").unwrap().to_string(),
                           version = VERSION,
                           fileext = fileext
                       );

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

    let n = settings.simulation.number_of_timesteps;

    // Create commuication channel for thread
    let (tx, rx) = mpsc::sync_channel::<IOWorkerMsg>(settings.environment.io_queue_size);

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

    // Run the simulation and send data to asynchronous to the IO-thread.
    for data in (&mut simulation).take(n) {
        tx.send(IOWorkerMsg::Data(data)).unwrap();
    }

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

    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).version(crate_version!()).get_matches();

    if matches.is_present("initial_condition") {
        let ic = match de::from_reader(io::stdin()) {
            Ok(s) => s,
            Err(e) => {
                error!("Can't read given initial condition. Error: {}", e);
                std::process::exit(1);
            }
        };

        println!("{:?}", ic);
    } else {
        run(matches.value_of("parameter_file").unwrap());
    }

    std::process::exit(0);
}
