#![crate_type = "bin"]
#![recursion_limit = "1024"]

extern crate stochasticsampling;
extern crate bincode;
#[macro_use]
extern crate clap;
extern crate env_logger;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate serde_cbor;
extern crate time;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
}

use bincode::serde::serialize_into;
use clap::App;
use errors::*;
use serde_cbor::{de, ser};
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use stochasticsampling::coordinates::particle::Particle;
use stochasticsampling::settings::{self, Settings};
use stochasticsampling::settings::OutputFormat;
use stochasticsampling::simulation::Simulation;
use stochasticsampling::simulation::Snapshot;

const VERSION: &'static str = env!("CARGO_PKG_VERSION");

fn main() {
    // initialize the env_logger implementation
    env_logger::init().unwrap();

    // error handling of runner
    if let Err(ref e) = run() {
        error!("error: {}", e);

        for e in e.iter().skip(1) {
            error!("caused by: {}", e);
        }

        // The backtrace is not always generated. Try to run this  with
        // `RUST_BACKTRACE=1`.
        if let Some(backtrace) = e.backtrace() {
            error!("backtrace: {:?}", backtrace);
        }

        ::std::process::exit(1);
    }

    ::std::process::exit(0);
}


/// Main function
fn run() -> Result<()> {
    // Parse command line
    let yaml = load_yaml!("cli.yml");
    let cli_matches = App::from_yaml(yaml).version(crate_version!()).get_matches();

    let settings_file_name = cli_matches.value_of("parameter_file").unwrap();

    let settings = settings::read_parameter_file(settings_file_name)
            .chain_err(|| "Error reading parameter file.")?;

    let init_type = if cli_matches.is_present("initial_condition") {
        InitType::Stdin
    } else {
        InitType::Random
    };

    let mut simulation = init_simulation(&settings, init_type)
        .chain_err(|| "Error during initialization of simulation.")?;
    let file = prepare_output_file(&settings).chain_err(|| "Cannot prepare output file.")?;

    Ok(run_simulation(&settings, file, &mut simulation)?)
}


/// Type of setting up initial condition.
enum InitType {
    Stdin,
    Random,
}


/// Returns an initialized simulation. Sets initial condition according to
/// `init_type` flag.
fn init_simulation(settings: &Settings, init_type: InitType) -> Result<Simulation> {

    // Setup simulation
    let mut simulation = Simulation::new(settings.clone());

    let initial_condition = match init_type {
        InitType::Stdin => {
            de::from_reader(io::stdin()).chain_err(|| "Can't read given initial condition.")?
        }
        InitType::Random => {
            Particle::randomly_placed_particles(settings.simulation.number_of_particles,
                                                settings.simulation.box_size,
                                                settings.simulation.seed)
        }
    };

    simulation.init(initial_condition);

    Ok(simulation)
}


/// Creates an output file. Already writes header for metadata.
fn prepare_output_file(settings: &Settings) -> Result<File> {
    let fileext = match settings.environment.output_format {
        OutputFormat::CBOR => "cbor",
        OutputFormat::Bincode => "bincode",
    };

    // Create and initialize output file
    let filename = format!("{prefix}-{time}_v{version}.{fileext}",
                           prefix = settings.environment.prefix,
                           time = &time::now().strftime("%Y-%m-%d_%H%M%S").unwrap().to_string(),
                           version = VERSION,
                           fileext = fileext);

    let filepath = Path::new(&settings.environment.output_dir).join(filename);

    let mut file = File::create(&filepath)
        .chain_err(|| format!("couldn't create output file '{}'.", filepath.display()))?;

    // Serialize settings as first object in file
    match settings.environment.output_format {
        OutputFormat::CBOR => ser::to_writer_sd(&mut file, &settings).unwrap(),
        OutputFormat::Bincode => {
            serialize_into(&mut file, &settings, bincode::SizeLimit::Infinite).unwrap()
        }
    }

    Ok(file)
}


/// Message type for the IO worker thread channel.
enum IOWorkerMsg {
    Quit,
    Data(Snapshot),
}


/// Spawns output thread and run simulation.
fn run_simulation(settings: &Settings,
                  mut file: File,
                  mut simulation: &mut Simulation)
                  -> Result<()> {
    let n = settings.simulation.number_of_timesteps;

    // Create communication channel for thread
    let (tx, rx) = mpsc::sync_channel::<IOWorkerMsg>(settings.environment.io_queue_size);

    // Copy output_format, so it can be captured by the thread closure.
    let output_format = settings.environment.output_format;

    // Spawn worker thread, that periodically flushes collections of simulation
    // states to disk.
    let io_worker = thread::spawn(move || -> Result<()> {
        loop {
            match rx.recv().unwrap() {
                IOWorkerMsg::Quit => break,
                IOWorkerMsg::Data(v) => {
                    // write all snapshots into one cbor file
                    match output_format {
                        OutputFormat::CBOR => {
                            ser::to_writer_sd(&mut file, &v)
                                .chain_err(||
                                    "Cannot write simulation output (format: CBOR).")?
                        }
                        OutputFormat::Bincode => {
                            serialize_into(&mut file, &v, bincode::SizeLimit::Infinite)
                                .chain_err(||
                                    "Cannot write simulation output (format: bincode).")?
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

    Ok(io_worker.join().unwrap().chain_err(|| "Error when flushing output to disk.")?)
}
