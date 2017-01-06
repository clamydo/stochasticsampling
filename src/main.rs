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
extern crate pbr;
extern crate serde_cbor;
extern crate time;

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
}

use bincode::serde::serialize_into;
use clap::App;
use errors::*;
use pbr::ProgressBar;
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
use stochasticsampling::simulation::output::Output;

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


fn create_filename(settings: &Settings) -> String {
    // Need to introduce placeholder `.ext`, since otherwise the patch version
    // number is chopped of later, when using `.with_extension()` method later.
    format!("{prefix}-{time}_v{version}.ext",
            prefix = settings.environment.prefix,
            time = &time::now().strftime("%Y-%m-%d_%H%M%S").unwrap().to_string(),
            version = VERSION)
}

/// Main function
fn run() -> Result<()> {
    // Parse command line
    let yaml = load_yaml!("cli.yml");
    let cli_matches = App::from_yaml(yaml).version(crate_version!()).get_matches();

    let settings_file_name = cli_matches.value_of("parameter_file").unwrap();

    let settings = settings::read_parameter_file(settings_file_name)
            .chain_err(|| "Error reading parameter file.")?;

    let init_type = if cli_matches.is_present("initial_condition_file") {
        InitType::File
    } else {
        if cli_matches.is_present("initial_condition") {
            InitType::Stdin
        } else {
            InitType::Random
        }
    };


    let output_dir = cli_matches.value_of("output_directory").unwrap();
    let filename = create_filename(&settings);
    let path = Path::new(&output_dir).join(filename).to_str().unwrap().to_string();

    let mut simulation = init_simulation(&settings, init_type)
        .chain_err(|| "Error during initialization of simulation.")?;
    let file = prepare_output_file(&settings, &path).chain_err(|| "Cannot prepare output file.")?;

    let show_progress = cli_matches.is_present("progress_bar");
    Ok(run_simulation(&settings, path.into(), file, &mut simulation, show_progress)?)
}


/// Type of setting up initial condition.
enum InitType {
    Stdin,
    File,
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
        InitType::File => {
            let f = match settings.environment.init_file {
                Some(ref fname) => File::open(fname).chain_err(|| "Unable to open input file.")?,
                None => bail!("No input file provided in the parameterfile."),
            };

            de::from_reader(f).chain_err(|| "Can't read given initial condition.")?
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
fn prepare_output_file(settings: &Settings, path: &str) -> Result<File> {
    let fileext = match settings.environment.output_format {
        OutputFormat::CBOR => "cbor",
        OutputFormat::Bincode => "bincode",
    };

    let filepath = Path::new(path).with_extension(fileext);

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
    Snapshot(Snapshot),
    Output(Output),
}


/// Spawns output thread and run simulation.
fn run_simulation(settings: &Settings,
                  path: String,
                  mut file: File,
                  mut simulation: &mut Simulation,
                  show_progress: bool)
                  -> Result<()> {
    let n = settings.simulation.number_of_timesteps;

    // Create communication channel for thread
    let (tx, rx) = mpsc::sync_channel::<IOWorkerMsg>(settings.environment.io_queue_size);

    // Copy output_format, so it can be captured by the thread closure.
    let output_format = settings.environment.output_format;

    // make copy, that is not moved into closure
    let filepath = path.clone();

    let mut snapshot_counter = 0;

    // Spawn worker thread, that periodically flushes collections of simulation
    // states to disk.
    let io_worker = thread::spawn(move || -> Result<()> {
        loop {
            match rx.recv().unwrap() {
                IOWorkerMsg::Quit => break,

                IOWorkerMsg::Snapshot(s) => {
                    snapshot_counter += 1;
                    let filepath = Path::new(&path)
                        .with_extension(format!("bincode.{}", snapshot_counter));

                    let mut snapshot_file = File::create(&filepath).chain_err(|| {
                            format!("couldn't create snapshot file '{}'.", filepath.display())
                        })?;

                    serialize_into(&mut snapshot_file, &s, bincode::SizeLimit::Infinite)
                        .chain_err(||
                            format!("Cannot write snapshot with number {}", snapshot_counter)
                        )?
                }

                IOWorkerMsg::Output(v) => {
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

    // Output sampled distribtuion for initial state. Scope, so `initial` is
    // directly discarted.
    {
        let mut initial = Output::default();
        initial.distribution = Some(simulation.get_distribution());
        initial.particles = settings.simulation
            .output
            .particle_head
            .and_then(|x| Some(simulation.get_particles_head(x)))
            .or_else(|| Some(simulation.get_particles()));

        tx.send(IOWorkerMsg::Output(initial)).unwrap();
    }

    let mut pb = ProgressBar::new(n as u64);
    pb.format("┫██░┣");

    // only show bar, if flag was present
    pb.show_bar = show_progress;
    pb.show_counter = show_progress;
    pb.show_percent = show_progress;
    pb.show_speed = show_progress;
    pb.show_time_left = show_progress;
    pb.show_message = show_progress;

    // Run the simulation and send data to asynchronous to the IO-thread.
    for timestep in 1..(n + 1) {
        pb.inc();
        simulation.do_timestep();

        // TODO: Refactor this ugly code

        // Build output
        let output = Output {
            distribution: settings.simulation.output.distribution_every_timestep.and_then(|x| {
                if timestep % x == 0 {
                    Some(simulation.get_distribution())
                } else {
                    None
                }
            }),
            flow_field: settings.simulation.output.flowfield_every_timestep.and_then(|x| {
                if timestep % x == 0 {
                    Some(simulation.get_flow_field())
                } else {
                    None
                }
            }),
            particles: settings.simulation.output.particle_every_timestep.and_then(|x| {
                if timestep % x == 0 {
                    settings.simulation
                        .output
                        .particle_head
                        .and_then(|x| Some(simulation.get_particles_head(x)))
                        .or_else(|| Some(simulation.get_particles()))
                } else {
                    None
                }
            }),
            timestep: timestep,
        };

        if output.distribution.is_some() || output.flow_field.is_some() ||
           output.particles.is_some() {
            tx.send(IOWorkerMsg::Output(output)).unwrap();
        }
    }
    // FIXME: substitute .ext with correct string
    pb.finish_print(&format!("done. Written '{}'.", filepath));

    // Stop worker
    tx.send(IOWorkerMsg::Quit).unwrap();

    Ok(io_worker.join().unwrap().chain_err(|| "Error when flushing output to disk.")?)
}
