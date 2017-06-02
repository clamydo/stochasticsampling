#![crate_type = "bin"]
#![recursion_limit = "1024"]

extern crate bincode;
#[macro_use]
extern crate clap;
extern crate colored;
extern crate env_logger;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate log;
extern crate pbr;
extern crate serde_cbor;
extern crate stochasticsampling;
extern crate time;

mod init;
mod output;
mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
}


use clap::App;
use colored::*;
use errors::*;
use init::InitType;
use output::path::OutputPath;
use output::worker::Worker;
use pbr::ProgressBar;
use std::path::Path;
use stochasticsampling::simulation::Simulation;
use stochasticsampling::simulation::output::OutputEntry;
use stochasticsampling::simulation::settings::{self, Settings};


const VERSION: &'static str = env!("CARGO_PKG_VERSION");
include!(concat!(env!("OUT_DIR"), "/version.rs"));

pub fn version() -> String {
    format!("{}-{}", VERSION, short_sha())
}

fn main() {
    // initialize the env_logger implementation
    env_logger::init().unwrap();

    // error handling of runner
    if let Err(ref e) = run() {
        error!("{}: {}", "error".red(), e);

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
    let cli_matches = App::from_yaml(yaml)
        .version(version().as_str())
        .get_matches();

    let settings_file_name = cli_matches.value_of("parameter_file").unwrap();

    let mut settings = settings::read_parameter_file(settings_file_name)
        .chain_err(|| "Error reading parameter file.")?;

    settings.set_version(&version());
    // drop mutability for safety
    let settings = settings;

    let init_type = if cli_matches.is_present("initial_condition") {
        InitType::Stdin
    } else if settings.environment.init_file.is_none() {
        if cli_matches.is_present("resume") {
            bail!("Cannot resume. Pleas provide snapshot in `init_file` field in parameter file.");
        }
        InitType::Random
    } else if cli_matches.is_present("resume") {
        InitType::Resume
    } else {
        InitType::File
    };


    let output_dir = Path::new(cli_matches.value_of("output_directory").unwrap());
    let path = OutputPath::new(output_dir, &settings.environment.prefix);

    let mut simulation = init::init_simulation(&settings, init_type)
        .chain_err(|| "Error during initialization of simulation.")?;

    let show_progress = cli_matches.is_present("progress_bar");

    path.create()
        .chain_err(|| "Cannot create output directory")?;

    let worker = Worker::new(
        settings.environment.io_queue_size,
        &path,
        settings.environment.output_format,
    )
            .chain_err(|| "Unable to create output thread.")?;

    worker
        .write_metadata(settings.clone())
        .chain_err(|| "Unable to write metadata to output.")?;

    Ok(run_simulation(&settings, &mut simulation, worker, show_progress)?)
}


/// Spawns output thread and run simulation.
fn run_simulation(
    settings: &Settings,
    mut simulation: &mut Simulation,
    out: Worker,
    show_progress: bool,
) -> Result<()> {

    let n = settings.simulation.number_of_timesteps;


    // Output sampled distribtuion for initial state. Scope, so `initial` is
    // directly discarted.
    {
        let mut initial = OutputEntry::default();
        initial.distribution = Some(simulation.get_distribution());
        initial.particles = if settings.simulation.output.particle_every_timestep.is_some() {
            settings
                .simulation
                .output
                .particle_head
                .and_then(|n| Some(simulation.get_particles_head(n)))
                .or_else(|| Some(simulation.get_particles()))
        } else {
            None
        };

        out.append(initial)
            .chain_err(|| "Unable to append initial condition.")?;
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

        // Build entry
        let entry = OutputEntry {
            distribution: settings
                .simulation
                .output
                .distribution_every_timestep
                .and_then(
                    |x| if timestep % x == 0 {
                        Some(simulation.get_distribution())
                    } else {
                        None
                    }
                ),
            flow_field: settings
                .simulation
                .output
                .flowfield_every_timestep
                .and_then(
                    |x| if timestep % x == 0 {
                        Some(simulation.get_flow_field())
                    } else {
                        None
                    }
                ),
            particles: settings
                .simulation
                .output
                .particle_every_timestep
                .and_then(
                    |x| if timestep % x == 0 {
                        settings
                            .simulation
                            .output
                            .particle_head
                            .and_then(|x| Some(simulation.get_particles_head(x)))
                            .or_else(|| Some(simulation.get_particles()))
                    } else {
                        None
                    }
                ),
            timestep: timestep,
        };

        if entry.distribution.is_some() || entry.flow_field.is_some() || entry.particles.is_some() {
            out.append(entry)
                .chain_err(|| "Unable to append simulation entry.")?;
        }

        match settings.simulation.output.snapshot_every_timestep {
            Some(x) if timestep % x == 0 => {
                let snapshot = simulation.get_snapshot();
                out.write_snapshot(snapshot)
            }
            _ => Ok(()),
        }?;
    }

    pb.finish_print(&format!("✓ {} ", "DONE".green().bold()));
    // TODO Why is this necessary?
    println!("");

    let snapshot = simulation.get_snapshot();
    out.write_snapshot(snapshot)
        .chain_err(|| "Error writing last snapshot.")?;

    print!("Write buffer to disk… ");
    let opath = out.get_output_filepath().to_str().unwrap().to_string();

    out.quit()?;

    println!("written '{}'.", opath);
    Ok(())
}
