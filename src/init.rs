use bincode::{self, Infinite};
use errors::*;
use serde_cbor;
use std::fs::File;
use std::io;
use std::path::Path;
use stochasticsampling::simulation::Simulation;
use stochasticsampling::simulation::particle::Particle;
use stochasticsampling::simulation::settings::Settings;


/// Type of setting up initial condition.
pub enum InitType {
    Stdin,
    File,
    Random,
    Resume,
}


/// Returns an initialized simulation. Sets initial condition according to
/// `init_type` flag.
pub fn init_simulation(settings: &Settings, init_type: InitType) -> Result<Simulation> {

    // Setup simulation
    let mut simulation = Simulation::new(settings.clone());

    match init_type {
        InitType::Stdin => {
            info!("Reading initial condition from standard input");
            let p = serde_cbor::de::from_reader(io::stdin())
                .chain_err(|| "Can't read given initial condition. Did you use CBOR format?")?;

            simulation.init(p);
        }
        InitType::File => {
            let fname = match settings.environment.init_file {
                Some(ref v) => v,
                None => bail!("No input file provided in the parameterfile."),
            };

            info!("Reading initial condition from {}", fname);
            let mut f = File::open(fname).chain_err(|| format!("Unable to open input file '{}'.", fname))?;

            match Path::new(&fname).extension() {
                Some(ext) => {
                    match ext.to_str().unwrap() {
                        "cbor" => {
                            let p = serde_cbor::de::from_reader(f)
                            .chain_err(|| "CBOR, Cannot read given initial condition.")?;
                            simulation.init(p);
                        }
                        "bincode" => {
                            let p = bincode::deserialize_from(&mut f, Infinite)
                            .chain_err(|| "Bincode, Cannot read given initial condition.")?;
                            simulation.init(p);
                        }
                        _ => bail!("Do not recognise file extension {}.", ext.to_str().unwrap()),
                    }
                }
                None => {
                    bail!("Missing file extension for initial condition, '{}'.
                           Cannot determine filetype.",
                          fname)
                }
            };

        }
        InitType::Random => {
            info!("Using isotropic initial condition.");
            let p = Particle::randomly_placed_particles(settings.simulation.number_of_particles,
                                                        settings.simulation.box_size,
                                                        settings.simulation.seed);

            simulation.init(p);
        }
        InitType::Resume => {
            info!("Resuming snapshot.");
            let mut f = match settings.environment.init_file {
                Some(ref fname) => {
                    info!("Reading snapshot from {}", *fname);
                    File::open(fname).chain_err(|| "Unable to open input file.")?
                }
                None => bail!("No input file provided in the parameterfile."),
            };

            let s =  bincode::deserialize_from(&mut f, Infinite)
                    .chain_err(|| "Cannot read given snapshot.")?;

            simulation.resume(s);
        }
    };

    Ok(simulation)
}
