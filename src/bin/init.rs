use crate::errors::*;
use crate::simulation::settings::{InitDistribution, Settings};
use crate::Simulation;
use bincode;
use log::info;
use lzma::LzmaReader;
use rmp_serde;
use serde::de::DeserializeOwned;
use serde_cbor;
use std::fs::File;
use std::io;
use std::path::Path;
use stochasticsampling::particle::Particle;

/// Type of setting up initial condition.
pub enum InitType {
    Stdin,
    File,
    Distribution,
    Resume,
}

fn read_from_file<T: DeserializeOwned>(fname: &Path) -> Result<T> {
    let f =
        File::open(fname).chain_err(|| format!("Unable to open file '{}'.", fname.display()))?;

    let mut r = LzmaReader::new_decompressor(f).chain_err(|| "LZMA reader cannot be created.")?;

    match Path::new(&fname).extension() {
        Some(ext) => match ext.to_str().unwrap() {
            "cbor-lzma" => {
                Ok(serde_cbor::de::from_reader(r)
                    .chain_err(|| "CBOR, cannot decode given file.")?)
            }
            "bincode-lzma" => Ok(bincode::deserialize_from(&mut r)
                .chain_err(|| "Bincode, cannot decode given file.")?),
            "msgpack-lzma" => {
                Ok(rmp_serde::from_read(r).chain_err(|| "MsgPack, cannot decode given file.")?)
            }
            _ => bail!("Do not recognise file extension {}.", ext.to_str().unwrap()),
        },
        None => bail!(
            "Missing file extension for initial condition, '{}'.
                           Cannot determine filetype.",
            fname.display()
        ),
    }
}

/// Returns an initialized simulation. Sets initial condition according to
/// `init_type` flag.
pub fn init_simulation(settings: &Settings, init_type: InitType) -> Result<Simulation> {
    // Setup simulation
    let mut simulation = Simulation::new(settings.clone());

    match init_type {
        InitType::Stdin => {
            info!("Reading initial condition from standard input");
            let p = rmp_serde::from_read(io::stdin())
                .chain_err(|| "Can't read given initial condition. Did you use MsgPack format?")?;

            simulation.init(p);
        }
        InitType::File => {
            let fname = match settings.environment.init_file {
                Some(ref v) => v,
                None => bail!("No input file provided in the parameterfile."),
            };

            info!("Reading initial condition from {}", fname);

            let path = Path::new(&fname);
            let p = read_from_file(&path).chain_err(|| "Cannot read init file.")?;
            simulation.init(p);
        }
        InitType::Distribution => {
            let p = match settings.simulation.init_distribution {
                InitDistribution::Isotropic => {
                    info!("Using isotropic initial condition.");
                    Particle::create_isotropic(
                        settings.simulation.number_of_particles,
                        &settings.simulation.box_size,
                        settings.simulation.seed,
                    )
                }
                InitDistribution::Homogeneous => {
                    info!("Using spatial homogeneous initial condition.");
                    Particle::create_homogeneous(
                        settings.simulation.number_of_particles,
                        settings.parameters.magnetic_reorientation
                            / settings.parameters.diffusion.rotational,
                        &settings.simulation.box_size,
                        settings.simulation.seed,
                    )
                }
            };

            simulation.init(p);
        }
        InitType::Resume => {
            info!("Resuming snapshot.");
            let fname = match settings.environment.init_file {
                Some(ref v) => v,
                None => bail!("No input file provided in the parameterfile."),
            };

            info!("Reading snapshot from {}", *fname);
            let path = Path::new(&fname);

            let s = read_from_file(&path)
                .chain_err(|| "Cannot read snapshot, resuming not possible.")?;
            simulation.resume(s);
        }
    };

    Ok(simulation)
}
