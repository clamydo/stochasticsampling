//! This module handles a TOML settings file.

use std::convert::From;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use toml;

/// Structure that holds settings, which are defined externally in a TOML file.
#[derive(RustcEncodable, RustcDecodable, Debug, Copy, Clone)]
pub struct Settings {
    pub simulation: SimulationSettings,
    pub parameters: Parameters,
}

/// Size of the simulation box an arbitary physical dimensions.
pub type BoxSize = (f64, f64);
/// Size of the discrete grid.
pub type GridSize = (usize, usize, usize);


/// Holds rotational and translational diffusion constants
#[derive(RustcEncodable, RustcDecodable, Debug, Copy, Clone)]
pub struct DiffusionConstants {
    pub translational: f64,
    pub rotational: f64,
}

/// Holds prefactors for active and magnetic stress
#[derive(RustcEncodable, RustcDecodable, Debug, Copy, Clone)]
pub struct StressPrefactors {
    pub active: f64,
    pub magnetic: f64,
}

/// Holds phyiscal parameters
#[derive(RustcEncodable, RustcDecodable, Debug, Copy, Clone)]
pub struct Parameters {
    pub self_propulsion_speed: f64,
    pub diffusion: DiffusionConstants,
    pub stress: StressPrefactors,
    /// Assumes that b points in x-direction
    pub magnetic_reoriantation: f64,
}

/// Holds simulation specific settings.
#[derive(RustcEncodable, RustcDecodable, Debug, Copy, Clone)]
pub struct SimulationSettings {
    pub box_size: BoxSize,
    pub grid_size: GridSize,
    pub number_of_cells: usize,
    pub number_of_particles: usize,
    pub number_of_timesteps: usize,
    pub timestep: f64,
    pub seed: [u64; 2],
}

/// Error type that merges all errors that can happen during loading and
/// parsing of the settings
/// file.
#[derive(Debug)]
pub enum SettingsError {
    Io(io::Error),
    Parser(toml::ParserError),
}

impl Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SettingsError::Io(ref e) => e.fmt(f),
            SettingsError::Parser(ref e) => write!(f, "{}", e),
        }
    }
}

impl Error for SettingsError {
    fn description(&self) -> &str {
        match *self {
            SettingsError::Io(ref e) => e.description(),
            SettingsError::Parser(ref e) => e.description(),
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            SettingsError::Io(ref e) => Some(e),
            SettingsError::Parser(ref e) => Some(e),
        }
    }
}

impl From<io::Error> for SettingsError {
    fn from(err: io::Error) -> SettingsError {
        SettingsError::Io(err)
    }
}

impl From<toml::ParserError> for SettingsError {
    fn from(err: toml::ParserError) -> SettingsError {
        SettingsError::Parser(err)
    }
}


/// Reads the content of a file `filename` into an string and return it.
fn read_from_file(filename: &str) -> Result<String, io::Error> {
    let mut f = try!(File::open(filename));
    let mut content = String::new();
    try!(f.read_to_string(&mut content));

    Ok(content)
}


/// Reads content of a file `param_file`, that should point to a valid TOML
/// file, and Parsers it.
/// Then returns the deserialised data in form of a Settings struct.
pub fn read_parameter_file(param_file: &str) -> Result<Settings, SettingsError> {
    // read .toml file into string
    let toml_string = try!(read_from_file(&param_file));

    let mut parser = toml::Parser::new(&toml_string);

    // desereialise
    match parser.parse() {
        Some(t) => Ok(toml::decode::<Settings>(toml::Value::Table(t)).unwrap()),
        None => Err(SettingsError::Parser(parser.errors[0].to_owned())),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_settings() {
        let settings = read_parameter_file("./test/parameter.toml").unwrap();

        assert_eq!(settings.parameters.diffusion.rotational, 0.5);
        assert_eq!(settings.parameters.diffusion.translational, 1.0);
        assert_eq!(settings.parameters.stress.active, 1.0);
        assert_eq!(settings.parameters.stress.magnetic, 1.0);
        assert_eq!(settings.parameters.magnetic_reoriantation, 1.0);
        assert_eq!(settings.simulation.box_size, (1., 1.));
        assert_eq!(settings.simulation.grid_size, (10, 10, 6));
        assert_eq!(settings.simulation.number_of_cells, 10);
        assert_eq!(settings.simulation.number_of_particles, 100);
        assert_eq!(settings.simulation.number_of_timesteps, 500);
        assert_eq!(settings.simulation.timestep, 0.1);
        assert_eq!(settings.simulation.seed, [1, 1]);
    }
}
