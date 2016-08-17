//! This module handles a TOML settings file.
extern crate toml;

use std::convert::From;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fs::File;
use std::io;
use std::io::prelude::*;

/// Structure that holds settings, which are defined externally in a TOML file.
#[derive(RustcEncodable, RustcDecodable)]
pub struct Settings {
    pub simulation: SimulationSettings,
}

/// Size of the simulation box an arbitary physical dimensions.
pub type BoxSize = (f64, f64);
/// Size of the discrete grid.
pub type GridSize = (usize, usize, usize);

/// Holds simulation specific settings.
#[derive(RustcEncodable, RustcDecodable)]
pub struct SimulationSettings {
    pub box_size: BoxSize,
    pub grid_size: GridSize,
    pub number_of_cells: usize,
    pub number_of_particles: usize,
    pub number_of_timesteps: usize,
    pub rotational_diffusion_constant: f64,
    pub timestep: f64,
    pub translational_diffusion_constant: f64,
}

/// Error type that merges all errors that can happen during loading and
/// parsing of the settings
/// file.
#[derive(Debug)]
pub enum SettingsError {
    Parser(String),
    Io(io::Error),
}

impl Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SettingsError::Io(ref e) => e.fmt(f),
            SettingsError::Parser(ref s) => write!(f, "{}", s),
        }
    }
}

impl Error for SettingsError {
    fn description(&self) -> &str {
        match *self {
            SettingsError::Io(ref e) => e.description(),
            SettingsError::Parser(ref s) => s,
        }
    }

    fn cause(&self) -> Option<&Error> {
        match *self {
            SettingsError::Io(ref e) => Some(e),
            SettingsError::Parser(_) => None,
        }
    }
}

impl From<io::Error> for SettingsError {
    fn from(err: io::Error) -> SettingsError {
        SettingsError::Io(err)
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

    // desereialise
    toml::decode_str::<Settings>(&toml_string)
        .ok_or_else(|| SettingsError::Parser("Settings file could not be Parserd.".to_owned()))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_settings() {
        let settings = read_parameter_file("./test/parameter.toml").unwrap();

        assert_eq!(settings.simulation.box_size, (1., 1.));
        assert_eq!(settings.simulation.grid_size, (10, 10, 6));
        assert_eq!(settings.simulation.number_of_cells, 10);
        assert_eq!(settings.simulation.number_of_particles, 100);
        assert_eq!(settings.simulation.number_of_timesteps, 500);
        assert_eq!(settings.simulation.rotational_diffusion_constant, 0.5);
        assert_eq!(settings.simulation.timestep, 0.1);
        assert_eq!(settings.simulation.translational_diffusion_constant, 1.0);
    }
}
