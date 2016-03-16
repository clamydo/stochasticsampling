extern crate toml;

use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::error::Error;
use std::convert::From;
use std::fmt::Display;
use std::fmt;

/// Structure that holds settings, which are defined externally in a TOML file.
#[derive(RustcEncodable, RustcDecodable)]
pub struct Settings {
    pub simulation: SimulationSettings,
}

/// Holds simulation specific settings.
#[derive(RustcEncodable, RustcDecodable)]
pub struct SimulationSettings {
    pub timestep: f64,
    pub number_of_particles: usize,
    pub number_of_timesteps: usize,
    pub number_of_cells: usize,
    pub diffusion_constant: f64,
}

/// Error type that merges all errors that can happen during loading and parsing of the settings
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


/// Reads content of a file `param_file`, that should point to a valid TOML file, and Parsers it.
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

        assert_eq!(settings.simulation.timestep, 0.1);
        assert_eq!(settings.simulation.number_of_particles, 100);
        assert_eq!(settings.simulation.number_of_timesteps, 500);
        assert_eq!(settings.simulation.number_of_cells, 10);
        assert_eq!(settings.simulation.diffusion_constant, 1.0);
    }
}
