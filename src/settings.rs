//! This module handles a TOML settings file.

use serde::Deserialize;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use toml;

/// Structure that holds settings, which are defined externally in a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub simulation: SimulationSettings,
    pub parameters: Parameters,
    pub environment: EnvironmentSettings,
}

/// Size of the simulation box an arbitary physical dimensions.
pub type BoxSize = (f64, f64);
/// Size of the discrete grid.
pub type GridSize = (usize, usize, usize);


/// Holds rotational and translational diffusion constants
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct DiffusionConstants {
    pub translational: f64,
    pub rotational: f64,
}

/// Holds prefactors for active and magnetic stress
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct StressPrefactors {
    pub active: f64,
    pub magnetic: f64,
}

/// Holds phyiscal parameters
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Parameters {
    pub diffusion: DiffusionConstants,
    pub stress: StressPrefactors,
    /// Assumes that b points in x-direction
    pub magnetic_reoriantation: f64,
}

/// Holds simulation specific settings.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SimulationSettings {
    pub box_size: BoxSize,
    pub grid_size: GridSize,
    pub number_of_particles: usize,
    pub number_of_timesteps: usize,
    pub timestep: f64,
    pub seed: [u64; 2],
}

// use enum_str macro to encode this variant into strings
serde_enum_str!(OutputFormat {
    CBOR("CBOR"),
    Bincode("bincode"),
});

/// Holds environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    pub output_dir: String,
    pub prefix: String,
    #[serde(default = "default_output_format")]
    pub output_format: OutputFormat,
}

fn default_output_format() -> OutputFormat {
    OutputFormat::CBOR
}

// Quickly implement meta error type for this module.
quick_error! {
    /// Error type including error that can happend during (de)serialization of
    /// the settings file.
    #[derive(Debug)]
    pub enum SettingsError {
        Io(err: io::Error) {
            display("I/O error: {}", err)
            from()
        }
        Parser(err: toml::ParserError) {
            display("Parser error: {}", err)
            from()
        }
        Devode(err: toml::DecodeError) {
            display("TOML decorder error: {}", err)
            from()
        }
    }
}


/// Reads the content of a file `filename` into an string and return it.
fn read_from_file(filename: &str) -> Result<String, io::Error> {
    let mut f = try!(File::open(filename));
    let mut content = String::new();
    f.read_to_string(&mut content)?;

    Ok(content)
}


/// Reads content of a file `param_file`, that should point to a valid TOML
/// file, and Parsers it.
/// Then returns the deserialized data in form of a Settings struct.
pub fn read_parameter_file(param_file: &str) -> Result<Settings, SettingsError> {
    // read .toml file into string
    let toml_string = read_from_file(&param_file)?;

    let mut parser = toml::Parser::new(&toml_string);

    // try to parse settings file
    match parser.parse() {
        // Choosing this more complicated way, to get better error messages.
        Some(t) => Ok(Settings::deserialize(&mut toml::Decoder::new(toml::Value::Table(t)))?),
        None => Err(parser.errors[0].to_owned().into()),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_settings() {

        let settings = read_parameter_file("./test/parameter.toml").unwrap();

        assert_eq!(settings.environment.output_dir, "./out/");
        assert_eq!(settings.environment.output_format, OutputFormat::Bincode);
        assert_eq!(settings.environment.prefix, "foo");
        assert_eq!(settings.parameters.diffusion.rotational, 0.5);
        assert_eq!(settings.parameters.diffusion.translational, 1.0);
        assert_eq!(settings.parameters.stress.active, 1.0);
        assert_eq!(settings.parameters.stress.magnetic, 1.0);
        assert_eq!(settings.parameters.magnetic_reoriantation, 1.0);
        assert_eq!(settings.simulation.box_size, (1., 1.));
        assert_eq!(settings.simulation.grid_size, (10, 10, 6));
        assert_eq!(settings.simulation.number_of_particles, 100);
        assert_eq!(settings.simulation.number_of_timesteps, 500);
        assert_eq!(settings.simulation.timestep, 0.1);
        assert_eq!(settings.simulation.seed, [1, 1]);
    }
}
