//! This module handles a TOML settings file.

use serde::Deserialize;
use std::fs::File;
use std::io::prelude::*;
use toml;

const DEFAULT_IO_QUEUE_SIZE: usize = 10;
const DEFAULT_OUTPUT_FORMAT: OutputFormat = OutputFormat::CBOR;

error_chain! {
    foreign_links {
        TOMLParser(toml::ParserError);
        TOMLDecoder(toml::DecodeError);
    }
}


/// Structure that holds settings, which are defined externally in a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub simulation: SimulationSettings,
    pub parameters: Parameters,
    pub environment: EnvironmentSettings,
}

/// Size of the simulation box an arbitary physical dimensions.
pub type BoxSize = [f64; 2];
/// Size of the discrete grid.
pub type GridSize = [usize; 3];


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
    /// Assumes that b points in y-direction
    pub magnetic_reorientation: f64,
}


/// Holds output configuration
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Output {
    #[serde(default)]
    pub distribution_every_timestep: Option<usize>,
    #[serde(default)]
    pub flowfield_every_timestep: Option<usize>,
    #[serde(default)]
    pub particle_head: Option<usize>,
    #[serde(default)]
    pub particle_every_timestep: Option<usize>,
}

/// Holds simulation specific settings.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SimulationSettings {
    pub box_size: BoxSize,
    pub grid_size: GridSize,
    pub number_of_particles: usize,
    pub number_of_timesteps: usize,
    pub output: Output,
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
    #[serde(default)]
    pub init_file: Option<String>,
    #[serde(default = "default_io_queue_size")]
    pub io_queue_size: usize,
    #[serde(default = "default_output_format")]
    pub output_format: OutputFormat,
    pub prefix: String,
}

/// Default value of IO queue size
fn default_io_queue_size() -> usize {
    DEFAULT_IO_QUEUE_SIZE
}

/// Default output format
fn default_output_format() -> OutputFormat {
    DEFAULT_OUTPUT_FORMAT
}


/// Reads the content of a file `filename` into an string and return it.
fn read_from_file(filename: &str) -> Result<String> {
    let mut f = File::open(filename).chain_err(|| "Unable to open file.")?;
    let mut content = String::new();

    f.read_to_string(&mut content).chain_err(|| "Unable to read file.")?;

    Ok(content)
}


/// Reads content of a file `param_file`, that should point to a valid TOML
/// file, and Parsers it.
/// Then returns the deserialized data in form of a Settings struct.
pub fn read_parameter_file(param_file: &str) -> Result<Settings> {
    // read .toml file into string
    let toml_string = read_from_file(param_file).chain_err(|| "Unable to read parameter file.")?;

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
        let settings_default = read_parameter_file("./test/parameter_no_defaults.toml").unwrap();

        assert_eq!(settings_default.environment.init_file, None);
        assert_eq!(settings.environment.init_file, Some("foo/bar.cbor".to_string()));
        assert_eq!(settings_default.environment.io_queue_size, DEFAULT_IO_QUEUE_SIZE);
        assert_eq!(settings.environment.io_queue_size, 50);
        assert_eq!(settings_default.environment.output_format, DEFAULT_OUTPUT_FORMAT);
        assert_eq!(settings.environment.output_format, OutputFormat::Bincode);
        assert_eq!(settings.environment.prefix, "foo");
        assert_eq!(settings.parameters.diffusion.rotational, 0.5);
        assert_eq!(settings.parameters.diffusion.translational, 1.0);
        assert_eq!(settings.parameters.stress.active, 1.0);
        assert_eq!(settings.parameters.stress.magnetic, 1.0);
        assert_eq!(settings.parameters.magnetic_reorientation, 1.0);
        assert_eq!(settings.simulation.box_size, [1., 1.]);
        assert_eq!(settings.simulation.grid_size, [10, 10, 6]);
        assert_eq!(settings.simulation.number_of_particles, 100);
        assert_eq!(settings.simulation.number_of_timesteps, 500);
        assert_eq!(settings.simulation.timestep, 0.1);
        assert_eq!(settings.simulation.seed, [1, 1]);

        assert_eq!(settings.simulation.output.distribution_every_timestep, Some(12));
        assert_eq!(settings_default.simulation.output.distribution_every_timestep, None);
        assert_eq!(settings.simulation.output.flowfield_every_timestep, Some(42));
        assert_eq!(settings_default.simulation.output.flowfield_every_timestep, None);
        assert_eq!(settings.simulation.output.particle_every_timestep, Some(100));
        assert_eq!(settings_default.simulation.output.particle_every_timestep, None);
        assert_eq!(settings.simulation.output.particle_head, Some(10));
        assert_eq!(settings_default.simulation.output.particle_head, None);
    }
}
