//! This module handles a TOML settings file.

use std::fs::File;
use std::io::prelude::*;
use toml;

const DEFAULT_IO_QUEUE_SIZE: usize = 1;
const DEFAULT_OUTPUT_FORMAT: OutputFormat = OutputFormat::MsgPack;
const DEFAULT_INIT_TYPE: InitDistribution = InitDistribution::Isotropic;

error_chain! {
    foreign_links {
        TOMLError(toml::de::Error);
    }
}


/// Structure that holds settings, which are defined externally in a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub simulation: SimulationSettings,
    pub parameters: Parameters,
    pub environment: EnvironmentSettings,
}

/// Size of the simulation box an arbitary physical dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BoxSize {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
/// Size of the discrete grid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GridSize {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub phi: usize,
    pub theta: usize,
}


/// Holds rotational and translational diffusion constants
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffusionConstants {
    pub translational: f64,
    pub rotational: f64,
}

/// Holds prefactors for active and magnetic stress
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StressPrefactors {
    pub active: f64,
    pub magnetic: f64,
}

/// Holds phyiscal parameters
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub diffusion: DiffusionConstants,
    pub stress: StressPrefactors,
    /// Assumes that b points in y-direction
    pub magnetic_reorientation: f64,
}


/// Holds output configuration
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Output {
    #[serde(default)]
    pub distribution: Option<usize>,
    #[serde(default = "default_final_snapshot")]
    pub final_snapshot: bool,
    #[serde(default)]
    pub flowfield: Option<usize>,
    #[serde(default)]
    pub particles_head: Option<usize>,
    #[serde(default)]
    pub particles: Option<usize>,
    #[serde(default = "default_initial_condition")]
    pub initial_condition: bool,
    #[serde(default)]
    pub snapshot: Option<usize>,
}

fn default_final_snapshot() -> bool {
    true
}

fn default_initial_condition() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InitDistribution {
    Isotropic,
    Homogeneous,
}


/// Holds simulation specific settings.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationSettings {
    pub box_size: BoxSize,
    pub grid_size: GridSize,
    #[serde(default = "default_init_distribution")]
    pub init_distribution: InitDistribution,
    pub number_of_particles: usize,
    pub number_of_timesteps: usize,
    pub output_at_timestep: Output,
    pub timestep: f64,
    pub seed: [u64; 2],
}

/// Default init type
fn default_init_distribution() -> InitDistribution {
    DEFAULT_INIT_TYPE
}


#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum OutputFormat {
    CBOR,
    Bincode,
    MsgPack,
}

/// Holds environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentSettings {
    #[serde(default)]
    pub init_file: Option<String>,
    #[serde(default = "default_io_queue_size")]
    pub io_queue_size: usize,
    #[serde(default = "default_output_format")]
    pub output_format: OutputFormat,
    pub prefix: String,
    #[serde(skip_deserializing)]
    version: String,
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

    f.read_to_string(&mut content).chain_err(
        || "Unable to read file.",
    )?;

    Ok(content)
}


/// Reads content of a file `param_file`, that should point to a valid TOML
/// file, and Parsers it.
/// Then returns the deserialized data in form of a Settings struct.
pub fn read_parameter_file(param_file: &str) -> Result<Settings> {
    // read .toml file into string
    let toml_string = read_from_file(param_file).chain_err(
        || "Unable to read parameter file.",
    )?;

    let mut settings: Settings = toml::from_str(&toml_string).chain_err(
        || "Unable to parse parameter file.",
    )?;

    settings.environment.version = "".to_string();

    check_settings(&settings)?;

    Ok(settings)
}


fn check_settings(s: &Settings) -> Result<()> {

    // TODO Check settings for sanity. For example, particles_head <=
    // number_of_particles
    let bs = s.simulation.box_size;

    if bs.x <= 0. || bs.y <= 0. || bs.z <= 0. {
        bail!("Box size is invalid. Must be bigger than 0: {:?}", bs)
    }

    if s.simulation.output_at_timestep.particles_head.is_some() {
        if s.simulation.number_of_particles <
            s.simulation.output_at_timestep.particles_head.unwrap()
        {
            bail!(
                "Cannot output more particles than available. `particles_head`
                   must be smaller or equal to `number_of_particles`"
            )
        }
    }

    Ok(())
}

impl Settings {
    pub fn set_version(&mut self, version: &str) {
        // save version to metadata
        self.environment.version = version.to_string();
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_settings() {
        let mut settings = read_parameter_file("./test/parameter.toml").unwrap();
        settings.set_version("version");
        let settings_default = read_parameter_file("./test/parameter_no_defaults.toml").unwrap();

        assert_eq!(settings_default.environment.init_file, None);
        assert_eq!(
            settings.environment.init_file,
            Some("foo/bar.cbor".to_string())
        );
        assert_eq!(
            settings_default.environment.io_queue_size,
            DEFAULT_IO_QUEUE_SIZE
        );
        assert_eq!(settings.environment.io_queue_size, 50);
        assert_eq!(
            settings_default.environment.output_format,
            DEFAULT_OUTPUT_FORMAT
        );
        assert_eq!(settings.environment.output_format, OutputFormat::Bincode);
        assert_eq!(settings.environment.prefix, "foo");
        assert_eq!(settings.environment.version, "version");
        assert_eq!(settings.parameters.diffusion.rotational, 0.5);
        assert_eq!(settings.parameters.diffusion.translational, 1.0);
        assert_eq!(settings.parameters.stress.active, 1.0);
        assert_eq!(settings.parameters.stress.magnetic, 1.0);
        assert_eq!(settings.parameters.magnetic_reorientation, 1.0);
        assert_eq!(
            settings.simulation.box_size,
            BoxSize {
                x: 1.,
                y: 2.,
                z: 3.,
            }
        );
        assert_eq!(
            settings.simulation.grid_size,
            GridSize {
                x: 11,
                y: 12,
                z: 13,
                phi: 6,
                theta: 7,
            }
        );
        assert_eq!(
            settings.simulation.init_distribution,
            InitDistribution::Homogeneous
        );
        assert_eq!(
            settings_default.simulation.init_distribution,
            InitDistribution::Isotropic
        );
        assert_eq!(settings.simulation.number_of_particles, 100);
        assert_eq!(settings.simulation.number_of_timesteps, 500);
        assert_eq!(settings.simulation.timestep, 0.1);
        assert_eq!(settings.simulation.seed, [1, 1]);

        assert_eq!(
            settings.simulation.output_at_timestep.distribution,
            Some(12)
        );
        assert_eq!(
            settings_default.simulation.output_at_timestep.distribution,
            None
        );

        assert_eq!(
            settings_default
                .simulation
                .output_at_timestep
                .final_snapshot,
            true
        );
        assert_eq!(settings.simulation.output_at_timestep.final_snapshot, false);

        assert_eq!(settings.simulation.output_at_timestep.flowfield, Some(42));
        assert_eq!(
            settings_default.simulation.output_at_timestep.flowfield,
            None
        );

        assert_eq!(settings.simulation.output_at_timestep.particles, Some(100));
        assert_eq!(
            settings_default.simulation.output_at_timestep.particles,
            None
        );

        assert_eq!(
            settings.simulation.output_at_timestep.particles_head,
            Some(10)
        );
        assert_eq!(
            settings_default
                .simulation
                .output_at_timestep
                .particles_head,
            None
        );

        assert_eq!(
            settings_default
                .simulation
                .output_at_timestep
                .initial_condition,
            true
        );
        assert_eq!(
            settings.simulation.output_at_timestep.initial_condition,
            false
        );

        assert_eq!(settings.simulation.output_at_timestep.snapshot, Some(666));
        assert_eq!(
            settings_default.simulation.output_at_timestep.snapshot,
            None
        );
    }

    #[test]
    #[should_panic]
    fn test_settings_unused_keys() {
        read_parameter_file("./test/parameter_unused.toml").unwrap();
    }
}
