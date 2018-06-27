pub mod distribution;
pub mod flowfield;
pub mod integrators;
pub mod magnetic_interaction;
pub mod mesh;
pub mod output;
pub mod particle;
pub mod polarization;
pub mod vector;

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
