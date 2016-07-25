pub mod modulofloat;
pub mod vector;

use self::vector::Mod64Vector2;

#[derive(Copy, Clone)]
pub struct Particle {
    pub position: Mod64Vector2,
    pub orientation: f64,
}
