pub mod modulofloat;
pub mod vector;

use self::vector::ModVector64;

#[derive(Copy, Clone)]
pub struct Particle {
    pub position: ModVector64,
}
