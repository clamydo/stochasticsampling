use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use super::modulofloat::Mf64;

#[derive(Copy, Clone)]
pub struct ModVector64 {
    pub x: Mf64,
    pub y: Mf64,
    pub z: Mf64,
}

impl ModVector64 {
    pub fn new(x: f64, y:f64, z: f64) -> ModVector64 {
        ModVector64 {
            x: Mf64::new(x),
            y: Mf64::new(y),
            z: Mf64::new(z),
        }
    }
}

impl Add for ModVector64 {
    type Output = ModVector64;

    fn add(self, other: ModVector64) -> ModVector64 {
        ModVector64 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for ModVector64 {
    type Output = ModVector64;

    fn sub(self, rhs: ModVector64) -> ModVector64 {
        ModVector64 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f64> for ModVector64 {
    type Output = ModVector64;

    fn mul(self, rhs: f64) -> ModVector64 {
        ModVector64 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_multiplication() {
        let p = ModVector64::new(0.5, -1.5, 0.75);
        let p3 = p * 3.;
        assert_eq!(*p3.x.tof64(), 0.5);
        assert_eq!(*p3.y.tof64(), 0.5);
        assert_eq!(*p3.z.tof64(), 0.25);
    }
}
