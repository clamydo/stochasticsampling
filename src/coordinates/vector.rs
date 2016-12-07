//! 2D and 3D modulo vector types.

use coordinates::modulofloat::Mf64;
use std::ops::{Add, AddAssign, Mul, Sub};

/// 3D modulo vector type
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Mod64Vector3 {
    /// x-coordinate
    pub x: Mf64,
    /// y-coordinate
    pub y: Mf64,
    /// z-coordinate
    pub z: Mf64,
}

impl Mod64Vector3 {
    /// Returns a modulo 3D vector with the given coordinates and quotients.
    pub fn new(x: f64, y: f64, z: f64, m: [f64;3]) -> Mod64Vector3 {
        Mod64Vector3 {
            x: Mf64::new(x, m[0]),
            y: Mf64::new(y, m[1]),
            z: Mf64::new(z, m[2]),
        }
    }
}

/// Implementation of modulo vector addition. **Caution**: This operation is
/// not commutative.
impl Add for Mod64Vector3 {
    type Output = Mod64Vector3;

    fn add(self, other: Mod64Vector3) -> Mod64Vector3 {
        Mod64Vector3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}


// Implement inplace adding a value
impl AddAssign for Mod64Vector3 {
    fn add_assign(&mut self, _rhs: Mod64Vector3) {
        self.x += _rhs.x;
        self.y += _rhs.y;
        self.z += _rhs.z;
    }
}

impl AddAssign<f64> for Mod64Vector3 {
    fn add_assign(&mut self, _rhs: f64) {
        self.x += _rhs;
        self.y += _rhs;
        self.z += _rhs;
    }
}

/// Implementation of modulo vector subtraction. **Caution**: This operation is
/// not commutative.
impl Sub for Mod64Vector3 {
    type Output = Mod64Vector3;

    fn sub(self, rhs: Mod64Vector3) -> Mod64Vector3 {
        Mod64Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

/// Implementation of elementwise modulo vector multiplication. **Caution**:
/// This operation is not commutative.
impl Mul<f64> for Mod64Vector3 {
    type Output = Mod64Vector3;

    fn mul(self, rhs: f64) -> Mod64Vector3 {
        Mod64Vector3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

// implement 2D version

/// 2D modulo vector type
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Mod64Vector2 {
    /// x-coordinate
    pub x: Mf64,
    /// y-coordinate
    pub y: Mf64,
}


impl Mod64Vector2 {
    /// Returns a modulo 3D vector with the given coordinates and quotients.
    pub fn new(x: f64, y: f64, m: [f64; 2]) -> Mod64Vector2 {
        Mod64Vector2 {
            x: Mf64::new(x, m[0]),
            y: Mf64::new(y, m[1]),
        }
    }
}

/// Implementation of modulo vector addition. **Caution**: This operation is
/// not commutative.
impl Add for Mod64Vector2 {
    type Output = Mod64Vector2;

    fn add(self, other: Mod64Vector2) -> Mod64Vector2 {
        Mod64Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

/// Implementation elementwise scalar addition.
impl Add<f64> for Mod64Vector2 {
    type Output = Mod64Vector2;

    fn add(self, other: f64) -> Mod64Vector2 {
        Mod64Vector2 {
            x: self.x + other,
            y: self.y + other,
        }
    }
}

// Implement inplace adding a value
impl AddAssign for Mod64Vector2 {
    fn add_assign(&mut self, _rhs: Mod64Vector2) {
        self.x += _rhs.x;
        self.y += _rhs.y;
    }
}

impl AddAssign<f64> for Mod64Vector2 {
    fn add_assign(&mut self, _rhs: f64) {
        self.x += _rhs;
        self.y += _rhs;
    }
}

/// Implementation of modulo vector subtraction. **Caution**: This operation is
/// not commutative.
impl Sub for Mod64Vector2 {
    type Output = Mod64Vector2;

    fn sub(self, rhs: Mod64Vector2) -> Mod64Vector2 {
        Mod64Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

/// Implementation of elementwise modulo vector multiplication. **Caution**:
/// This operation is not commutative.
impl Mul<f64> for Mod64Vector2 {
    type Output = Mod64Vector2;

    fn mul(self, rhs: f64) -> Mod64Vector2 {
        Mod64Vector2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_multiplication_3() {
        let p = Mod64Vector3::new(0.5, -1.25, 0.75, (1., 1., 1.));
        let p3 = p * 3.;
        assert_eq!(*p3.x.as_ref(), 0.5);
        assert_eq!(*p3.y.as_ref(), 0.25);
        assert_eq!(*p3.z.as_ref(), 0.25);
    }

    #[ignore]
    quickcheck!{
        fn scalar_multiplication_3_qc(x: f64, y: f64, z: f64, rhs: f64) -> bool {
            const DIV: f64 = 3.45;
            let boxsize = [3.45, 3.45, 3.45];
            let a = Mod64Vector3::new(x, y, z, boxsize);
            let b = a * rhs;
            0. <= *b.x.as_ref() && *b.x.as_ref() < DIV && 0. <= *b.y.as_ref() &&
            *b.y.as_ref() < DIV && 0. <= *b.z.as_ref() && *b.z.as_ref() < DIV
        }
    }

    #[test]
    fn scalar_multiplication_2() {
        let p = Mod64Vector2::new(0.5, -1.5, (1., 1.));
        let p3 = p * 3.;
        assert_eq!(*p3.x.as_ref(), 0.5);
        assert_eq!(*p3.y.as_ref(), 0.5);
    }

    #[ignore]
    quickcheck!{
        fn scalar_multiplication_2_qc(x: f64, y: f64, rhs: f64) -> bool {
            let a = Mod64Vector2::new(x, y, (1., 1.));
            let b = a * rhs;
            0. <= *b.x.as_ref() && *b.x.as_ref() < 1. && 0. <= *b.y.as_ref() && *b.y.as_ref() < 1.
        }
    }
}
