use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use super::modulofloat::Mf64;

#[derive(Copy, Clone)]
pub struct Mod64Vector3 {
    pub x: Mf64,
    pub y: Mf64,
    pub z: Mf64,
    pub mx: f64, // divisor
    pub my: f64, // divisor
    pub mz: f64, // divisor
}

impl Mod64Vector3 {
    /// Marked as unsafe, beacuse having `m <= 0` will lead to unwanted results.
    pub unsafe fn new(x: f64, y: f64, z: f64, m: (f64, f64, f64)) -> Mod64Vector3 {
        Mod64Vector3 {
            x: Mf64::new(x, m.0),
            y: Mf64::new(y, m.1),
            z: Mf64::new(z, m.2),
            mx: m.0,
            my: m.1,
            mz: m.2,
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
            mx: self.mx,
            my: self.my,
            mz: self.mz,
        }
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
            mx: self.mx,
            my: self.my,
            mz: self.mz,
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
            mx: self.mx,
            my: self.my,
            mz: self.mz,
        }
    }
}

// implement 2D version

#[derive(Copy, Clone)]
pub struct Mod64Vector2 {
    pub x: Mf64,
    pub y: Mf64,
    pub mx: f64,
    pub my: f64,
}

impl Mod64Vector2 {
    /// Marked as unsafe, beacuse having `m <= 0` will lead to unwanted results.
    pub unsafe fn new(x: f64, y: f64, m: (f64, f64)) -> Mod64Vector2 {
        Mod64Vector2 {
            x: Mf64::new(x, m.0),
            y: Mf64::new(y, m.1),
            mx: m.0,
            my: m.1,
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
            mx: self.mx,
            my: self.my,
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
            mx: self.mx,
            my: self.my,
        }
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
            mx: self.mx,
            my: self.my,
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
            mx: self.mx,
            my: self.my,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_multiplication_3() {
        let p = unsafe { Mod64Vector3::new(0.5, -1.25, 0.75, (1., 1., 1.)) };
        let p3 = p * 3.;
        assert_eq!(*p3.x.as_ref(), 0.5);
        assert_eq!(*p3.y.as_ref(), 0.25);
        assert_eq!(*p3.z.as_ref(), 0.25);
    }

    #[quickcheck]
    #[ignore]
    fn scalar_multiplication_3_qc(x: f64, y: f64, z: f64, rhs: f64) -> bool {
        const DIV: f64 = 3.45;
        let boxsize = (3.45, 3.45, 3.45);
        let a = unsafe { Mod64Vector3::new(x, y, z, boxsize) };
        let b = a * rhs;
        0. <= *b.x.as_ref() && *b.x.as_ref() < DIV && 0. <= *b.y.as_ref() &&
        *b.y.as_ref() < DIV && 0. <= *b.z.as_ref() && *b.z.as_ref() < DIV
    }

    #[test]
    fn scalar_multiplication_2() {
        let p = unsafe { Mod64Vector2::new(0.5, -1.5, (1., 1.)) };
        let p3 = p * 3.;
        assert_eq!(*p3.x.as_ref(), 0.5);
        assert_eq!(*p3.y.as_ref(), 0.5);
    }

    #[quickcheck]
    #[ignore]
    fn scalar_multiplication_2_qc(x: f64, y: f64, rhs: f64) -> bool {
        let a = unsafe { Mod64Vector2::new(x, y, (1., 1.)) };
        let b = a * rhs;
        0. <= *b.x.as_ref() && *b.x.as_ref() < 1. && 0. <= *b.y.as_ref() && *b.y.as_ref() < 1.
    }
}
