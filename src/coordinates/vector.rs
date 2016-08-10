use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use super::modulofloat::Mf64;

#[derive(Copy, Clone)]
pub struct Mod64Vector3 {
    pub x: Mf64,
    pub y: Mf64,
    pub z: Mf64,
}

impl Mod64Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Mod64Vector3 {
        Mod64Vector3 {
            x: Mf64::new(x),
            y: Mf64::new(y),
            z: Mf64::new(z),
        }
    }
}

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

#[derive(Copy, Clone)]
pub struct Mod64Vector2 {
    pub x: Mf64,
    pub y: Mf64,
}

impl Mod64Vector2 {
    pub fn new(x: f64, y: f64) -> Mod64Vector2 {
        Mod64Vector2 {
            x: Mf64::new(x),
            y: Mf64::new(y),
        }
    }
}

impl Add for Mod64Vector2 {
    type Output = Mod64Vector2;

    fn add(self, other: Mod64Vector2) -> Mod64Vector2 {
        Mod64Vector2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Mod64Vector2 {
    type Output = Mod64Vector2;

    fn sub(self, rhs: Mod64Vector2) -> Mod64Vector2 {
        Mod64Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

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
        let p = Mod64Vector3::new(0.5, -1.25, 0.75);
        let p3 = p * 3.;
        assert_eq!(*p3.x.as_ref(), 0.5);
        assert_eq!(*p3.y.as_ref(), 0.25);
        assert_eq!(*p3.z.as_ref(), 0.25);
    }

    #[quickcheck]
    #[ignore]
    fn scalar_multiplication_3_qc(x: f64, y: f64, z: f64, rhs: f64) -> bool {
        let a = Mod64Vector3::new(x, y, z);
        let b = a * rhs;
        0. <= *b.x.as_ref() && *b.x.as_ref() < 1. && 0. <= *b.y.as_ref() && *b.y.as_ref() < 1. &&
        0. <= *b.z.as_ref() && *b.z.as_ref() < 1.
    }

    #[test]
    fn scalar_multiplication_2() {
        let p = Mod64Vector2::new(0.5, -1.5);
        let p3 = p * 3.;
        assert_eq!(*p3.x.as_ref(), 0.5);
        assert_eq!(*p3.y.as_ref(), 0.5);
    }

    #[quickcheck]
    #[ignore]
    fn scalar_multiplication_2_qc(x: f64, y: f64, rhs: f64) -> bool {
        let a = Mod64Vector2::new(x, y);
        let b = a * rhs;
        0. <= *b.x.as_ref() && *b.x.as_ref() < 1. && 0. <= *b.y.as_ref() && *b.y.as_ref() < 1.
    }
}
