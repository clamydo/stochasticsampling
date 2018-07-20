// Move unit test into own file
#[cfg(test)]
#[path = "./mod_test.rs"]
mod mod_test;

pub mod modifiers;

use particle::{CosSinOrientation, Particle, ParticleVector};
use BoxSize;

#[derive(Clone, Copy)]
pub struct OriginalParticle {
    pub vector: ParticleVector,
    pub orientation_angles: CosSinOrientation,
}

pub struct LangevinBuilder(OriginalParticle);
pub struct Modification {
    old: OriginalParticle,
    delta: ParticleVector,
}

impl LangevinBuilder {
    pub fn new(p: &Particle) -> LangevinBuilder {
        // Simpler, but calculates trigonometric functions under the hood
        // LangevinBuilder(p.into())

        // cache trigonometric functions for performance
        let cs = CosSinOrientation::from_orientation(&p.orientation);
        LangevinBuilder(OriginalParticle {
            vector: ParticleVector {
                position: p.position.to_vector(),
                orientation: cs.to_orientation_vector(),
            },
            orientation_angles: cs,
        })
    }

    pub fn with(self, f: fn(OriginalParticle, ParticleVector) -> ParticleVector) -> Modification {
        Modification {
            old: self.0,
            delta: f(self.0, ParticleVector::zero()),
        }
    }

    pub fn with_param<T>(
        self,
        f: fn(OriginalParticle, ParticleVector, T) -> ParticleVector,
        p: T,
    ) -> Modification {
        Modification {
            old: self.0,
            delta: f(self.0, ParticleVector::zero(), p),
        }
    }
}

pub struct TimeStep(pub f64);

impl Modification {
    pub fn with(self, f: fn(OriginalParticle, ParticleVector) -> ParticleVector) -> Modification {
        Modification {
            old: self.old,
            delta: f(self.old, self.delta),
        }
    }

    pub fn with_param<T>(
        self,
        f: fn(OriginalParticle, ParticleVector, T) -> ParticleVector,
        p: T,
    ) -> Modification {
        Modification {
            old: self.old,
            delta: f(self.old, self.delta, p),
        }
    }

    pub fn conditional_with(
        self,
        condition: bool,
        f: fn(OriginalParticle, ParticleVector) -> ParticleVector,
    ) -> Modification {
        if condition {
            Modification {
                old: self.old,
                delta: f(self.old, self.delta),
            }
        } else {
            self
        }
    }

    pub fn conditional_with_param<T>(
        self,
        condition: bool,
        f: fn(OriginalParticle, ParticleVector, T) -> ParticleVector,
        p: Option<T>,
    ) -> Modification {
        if condition {
            match p {
                Some(p) => Modification {
                    old: self.old,
                    delta: f(self.old, self.delta, p),
                },
                None => self,
            }
        } else {
            self
        }
    }

    pub fn step(self, timestep: &TimeStep) -> Modification {
        Modification {
            old: self.old,
            delta: self.delta * timestep.0,
        }
    }

    pub fn finalize(self, bs: &BoxSize) -> Particle {
        let mut p = Particle::from(self.old.vector + self.delta);
        p.pbc(bs);
        p
    }
}
