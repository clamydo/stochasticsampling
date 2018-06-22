// Move unit test into own file
#[cfg(test)]
#[path = "./mod_test.rs"]
mod mod_test;

pub mod modifiers;

use simulation::particle::{Orientation, Particle, ParticleVector, Position};
use simulation::settings::BoxSize;

pub struct LangevinBuilder(ParticleVector);
pub struct Modification {
    old: ParticleVector,
    delta: ParticleVector,
}

impl LangevinBuilder {
    pub fn new(p: &Particle) -> LangevinBuilder {
        LangevinBuilder(p.into())
    }

    pub fn with(self, f: fn(ParticleVector, ParticleVector) -> ParticleVector) -> Modification {
        Modification {
            old: self.0,
            delta: f(self.0, ParticleVector::zero()),
        }
    }

    pub fn with_param<T>(
        self,
        f: fn(ParticleVector, ParticleVector, T) -> ParticleVector,
        p: T,
    ) -> Modification {
        Modification {
            old: self.0,
            delta: f(self.0, ParticleVector::zero(), p),
        }
    }
}

pub struct TimeStep(f64);

impl Modification {
    pub fn with(self, f: fn(ParticleVector, ParticleVector) -> ParticleVector) -> Modification {
        Modification {
            old: self.old,
            delta: f(self.old, self.delta),
        }
    }

    pub fn with_param<T>(
        self,
        f: fn(ParticleVector, ParticleVector, T) -> ParticleVector,
        p: T,
    ) -> Modification {
        Modification {
            old: self.old,
            delta: f(self.old, self.delta, p),
        }
    }

    pub fn step(self, timestep: TimeStep) -> Modification {
        Modification {
            old: self.old,
            delta: self.delta * timestep.0,
        }
    }

    pub fn finalize(self, bs: BoxSize) -> Particle {
        let mut p = Particle::from(self.old + self.delta);
        p.pbc(bs);
        p
    }
}
