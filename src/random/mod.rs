use rand::distributions::normal::StandardNormal;
use rand::{Rng, ThreadRng};

pub struct NormalDistributionIterator {
    rng: ThreadRng,
}

impl NormalDistributionIterator {
    pub fn new() -> NormalDistributionIterator {
        NormalDistributionIterator {
            // caching thread-local random number generator
            rng: ::rand::thread_rng(),
        }
    }

    pub fn sample(&mut self) -> f64 {
        let StandardNormal(x) = self.rng.gen();
        x
    }
}

impl Default for NormalDistributionIterator {
    fn default() -> NormalDistributionIterator {
        NormalDistributionIterator::new()
    }
}

// Implement iterator trait for `NormalDistributionIteratorIterator`
impl Iterator for NormalDistributionIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let StandardNormal(x) = self.rng.gen();
        Some(x)
    }
}
