extern crate vergen;

use vergen::{SHORT_SHA, vergen};

fn main() {
    vergen(SHORT_SHA).unwrap();
}
