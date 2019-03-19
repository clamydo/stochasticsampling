#![allow(clippy::float_cmp, clippy::unreadable_literal)]
use super::*;
use crate::particle::Particle;
use crate::test_helper::equal_floats;
use crate::Float;
use crate::{BoxSize, GridSize};

#[test]
fn test_polarization_from_distribution() {
    let bs = BoxSize {
        x: 3.,
        y: 3.,
        z: 1.,
    };
    let gs = GridSize {
        x: 3,
        y: 3,
        z: 1,
        phi: 35,
        theta: 5,
    };

    let gw = GridWidth::new(gs, bs);

    let p1 = Particle::new(0.0, 0.0, 0.0, PI, PI / 2., &bs);
    let p2 = Particle::new(1.5, 2.5, 0.0, 0.269279370307697, 0.942477796076938, &bs);
    let mut d = Distribution::new(gs, bs);
    let p = vec![p1, p2];
    let n = p.len() as Float;
    d.sample_from(&p);

    let mut p = DirectorField::new(gs, gw);

    p.from_distribution(&d, None);

    assert_eq!(p.field[[0, 0, 0, 0]].re, -1.0 / n);
    assert_eq!((p.field[[1, 0, 0, 0]].re * 10e14).round(), 0.0 / n);
    assert_eq!((p.field[[2, 0, 0, 0]].re * 10e14).round(), 0.0 / n);

    // set singular entry to zero
    p.field
        .slice_mut(s![.., 0, 0, 0])
        .map_inplace(|v| *v = (0.0).into());

    assert!(equal_floats(
        p.field[[0, 1, 2, 0]].re,
        0.7798623362492354 / n
    ));
    assert!(equal_floats(
        p.field[[1, 1, 2, 0]].re,
        0.2152283291933436 / n
    ));
    assert!(equal_floats(
        p.field[[2, 1, 2, 0]].re,
        0.5877852522924731 / n
    ));

    // set singular entry to zero
    p.field
        .slice_mut(s![.., 1, 2, 0])
        .map_inplace(|v| *v = (0.0).into());

    // check the rest
    for (i, v) in p.field.iter().enumerate() {
        assert!(
            *v == (0.0).into(),
            "Value at index {} should be zero, but is {}.",
            i,
            v
        );
    }
}
