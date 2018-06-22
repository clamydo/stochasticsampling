use std::ops::Shr;


struct LangevinBuilder {
    orig: i32,
}

#[derive(Debug)]
struct Wrapped((i32, u32));

impl<F> Shr<F> for LangevinBuilder
    where F: FnOnce(i32, u32) -> (i32, u32)
{
    type Output = Wrapped;

    fn shr(self, f: F) -> Wrapped {
        Wrapped(f(self.orig, 0))
    }
}


impl<F> Shr<F> for Wrapped
    where F: FnOnce(i32, u32) -> (i32, u32)
{
    type Output = Wrapped;

    fn shr(self, f: F) -> Wrapped {
        Wrapped(f((self.0).0, (self.0).1))
    }
}

fn a(state: i32, delta: u32) -> (i32, u32) {
    (state, delta + 2)
}

fn b(state: i32, delta: u32) -> (i32, u32) {
    (state, delta * 2)
}

impl LangevinBuilder {
    fn with(self, f: fn(i32, u32) -> (i32, u32)) -> Wrapped {
        Wrapped(f(self.orig, 0))
    }
}
impl Wrapped {
    fn with(self, f: fn(i32, u32) -> (i32, u32)) -> Wrapped {
        Wrapped(f((self.0).0, (self.0).1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langevin_builder() {

        let l = LangevinBuilder { orig: 15 } >> a >> b;
        println!("{:?}", l);
        let m = LangevinBuilder { orig: 15 } >> b >> a;
        println!("{:?}", m);

        let l = LangevinBuilder { orig: 15 };
        let l = l.with(a).with(b);
        println!("{:?}", l);
        let l = LangevinBuilder { orig: 15 };
        let l = l.with(b).with(a);
        println!("{:?}", l);

        assert!(false);
    }
}
