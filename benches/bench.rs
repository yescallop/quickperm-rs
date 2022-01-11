#![feature(test)]

extern crate test;

use quickperm::meta::{Const, MetaPerm};
use test::{black_box, Bencher};

macro_rules! gen_perm {
    ($group:ident, $($x:literal),+ $(,)?) => {
        $(
            paste::paste! {
                #[bench]
                fn [<perm_ $x>](b: &mut Bencher) {
                    b.iter(|| {
                        let mut perm = MetaPerm::<Const<$x>>::new_const();
                        while let Some(x) = perm.gen() {
                            black_box(x);
                        }
                    });
                }
            }
        )+
    };
}

gen_perm!(group, 3, 4, 5, 6, 7, 8, 9);
