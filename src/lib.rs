//! # quickperm
//!
//! A minimum-overhead Rust implementation of the [QuickPerm Algorithm](https://www.quickperm.org/).
//!
//! This crate provides quick approaches for generating permutations of both constant and dynamic length.
//!
//! ## To-Do List
//!
//! * Idiomatic way to break from permuting (macros?).
//!
//! * *k*-permutations.
//!
//! ## Examples
//!
//! The following example permutes the array in place and prints all possible permutations:
//!
//! ```
//! use quickperm::Perm;
//! let mut arr = [1, 2, 3];
//! arr.permute(|perm| println!("{:?}", perm));
//! ```

#![no_std]
#![warn(missing_docs, rust_2018_idioms)]

extern crate alloc;

pub mod meta;

use meta::*;

/// Trait for permuting arrays and slices.
pub trait Perm<T, C: Container>: internal::Sealed {
    /// Walks through all possible permutations in place.
    fn permute(&mut self, f: impl Fn(&Self));
}

impl<T, const N: usize> Perm<T, Const<N>> for [T; N] {
    #[inline]
    fn permute(&mut self, f: impl Fn(&[T; N])) {
        let mut mp = MetaPerm::from_array(self);
        loop {
            f(self);
            if let Some(p) = mp.gen() {
                // SAFETY: We created `mp` with the same length as the array.
                unsafe { p.swap_unchecked(self) }
            } else {
                break;
            }
        }
    }
}

impl<T> Perm<T, Dyn> for [T] {
    #[inline]
    fn permute(&mut self, f: impl Fn(&[T])) {
        let mut mp = MetaPerm::new(self.len());
        loop {
            f(self);
            if let Some(p) = mp.gen() {
                // SAFETY: We created `mp` with the same length as the slice.
                unsafe { p.swap_unchecked(self) }
            } else {
                break;
            }
        }
    }
}

mod internal {
    pub trait Sealed {}

    impl<T> Sealed for [T] {}
    impl<T, const N: usize> Sealed for [T; N] {}
}
