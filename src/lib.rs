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
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod meta;

use meta::*;

/// Trait for permuting arrays and slices.
pub trait Perm<T>: internal::Sealed + AsMut<[T]> {
    /// Permutes the target in place.
    fn permute(&mut self, f: impl Fn(&Self));
}

impl<T, const N: usize> Perm<T> for [T; N] {
    #[inline]
    fn permute(&mut self, f: impl Fn(&[T; N])) {
        if self.len() < 2 {
            f(self)
        } else {
            let mut mp = MetaPerm::from_array(self);
            mp.permute(self, f);
        }
    }
}

impl<T> Perm<T> for [T] {
    #[inline]
    fn permute(&mut self, f: impl Fn(&[T])) {
        if self.len() < 2 {
            f(self)
        } else {
            let mut mp = MetaPerm::new(self.len());
            mp.permute(self, f);
        }
    }
}

mod internal {
    pub trait Sealed {}

    impl<T> Sealed for [T] {}
    impl<T, const N: usize> Sealed for [T; N] {}
}
