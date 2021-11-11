//! # quickperm
//!
//! A minimum-overhead Rust implementation of the [QuickPerm Algorithm](https://www.quickperm.org/).
//!
//! This crate provides quick approaches for generating permutations of both constant and dynamic size.

#![no_std]
#![warn(missing_docs, rust_2018_idioms)]

/// Walks through all possible permutations within an array.
/// 
/// # Examples
/// 
/// The following example prints all possible permutations of [1, 2, 3]:
/// 
/// ```
/// use quickperm::permute_array;
/// 
/// let mut arr = [1, 2, 3];
/// 
/// permute_array!(arr => {
///     println!("{:?}", arr);
/// });
/// ```
/// 
/// Note that it is not allowed to mutate the permuted array from within the block:
/// 
/// ```compile_fail
/// use quickperm::permute_array;
/// 
/// let mut arr = [1, 2, 3];
/// 
/// permute_array!(arr => {
///     println!("{:?}", arr);
///     arr[0] = 0; 
/// });
/// ```
/// 
/// This restriction avoids producing the wrong result by mistake.
#[macro_export]
macro_rules! permute_array {
    ($x:ident => $body:block) => {
        let mut perm = $crate::ConstPerm::from_array(&$x);
        loop {
            let _lock = &$x;
            $body;
            drop(_lock);
            if !perm.permute(&mut $x) {
                break;
            }
        }
    };
}

use core::{fmt, mem, num::NonZeroUsize, slice};

/// A pair of distinct indexes with an optional upper bound.
///
/// An `IndexPair` is associated with a certain linear data structure,
/// in which the corresponding pair of elements may be swapped to produce
/// a unique permutation within that structure.
///
/// The const generic `N` indicates the upper bound of indexes (exclusive),
/// where `N = 0` indicates no upper bound.
#[derive(Debug, Clone, Copy)]
pub struct IndexPair<const N: usize>(usize, NonZeroUsize);

impl<const N: usize> IndexPair<N> {
    /// Creates an `IndexPair` from a pair of indexes.
    ///
    /// # Safety
    ///
    /// * `j` and `i` must not be equal.
    ///
    /// * `i` must not be zero.
    ///
    /// * Both `j` and `i` must be less than `N` if `N != 0`.
    unsafe fn new(j: usize, i: usize) -> IndexPair<N> {
        // SAFETY: The caller must ensure that `i` is non-zero.
        IndexPair(j, NonZeroUsize::new_unchecked(i))
    }

    /// Returns the pair of indexes as a tuple.
    #[inline]
    pub fn get(self) -> (usize, usize) {
        (self.0, self.1.get())
    }

    /// Swaps the corresponding pair of elements in a slice.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len()` does not equal the upper bound of indexes,
    /// or if any of the indexes is out of bounds.
    ///
    /// This function simply panics when called with a slice of length greater than the
    /// upper bound of indexes, because it might suggest some mistake in the code.
    /// If such behavior is really needed, shortens the slice to fit the bound instead.
    #[inline]
    pub fn swap<T>(self, slice: &mut [T]) {
        let (j, i) = self.get();
        if N != 0 {
            assert!(N == slice.len(), "slice.len() does not fit the upper bound");
            // Here `j` and `i` are less than `N` and thus inside `slice`.
        } else {
            assert!(j < slice.len() && i < slice.len(), "index out of bounds");
        }

        let ptr = slice.as_mut_ptr();
        // SAFETY: `j` and `i` are checked to be inside `slice`, so
        // the resulting pointers `pj` and `pi` are valid and aligned.
        //
        // `pj` and `pi` never point to the same element since `j` and `i` are not equal.
        unsafe {
            let pj = ptr.add(j);
            let pi = ptr.add(i);
            mem::swap(&mut *pj, &mut *pi);
        }
    }

    /// Swaps the corresponding pair of elements in a slice,
    /// without doing bounds checking.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if any of the following conditions is violated:
    ///
    /// * The length of the slice must not be less than the upper bound of indexes.
    ///   This is forbidden even if neither of the indexes is out of bounds.
    ///
    /// * Both indexes must not be out of bounds.
    #[inline]
    pub unsafe fn swap_unchecked<T>(self, slice: &mut [T]) {
        let (j, i) = self.get();
        let ptr = slice.as_mut_ptr();
        // SAFETY: The caller must ensure that `j` and `i` are inside `slice`.
        //
        // `pj` and `pi` never point to the same element since `j` and `i` are not equal.
        let pj = ptr.add(j);
        let pi = ptr.add(i);
        mem::swap(&mut *pj, &mut *pi);
    }
}

impl<const N: usize> fmt::Display for IndexPair<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

/// Constant-sized permutation generator.
///
/// The const generic `N` indicates the length of permutations, which is at most
/// `u8::MAX` since the inner implementation uses a `u8` array to store the state.
/// This constraint, if violated, will trigger a compile-time error of overflow.
#[derive(Debug)]
#[repr(C)]
pub struct ConstPerm<const N: usize> {
    i: u8,
    p_headless: [u8; N],
}

impl<const N: usize> ConstPerm<N> {
    // Our lovely array `p` but headless.
    // p_headless = [1, 2, 3, ..., N]
    #[deny(const_err)]
    const P_HEADLESS: [u8; N] = {
        let mut out = [0; N];
        let mut i = 0;
        while i < N {
            out[i] = i as u8 + 1;
            i += 1;
        }
        out
    };

    /// Creates a new `ConstPerm` of length `N`.
    ///
    /// # Compile-Time Errors
    ///
    /// An `N` greater than `u8::MAX` will trigger a compile-time error of overflow.
    pub const fn new() -> ConstPerm<N> {
        ConstPerm {
            i: 1,
            p_headless: Self::P_HEADLESS,
        }
    }

    /// Creates a new `ConstPerm` of the same length as an array.
    ///
    /// # Compile-Time Errors
    ///
    /// An array of length greater than `u8::MAX` will trigger a compile-time error of overflow.
    pub const fn from_array<T>(_arr: &[T; N]) -> ConstPerm<N> {
        ConstPerm::<N>::new()
    }

    /// Returns an index pair for producing the next unique permutation, or `None`
    /// if all permutations are exhausted.
    pub fn next(&mut self) -> Option<IndexPair<N>> {
        let i = self.i as usize;
        if i >= N {
            // All permutations are exhausted.
            return None;
        }

        // Give the fancy head back to `p`.
        // p = [i, 1, 2, ..., N]
        //
        // SAFETY: It is safe to cast `self` to a mutable `u8` slice of length `N + 1`
        // because `ConstPerm<N>` has a size of `N + 1` and an alignment of 1.
        let p = unsafe { slice::from_raw_parts_mut(self as *mut _ as *mut u8, N + 1) };

        // Decrement `p[i]` by 1.
        let pi = &mut p[i];
        *pi -= 1;
        // If `i` is odd, then let `j = p[i]` otherwise let `j = 0`.
        let j = if i & 1 != 0 { *pi as usize } else { 0 };

        // SAFETY: `i` is non-zero and less than `N` since:
        // 1) `i` was initially 1, and after each iteration we have `1 <= i <= N`;
        // 2) this function would've returned when `i == N`.
        //
        // `j` does not equal `i` and is less than `N` since:
        // 1) `p[i]` was initially `i`, but has been decremented at least once
        //    and is thus less than `i`;
        // 2) `j` is either `p[i]` or 0, which is less than `i` and thus `N`.
        let out = unsafe { IndexPair::new(j, i) };

        let mut i = 1;
        loop {
            // SAFETY: `i` is never greater than `N` since:
            // 1) this loop is only reachable for `N >= 2`;
            // 2) the last element of `p` is initially `N` and never altered, which,
            //    when reached, will immediately break the loop with `i == N`.
            let pi = unsafe { p.get_unchecked_mut(i) };
            if *pi != 0 {
                break;
            }
            *pi = i as u8;
            i += 1;
        }
        // Here we have `1 <= i <= N` since `i` was incremented from 1.

        // Now drop `p` because it would be UB to set `self.i` with `p` in scope.
        drop(p);
        // Although we could just set `p[0]`, it is clearer to set the field instead.
        self.i = i as u8;
        Some(out)
    }

    /// Swaps two elements in a slice to produce the next unique permutation.
    ///
    /// This function returns `true` if a next permutation is produced,
    /// or `false` if all permutations are exhausted.
    ///
    /// # Panics
    ///
    /// Panics if `slice.len() != N`.
    ///
    /// This function simply panics when called with a slice of length greater than `N`,
    /// because it might suggest some mistake in the code.
    /// If such behavior is really needed, shortens the slice to fit the bound instead.
    pub fn permute<T>(&mut self, slice: &mut [T]) -> bool {
        match self.next() {
            Some(i) => {
                i.swap(slice);
                true
            }
            None => false,
        }
    }
}

extern crate alloc;

use alloc::boxed::Box;

// TODO: A few things to do before making it public:
// 1) determine whether another integer type should be used here;
// 2) finish the impl and the doc.
#[allow(unused)]
#[derive(Debug)]
struct DynPerm {
    p: Box<[usize]>,
    i: usize,
}
