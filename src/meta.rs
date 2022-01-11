//! Meta-permutation generators.

use alloc::vec::Vec;
use core::{
    fmt,
    mem::{self, ManuallyDrop},
    num::NonZeroUsize,
};
pub(crate) use internal::Meta;

/// A pair of distinct indexes.
///
/// An `IndexPair` is associated with a certain linear data structure,
/// in which the corresponding pair of elements may be swapped to produce
/// a unique permutation within that structure.
#[derive(Debug, Clone, Copy)]
pub struct IndexPair(usize, NonZeroUsize);

impl IndexPair {
    /// Creates an `IndexPair` from a pair of indexes.
    ///
    /// # Safety
    ///
    /// * `j` and `i` must not be equal.
    ///
    /// * `i` must not be zero.
    #[inline]
    pub unsafe fn new(j: usize, i: usize) -> IndexPair {
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
    /// Panics if either of the indexes is out of bounds.
    #[inline]
    pub fn swap<T>(self, slice: &mut [T]) {
        let (j, i) = self.get();
        assert!(j < slice.len() && i < slice.len(), "index out of bounds");

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
    /// Behavior is undefined if either of the indexes is out of bounds.
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

impl fmt::Display for IndexPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

/// Generic meta-permutation generator.
#[derive(Debug)]
pub struct MetaPerm<M: Meta> {
    meta: M,
}

impl<const N: usize> MetaPerm<Const<N>> {
    /// Creates a new `MetaPerm` of constant length `N`.
    ///
    /// # Compile-Time Errors
    ///
    /// An `N` greater than `u8::MAX` will trigger a compile-time error of overflow.
    #[inline]
    pub const fn new_const() -> Self {
        MetaPerm {
            meta: Const {
                i: 1,
                p_headless: Const::<N>::P_HEADLESS,
            },
        }
    }

    /// Creates a new `MetaPerm` of the same length as an array.
    ///
    /// # Compile-Time Errors
    ///
    /// An array of length greater than `u8::MAX` will trigger a compile-time error of overflow.
    #[inline]
    pub const fn from_array<T>(_arr: &[T; N]) -> Self {
        Self::new_const()
    }
}

impl MetaPerm<Dyn> {
    /// Creates a new `MetaPerm` of dynamic length `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than `u8::MAX`.
    #[inline]
    pub fn new(n: usize) -> Self {
        assert!(n <= u8::MAX as usize);

        let mut p = ManuallyDrop::new(Vec::<u8>::with_capacity(n + 1));
        let p = p.as_mut_ptr();
        // SAFETY: The capacity is `n + 1`, exactly the number of bytes written.
        unsafe {
            for i in 0..n + 1 {
                *p.add(i) = i as u8;
            }
        }

        MetaPerm {
            meta: Dyn {
                p,
                n: n as u8,
                i: 1,
            },
        }
    }
}

impl<M: Meta> MetaPerm<M> {
    /// Returns an index pair for producing the next unique permutation, or `None`
    /// if all permutations are exhausted.
    #[inline]
    pub fn gen(&mut self) -> Option<IndexPair> {
        self.meta.gen()
    }
}

/// Constant-sized meta-permutation generator.
///
/// The const generic `N` indicates the length of permutations, which is at most
/// `u8::MAX` since the inner implementation uses a `u8` array to store the state.
/// This constraint, if violated, will trigger a compile-time error of overflow.
#[derive(Debug)]
#[repr(C)]
pub struct Const<const N: usize> {
    // FIXME: Make `i` fit in a register.
    i: u8,
    p_headless: [u8; N],
}

impl<const N: usize> Const<N> {
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
}

// SAFETY: A `Const<N>` can be interpreted as a valid `u8` array of length `N + 1`,
// because `Const<N>` has a size of `N + 1` and an alignment of 1.
// The entire array is properly initialized in `MetaPerm::new_const`.
unsafe impl<const N: usize> Meta for Const<N> {
    #[inline]
    fn get_nip(&mut self) -> (u8, u8, *mut u8) {
        let p = self as *mut _ as *mut u8;
        (N as u8, self.i, p)
    }

    #[inline]
    fn set_i(&mut self, i: u8) {
        self.i = i;
    }
}

/// Dynamic-sized meta-permutation generator.
#[derive(Debug)]
pub struct Dyn {
    p: *mut u8,
    n: u8,
    i: u8,
}

impl Drop for Dyn {
    #[inline]
    fn drop(&mut self) {
        let size = self.n as usize + 1;
        // SAFETY: `p` is allocated via `Vec` with capacity `n + 1`.
        // The entire array is properly initialized in `MetaPerm::new`.
        unsafe {
            Vec::from_raw_parts(self.p, size, size);
        }
    }
}

// SAFETY: `p` is allocated via `Vec` with capacity `n + 1`.
// The entire array is properly initialized in `MetaPerm::new`.
unsafe impl Meta for Dyn {
    #[inline]
    fn get_nip(&mut self) -> (u8, u8, *mut u8) {
        (self.n, self.i, self.p)
    }

    #[inline]
    fn set_i(&mut self, i: u8) {
        self.i = i;
    }
}

mod internal {
    use super::IndexPair;

    /// Helper trait for a meta-permutation generator.
    ///
    /// This trait requires three variables `n`, `i`, `p`.
    ///
    /// # Safety
    ///
    /// Implementations must ensure that `n`, `i`, `p` satisfy the following conditions:
    ///
    /// * `n` and `p` must not be altered at any time.
    ///
    /// * `i` must be initialized with `1` and may only be altered through `Meta::gen`.
    ///
    /// * `p` must point to the first element of a valid `u8` array of length `n + 1`.
    ///   The elements in this array, except the first one, must be initialized with
    ///   values from `1` to `n` in order, and may only be altered through `Meta::gen`.
    pub unsafe trait Meta {
        /// Returns `(n, i, p)`.
        fn get_nip(&mut self) -> (u8, u8, *mut u8);
        /// Sets `i`.
        fn set_i(&mut self, i: u8);

        #[inline]
        fn gen(&mut self) -> Option<IndexPair> {
            let (n, i, p) = self.get_nip();
            if i >= n {
                // All permutations are exhausted.
                return None;
            }
            let i = i as usize;

            // Decrement `p[i]` by 1.
            // SAFETY: `i` is checked to be less than `n`.
            let pi = unsafe { &mut *p.add(i) };
            *pi -= 1;
            // If `i` is odd, then let `j = p[i]` otherwise let `j = 0`.
            let j = if i & 1 != 0 { *pi as usize } else { 0 };

            // SAFETY: `i` is non-zero and less than `n` since:
            // 1) `i` was initially 1, and after each iteration we have `1 <= i <= n`;
            // 2) this function would've returned when `i == n`.
            //
            // `j` does not equal `i` and is less than `n` since:
            // 1) `p[i]` was initially `i`, but has been decremented at least once
            //    and is thus less than `i`;
            // 2) `j` is either `p[i]` or 0, which is less than `i` and thus than `n`.
            let out = unsafe { IndexPair::new(j, i) };

            let mut i = 1;
            loop {
                // SAFETY: `i` is never greater than `n` since:
                // 1) this loop is only reachable for `n >= 2`;
                // 2) the last element of `p` is initially `n` and never altered, which,
                //    when reached, will immediately break the loop with `i == n`.
                let pi = unsafe { &mut *p.add(i) };
                if *pi != 0 {
                    break;
                }
                *pi = i as u8;
                i += 1;
            }
            // Here we have `1 <= i <= n` since `i` was incremented from 1.
            self.set_i(i as u8);

            Some(out)
        }
    }
}
