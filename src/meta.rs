//! Meta-permutation generators.

use alloc::{boxed::Box, vec::Vec};
use core::{fmt, mem, num::NonZeroUsize};
pub(crate) use internal::Container;

use crate::Perm;

#[cfg(any(target_pointer_width = "8", target_pointer_width = "16"))]
type UFast = usize;
#[cfg(any(
    target_pointer_width = "32",
    target_pointer_width = "64",
    target_pointer_width = "128"
))]
type UFast = u32;

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
        IndexPair(j, unsafe { NonZeroUsize::new_unchecked(i) })
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
        let _ = (&slice[j], &slice[i]);

        // SAFETY: `j` and `i` are checked to be inside `slice`.
        unsafe { self.swap_unchecked(slice) }
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
        // `pj` and `pi` never point to the same element since `j` and `i` are not equal.
        unsafe {
            let pj = ptr.add(j);
            let pi = ptr.add(i);
            mem::swap(&mut *pj, &mut *pi);
        }
    }
}

impl fmt::Display for IndexPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

/// Generic meta-permutation generator.
///
/// The length of a `MetaPerm` may not exceed [`i32::MAX`] on a 32-bit or higher target,
/// or [`isize::MAX`] on a 16-bit or lower target.
#[derive(Debug)]
pub struct MetaPerm<C: Container> {
    container: C,
}

impl<const N: usize> MetaPerm<Const<N>> {
    /// Creates a new `MetaPerm` of constant length `N`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is less than 2 or too large.
    #[inline]
    pub const fn new_const() -> Self {
        assert!(N >= 2 && N >> (UFast::BITS - 1) == 0);
        Self {
            container: Const::INIT,
        }
    }

    /// Creates a new `MetaPerm` of the same length as an array.
    ///
    /// # Panics
    ///
    /// Panics if the array has a length less than 2 or too large.
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
    /// Panics if `n` is less than 2 or too large.
    #[inline]
    pub fn new(n: usize) -> Self {
        assert!(n >= 2 && n >> (UFast::BITS - 1) == 0);
        let mut vec = Vec::<UFast>::with_capacity(n);

        unsafe {
            // SAFETY: The index is always less than the capacity `n`.
            let p = vec.as_mut_ptr();
            for i in 0..n {
                // This won't overflow because we have at least one extra bit.
                *p.add(i) = i as UFast + 1;
            }
            // SAFETY: We just initialized the entire `Vec`.
            vec.set_len(n);
        }

        MetaPerm {
            container: Dyn {
                inner: vec.into_boxed_slice(),
            },
        }
    }
}

impl<C: Container> MetaPerm<C> {
    /// Returns the length of this `MetaPerm`.
    #[inline]
    pub fn len(&mut self) -> usize {
        self.container.len()
    }

    /// Returns an index pair for producing the next unique permutation
    /// **of even index**, or `None` if all permutations are exhausted.
    ///
    /// Permutations of odd index are produced by swapping the 0th and 1st elements.
    #[inline]
    pub fn gen_even(&mut self) -> Option<IndexPair> {
        self.container.gen_even()
    }

    /// Permutes the target with this `MetaPerm` in place.
    ///
    /// This `MetaPerm` will be reset after permuting.
    #[inline]
    pub fn permute<T, P: Perm<T> + ?Sized>(&mut self, target: &mut P, f: impl Fn(&P)) {
        assert!(self.len() == target.as_mut().len(), "length mismatch");
        loop {
            // Even indexes.
            f(target);

            // Swap the 0th and 1st elements for odd indexes.
            // SAFETY: `Container`s assure they have a length no less than 2.
            unsafe { IndexPair::new(0, 1).swap_unchecked(target.as_mut()) }
            f(target);

            if let Some(p) = self.gen_even() {
                // SAFETY: We have checked that the lengths are equal.
                unsafe { p.swap_unchecked(target.as_mut()) }
            } else {
                break;
            }
        }
    }
}

/// Constant-sized container used by `MetaPerm`.
///
/// The const generic `N` indicates the length of permutations.
#[derive(Debug)]
#[repr(C)]
pub struct Const<const N: usize> {
    inner: [UFast; N],
}

impl<const N: usize> Const<N> {
    // [1, 2, ..., N].
    const INIT: Self = Const {
        inner: {
            let mut out = [0; N];
            let mut i = 0;
            while i < N {
                out[i] = i as UFast + 1;
                i += 1;
            }
            out
        },
    };
}

// SAFETY: The entire array is properly initialized in `MetaPerm::new_const`.
unsafe impl<const N: usize> Container for Const<N> {
    #[inline]
    fn len(&self) -> usize {
        N
    }

    #[inline]
    fn ptr(&mut self) -> *mut UFast {
        self.inner.as_mut_ptr()
    }
}

/// Dynamic-sized container used by `MetaPerm`.
#[derive(Debug)]
pub struct Dyn {
    inner: Box<[UFast]>,
}

// SAFETY: The entire slice is properly initialized in `MetaPerm::new`.
unsafe impl Container for Dyn {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn ptr(&mut self) -> *mut UFast {
        self.inner.as_mut_ptr()
    }
}

mod internal {
    use super::*;

    /// Trait for a container used by `MetaPerm`.
    ///
    /// This trait requires two variables, a length `n` and a pointer `p`.
    ///
    /// # Safety
    ///
    /// Implementations must ensure that `n` and `p` satisfy the following conditions:
    ///
    /// * `n` must not be less than 2.
    ///
    /// * `n` and `p` must not be altered at any time.
    ///
    /// * `p` must point to the first element of a valid `usize` array of length `n`.
    ///   The elements in this array, except the first one, must be initialized with
    ///   values from `2` to `n` in order, and may only be altered through `Container::gen`.
    pub unsafe trait Container {
        /// Returns the length.
        fn len(&self) -> usize;

        /// Returns the pointer.
        fn ptr(&mut self) -> *mut UFast;

        #[inline]
        fn gen_even(&mut self) -> Option<IndexPair> {
            let n = self.len();
            let mut p = self.ptr();

            // This loop is perf-sensitive as benchmarked.
            // Ideally for `Const` it yields such asm on x86-64:
            //
            // .LBB3_3:
            //     mov	dword ptr [rsp + 4*rax + 28], eax
            //     mov	edx, dword ptr [rsp + 4*rax + 32]
            //     add	rax, 1
            //     test	edx, edx
            //     je	.LBB3_3
            let mut i: UFast = 2;
            let pi = loop {
                // SAFETY: `p` is never out of bounds since:
                // 1) this loop is only reachable for `n >= 2`;
                // 2) the last element of `p` is initially `n` and never altered,
                //    which, when reached, will immediately break the loop.
                unsafe {
                    p = p.add(1);
                    if *p != 0 {
                        break &mut *p;
                    }
                    *p = i;
                }
                i += 1;
            };

            if i >= n as UFast {
                // All permutations are exhausted.
                return None;
            }

            // Decrement `p[i]` by 1.
            *pi -= 1;
            // If `i` is odd, then let `j = p[i]` otherwise let `j = 0`.
            let j = if i & 1 != 0 { *pi } else { 0 };

            // SAFETY: `i` is non-zero and less than `n` as mentioned above.
            //
            // `j` does not equal `i` and is less than `n` since:
            // 1) `p[i]` was initially `i`, but has been decremented at least once
            //    and is thus less than `i`;
            // 2) `j` is either `p[i]` or 0, which is less than `i` and thus than `n`.
            Some(unsafe { IndexPair::new(j as usize, i as usize) })
        }
    }
}
