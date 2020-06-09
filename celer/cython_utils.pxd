# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause
from cython cimport floating
cimport numpy as np

ctypedef np.uint8_t uint8

cdef int LASSO
cdef int LOGREG

cdef floating ST(floating, floating) nogil

cdef floating dual(int, int, floating, floating, floating *, floating *) nogil
cdef floating primal(int, floating, int, floating *, floating *,
                            int, floating *) nogil
cdef void create_dual_pt(int, int, floating, floating *, floating *, floating *) nogil

cdef floating Nh(floating) nogil
cdef floating sigmoid(floating) nogil
# cdef floating log_1pexp(floating) nogil

cdef floating fdot(int *, floating *, int *, floating *, int *) nogil
cdef floating fasum(int *, floating *, int *) nogil
cdef void faxpy(int *, floating *, floating *, int *, floating *, int *) nogil
cdef floating fnrm2(int * , floating *, int *) nogil
cdef void fcopy(int *, floating *, int *, floating *, int *) nogil
cdef void fscal(int *, floating *, floating *, int *) nogil

cdef void fposv(char *, int *, int *, floating *,
                     int *, floating *, int *, int *) nogil

cdef int create_accel_pt(
    int, int, int, int, floating, floating *, floating *,
    floating *, floating[:, :], floating[:, :], floating[:], floating[:])


cpdef void compute_Xw(
    bint, int, floating[:], floating[:],
    floating[:], bint, floating[::1, :],
    floating[:], int[:], int[:], floating[:])


cpdef void compute_norms_X_col(
    bint, floating[:], int, floating[::1, :],
    floating[:], int[:], int[:], floating[:])


cdef floating compute_dual_scaling(
        bint, floating[:], floating[::1, :], floating[:],
        int[:], int[:], int, int[:], int[:], floating[:], bint, bint) nogil


cdef void set_prios(
    bint, floating[:], floating[::1, :], floating[:], int[:],
    int[:], floating[:], floating[:], uint8 *, floating, int *, bint) nogil