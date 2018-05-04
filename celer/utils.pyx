# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

cimport cython

from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy, dscal
from scipy.linalg.cython_blas cimport sdot, sasum, saxpy, snrm2, scopy, sscal
from libc.math cimport fabs
from cython cimport floating


cdef inline double fmax(double x, double y) nogil:
    return x if x > y else y


cdef inline double fmin(double x, double y) nogil:
    return y if x > y else y


# cdef inline double fsign(double x) nogil :
#     if x == 0.:
#         return 0.
#     elif x > 0.:
#         return 1.
#     else:
#         return - 1.


cdef inline double ST(double u, double x) nogil:
    if x > u:
        return x - u
    elif x < - u:
        return x + u
    else:
        return 0


# cdef double abs_max(int n, double * a) nogil:
#     cdef int ii
#     cdef double m = 0.
#     cdef double d
#     for ii in range(n):
#         d = fabs(a[ii])
#         if d > m:
#             m = d
#     return m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double primal_value(double alpha, int n_samples, double * R,
                         int n_features, double * w) nogil:
    cdef int inc = 1
    # regularization term: alpha ||w||_1
    cdef double p_obj = alpha * dasum(&n_features, w, &inc)
    # R is passed as a pointer so no need to & it
    p_obj += ddot(&n_samples, R, &inc, R, &inc) / (2. * n_samples)
    return p_obj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double dual_value(int n_samples, double alpha, double norm_y2,
                       double * theta, double * y) nogil:
    """Theta must be feasible"""
    cdef int i
    cdef double d_obj = 0.
    for i in range(n_samples):
        d_obj -= (y[i] / (alpha * n_samples) - theta[i]) ** 2
    d_obj *= 0.5 * alpha ** 2 * n_samples
    d_obj += norm_y2 / (2. * n_samples)
    return d_obj


cdef floating fused_dot(int * n, floating * x, int * inc1, floating * y,
                        int * inc2) nogil:
    if floating is double:
        return ddot(n, x, inc1, y, inc2)
    else:
        return sdot(n, x, inc1, y, inc2)


cdef floating fused_asum(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dasum(n, x, inc)
    else:
        return sasum(n, x, inc)

cdef void fused_axpy(int * n, floating * alpha, floating * x, int * incx,
                     floating * y, int * incy) nogil:
     if floating is double:
         daxpy(n, alpha, x, incx, y, incy)
     else:
         saxpy(n, alpha, x, incx, y, incy)


cdef floating fused_nrm2(int * n, floating * x, int * inc) nogil:
     if floating is double:
         return dnrm2(n, x, inc)
     else:
         return snrm2(n, x, inc)


cdef void fused_copy(int * n, floating * x, int * incx, floating * y,
                     int * incy) nogil:
    if floating is double:
        dcopy(n, x, incx, y, incy)
    else:
        scopy(n, x, incx, y, incy)


cdef void fused_scal(int * n, floating * alpha, floating * x,
                     int * incx) nogil:
    if floating is double:
        dscal(n, alpha, x, incx)
    else:
        sscal(n, alpha, x, incx)
