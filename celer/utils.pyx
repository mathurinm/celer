# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

cimport cython

from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy, dscal
from scipy.linalg.cython_blas cimport sdot, sasum, saxpy, snrm2, scopy, sscal
from scipy.linalg.cython_lapack cimport sposv, dposv
from libc.math cimport fabs
from cython cimport floating


cdef floating fdot(int * n, floating * x, int * inc1, floating * y,
                        int * inc2) nogil:
    if floating is double:
        return ddot(n, x, inc1, y, inc2)
    else:
        return sdot(n, x, inc1, y, inc2)


cdef floating fasum(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dasum(n, x, inc)
    else:
        return sasum(n, x, inc)


cdef void faxpy(int * n, floating * alpha, floating * x, int * incx,
                     floating * y, int * incy) nogil:
    if floating is double:
        daxpy(n, alpha, x, incx, y, incy)
    else:
        saxpy(n, alpha, x, incx, y, incy)


cdef floating fnrm2(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dnrm2(n, x, inc)
    else:
        return snrm2(n, x, inc)


cdef void fcopy(int * n, floating * x, int * incx, floating * y,
                     int * incy) nogil:
    if floating is double:
        dcopy(n, x, incx, y, incy)
    else:
        scopy(n, x, incx, y, incy)


cdef void fscal(int * n, floating * alpha, floating * x,
                     int * incx) nogil:
    if floating is double:
        dscal(n, alpha, x, incx)
    else:
        sscal(n, alpha, x, incx)


cdef void fposv(char * uplo, int * n, int * nrhs, floating * a,
                     int * lda, floating * b, int * ldb, int * info) nogil:
    if floating is double:
        dposv(uplo, n, nrhs, a, lda, b, ldb, info)
    else:
        sposv(uplo, n, nrhs, a, lda, b, ldb, info)


cdef inline floating ST(floating u, floating x) nogil:
    if x > u:
        return x - u
    elif x < - u:
        return x + u
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating primal_value(floating alpha, int n_samples, floating * R,
                         int n_features, floating * w) nogil:
    cdef int inc = 1
    # regularization term: alpha ||w||_1
    cdef floating p_obj = alpha * fasum(&n_features, w, &inc)
    # R is passed as a pointer so no need to & it
    p_obj += fdot(&n_samples, R, &inc, R, &inc) / (2. * n_samples)
    return p_obj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating dual_value(int n_samples, floating alpha, floating norm_y2,
                       floating * theta, floating * y) nogil:
    """Theta must be feasible"""
    cdef int i
    cdef floating d_obj = 0.
    for i in range(n_samples):
        d_obj -= (y[i] / (alpha * n_samples) - theta[i]) ** 2
    d_obj *= 0.5 * alpha ** 2 * n_samples
    d_obj += norm_y2 / (2. * n_samples)
    return d_obj
