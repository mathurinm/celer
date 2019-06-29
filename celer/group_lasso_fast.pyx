#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport fabs, sqrt

from .utils cimport primal_value, dual_value, ST
from .utils cimport fdot, fasum, faxpy, fnrm2, fcopy, fscal

ctypedef np.uint8_t uint8

cdef:
    int inc = 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void group_lasso(
    bint is_sparse,
    int n_samples, int n_features, floating[::1, :] X, int[:] grp_indices,
    int[:] grp_ptr,
    floating[:] X_data, int[:] X_indices, int[:] X_indptr, floating[:] X_mean,
    floating[:] y, floating alpha, bint center, floating[:] w, floating[:] R,
    floating[:] theta, floating[:] lc_groups,
    floating norm_y2, floating eps, int max_epochs, int gap_freq,
    int verbose=0):

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int n_groups = lc_groups.shape[0]

    cdef int i, j, g, k, startptr, endptr, epoch
    cdef int max_group_size = 0
    for g in range(n_groups):
        max_group_size = max(max_group_size, grp_ptr[g + 1] - grp_ptr[g])

    cdef floating old_w_g = np.zeros(max_group_size, dtype=dtype)
    # cdef
    # X_mean_g, w_g
    cdef int inc = 1

    cdef floating gap, p_obj, d_obj, dual_scale
    cdef floating highest_d_obj = 0.
    cdef floating tmp, R_sum

    for epoch in range(max_epochs):
        if epoch % gap_freq == 1:
            # theta = R / (alpha * n_samples)
            fcopy(&n_samples, &R[0], &inc, &theta[0], &inc)
            tmp = 1. / (alpha * n_samples)
            fscal(&n_samples, &tmp, &theta[0], &inc)

            dual_scale = compute_dual_scaling(
                is_sparse,
                n_features, n_samples, &theta[0], X, X_data, X_indices, X_indptr,
                ws_size, &C[0], &dummy_screened[0], X_mean, center)

            if dual_scale > 1. :
                tmp = 1. / dual_scale
                fscal(&n_samples, &tmp, &theta[0], &inc)

            d_obj = dual_value(n_samples, alpha, norm_y2, &theta[0], &y[0])

            if d_obj > highest_d_obj:
                highest_d_obj = d_obj
            p_obj = primal_value(alpha, n_samples, &R[0], n_features, &w[0])
            gap = p_obj - highest_d_obj

            if verbose:
                print("Epoch %d, primal %.10f, gap: %.2e" % (epoch, p_obj, gap))
            if gap < eps:
                if verbose:
                    print("Exit epoch %d, gap: %.2e < %.2e" % \
                        (epoch, gap, eps))
                break

        for g in range(n_groups):
            if lc_groups[g] == 0.:
                    continue
            for k in range(grp_ptr[g + 1] - grp_ptr[g]):
                j = grp_indices[k]
                old_w_g[k] = w[j]

                if is_sparse:
                    X_mean_j = X_mean[j]
                    startptr, endptr = X_indptr[j], X_indptr[j + 1]
                    for i in range(startptr, endptr):
                        w[j] += R[X_indices[i]] * X_data[i] / lc_groups[g] ** 2
                    if center:
                        R_sum = 0.
                        for i in range(n_samples):
                            R_sum += R[i]
                        w[j] -= R_sum * X_mean_j / norms_X_col[j] ** 2
                else:
                    w[j] += fdot(&n_samples, &X[0, j], &inc, &R[0], &inc) / norms_X_col[j] ** 2

            BST(w, alpha / lc_groups[g] ** 2 * n_samples)

            # R -= (w_j - old_w_j) * (X[:, j] - X_mean[j])
            for k in range(grp_ptr[g + 1] - grp_ptr[g]):
                j = grp_indices[k]
                tmp = w[j] - old_w[k]
                if tmp != 0.:
                    if is_sparse:
                        for i in range(startptr, endptr):
                            R[X_indices[i]] -= tmp *  X_data[i]
                        if center:
                            for i in range(n_samples):
                                R[i] += X_mean_j * tmp
                    else:
                        tmp = -tmp
                        faxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0], &inc)
