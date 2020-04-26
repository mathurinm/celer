#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport fabs, sqrt

from .cython_utils cimport fdot, fasum, faxpy, fnrm2, fcopy, fscal, dual, LASSO

ctypedef np.uint8_t uint8

cdef:
    int inc = 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef floating primal_grplasso(
        floating alpha, floating[:] R, int[::1] grp_ptr,
        int[::1] grp_indices, floating[:] w):
    cdef floating nrm = 0.
    cdef int j, k, g
    cdef int n_samples = R.shape[0]
    cdef int n_groups = grp_ptr.shape[0] - 1
    cdef floating p_obj = fnrm2(&n_samples, &R[0], &inc) ** 2 / (2 * n_samples)
    for g in range(n_groups):
        nrm = 0.
        for k in range(grp_ptr[g], grp_ptr[g + 1]):
            j = grp_indices[k]
            nrm += w[j] ** 2
        p_obj += alpha * sqrt(nrm)
    return p_obj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef floating dscal_grplasso(
        bint is_sparse, floating[::1] theta, int[::1] grp_ptr,
        int[::1] grp_indices, floating[::1, :] X, floating[::1] X_data,
        int[::1] X_indices, int[::1] X_indptr, floating[::1] X_mean,
        bint center):
    cdef floating Xj_theta, tmp
    cdef floating scal = 0.
    cdef floating theta_sum = 0.
    cdef int i, j, g, k, startptr, endptr
    cdef int n_groups = grp_ptr.shape[0] - 1
    cdef int n_samples = theta.shape[0]

    if is_sparse:
        if center:
            for i in range(n_samples):
                theta_sum += theta[i]

    for g in range(n_groups):
        tmp = 0
        for k in range(grp_ptr[g], grp_ptr[g + 1]):
            j = grp_indices[k]
            if is_sparse:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                Xj_theta = 0.
                for i in range(startptr, endptr):
                    Xj_theta += X_data[i] * theta[X_indices[i]]
                if center:
                    Xj_theta -= theta_sum * X_mean[j]
            else:
                Xj_theta = fdot(&n_samples, &theta[0], &inc, &X[0, j], &inc)
            tmp += Xj_theta ** 2

        scal = max(scal, sqrt(tmp))
    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_prios_grplasso(
        bint is_sparse, int pb,floating[::1] theta, floating[::1, :] X,
        floating[::1] X_data, int[::1] X_indices, int[::1] X_indptr,
        floating[::1] norms_X_grp, int[::1] grp_ptr, int[::1] grp_indices,
        floating[::1] prios, uint8[::1] screened, floating radius,
        int * n_screened) nogil:
    cdef int i, j, k, g, startptr, endptr
    cdef floating nrm_Xgtheta, Xj_theta
    cdef int n_groups = grp_ptr.shape[0] - 1
    cdef int n_samples = theta.shape[0]

    for g in range(n_groups):
        if screened[g] or norms_X_grp[g] == 0.:
            prios[g] = 10000
            continue
        nrm_Xgtheta = 0
        for k in range(grp_ptr[g], grp_ptr[g + 1]):
            j = grp_indices[k]
            if is_sparse:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                Xj_theta = 0.
                for i in range(startptr, endptr):
                    Xj_theta += X_data[i] * theta[X_indices[i]]
            else:
                Xj_theta = fdot(&n_samples, &theta[0], &inc, &X[0, j], &inc)
            nrm_Xgtheta += Xj_theta ** 2
        nrm_Xgtheta = sqrt(nrm_Xgtheta)


        prios[g] = (1. - nrm_Xgtheta) / norms_X_grp[g]

        if prios[g] > radius:
            screened[g] = True
            n_screened[0] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef group_lasso(
        bint is_sparse, floating[::1, :] X, int[::1] grp_indices, int[::1] grp_ptr,
        floating[::1] X_data, int[::1] X_indices, int[::1] X_indptr, floating[::1] X_mean,
        floating[:] y, floating alpha, floating[:] w, floating[:] R,
        floating[::1] theta, floating[:] lc_groups, floating eps, int max_epochs,
        int gap_freq, int verbose=0):

    cdef bint center = False

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int n_samples = y.shape[0]
    cdef int n_features = w.shape[0]
    cdef int n_groups = lc_groups.shape[0]
    cdef floating norm_y2 = fnrm2(&n_samples, &y[0], &inc) ** 2
    # print("nrmyy", norm_y2)

    cdef int i, j, g, k, startptr, endptr, epoch
    cdef int max_group_size = 0
    for g in range(n_groups):
        max_group_size = max(max_group_size, grp_ptr[g + 1] - grp_ptr[g])

    cdef floating[:] old_w_g = np.zeros(max_group_size, dtype=dtype)

    cdef floating gap, p_obj, d_obj, dual_scale, X_mean_j
    cdef floating highest_d_obj = 0.
    cdef floating tmp, R_sum, norm_wg, bst_scal

    for epoch in range(max_epochs):
        # print("epoch", epoch)
        if epoch % gap_freq == 1:
            # print("gap")
            # theta = R / (alpha * n_samples)
            fcopy(&n_samples, &R[0], &inc, &theta[0], &inc)
            tmp = 1. / (alpha * n_samples)
            fscal(&n_samples, &tmp, &theta[0], &inc)

            dual_scale = dscal_grplasso(
                is_sparse, theta, grp_ptr,
                grp_indices, X, X_data, X_indices, X_indptr, X_mean, center)
            # print("dscal", dual_scale)

            if dual_scale > 1. :
                tmp = 1. / dual_scale
                fscal(&n_samples, &tmp, &theta[0], &inc)

            # dual value is the same as for the Lasso
            # print(np.array(theta))
            d_obj = dual(LASSO, n_samples, alpha, norm_y2, &theta[0], &y[0])
            # print(d_obj)
            if d_obj > highest_d_obj:
                highest_d_obj = d_obj
            p_obj = primal_grplasso(alpha, R, grp_ptr, grp_indices, w)
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
            norm_wg = 0.
            for k in range(grp_ptr[g + 1] - grp_ptr[g]):
                j = grp_indices[k + grp_ptr[g]]
                old_w_g[k] = w[j]

                if is_sparse:
                    X_mean_j = X_mean[j]
                    startptr, endptr = X_indptr[j], X_indptr[j + 1]
                    for i in range(startptr, endptr):
                        w[j] += R[X_indices[i]] * X_data[i] / lc_groups[g]
                    if center:
                        R_sum = 0.
                        for i in range(n_samples):
                            R_sum += R[i]
                        w[j] -= R_sum * X_mean_j / lc_groups[g]
                else:
                    w[j] += fdot(&n_samples, &X[0, j], &inc, &R[0],
                                 &inc) / lc_groups[g]
                norm_wg += w[j] ** 2
            norm_wg = sqrt(norm_wg)
            bst_scal = max(0., 1. - alpha / lc_groups[g] * n_samples / norm_wg)

            for k in range(grp_ptr[g + 1] - grp_ptr[g]):
                j = grp_indices[grp_ptr[g] + k]
                # perform BST:
                w[j] *= bst_scal
                # R -= (w_j - old_w_j) * (X[:, j] - X_mean[j])
                tmp = w[j] - old_w_g[k]
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

    return gap


