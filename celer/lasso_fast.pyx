#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport fabs, sqrt

from .cython_utils cimport fdot, fasum, faxpy, fnrm2, fcopy, fscal, fposv
from .cython_utils cimport (primal, dual, create_dual_pt, create_accel_pt,
                            sigmoid, ST, LASSO, LOGREG, compute_dual_scaling,
                            set_prios)
ctypedef np.uint8_t uint8

cdef:
    int inc = 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def celer(
        # bint is_sparse, int pb, floating[::1, :] X, floating[:] X_data,
        bint is_sparse, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] X_mean,
        floating[:] y, floating alpha, floating[:] w, floating[:] R,
        floating[:] theta, floating[:] norms_X_col, int max_iter,
        int max_epochs, int gap_freq=10, float tol_ratio_inner=0.3,
        float tol=1e-6, int p0=100, int verbose=0,
        int verbose_inner=0, int use_accel=1, int prune=0, bint positive=0):
    """R/Xw and w are modified in place and assumed to match."""

    cdef int pb = LASSO
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int n_features = w.shape[0]
    cdef int n_samples = y.shape[0]

    if p0 > n_features:
        p0 = n_features

    cdef int i, j, t, startptr, endptr
    cdef int inc = 1
    cdef floating tmp
    cdef int ws_size = 0
    cdef int nnz = 0
    cdef floating p_obj, d_obj, highest_d_obj, gap, radius
    cdef floating scal
    cdef int n_screened = 0
    cdef bint center = False
    cdef floating X_mean_j
    cdef floating[:] prios = np.empty(n_features, dtype=dtype)
    cdef uint8[:] screened = np.zeros(n_features, dtype=np.uint8)

    if is_sparse:
        # center = X_mean.any():
        for j in range(n_features):
            if X_mean[j]:
                center = True
                break

    cdef floating norm_y2 = fnrm2(&n_samples, &y[0], &inc) ** 2

    cdef floating[:] gaps = np.zeros(max_iter, dtype=dtype)

    cdef floating[:] theta_inner = np.zeros(n_samples, dtype=dtype)

    # passed to inner solver
    # and potentially used for screening if it gives a better d_obj
    cdef floating d_obj_from_inner = 0.

    cdef int[:] dummy_C = np.zeros(1, dtype=np.int32) # initialize with dummy value
    cdef int[:] all_features = np.arange(n_features, dtype=np.int32)

    for t in range(max_iter):
        if t != 0:
            create_dual_pt(pb, n_samples, alpha, &theta[0], &R[0], &y[0])

            scal = compute_dual_scaling(
                is_sparse, pb, n_features, n_samples, &theta[0], X, X_data,
                X_indices, X_indptr, n_features, &dummy_C[0], &screened[0],
                X_mean, center, positive)

            if scal > 1. :
                tmp = 1. / scal
                fscal(&n_samples, &tmp, &theta[0], &inc)

            d_obj = dual(pb, n_samples, alpha, norm_y2, &theta[0], &y[0])

            # also test dual point returned by inner solver after 1st iter:
            scal = compute_dual_scaling(
                is_sparse, pb, n_features, n_samples, &theta_inner[0],
                X, X_data, X_indices, X_indptr,
                n_features, &dummy_C[0], &screened[0], X_mean, center, positive)
            if scal > 1.:
                tmp = 1. / scal
                fscal(&n_samples, &tmp, &theta_inner[0], &inc)

            d_obj_from_inner = dual(
                pb, n_samples, alpha, norm_y2, &theta_inner[0], &y[0])
        else:
            d_obj = dual(pb, n_samples, alpha, norm_y2, &theta[0], &y[0])

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            fcopy(&n_samples, &theta_inner[0], &inc, &theta[0], &inc)
            theta_to_use = theta

        if t == 0 or d_obj > highest_d_obj:
            highest_d_obj = d_obj
            # TODO implement a best_theta

        p_obj = primal(pb, alpha, n_samples, &R[0], &y[0], n_features, &w[0])
        gap = p_obj - highest_d_obj
        gaps[t] = gap  # TODO useful?

        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")

        if gap < tol:
            if verbose:
                print("\nEarly exit, gap: %.2e < %.2e" % (gap, tol))
            break

        radius = sqrt(2 * gap / n_samples) / alpha
        set_prios(
            is_sparse, pb, n_samples, n_features, &theta[0], X, X_data,
            X_indices, X_indptr, &norms_X_col[0], &prios[0], &screened[0],
            radius, &n_screened, positive)

        if prune:
            nnz = 0
            for j in range(n_features):
                if w[j] != 0:
                    prios[j] = -1.
                    nnz += 1

            if t == 0:
                ws_size = p0 if nnz == 0 else nnz
            else:
                ws_size = 2 * nnz

        else:
            for j in range(n_features):
                if w[j] != 0:
                    prios[j] = - 1  # include active features
            if t == 0:
                ws_size = p0
            else:
                for j in range(ws_size):
                    if not screened[C[j]]:
                        # include previous features, if not screened
                        prios[C[j]] = -1
                ws_size = 2 * ws_size

        if ws_size > n_features - n_screened:
            ws_size = n_features - n_screened


        # if ws_size === n_features then argpartition will break:
        if ws_size == n_features:
            C = all_features
        else:
            C = np.argpartition(np.asarray(prios), ws_size)[:ws_size].astype(np.int32)
            C.sort()
        if prune:
            tol_inner = tol_ratio_inner * gap
        else:
            tol_inner = tol

        if verbose:
            print(", %d feats in subpb (%d left)" % (len(C), n_features - n_screened))
        # calling inner solver which will modify w and R inplace
        inner_solver(
            is_sparse,
            n_samples, n_features, ws_size, X, X_data, X_indices, X_indptr, X_mean,
            y, alpha, center, w, R, C, theta_inner, norms_X_col,
            norm_y2, tol_inner, max_epochs=max_epochs,
            gap_freq=gap_freq, verbose=verbose_inner,
            use_accel=use_accel, positive=positive)

    return (np.asarray(w), np.asarray(theta),
            np.asarray(gaps[:t + 1]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void inner_solver(
    bint is_sparse,
    int n_samples, int n_features, int ws_size, floating[::1, :] X,
    floating[:] X_data, int[:] X_indices, int[:] X_indptr, floating[:] X_mean,
    floating[:] y, floating alpha, bint center, floating[:] w, floating[:] R,
    int[:] C, floating[:] theta, floating[:] norms_X_col,
    floating norm_y2, floating eps, int max_epochs, int gap_freq,
    int verbose=0, int K=6, int use_accel=1, bint positive=0):

    cdef int pb = LASSO
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int i, j, k, startptr, endptr, epoch
    cdef floating old_w_j, X_mean_j, w_Cj
    cdef int inc = 1
    cdef uint8[:] dummy_screened = np.zeros(1, dtype=np.uint8)

    cdef floating[:] thetaccel = np.empty(n_samples, dtype=dtype)
    cdef floating gap, p_obj, d_obj, d_obj_accel, scal
    cdef floating highest_d_obj = 0. # d_obj is always >=0 so this gets replaced
    # at first d_obj computation. highest_d_obj corresponds to theta = 0.
    cdef floating tmp, R_sum
    # acceleration variables:
    cdef floating[:, :] last_K_res = np.empty([K, n_samples], dtype=dtype)
    cdef floating[:, :] U = np.empty([K - 1, n_samples], dtype=dtype)
    cdef floating[:, :] UtU = np.empty([K - 1, K - 1], dtype=dtype)
    cdef floating[:] onesK = np.ones(K - 1, dtype=dtype)

    # solving linear system in cython
    # doc at https://software.intel.com/en-us/node/468894
    cdef char * char_U = 'U'
    cdef int Kminus1 = K - 1
    cdef int one = 1
    cdef floating sum_z
    cdef int info_dposv

    for epoch in range(max_epochs):
        if epoch % gap_freq == 1:
            # theta = R / (alpha * n_samples)
            fcopy(&n_samples, &R[0], &inc, &theta[0], &inc)
            tmp = 1. / (alpha * n_samples)
            fscal(&n_samples, &tmp, &theta[0], &inc)

            scal = compute_dual_scaling(
                is_sparse, pb,
                n_features, n_samples, &theta[0], X, X_data, X_indices, X_indptr,
                ws_size, &C[0], &dummy_screened[0], X_mean, center, positive)

            if scal > 1. :
                tmp = 1. / scal
                fscal(&n_samples, &tmp, &theta[0], &inc)

            d_obj = dual(pb, n_samples, alpha, norm_y2, &theta[0], &y[0])

            if use_accel: # also compute accelerated dual_point
                if epoch // gap_freq < K:
                    # last_K_res[it // f_gap] = R:
                    fcopy(&n_samples, &R[0], &inc,
                          &last_K_res[epoch // gap_freq, 0], &inc)
                else:
                    for k in range(K - 1):
                        fcopy(&n_samples, &last_K_res[k + 1, 0], &inc,
                              &last_K_res[k, 0], &inc)
                    fcopy(&n_samples, &R[0], &inc, &last_K_res[K - 1, 0], &inc)
                    for k in range(K - 1):
                        for i in range(n_samples):
                            U[k, i] = last_K_res[k + 1, i] - last_K_res[k, i]

                    for k in range(K - 1):
                        for j in range(k, K - 1):
                            UtU[k, j] = fdot(&n_samples, &U[k, 0], &inc,
                                              &U[j, 0], &inc)
                            UtU[j, k] = UtU[k, j]

                    # refill onesK with ones because it has been overwritten
                    # by dposv
                    for k in range(K - 1):
                        onesK[k] = 1

                    fposv(char_U, &Kminus1, &one, &UtU[0, 0], &Kminus1,
                           &onesK[0], &Kminus1, &info_dposv)

                    # onesK now holds the solution in x to UtU dot x = onesK
                    if info_dposv != 0:
                        if verbose:
                            print("linear system solving failed")
                        # don't use accel for this iteration
                        for k in range(K - 2):
                            onesK[k] = 0
                        onesK[K - 2] = 1

                    sum_z = 0
                    for k in range(K - 1):
                        sum_z += onesK[k]
                    for k in range(K - 1):
                        onesK[k] /= sum_z

                    for i in range(n_samples):
                        thetaccel[i] = 0.
                    for k in range(K - 1):
                        for i in range(n_samples):
                            thetaccel[i] += onesK[k] * last_K_res[k, i]

                    tmp = 1. / (alpha * n_samples)
                    fscal(&n_samples, &tmp, &thetaccel[0], &inc)

                    scal = compute_dual_scaling(
                        is_sparse, pb, n_features, n_samples, &thetaccel[0], X,
                        X_data, X_indices, X_indptr, ws_size, &C[0],
                        &dummy_screened[0], X_mean, center, positive)

                    if scal > 1. :
                        tmp = 1. / scal
                        fscal(&n_samples, &tmp, &thetaccel[0], &inc)

                    d_obj_accel = dual(
                        pb, n_samples, alpha, norm_y2, &thetaccel[0], &y[0])

                    if d_obj_accel > d_obj:
                        d_obj = d_obj_accel
                        # theta = theta_accel (theta is defined as
                        # theta_inner in outer loop)
                        fcopy(&n_samples, &thetaccel[0], &inc, &theta[0], &inc)

            if d_obj > highest_d_obj:
                highest_d_obj = d_obj

            # CAUTION: I have not yet written the code to include a best_theta.
            # This is of no consequence as long as screening is not performed.
            # Otherwise dgap and theta might disagree.

            # we pass full w and will ignore zero values
            p_obj = primal(
                pb, alpha, n_samples, &R[0], &y[0], n_features, &w[0])
            gap = p_obj - highest_d_obj

            if verbose:
                print("Inner epoch %d, primal %.10f, gap: %.2e" % (epoch, p_obj, gap))
            if gap < eps:
                if verbose:
                    print("Inner: exit epoch %d, gap: %.2e < %.2e" % \
                        (epoch, gap, eps))
                break

        for k in range(ws_size):
            # update feature j in place
            j = C[k]
            if norms_X_col[j] == 0.:
                continue
            old_w_j = w[j]
            if is_sparse:
                X_mean_j = X_mean[j]
                startptr, endptr = X_indptr[j], X_indptr[j + 1]
                for i in range(startptr, endptr):
                    w[j] += R[X_indices[i]] * X_data[i] / norms_X_col[j] ** 2
                if center:
                    R_sum = 0.
                    for i in range(n_samples):
                        R_sum += R[i]
                    w[j] -= R_sum * X_mean_j / norms_X_col[j] ** 2
            else:
                w[j] += fdot(&n_samples, &X[0, j], &inc, &R[0], &inc) / norms_X_col[j] ** 2

            # perform ST in place:
            if positive and w[j] <= 0.:
                w[j] = 0.
            else:
                w[j] = ST(w[j], alpha / norms_X_col[j] ** 2 * n_samples)

            # R -= (w_j - old_w_j) * (X[:, j] - X_mean[j])
            tmp = old_w_j - w[j]
            if tmp != 0.:
                if is_sparse:
                    for i in range(startptr, endptr):
                        R[X_indices[i]] += tmp *  X_data[i]
                    if center:
                        for i in range(n_samples):
                            R[i] -= X_mean_j * tmp
                else:
                    faxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0], &inc)
    else:
        print("!!! Inner solver did not converge at epoch %d, gap: %.2e > %.2e" % \
            (epoch, gap, eps))

