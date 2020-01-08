#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np

cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport fabs, sqrt, exp

from .cython_utils cimport fdot, faxpy, fcopy, fposv, fscal, fnrm2
from .cython_utils cimport (primal, dual, create_dual_pt, create_accel_pt,
                            sigmoid, ST, LOGREG, compute_dual_scaling,
                            set_prios)

ctypedef np.uint8_t uint8

cdef:
    int inc = 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def celer_logreg(
        bint is_sparse, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] X_mean,
        floating[:] y, floating alpha, floating[:] w, floating[:] Xw,
        floating[:] theta, floating[:] norms_X_col, int max_iter,
        int max_epochs,
        int gap_freq=10, floating tol_ratio_inner=0.3, floating tol=1e-6,
        int p0=100, int verbose=0, int verbose_inner=0,
        int use_accel=1, int prune=0, int K=6, int better_lc=1):
    """Xw and w are modified in place and assumed to match."""

    cdef int pb = LOGREG
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

    cdef floating[:] gaps = np.zeros(max_iter, dtype=dtype)

    cdef floating[:] theta_inner = np.zeros(n_samples, dtype=dtype)
    # passed to inner solver
    # and potentially used for screening if it gives a better d_obj
    cdef floating d_obj_from_inner = 0.

    cdef int[:] C
    cdef int[:] dummy_C = np.zeros(1, dtype=np.int32) # initialize with dummy value
    cdef int[:] all_features = np.arange(n_features, dtype=np.int32)


    for t in range(max_iter):
        create_dual_pt(pb, n_samples, alpha, &theta[0], &Xw[0], &y[0])

        scal = compute_dual_scaling(
            is_sparse, pb, n_features, n_samples, &theta[0], X, X_data, X_indices,
            X_indptr, n_features, &dummy_C[0], &screened[0], X_mean, center, 0)

        if scal > 1. :
            tmp = 1. / scal
            fscal(&n_samples, &tmp, &theta[0], &inc)

        d_obj = dual(pb, n_samples, alpha, tmp, &theta[0], &y[0])

        # also test dual point returned by inner solver after 1st iter:
        if t != 0:
            scal = compute_dual_scaling(
                is_sparse, pb, n_features, n_samples, &theta_inner[0],
                X, X_data, X_indices, X_indptr,
                n_features, &dummy_C[0], &screened[0], X_mean, center, 0)
            if scal > 1.:
                tmp = 1. / scal
                fscal(&n_samples, &tmp, &theta_inner[0], &inc)

            d_obj_from_inner = dual(pb, n_samples, alpha, tmp, &theta_inner[0],
                                    &y[0])

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            fcopy(&n_samples, &theta_inner[0], &inc, &theta[0], &inc)

        if t == 0 or d_obj > highest_d_obj:
            highest_d_obj = d_obj
            # TODO implement a best_theta

        p_obj = primal(pb, alpha, n_samples, &Xw[0], &y[0], n_features, &w[0])
        gap = p_obj - highest_d_obj
        gaps[t] = gap

        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")


        if gap < tol:
            if verbose:
                print("\nEarly exit, gap: %.2e < %.2e" % (gap, tol))
            break

        radius = sqrt(gap / 2.) / alpha
        set_prios(
            is_sparse, pb, n_samples, n_features, &theta[0], X, X_data,
            X_indices, X_indptr, &norms_X_col[0], &prios[0], &screened[0],
            radius, &n_screened, 0)

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
            np.asarray(C).sort()
        if prune:
            tol_inner = tol_ratio_inner * gap
        else:
            tol_inner = tol
        if verbose:
            print(", %d feats in subpb (%d left)" % (len(C), n_features - n_screened))
        # calling inner solver which will modify w and R inplace
        inner_solver_logreg(
            is_sparse,
            n_samples, n_features, ws_size, X, X_data, X_indices, X_indptr,
            X_mean, y, alpha, center, w, Xw, C, theta_inner, norms_X_col,
            tol_inner, max_epochs, gap_freq,
            verbose=verbose_inner,
            K=K, use_accel=use_accel, better_lc=better_lc)

    return (np.asarray(w), np.asarray(theta),
            np.asarray(gaps[:t + 1]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int inner_solver_logreg(
    bint is_sparse,
    int n_samples, int n_features, int ws_size, floating[::1, :] X,
    floating[:] X_data, int[:] X_indices, int[:] X_indptr,
    floating[:] X_mean, floating[:] y, floating alpha, bint center,
    floating[:] w,
    floating[:] Xw, int[:] C, floating[:] theta, floating[:] norms_X_col,
    floating eps, int max_epochs, int gap_freq,
    int verbose=0, int K=6, int use_accel=1, int better_lc=1):

    cdef int pb = LOGREG
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    # TODO issue ?
    cdef floating[:] inv_lc = 4. / np.asarray(norms_X_col) ** 2

    cdef int i, j, k, startptr, endptr, epoch
    cdef floating old_w_j, X_mean_j, w_Cj
    cdef int inc = 1
    cdef uint8[:] dummy_screened = np.zeros(1, dtype=np.uint8)

    cdef int idx  # used as shortcut for X_indices[i]

    cdef floating tmp
    cdef double tmp_exp
    cdef floating[:] thetaccel = np.empty(n_samples, dtype=dtype)
    cdef floating gap, p_obj, d_obj, d_obj_accel, scal
    cdef floating highest_d_obj = 0.

    # acceleration variables:
    cdef floating[:, :] last_K_Xw = np.empty([K, n_samples], dtype=dtype)
    cdef floating[:, :] U = np.empty([K - 1, n_samples], dtype=dtype)
    cdef floating[:, :] UtU = np.empty([K - 1, K - 1], dtype=dtype)
    cdef floating[:] onesK = np.ones(K - 1, dtype=dtype)

    cdef int info_dposv

    for epoch in range(max_epochs):
        if epoch != 0 and epoch % gap_freq == 0:
            create_dual_pt(pb, n_samples, alpha, &theta[0], &Xw[0], &y[0])

            scal = compute_dual_scaling(
                is_sparse, pb, n_features, n_features,
                &theta[0], X, X_data, X_indices, X_indptr,
                ws_size, &C[0], &dummy_screened[0], X_mean, center, 0)

            if scal > 1. :
                tmp = 1. / scal
                fscal(&n_samples, &tmp, &theta[0], &inc)

            d_obj = dual(pb, n_samples, alpha, tmp, &theta[0], &y[0])

            if use_accel: # also compute accelerated dual_point
                info_dposv = create_accel_pt(
                    pb, n_samples, n_features, K, epoch, gap_freq, alpha,
                    &Xw[0], &thetaccel[0], &last_K_Xw[0, 0], U, UtU, onesK, y)

                if info_dposv != 0 and verbose:
                    print("linear system solving failed")

                scal = compute_dual_scaling(
                    is_sparse, pb, n_features, n_samples, &thetaccel[0], X, X_data,
                    X_indices, X_indptr, ws_size, &C[0],
                    &dummy_screened[0], X_mean, center, 0)

                if scal > 1. :
                    tmp = 1. / scal
                    fscal(&n_samples, &tmp, &thetaccel[0], &inc)

                d_obj_accel = dual(pb, n_samples, alpha, tmp, &thetaccel[0], &y[0])
                # CAUTION: I have not yet written the code to include a best_theta.
                # This is of no consequence as long as screening is not performed.
                # Otherwise dgap and theta might disagree.
                if d_obj_accel > d_obj:
                    if verbose:
                        print("Win: %.2e vs %.2e" % (p_obj - d_obj_accel, p_obj - d_obj))
                    d_obj = d_obj_accel
                    # theta = theta_accel (theta is defined as
                    # theta_inner in outer loop)
                    fcopy(&n_samples, &thetaccel[0], &inc, &theta[0], &inc)

            if d_obj > highest_d_obj:
                highest_d_obj = d_obj
                        # we pass full w and will ignore zero values
            p_obj = primal(
                pb, alpha, n_samples, &Xw[0], &y[0], n_features, &w[0])
            gap = p_obj - highest_d_obj

            if verbose:
                print("Inner epoch %d, primal %.10f, gap: %.2e" % (epoch, p_obj, gap))
            if gap < eps:
                if verbose:
                    print("Inner: early exit at epoch %d, gap: %.2e < %.2e" % \
                        (epoch, gap, eps))
                break

        for k in range(ws_size):
            # update feature j cyclically
            j = C[k]
            old_w_j = w[j]
            if is_sparse:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                if better_lc:
                    tmp = 0.
                    for i in range(startptr, endptr):
                        tmp_exp = exp(Xw[X_indices[i]])
                        tmp += X_data[i] ** 2 * tmp_exp / (1. + tmp_exp) ** 2
                    inv_lc[j] = 1. / tmp
            else:
                if better_lc:
                    tmp = 0.
                    for i in range(n_samples):
                        tmp_exp = exp(Xw[i])
                        tmp += (X[i, j] ** 2) * tmp_exp / (1. + tmp_exp) ** 2
                    inv_lc[j] = 1. / tmp

            tmp = 0.  # tmp = dot(Xj, y * sigmoid(-y * w)) / lc[j]
            if is_sparse:
                for i in range(startptr, endptr):
                    idx = X_indices[i]
                    tmp += X_data[i] * y[idx] * sigmoid(- y[idx] * Xw[idx])
            else:
                for i in range(n_samples):
                    tmp += X[i, j] * y[i] * sigmoid(- y[i] * Xw[i])

            w[j] = ST(w[j] + tmp * inv_lc[j], alpha * inv_lc[j])

            tmp = w[j] - old_w_j
            if tmp != 0.:
                if is_sparse:
                    for i in range(startptr, endptr):
                        Xw[X_indices[i]] += tmp * X_data[i]
                else:
                    faxpy(&n_samples, &tmp, &X[0, j], &inc, &Xw[0], &inc)

    else:
        print("!!! Inner solver did not converge at epoch %d, gap: %.2e > %.2e" % \
            (epoch, gap, eps))

    return epoch



