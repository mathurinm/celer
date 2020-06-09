#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport fabs, sqrt

from .cython_utils cimport (fdot, fasum, faxpy, fnrm2, fcopy, fscal, dual,
                            LASSO, LOGREG, create_accel_pt)


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
cpdef floating dnorm_grp(
        bint is_sparse, floating[::1] theta, int[::1] grp_ptr,
        int[::1] grp_indices, floating[::1, :] X, floating[::1] X_data,
        int[::1] X_indices, int[::1] X_indptr, floating[::1] X_mean,
        int ws_size, int[:] C, bint center):
    """Dual norm in the group case, i.e. L2/infty ofter groups."""
    cdef floating Xj_theta, tmp
    cdef floating scal = 0.
    cdef floating theta_sum = 0.
    cdef int i, j, g, g_idx, k, startptr, endptr
    cdef int n_groups = grp_ptr.shape[0] - 1
    cdef int n_samples = theta.shape[0]

    if is_sparse:
        if center:
            for i in range(n_samples):
                theta_sum += theta[i]

    if ws_size == n_groups:  # max over all groups
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
                    Xj_theta = fdot(&n_samples, &theta[0], &inc, &X[0, j],
                                    &inc)
                tmp += Xj_theta ** 2

            scal = max(scal, sqrt(tmp))

    else:  # scaling only with features in C
        for g_idx in range(ws_size):
            g = C[g_idx]
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
                    Xj_theta = fdot(&n_samples, &theta[0], &inc, &X[0, j],
                                    &inc)
                tmp += Xj_theta ** 2

            scal = max(scal, sqrt(tmp))
    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_prios_grp(
        bint is_sparse, int pb, floating[::1] theta, floating[::1, :] X,
        floating[::1] X_data, int[::1] X_indices, int[::1] X_indptr,
        floating[::1] norms_X_grp, int[::1] grp_ptr, int[::1] grp_indices,
        floating[::1] prios, int[::1] screened, floating radius,
        int * n_screened):
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
cpdef celer_grp(
        bint is_sparse, int pb, floating[::1, :] X, int[::1] grp_indices,
        int[::1] grp_ptr, floating[::1] X_data, int[::1] X_indices,
        int[::1] X_indptr, floating[::1] X_mean, floating[:] y, floating alpha,
        floating[:] w, floating[:] R, floating[::1] theta,
        floating[::1] norms_X_grp, floating eps, int max_iter, int max_epochs,
        int gap_freq, floating tol_ratio_inner=0.3, int p0=100, bint prune=1, bint use_accel=1,
        bint verbose=0):

    pb = LASSO
    cdef int verbose_in = max(0, verbose - 1)

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int n_samples = y.shape[0]
    cdef int n_features = w.shape[0]
    cdef int n_groups = norms_X_grp.shape[0]

    cdef floating norm_y2 = fnrm2(&n_samples, &y[0], &inc) ** 2
    cdef floating[::1] lc_groups = np.square(norms_X_grp)

    cdef int[:] all_groups = np.arange(n_groups, dtype=np.int32)
    cdef int[:] dummy_C = np.zeros(1, dtype=np.int32)
    cdef int[:] C

    cdef int n_screened = 0
    cdef int i, j, g, g_idx, k, startptr, endptr, epoch, t
    cdef int nnz, ws_size
    cdef floating[::1] prios = np.empty(n_groups, dtype=dtype)
    cdef int[::1] screened = np.zeros(n_groups, dtype=np.int32)
    cdef int max_group_size = 0

    cdef bint center = False
    if is_sparse:
        # center = X_mean.any():
        for j in range(n_features):
            if X_mean[j]:
                center = True
                break

    for g in range(n_groups):
        max_group_size = max(max_group_size, grp_ptr[g + 1] - grp_ptr[g])

    cdef floating[:] old_w_g = np.zeros(max_group_size, dtype=dtype)

    cdef floating[::1] gaps = np.zeros(max_iter, dtype=dtype)
    cdef floating[::1] theta_inner = np.zeros(n_samples, dtype=dtype)
    cdef floating[::1] thetacc = np.empty(n_samples, dtype=dtype)

    cdef floating gap, p_obj, d_obj, scal, X_mean_j
    cdef floating gap_in, p_obj_in, d_obj_in, tol_in, d_obj_accel
    cdef floating d_obj_from_inner
    cdef floating highest_d_obj = 0.
    cdef floating highest_d_obj_in = 0.
    cdef floating tmp, R_sum, norm_wg, bst_scal
    cdef floating radius = 10000 # TODO

    # acceleration variables:
    cdef int K = 6
    cdef floating[:, :] last_K_R = np.empty([K, n_samples], dtype=dtype)
    cdef floating[:, :] U = np.empty([K - 1, n_samples], dtype=dtype)
    cdef floating[:, :] UtU = np.empty([K - 1, K - 1], dtype=dtype)
    cdef floating[:] onesK = np.ones(K - 1, dtype=dtype)

    cdef int info_dposv

    for t in range(max_iter):
        # if t != 0: TODO potential speedup at iteration 0
        fcopy(&n_samples, &R[0], &inc, &theta[0], &inc)
        tmp = 1. / (alpha * n_samples)
        fscal(&n_samples, &tmp, &theta[0], &inc)

        scal = dnorm_grp(
            is_sparse, theta, grp_ptr, grp_indices, X, X_data, X_indices,
            X_indptr, X_mean, n_groups, dummy_C, center)

        if scal > 1. :
            tmp = 1. / scal
            fscal(&n_samples, &tmp, &theta[0], &inc)

        d_obj = dual(pb, n_samples, alpha, norm_y2, &theta[0], &y[0])

        if t > 0:
            pass
            # also test dual point returned by inner solver after 1st iter:
            scal = dnorm_grp(
                    is_sparse, theta_inner, grp_ptr, grp_indices, X, X_data,
                    X_indices, X_indptr, X_mean, n_groups, dummy_C, center)
            if scal > 1.:
                tmp = 1. / scal
                fscal(&n_samples, &tmp, &theta_inner[0], &inc)

            d_obj_from_inner = dual(
                pb, n_samples, alpha, norm_y2, &theta_inner[0], &y[0])

            if d_obj_from_inner > d_obj:
                d_obj = d_obj_from_inner
                fcopy(&n_samples, &theta_inner[0], &inc, &theta[0], &inc)

        if t == 0 or d_obj > highest_d_obj:
            highest_d_obj = d_obj
            # TODO implement a best_theta

        p_obj = primal_grplasso(alpha, R, grp_ptr, grp_indices, w)
        gap = p_obj - highest_d_obj
        gaps[t] = gap

        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")

        if gap < eps:
            if verbose:
                print("\nEarly exit, gap: %.2e < %.2e" % (gap, eps))
            break

        # if pb == LASSO:
        radius = sqrt(2 * gap / n_samples) / alpha
        # elif pb == LOGREG:
            # radius = sqrt(gap / 2.) / alpha

        set_prios_grp(
            is_sparse, pb, theta, X, X_data, X_indices, X_indptr, lc_groups,
            grp_ptr, grp_indices, prios, screened, radius, &n_screened)

        if prune:
            nnz = 0
            for g in range(n_groups):
                # TODO this is a hack, will fail for sparse group lasso
                if w[grp_indices[grp_ptr[g]]] != 0:
                    prios[g] = -1.
                    nnz += 1

            if t == 0:
                ws_size = p0 if nnz == 0 else nnz
            else:
                ws_size = 2 * nnz

        else:
            for g in range(n_groups):
                if w[grp_indices[grp_ptr[g]]] != 0:
                    prios[g] = - 1  # include active features
            if t == 0:
                ws_size = p0
            else:
                for g in range(ws_size):
                    if not screened[C[g]]:
                        prios[C[g]] = -1
                ws_size = 2 * ws_size

        if ws_size > n_groups - n_screened:
            ws_size = n_groups - n_screened

        # if ws_size == n_groups then argpartition will break:
        if ws_size == n_groups:
            C = all_groups
        else:
            C = np.argpartition(np.asarray(prios),
                                ws_size)[:ws_size].astype(np.int32)
        if prune:
            tol_in = 0.3 * gap
        else:
            tol_in = eps

        if verbose:
            print(", %d groups in subpb (%d left)" %
                  (len(C), n_groups - n_screened))

        highest_d_obj_in = 0.
        for epoch in range(max_epochs):
            if epoch != 0 and epoch % gap_freq == 0:
                fcopy(&n_samples, &R[0], &inc, &theta_inner[0], &inc)
                tmp = 1. / (alpha * n_samples)
                fscal(&n_samples, &tmp, &theta_inner[0], &inc)

                scal = dnorm_grp(
                    is_sparse, theta_inner, grp_ptr, grp_indices, X, X_data,
                    X_indices, X_indptr, X_mean, ws_size, C, center)

                if scal > 1. :
                    tmp = 1. / scal
                    fscal(&n_samples, &tmp, &theta_inner[0], &inc)

                # dual value is the same as for the Lasso
                d_obj_in = dual(
                    pb, n_samples, alpha, norm_y2, &theta_inner[0], &y[0])

                if use_accel: # also compute accelerated dual_point
                    info_dposv = create_accel_pt(
                        LASSO, n_samples, epoch, gap_freq, alpha, &R[0],
                        &thetacc[0], &last_K_R[0, 0], U, UtU, onesK, y)

                    # if info_dposv != 0 and verbose:
                    #     print("linear system solving failed")

                    if epoch // gap_freq >= K:
                        scal = dnorm_grp(
                            is_sparse, thetacc, grp_ptr, grp_indices, X,
                            X_data, X_indices, X_indptr, X_mean, ws_size, C,
                            center)

                        if scal > 1.:
                            tmp = 1. / scal
                            fscal(&n_samples, &tmp, &thetacc[0], &inc)

                        d_obj_accel = dual(pb, n_samples, alpha, norm_y2,
                                           &thetacc[0], &y[0])
                        if d_obj_accel > d_obj_in:
                            d_obj_in = d_obj_accel
                            fcopy(&n_samples, &thetacc[0], &inc,
                            &theta_inner[0], &inc)


                if d_obj_in > highest_d_obj_in:
                    highest_d_obj_in = d_obj_in
                p_obj_in = primal_grplasso(alpha, R, grp_ptr, grp_indices, w)
                gap_in = p_obj_in - highest_d_obj_in

                if verbose_in:
                    print("Epoch %d, primal %.10f, gap: %.2e" %
                          (epoch, p_obj_in, gap_in))
                if gap_in < tol_in:
                    if verbose_in:
                        print("Exit epoch %d, gap: %.2e < %.2e" %
                              (epoch, gap_in, tol_in))
                    break

            for g_idx in range(ws_size):
                g = C[g_idx]
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
                bst_scal = max(0.,
                               1. - alpha / lc_groups[g] * n_samples / norm_wg)

                for k in range(grp_ptr[g + 1] - grp_ptr[g]):
                    j = grp_indices[grp_ptr[g] + k]
                    # perform BST:
                    w[j] *= bst_scal
                    # R -= (w_j - old_w_j) * (X[:, j] - X_mean[j])
                    tmp = old_w_g[k] - w[j]
                    if tmp != 0.:
                        if is_sparse:
                            startptr, endptr = X_indptr[j], X_indptr[j + 1]
                            for i in range(startptr, endptr):
                                R[X_indices[i]] += tmp *  X_data[i]
                            if center:
                                X_mean_j = X_mean[j]
                                for i in range(n_samples):
                                    R[i] -= X_mean_j * tmp
                        else:
                            faxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0],
                                  &inc)

    return np.asarray(w), np.asarray(theta), np.asarray(gaps[:t + 1])


