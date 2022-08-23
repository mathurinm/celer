#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
cimport numpy as np
cimport cython
import warnings

from cython cimport floating
from libc.math cimport fabs, sqrt, exp, INFINITY
from sklearn.exceptions import ConvergenceWarning

from .cython_utils cimport fdot, fasum, faxpy, fnrm2, fcopy, fscal, fposv
from .cython_utils cimport (primal, dual, create_dual_pt, create_accel_pt,
                            sigmoid, ST, LASSO, LOGREG, dnorm_enet,
                            set_prios, fweighted_norm_w2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def celer(
        bint is_sparse, int pb, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] X_mean,
        floating[:] y, floating alpha, floating l1_ratio, floating[:] w, floating[:] Xw,
        floating[:] theta, floating[:] norms_X_col, floating[:] weights,
        int max_iter, int max_epochs, int gap_freq=10,
        float tol=1e-6, int p0=100, int verbose=0,
        int use_accel=1, int prune=0, bint positive=0,
        int better_lc=1):
    """R/Xw and w are modified in place and assumed to match.
    Weights must be > 0, features with weights equal to np.inf are ignored.
    WARNING for Logreg the datafit is a sum, while for Lasso it is a mean.
    """
    assert pb in (LASSO, LOGREG)

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int inc = 1
    cdef int verbose_in = max(0, verbose - 1)
    cdef int n_features = w.shape[0]
    cdef int n_samples = y.shape[0]

    # scale stopping criterion: multiply tol by primal value at w = 0
    if pb == LASSO:
        # actually for Lasso, omit division by 2 to match sklearn
        tol *= fnrm2(&n_samples, &y[0], &inc) ** 2 / n_samples
    elif pb == LOGREG:
        tol *= n_samples * np.log(2)

    if p0 > n_features:
        p0 = n_features

    cdef int t = 0
    cdef int i, j, k, idx, startptr, endptr, epoch
    cdef int ws_size = 0
    cdef int nnz = 0
    cdef floating gap = -1  # initialized for the warning if max_iter=0
    cdef floating p_obj, d_obj, highest_d_obj, radius, tol_in
    cdef floating gap_in, p_obj_in, d_obj_in, d_obj_accel, highest_d_obj_in
    cdef floating theta_scaling, R_sum, tmp, tmp_exp, dnorm_XTtheta
    cdef int n_screened = 0
    cdef bint center = False
    cdef floating old_w_j, X_mean_j
    cdef floating[:] prios = np.empty(n_features, dtype=dtype)
    cdef int[:] screened = np.zeros(n_features, dtype=np.int32)
    cdef int[:] notin_ws = np.zeros(n_features, dtype=np.int32)


    # acceleration variables:
    cdef int K = 6
    cdef floating[:, :] last_K_Xw = np.empty([K, n_samples], dtype=dtype)
    cdef floating[:, :] U = np.empty([K - 1, n_samples], dtype=dtype)
    cdef floating[:, :] UtU = np.empty([K - 1, K - 1], dtype=dtype)
    cdef floating[:] onesK = np.ones(K - 1, dtype=dtype)
    cdef int info_dposv

    if is_sparse:
        # center = X_mean.any():
        for j in range(n_features):
            if X_mean[j]:
                center = True
                break

    # TODO this is used only for logreg, L97 is misleading and deserves a comment/refactoring
    cdef floating[:] inv_lc = np.zeros(n_features, dtype=dtype)

    for j in range(n_features):
        # can have 0 features when performing CV on sparse X
        if norms_X_col[j]:
            if pb == LOGREG:
                inv_lc[j] = 4. / norms_X_col[j] ** 2
            else:
                inv_lc[j] = 1. / norms_X_col[j] ** 2

    cdef floating norm_y2 = fnrm2(&n_samples, &y[0], &inc) ** 2
    cdef floating weighted_norm_w2 = fweighted_norm_w2(w, weights)
    theta_scaling = 1.0

    # max_iter + 1 is to deal with max_iter=0
    cdef floating[:] gaps = np.zeros(max_iter + 1, dtype=dtype)
    gaps[0] = -1

    cdef floating[:] theta_in = np.zeros(n_samples, dtype=dtype)
    cdef floating[:] thetacc = np.zeros(n_samples, dtype=dtype)
    cdef floating d_obj_from_inner = 0.

    cdef int[:] ws
    cdef int[:] all_features = np.arange(n_features, dtype=np.int32)

    for t in range(max_iter):
        if t != 0:
            create_dual_pt(pb, n_samples, &theta[0], &Xw[0], &y[0])

            dnorm_XTtheta = dnorm_enet(
                is_sparse, theta, w, X, X_data, X_indices, X_indptr, screened,
                X_mean, weights, center, positive, alpha, l1_ratio)

            if dnorm_XTtheta > alpha * l1_ratio:
                theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                fscal(&n_samples, &theta_scaling, &theta[0], &inc)
            else:
                theta_scaling = 1.

            #  compute ||w||^2 only for Enet
            if l1_ratio != 1:
                weighted_norm_w2 = fweighted_norm_w2(w, weights)

            d_obj = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                theta_scaling**2*weighted_norm_w2, &theta[0], &y[0])

            # also test dual point returned by inner solver after 1st iter:
            dnorm_XTtheta = dnorm_enet(
                is_sparse, theta_in, w, X, X_data, X_indices, X_indptr,
                screened, X_mean, weights, center, positive, alpha, l1_ratio)

            if dnorm_XTtheta  > alpha * l1_ratio:
                theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                fscal(&n_samples, &theta_scaling, &theta_in[0], &inc)
            else:
                theta_scaling = 1.

            d_obj_from_inner = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                    theta_scaling**2*weighted_norm_w2, &theta_in[0], &y[0])
        else:
            d_obj = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                theta_scaling**2*weighted_norm_w2, &theta[0], &y[0])

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            fcopy(&n_samples, &theta_in[0], &inc, &theta[0], &inc)

        highest_d_obj = d_obj  # TODO monotonicity could be enforced but it
        # would add yet another variable, best_theta. I'm not sure it brings
        # anything.

        p_obj = primal(pb, alpha, l1_ratio, Xw, y, w, weights)
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")

        if gap <= tol:
            if verbose:
                print("\nEarly exit, gap: %.2e < %.2e" % (gap, tol))
            break

        if pb == LASSO:
            radius = sqrt(2 * gap / n_samples)
        else:
            radius = sqrt(gap / 2.)
        set_prios(
            is_sparse, theta, w, alpha, l1_ratio, X, X_data, X_indices, X_indptr, norms_X_col,
            weights, prios, screened, radius, &n_screened, positive)

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
                    if not screened[ws[j]]:
                        # include previous features, if not screened
                        prios[ws[j]] = -1
                ws_size = 2 * ws_size
        if ws_size > n_features - n_screened:
            ws_size = n_features - n_screened


        # if ws_size === n_features then argpartition will break:
        if ws_size == n_features:
            ws = all_features
        else:
            ws = np.argpartition(np.asarray(prios), ws_size)[:ws_size].astype(np.int32)

        for j in range(n_features):
            notin_ws[j] = 1
        for idx in range(ws_size):
            notin_ws[ws[idx]] = 0

        if prune:
            tol_in = 0.3 * gap
        else:
            tol_in = tol

        if verbose:
            print(", %d feats in subpb (%d left)" %
                  (len(ws), n_features - n_screened))

        # calling inner solver which will modify w and R inplace
        highest_d_obj_in = 0
        for epoch in range(max_epochs):
            if epoch != 0 and epoch % gap_freq == 0:
                create_dual_pt(
                    pb, n_samples, &theta_in[0], &Xw[0], &y[0])

                dnorm_XTtheta  = dnorm_enet(
                    is_sparse, theta_in, w, X, X_data, X_indices, X_indptr,
                    notin_ws, X_mean, weights, center, positive, alpha, l1_ratio)

                if dnorm_XTtheta  > alpha * l1_ratio:
                    theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                    fscal(&n_samples, &theta_scaling, &theta_in[0], &inc)
                else:
                    theta_scaling = 1.

                # update norm_w2 in inner loop for Enet only
                if l1_ratio != 1:
                    weighted_norm_w2 = fweighted_norm_w2(w, weights)
                d_obj_in = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                    theta_scaling**2*weighted_norm_w2, &theta_in[0], &y[0])

                if use_accel: # also compute accelerated dual_point
                    info_dposv = create_accel_pt(
                        pb, n_samples, epoch, gap_freq, &Xw[0],
                        &thetacc[0], &last_K_Xw[0, 0], U, UtU, onesK, y)

                    if info_dposv != 0 and verbose_in:
                        pass
                        # print("linear system solving failed")

                    if epoch // gap_freq >= K:
                        dnorm_XTtheta  = dnorm_enet(
                            is_sparse, thetacc, w, X, X_data, X_indices,
                            X_indptr, notin_ws, X_mean, weights, center,
                            positive, alpha, l1_ratio)

                        if dnorm_XTtheta  > alpha * l1_ratio:
                            theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                            fscal(&n_samples, &theta_scaling, &thetacc[0], &inc)
                        else:
                            theta_scaling = 1.

                        d_obj_accel = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                                theta_scaling**2*weighted_norm_w2, &thetacc[0], &y[0])
                        if d_obj_accel > d_obj_in:
                            d_obj_in = d_obj_accel
                            fcopy(&n_samples, &thetacc[0], &inc,
                                  &theta_in[0], &inc)

                if d_obj_in > highest_d_obj_in:
                    highest_d_obj_in = d_obj_in

                # CAUTION: code does not yet include a best_theta.
                # Can be an issue in screening: dgap and theta might disagree.

                p_obj_in = primal(pb, alpha, l1_ratio, Xw, y, w, weights)
                gap_in = p_obj_in - highest_d_obj_in

                if verbose_in:
                    print("Epoch %d, primal %.10f, gap: %.2e" %
                          (epoch, p_obj_in, gap_in))
                if gap_in < tol_in:
                    if verbose_in:
                        print("Exit epoch %d, gap: %.2e < %.2e" % \
                              (epoch, gap_in, tol_in))
                    break

            for k in range(ws_size):
                j = ws[k]
                if norms_X_col[j] == 0. or weights[j] == INFINITY:
                    continue
                old_w_j = w[j]

                if pb == LASSO:
                    if is_sparse:
                        X_mean_j = X_mean[j]
                        startptr, endptr = X_indptr[j], X_indptr[j + 1]
                        for i in range(startptr, endptr):
                            w[j] += Xw[X_indices[i]] * X_data[i] / \
                                    norms_X_col[j] ** 2
                        if center:
                            R_sum = 0.
                            for i in range(n_samples):
                                R_sum += Xw[i]
                            w[j] -= R_sum * X_mean_j / norms_X_col[j] ** 2
                    else:
                        w[j] += fdot(&n_samples, &X[0, j], &inc, &Xw[0],
                                     &inc) / norms_X_col[j] ** 2

                    if positive and w[j] <= 0.:
                        w[j] = 0.
                    else:
                        if l1_ratio != 1.:
                            w[j] = ST(
                                w[j],
                                alpha * l1_ratio / norms_X_col[j] ** 2 * n_samples * weights[j]) / \
                                (1 + alpha * (1 - l1_ratio) * weights[j] /  norms_X_col[j] ** 2 * n_samples)
                        else:
                            w[j] = ST(
                                w[j],
                                alpha / norms_X_col[j] ** 2 * n_samples * weights[j])

                    # R -= (w_j - old_w_j) * (X[:, j] - X_mean[j])
                    tmp = old_w_j - w[j]
                    if tmp != 0.:
                        if is_sparse:
                            for i in range(startptr, endptr):
                                Xw[X_indices[i]] += tmp * X_data[i]
                            if center:
                                for i in range(n_samples):
                                    Xw[i] -= X_mean_j * tmp
                        else:
                            faxpy(&n_samples, &tmp, &X[0, j], &inc,
                                  &Xw[0], &inc)
                else:
                    if is_sparse:
                        startptr = X_indptr[j]
                        endptr = X_indptr[j + 1]
                        if better_lc:
                            tmp = 0.
                            for i in range(startptr, endptr):
                                tmp_exp = exp(Xw[X_indices[i]])
                                tmp += X_data[i] ** 2 * tmp_exp / \
                                       (1. + tmp_exp) ** 2
                            inv_lc[j] = 1. / tmp
                    else:
                        if better_lc:
                            tmp = 0.
                            for i in range(n_samples):
                                tmp_exp = exp(Xw[i])
                                tmp += (X[i, j] ** 2) * tmp_exp / \
                                       (1. + tmp_exp) ** 2
                            inv_lc[j] = 1. / tmp

                    tmp = 0.  # tmp = dot(Xj, y * sigmoid(-y * w)) / lc[j]
                    if is_sparse:
                        for i in range(startptr, endptr):
                            idx = X_indices[i]
                            tmp += X_data[i] * y[idx] * \
                                   sigmoid(- y[idx] * Xw[idx])
                    else:
                        for i in range(n_samples):
                            tmp += X[i, j] * y[i] * sigmoid(- y[i] * Xw[i])

                    w[j] = ST(w[j] + tmp * inv_lc[j],
                              alpha * inv_lc[j] * weights[j])

                    tmp = w[j] - old_w_j
                    if tmp != 0.:
                        if is_sparse:
                            for i in range(startptr, endptr):
                                Xw[X_indices[i]] += tmp * X_data[i]
                        else:
                            faxpy(&n_samples, &tmp, &X[0, j], &inc,
                                  &Xw[0], &inc)
        else:
            warnings.warn(
                'Inner solver did not converge at ' +
                f'epoch: {epoch}, gap: {gap_in:.2e} > {tol_in:.2e}',
                ConvergenceWarning)
    else:
        warnings.warn(
            'Objective did not converge: duality ' +
            f'gap: {gap}, tolerance: {tol}. Increasing `tol` may make the' +
            ' solver faster without affecting the results much. \n' +
            'Fitting data with very small alpha causes precision issues.',
            ConvergenceWarning)

    return np.asarray(w), np.asarray(theta), np.asarray(gaps[:t + 1])

