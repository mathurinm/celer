# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import time

import numpy as np
cimport numpy as np

from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy, dscal
from scipy.linalg.cython_lapack cimport dposv

from libc.math cimport fabs, sqrt, ceil
cimport cython
from utils cimport fmax, primal_value, dual_value, ST, GEOM_GROWTH, LIN_GROWTH


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double compute_dual_scaling_sparse(int n_features, double * theta,
                          double[:] X_data, int[:] X_indices,
                          int[:] X_indptr, int ws_size, int * C) nogil:
    """compute norm(X.T.dot(theta), ord=inf),
    with X restricted to features (columns) with indices in array C.
    if ws_size == n_features, C=np.arange(n_features is used)"""
    cdef double Xj_theta
    cdef double scal = 1.
    cdef int j
    cdef int Cj
    cdef int i
    cdef int startptr
    cdef int endptr
    if ws_size == n_features: # scaling wrt all features
        for j in range(n_features):
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            Xj_theta = 0
            for i in range(startptr, endptr):
                Xj_theta += X_data[i] * theta[X_indices[i]]
            Xj_theta = fabs(Xj_theta)
            scal = max(scal, Xj_theta)
    else: # scaling wrt features in C only
        for j in range(ws_size):
            Cj = C[j]
            startptr = X_indptr[Cj]
            endptr = X_indptr[Cj + 1]
            Xj_theta = 0
            for i in range(startptr, endptr):
                Xj_theta += X_data[i] * theta[X_indices[i]]
            Xj_theta = fabs(Xj_theta)

            scal = max(scal, Xj_theta)
    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_feature_prios_sparse(int n_features, double * theta,
                            double[:] X_data, int[:] X_indices,
                            int[:] X_indptr,
                            double * norms_X_col,
                            double * prios) nogil:
    cdef int j
    cdef int i
    cdef double Xj_theta
    cdef int startptr
    cdef int endptr

    for j in range(n_features):
        Xj_theta = 0
        startptr = X_indptr[j]
        endptr = X_indptr[j + 1]
        for i in range(startptr, endptr):
            Xj_theta += theta[X_indices[i]] * X_data[i]
        prios[j] = fabs(fabs(Xj_theta) - 1.) / norms_X_col[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def celer_sparse(double[:] X_data,
               int[:] X_indices,
               int[:] X_indptr,
               double[:] y,
               double alpha,
               double[:] beta_init,
               int max_iter,
               int max_epochs_inner,
               int gap_freq=10,
               float tol_ratio_inner=0.3,
               float tol=1e-6,
               int p0=100,
               int screening=0,
               int verbose=0,
               int verbose_inner=0,
               int use_accel=1,
               int growth=GEOM_GROWTH,
               int growth_factor=2,
               int return_ws_size=0,
               int prune=0,
               ):

    cdef int n_features = beta_init.shape[0]
    cdef double t0 = time.time()
    if p0 > n_features:
        p0 = n_features

    cdef int startptr, endptr

    cdef int n_samples = y.shape[0]
    cdef double[:] beta = np.empty(n_features)
    cdef int j  # features
    cdef int i  # samples
    cdef int ii
    cdef int t  # outer loop
    cdef int inc = 1
    cdef double tmp
    cdef int ws_size = 0
    cdef double p_obj
    cdef double d_obj
    cdef double highest_d_obj
    cdef double scal
    cdef double gap
    cdef double[:] prios = np.empty(n_features, dtype=float)
    cdef double[:] norms_X_col = np.empty(n_features)

    # compute norms_X_col
    for j in range(n_features):
        startptr = X_indptr[j]
        endptr = X_indptr[j + 1]
        tmp = 0.
        for i in range(startptr, endptr):
            tmp += X_data[i] ** 2
        norms_X_col[j] = sqrt(tmp)

    cdef double norm_y2 = dnrm2(&n_samples, &y[0], &inc) ** 2
    cdef double[:] invnorm_Xcols_2 = np.empty(n_features)
    cdef double[:] alpha_invnorm_Xcols_2 = np.empty(n_features)

    for j in range(n_features):
        beta[j] = beta_init[j]
        invnorm_Xcols_2[j] = 1. / norms_X_col[j] ** 2
        alpha_invnorm_Xcols_2[j] = alpha * invnorm_Xcols_2[j]

    cdef double[:] times = np.zeros(max_iter)
    cdef double[:] gaps = np.zeros(max_iter)
    cdef double[:] epochs = np.zeros(max_iter)
    cdef double[:] ws_sizes = np.zeros(max_iter)

    cdef double[:] R = np.zeros(n_samples)
    cdef double[:] theta = np.zeros(n_samples)
    cdef double[:] theta_inner = np.zeros(n_samples)  # passed to inner solver
    # and potentially used for screening if it gives a better d_obj
    cdef double d_obj_from_inner = 0

    cdef int[:] dummy_C = np.zeros(1, dtype=np.int32) # initialize with dummy value

    for t in range(max_iter):
        dcopy(&n_samples, &y[0], &inc, &R[0], &inc)
        for j in range(n_features):
            if beta[j] == 0.:
                continue
            else:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                for ii in range(startptr, endptr):
                    R[X_indices[ii]] -= beta[j] * X_data[ii]

        # theta = R / (alpha * n_samples)
        dcopy(&n_samples, &R[0], &inc, &theta[0], &inc)
        tmp = 1. / (alpha * n_samples)
        dscal(&n_samples, &tmp, &theta[0], &inc)

        scal = compute_dual_scaling_sparse(n_features, &theta[0],
                                   X_data, X_indices, X_indptr,
                                   n_features, &dummy_C[0])

        if scal > 1. :
            tmp = 1. / scal
            dscal(&n_samples, &tmp, &theta[0], &inc)

        d_obj = dual_value(n_samples, alpha, norm_y2, &theta[0],
                           &y[0])

        # also test dual point returned by inner solver after 1st iter:
        if t != 0:
            scal = compute_dual_scaling_sparse(n_features, &theta_inner[0],
                                       X_data, X_indices, X_indptr,
                                       n_features, &dummy_C[0])
            if scal > 1.:
                tmp = 1. / scal
                dscal(&n_samples, &tmp, &theta_inner[0], &inc)

            d_obj_from_inner = dual_value(n_samples, alpha, norm_y2,
                                          &theta_inner[0], &y[0])

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            dcopy(&n_samples, &theta_inner[0], &inc, &theta[0], &inc)

        if t == 0 or d_obj > highest_d_obj:
            highest_d_obj = d_obj

        p_obj = primal_value(alpha, n_samples, &R[0], n_features, &beta[0])
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        times[t] = time.time() - t0

        if verbose:
            print("############ Iteration %d  #################" % t)
            print("Primal {:.10f}".format(p_obj))
            print("Dual {:.10f}".format(highest_d_obj))
            print("Log gap %.2e" % gap)

        if gap < tol:
            if verbose:
                print("Early exit, gap: %.2e < %.2e" % (gap, tol))
            break

        set_feature_prios_sparse(n_features, &theta[0],
                          X_data, X_indices, X_indptr,
                          &norms_X_col[0], &prios[0])

        if prune:
            for j in range(n_features):
                if beta[j] != 0:
                    prios[j] = - 1
            if t == 0:
                ws_size = p0
            else:
                for j in range(ws_size):
                    prios[C[j]] = -1
                ws_size = min(n_features, 2 * ws_size)

        else:
            ws_size = 0
            for j in range(n_features):
                if beta[j] != 0:
                    prios[j] = -1.
                    ws_size += 1

            if growth == LIN_GROWTH:
                ws_size += growth_factor
            elif growth == GEOM_GROWTH:
                ws_size *= growth_factor

            if t == 0:
                ws_size = p0
            if ws_size > n_features:
                ws_size = n_features

        ws_sizes[t] = ws_size

        C = np.argpartition(np.asarray(prios), ws_size)[:ws_size].astype(np.int32)
        C.sort()
        if prune:
            tol_inner = tol
        else:
            tol_inner = tol_ratio_inner * gap

        if verbose:
            print("Solving subproblem with %d constraints" % len(C))
        # calling inner solver which will modify beta and R inplace
        epochs[t] = inner_solver_sparse(
            n_samples, n_features, ws_size, X_data, X_indices, X_indptr,
            y, alpha, beta, R, C, theta_inner, invnorm_Xcols_2,
            alpha_invnorm_Xcols_2,
            norm_y2, tol_inner, max_epochs=max_epochs_inner,
            gap_freq=gap_freq, verbose=verbose_inner,
            use_accel=use_accel)

    if return_ws_size:
        return (np.asarray(beta), np.asarray(theta),
                np.asarray(gaps[:t + 1]), np.asarray(times[:t + 1]),
                np.asarray(ws_sizes[:t + 1]))

    return (np.asarray(beta), np.asarray(theta),
            np.asarray(gaps[:t + 1]), np.asarray(times[:t + 1]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int inner_solver_sparse(int n_samples, int n_features, int ws_size,
                   double[:] X_data,
                   int[:] X_indices,
                   int[:] X_indptr,
                   double[:] y,
                   double alpha,
                   double[:] beta,
                   double[:] R,
                   int[:] C,
                   double[:] theta,
                   double[:] invnorm_Xcols_2,
                   double[:] alpha_invnorm_Xcols_2,
                   double norm_y2,
                   double eps,
                   int max_epochs,
                   int gap_freq,
                   int verbose=0,
                   int K=6,
                   int use_accel=1,
                   ):

    cdef int startptr
    cdef int endptr

    cdef int i # to iterate over samples.
    cdef int ii
    cdef int jj  # to iterate over features
    cdef int j
    cdef int k
    cdef int epoch
    cdef int start
    cdef int stop
    cdef double old_beta_j
    cdef double beta_Cj
    cdef int inc = 1
    # gap related:
    cdef double gap
    cdef double[:] gaps = np.zeros(max_epochs // gap_freq)

    # cdef double[:] theta = np.empty(n_samples)
    cdef double[:] thetaccel = np.empty(n_samples)
    cdef double dual_scale
    cdef double d_obj
    cdef double highest_d_obj = 0. # d_obj is always >=0 so this gets replaced
    # at first d_obj computation. highest_d_obj corresponds to theta = 0.
    cdef double tmp
    # acceleration variables:
    cdef double[:, :] last_K_res = np.empty([K, n_samples])
    cdef double[:, :] U = np.empty([K - 1, n_samples])
    cdef double[:, :] UtU = np.empty([K - 1, K - 1])
    cdef double[:] onesK = np.ones(K - 1)
    cdef double dual_scale_accel
    cdef double d_obj_accel

    # solving linear system in cython
    # doc at https://software.intel.com/en-us/node/468894
    cdef char char_U = 'U'
    cdef int Kminus1 = K - 1
    cdef int one = 1
    cdef double sum_z
    cdef int info_dposv

    for i in range(n_samples):
        R[i] = y[i]
    for j in range(ws_size):
        beta_Cj = beta[C[j]]
        if beta_Cj == 0.:
            continue
        else:
            startptr = X_indptr[C[j]]
            endptr = X_indptr[C[j] + 1]
            for ii in range(startptr, endptr):
                R[X_indices[ii]] -= beta_Cj * X_data[ii]


    for epoch in range(max_epochs):
        if epoch % gap_freq == 1:
            # theta = R / (alpha * n_samples)
            dcopy(&n_samples, &R[0], &inc, &theta[0], &inc)
            tmp = 1. / (alpha * n_samples)
            dscal(&n_samples, &tmp, &theta[0], &inc)

            dual_scale = compute_dual_scaling_sparse(
                n_features, &theta[0], X_data, X_indices, X_indptr,
                ws_size, &C[0])

            if dual_scale > 1. :
                tmp = 1. / dual_scale
                dscal(&n_samples, &tmp, &theta[0], &inc)

            d_obj = dual_value(n_samples, alpha, norm_y2, &theta[0],
                               &y[0])

            if use_accel: # also compute accelerated dual_point
                if epoch // gap_freq < K:
                    # last_K_res[it // f_gap] = R:
                    dcopy(&n_samples, &R[0], &inc,
                          &last_K_res[epoch // gap_freq, 0], &inc)
                else:
                    for k in range(K - 1):
                        dcopy(&n_samples, &last_K_res[k + 1, 0], &inc,
                              &last_K_res[k, 0], &inc)
                    dcopy(&n_samples, &R[0], &inc, &last_K_res[K - 1, 0], &inc)
                    for k in range(K - 1):
                        for i in range(n_samples):
                            U[k, i] = last_K_res[k + 1, i] - last_K_res[k, i]

                    for k in range(K - 1):
                        for jj in range(k, K - 1):
                            UtU[k, jj] = ddot(&n_samples, &U[k, 0], &inc,
                                              &U[jj, 0], &inc)
                            UtU[jj, k] = UtU[k, jj]

                    # refill onesK with ones because it has been overwritten
                    # by dposv
                    for k in range(K - 1):
                        onesK[k] = 1

                    dposv(&char_U, &Kminus1, &one, &UtU[0, 0], &Kminus1,
                           &onesK[0], &Kminus1,
                           &info_dposv)

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

                    tmp = 1 / (alpha * n_samples)
                    dscal(&n_samples, &tmp, &thetaccel[0], &inc)

                    dual_scale_accel = compute_dual_scaling_sparse(
                        n_features, &thetaccel[0], X_data, X_indices, X_indptr,
                        ws_size, &C[0])

                    if dual_scale_accel > 1. :
                        tmp = 1. / dual_scale_accel
                        dscal(&n_samples, &tmp, &thetaccel[0], &inc)

                    d_obj_accel = dual_value(n_samples, alpha, norm_y2,
                                             &thetaccel[0], &y[0])

                    if d_obj_accel > d_obj:
                        d_obj = d_obj_accel
                        # theta = theta_accel (theta is defined as
                        # theta_inner in outer loop)
                        dcopy(&n_samples, &thetaccel[0], &inc, &theta[0], &inc)

            if d_obj > highest_d_obj:
                highest_d_obj = d_obj

            # CAUTION: I have not yet written the code to include a best_theta.
            # This is of no consequence as long as screening is not performed. Otherwise dgap and theta might disagree.

            # we pass full beta and will ignore zero values
            gap = primal_value(alpha, n_samples, &R[0], n_features,
                               &beta[0]) - highest_d_obj
            gaps[epoch / gap_freq] = gap

            if verbose:
                print("Inner epoch %d, gap: %.2e" % (epoch, gap))
                print("primal %.9f" % (gap + highest_d_obj))
            if gap < eps:
                if verbose:
                    print("Inner: early exit at epoch %d, gap: %.2e < %.2e" % \
                        (epoch, gap, eps))
                break

        for k in range(ws_size):
            # update feature k in place, cyclically
            j = C[k]
            old_beta_j = beta[j]
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            for ii in range(startptr, endptr):
                beta[j] += R[X_indices[ii]] * X_data[ii] * invnorm_Xcols_2[j]
            # perform ST in place:
            beta[j] = ST(alpha_invnorm_Xcols_2[j] * n_samples, beta[j])
            tmp = beta[j] - old_beta_j

            # R -= (beta_j - old_beta_j) * X[:, j]
            if tmp != 0.:
                for ii in range(startptr, endptr):
                    R[X_indices[ii]] -= tmp *  X_data[ii]
    else:
        print("!!! Inner solver did not converge at epoch %d, gap: %.2e > %.2e" % \
            (epoch, gap, eps))

    return epoch
