#cython: language_level=3
cimport cython
cimport numpy as np

import numpy as np
import warnings
from cython cimport floating
from libc.math cimport fabs, sqrt
from numpy.math cimport INFINITY
from sklearn.exceptions import ConvergenceWarning

from .cython_utils cimport fscal, fcopy, fnrm2, fdot, faxpy
from .cython_utils cimport LASSO, create_accel_pt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void BST(int n_tasks, floating * x, floating u) nogil:
    cdef int inc = 1
    cdef int k
    cdef floating tmp
    cdef floating norm_x = fnrm2(&n_tasks, x, &inc)
    if norm_x < u:
        for k in range(n_tasks):
            x[k] = 0.
    else:
        tmp = 1. - u / norm_x
        fscal(&n_tasks, &tmp, x, &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating dual_scaling_mtl(
        int n_features, int n_samples, int n_tasks, floating[::1, :] theta,
        floating[::1, :] X, int ws_size, int * C, int * screened,
        floating * Xj_theta) nogil:
    cdef int ind, j, k
    cdef int inc = 1
    cdef floating tmp
    cdef floating dnorm_XTtheta = 0.

    if ws_size == n_features:
        for j in range(n_features):
            if screened[j]:
                continue
            for k in range(n_tasks):
                Xj_theta[k] = fdot(&n_samples, &theta[0, k], &inc, &X[0, j], &inc)
            tmp = fnrm2(&n_tasks, &Xj_theta[0], &inc)
            if tmp > dnorm_XTtheta:
                dnorm_XTtheta = tmp
    else:
        for ind in range(ws_size):
            j = C[ind]
            for k in range(n_tasks):
                Xj_theta[k] = fdot(&n_samples, &theta[0, k], &inc, &X[0, j], &inc)
            tmp = fnrm2(&n_tasks, &Xj_theta[0], &inc)
            if tmp > dnorm_XTtheta:
                dnorm_XTtheta = tmp
    return dnorm_XTtheta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_prios_mtl(
        floating[:, ::1] W, int[:] screened,
        floating[::1, :] X, floating[::1, :] theta, floating alpha, floating[:] norms_X_col,
        floating[:] Xj_theta, floating[:] prios, floating radius,
        int * n_screened) nogil:
    cdef int j, k
    cdef int inc = 1
    cdef floating nrm = 0.
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_tasks = W.shape[1]

    for j in range(n_features):
        if screened[j]:
            prios[j] = INFINITY
            continue
        for k in range(n_tasks):
            Xj_theta[k] = fdot(&n_samples, &theta[0, k], &inc, &X[0, j], &inc)

        nrm = fnrm2(&n_tasks, &Xj_theta[0], &inc)
        prios[j] = (alpha - nrm) / norms_X_col[j]
        if prios[j] > radius:
            # screen only if W[j, :] is zero:
            for k in range(n_tasks):
                if W[j, k] != 0:
                    break
            else:
                screened[j] = True
                n_screened[0] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating dual_mtl(
        int n_samples, int n_tasks, floating[::1, :] theta, floating[::1, :] Y,
        floating norm_Y2) nogil:
    cdef int inc = 1
    cdef int i, k
    cdef floating d_obj = 0.

    for k in range(n_tasks):
        for i in range(n_samples):
            d_obj -= (Y[i, k] / n_samples - theta[i, k]) ** 2
    d_obj *= 0.5 * n_samples
    d_obj += norm_Y2 / (2. * n_samples)
    return d_obj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating primal_mtl(
        int n_samples, int n_features, int n_tasks,
        floating[:, ::1] W, floating alpha, floating[::1, :] R) nogil:
    cdef int inc = 1
    cdef int j, k
    cdef int n_obs = n_samples * n_tasks
    cdef floating p_obj = fnrm2(&n_obs, &R[0, 0], &inc) ** 2 / (2. * n_samples)

    for j in range(n_features):
        for k in range(n_tasks):
            if W[j, k]:
                p_obj += alpha * fnrm2(&n_tasks, &W[j, 0], &inc)
                break

    return p_obj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def celer_mtl(
        floating[::1, :] X, floating[::1, :] Y, floating alpha,
        floating[:, ::1] W, floating[::1, :] R, floating[::1, :] theta,
        floating[:] norms_X_col, int max_iter, int max_epochs,
        int gap_freq=10, floating tol_ratio=0.3, float tol=1e-6, int p0=100,
        int verbose=0, bint use_accel=1, bint prune=1,
        int K=6):

    cdef int verbose_inner = max(0, verbose - 1)
    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int n_samples = Y.shape[0]
    cdef int n_tasks = Y.shape[1]
    cdef int n_features = W.shape[0]


    if p0 > n_features:
        p0 = n_features

    cdef int i, j, k, t
    cdef int inc = 1
    cdef floating tmp, theta_scaling
    cdef int n_obs = n_samples * n_tasks
    cdef int ws_size
    cdef int nnz = 0
    cdef floating p_obj, d_obj, highes_d_obj, gap, radius, dnorm_XTtheta
    cdef int n_screened = 0
    cdef floating[:] prios = np.empty(n_features, dtype=dtype)
    cdef int[:] screened = np.zeros(n_features, dtype=np.int32)
    cdef floating[:] Xj_theta = np.empty(n_tasks, dtype=dtype)

    cdef floating norm_Y2 = fnrm2(&n_obs, &Y[0, 0], &inc) ** 2
    # scale tolerance to account for small or large Y:
    tol *= norm_Y2 / n_samples

    cdef floating[::1, :] theta_inner = np.zeros((n_samples, n_tasks),
                                                  dtype=dtype, order='F')
    cdef floating[::1, :] theta_to_use

    cdef floating d_obj_from_inner = 0
    cdef int[:] dummy_C = np.zeros(1, dtype=np.int32)
    cdef int[:] all_features = np.arange(n_features, dtype=np.int32)
    cdef int[:] C
    cdef floating tol_inner

    for t in range(max_iter):
        # if t != 0: TODO
        p_obj = primal_mtl(n_samples, n_features, n_tasks, W, alpha, R)
        # theta = R :
        fcopy(&n_obs, &R[0, 0], &inc, &theta[0, 0], &inc)

        dnorm_XTtheta = dual_scaling_mtl(
            n_features, n_samples, n_tasks, theta, X, n_features,
            &dummy_C[0], &screened[0], &Xj_theta[0])

        if dnorm_XTtheta > alpha:
            theta_scaling = alpha / dnorm_XTtheta
            fscal(&n_obs, &theta_scaling, &theta[0, 0], &inc)
        d_obj = dual_mtl(n_samples, n_tasks, theta, Y, norm_Y2)

        if t > 0:
            dnorm_XTtheta = dual_scaling_mtl(
                n_features, n_samples, n_tasks, theta_inner, X,
                n_features, &dummy_C[0], &screened[0], &Xj_theta[0])

            if dnorm_XTtheta > alpha:
                theta_scaling = alpha / dnorm_XTtheta
                fscal(&n_obs, &theta_scaling, &theta_inner[0, 0], &inc)
            d_obj_from_inner = dual_mtl(
                n_samples, n_tasks, theta_inner, Y, norm_Y2)
            if d_obj_from_inner > d_obj:
                d_obj = d_obj_from_inner

        gap = p_obj - d_obj
        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")

        if gap <= tol + 1e-16:
            if verbose:
                print("\nEarly exit, gap %.2e < %.2e" % (gap, tol))
            break

        radius = sqrt(2 * gap / n_samples)
        # TODO prios could be computed along with scaling
        set_prios_mtl(
            W, screened, X, theta, alpha, norms_X_col, Xj_theta, prios, radius,
            &n_screened)

        if t == 0:
            ws_size = p0
            # prios[j] = -1 if W[j, :].any()
            for j in range(n_features):
                for k in range(n_tasks):
                    if W[j, k]:
                        prios[j] = -1
                        break
        else:
            nnz = 0
            if prune:
                for j in range(n_features):
                    if W[j, 0]:
                        prios[j] = -1
                        nnz += 1
                ws_size = 2 * nnz
            else:
                for k in range(ws_size):
                    if not screened[C[k]]:
                        prios[C[k]] = -1
                ws_size = 2 * ws_size
        ws_size = min(n_features - n_screened, ws_size)

        if ws_size == n_features:
            C = all_features
        else:
            C = np.sort(np.argpartition(prios, ws_size)[:ws_size].astype(np.int32))

        if prune:
            tol_inner = tol_ratio * gap
        else:
            tol_inner = tol
        if verbose:
            print(", %d feats in subpb (%d left)" % (len(C), n_features - n_screened))

        inner_solver(
            n_samples, n_features, n_tasks, ws_size, X, Y, alpha, W, R, C,
            theta_inner, norms_X_col, norm_Y2, tol_inner, max_epochs,
            gap_freq, verbose_inner, use_accel, K)

    else:
        warnings.warn(
            'Objective did not converge: duality ' +
            f'gap: {gap}, tolerance: {tol}. Increasing `tol` may make the' +
            ' solver faster without affecting the results much. \n' +
            'Fitting data with very small alpha causes precision issues.',
            ConvergenceWarning)
    return (np.asarray(W), np.asarray(theta), gap)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void inner_solver(
        int n_samples, int n_features, int n_tasks, int ws_size,
        floating[::1, :] X, floating[::1, :]  Y, floating alpha,
        floating[:, ::1] W, floating[::1, :] R, int[:] C,
        floating[::1, :] theta, floating[:] norms_X_col,
        floating norm_Y2, floating eps, int max_epochs,
        int gap_freq, bint verbose, bint use_accel=1,
        int K=6):

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef floating p_obj, d_obj, gap
    cdef floating highest_d_obj = 0.
    cdef int i, j, k, epoch, ind
    cdef floating[:] old_Wj = np.empty(n_tasks, dtype=dtype)
    cdef int inc = 1
    cdef int n_obs = n_samples * n_tasks
    cdef floating tmp, dnorm_XTtheta, theta_scaling
    cdef int[:] dummy_screened = np.zeros(1, dtype=np.int32)
    cdef floating[:] Xj_theta = np.empty(n_tasks, dtype=dtype)


    # acceleration:
    cdef floating[::1, :] theta_acc = np.empty([n_samples, n_tasks],
                                          dtype=dtype, order='F')
    cdef floating d_obj_acc = 0
    cdef floating[:, :] last_K_R = np.empty([K, n_obs], dtype=dtype)
    cdef floating[:, :] U = np.empty([K - 1, n_obs], dtype=dtype)
    cdef floating[:, :] UtU = np.empty([K - 1, K - 1], dtype=dtype)
    cdef floating[:] onesK = np.ones(K - 1, dtype=dtype)
    # doc at https://software.intel.com/en-us/node/468894
    cdef char * char_U = 'U'
    cdef int Kminus1 = K - 1
    cdef int one = 1
    cdef floating sum_z
    cdef int info_dposv
    ####################

    for epoch in range(max_epochs):
        if epoch > 0 and epoch % gap_freq == 0:
            p_obj = primal_mtl(n_samples, n_features, n_tasks, W, alpha, R)
            fcopy(&n_obs, &R[0, 0], &inc, &theta[0, 0], &inc)

            tmp = 1. / n_samples
            fscal(&n_obs, &tmp, &theta[0, 0], &inc)

            dnorm_XTtheta = dual_scaling_mtl(
                n_features, n_samples, n_tasks, theta, X, ws_size,
                &C[0], &dummy_screened[0], &Xj_theta[0])

            if dnorm_XTtheta > alpha:
                theta_scaling = alpha / dnorm_XTtheta
                fscal(&n_obs, &theta_scaling, &theta[0, 0], &inc)
            d_obj = dual_mtl(n_samples, n_tasks, theta, Y, norm_Y2)

            if use_accel:
                create_accel_pt(
                    LASSO, n_obs, epoch, gap_freq,
                    &R[0, 0], &theta_acc[0, 0], &last_K_R[0, 0], U, UtU,
                    onesK, onesK)  # passing onesK as y which is ignored
                    # account for wrong n_samples passed to create_accel_pt
                tmp = n_tasks
                fscal(&n_obs, &tmp, &theta_acc[0, 0], &inc)
                if epoch // gap_freq >= K:
                    dnorm_XTtheta = dual_scaling_mtl(
                        n_features, n_samples, n_tasks, theta_acc, X, ws_size,
                        &C[0], &dummy_screened[0], &Xj_theta[0])

                    if dnorm_XTtheta > alpha:
                        theta_scaling = alpha / dnorm_XTtheta
                        fscal(&n_obs, &theta_scaling, &theta_acc[0, 0], &inc)
                    d_obj_acc = dual_mtl(
                        n_samples, n_tasks, theta_acc, Y, norm_Y2)
                    if d_obj_acc > d_obj:
                        d_obj = d_obj_acc
                        fcopy(&n_obs, &theta_acc[0, 0], &inc, &theta[0, 0],
                              &inc)
            highest_d_obj = max(highest_d_obj, d_obj)
            gap = p_obj - highest_d_obj
            if verbose:
                print("Inner epoch %d, primal %.10f, gap: %.2e" % (epoch, p_obj, gap))
            if gap < eps:
                if verbose:
                    print("Inner: early exit at epoch %d, gap: %.2e < %.2e" % \
                        (epoch, gap, eps))
                break

        for ind in range(ws_size):
            j = C[ind]
            fcopy(&n_tasks, &W[j, 0], &inc, &old_Wj[0], &inc)

            for k in range(n_tasks):
                tmp = fdot(&n_samples, &X[0, j], &inc, &R[0, k], &inc)
                W[j, k] += tmp / norms_X_col[j] ** 2
            BST(n_tasks, &W[j, 0], alpha / norms_X_col[j] ** 2 * n_samples)

            for k in range(n_tasks):
                tmp = old_Wj[k] - W[j, k]
                if tmp != 0.:
                    # for i in range(n_samples):
                    #     R[i, k] += tmp * X[i, j]
                    faxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0, k], &inc)

