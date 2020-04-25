# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import warnings
import numpy as np

from scipy import sparse
from numpy.linalg import norm
from sklearn.utils import check_array
from sklearn.exceptions import ConvergenceWarning

from .lasso_fast import celer
from .group_lasso_fast import group_lasso
from .cython_utils import compute_norms_X_col, compute_Xw
from .multitask_fast import celer_mtl
from .PN_logreg import newton_celer

LASSO = 0
LOGREG = 1
GRPLASSO = 2


def celer_path(X, y, pb, eps=1e-3, n_alphas=100, alphas=None,
               coef_init=None, max_iter=20, gap_freq=10, max_epochs=50000,
               p0=10, verbose=0, verbose_inner=0, tol=1e-6, prune=0,
               groups=None, return_thetas=False, X_offset=None, X_scale=None,
               return_n_iter=False, positive=False):
    r"""Compute optimization path with Celer as inner solver.

    With `n = len(y)` the number of samples, the losses are:

    Lasso:

    .. math::
        \frac{||y - X w||_2^2}{2 n} + \alpha ||w||_1
    Logreg:

    .. math::
        \sum_{i=1}^n \text{log} \,(1 + e^{-y_i x_i^\top w}) +
        \alpha  ||w||_1

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data or column
        sparse format (CSC) to avoid unnecessary memory duplication.

    y : ndarray, shape (n_samples,)
        Target values

    pb : "lasso" | "logreg" | "grouplasso"
        Optimization problem to solve.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min = 1e-3 * alpha_max``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    coef_init : ndarray, shape (n_features,) | None, optional, (defualt=None)
        Initial value of coefficients. If None, np.zeros(n_features) is used.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    gap_freq : int, optional
        Number of (block) coordinate descent epochs between each duality gap
        computations.

    max_epochs : int, optional
        Maximum number of (block) CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    verbose : bool or integer, optional
        Amount of verbosity.

    verbose_inner : bool or integer
        Amount of verbosity in the inner solver.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    groups : int or list of ints or list of list of ints, optional
        Used for the group Lasso only. See the documentation of the
        GroupLasso class.

    return_thetas : bool, optional
        If True, dual variables along the path are returned.

    use_PN : bool, optional
        If pb == "logreg", use ProxNewton solver instead of coordinate
        descent.

    X_offset : np.array, shape (n_features,), optional
        Used to center sparse X without breaking sparsity. Mean of each column.
        See sklearn.linear_model.base._preprocess_data().

    X_scale : np.array, shape (n_features,), optional
        Used to scale centered sparse X without breaking sparsity. Norm of each
        centered column. See sklearn.linear_model.base._preprocess_data().

    return_n_iter : bool, optional
        If True, number of iterations along the path are returned.

    positive : bool, optional (default=False)
        If True and pb == "lasso", forces the coefficients to be positive.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        Duality gaps returned by the solver along the path.

    thetas : array, shape (n_alphas, n_samples)
        The dual variables along the path.
        (Is returned only when ``return_thetas`` is set to True).
    """
    assert pb in ("lasso", "logreg")
    if pb == "lasso":
        pb = LASSO
    elif pb.lower() == "logreg":
        pb = LOGREG
        if set(y) - set([-1.0, 1.0]):
            raise ValueError(
                "y must contain only -1. or 1 values. Got %s " % (set(y)))
    elif pb.lower() == "grouplasso":
        pb = GRPLASSO
        solver = "celer"
        if groups is None:
            raise ValueError(
                "Groups must be specified for the group lasso problem.")
        grp_ptr, grp_indices = _grp_converter(groups, X.shape[1])
        n_groups = len(grp_ptr) - 1
    else:
        raise ValueError("Unsupported problem: %s" % pb)

    is_sparse = sparse.issparse(X)

    X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                    order='F', copy=False)
    y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                    ensure_2d=False)

    n_samples, n_features = X.shape

    if X_offset is not None:
        # As sparse matrices are not actually centered we need this
        # to be passed to the CD solver.
        X_sparse_scaling = X_offset / X_scale
        X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
    else:
        X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    if alphas is None:
        if pb == LASSO:
            if positive:
                alpha_max = np.max(X.T.dot(y)) / n_samples
            else:
                alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
        elif pb == LOGREG:
            alpha_max = norm(X.T @ y, ord=np.inf) / 2.
        elif pb == GRPLASSO:
            alpha_max = 0
            for g in range(n_groups):
                X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
                alpha_max = max(alpha_max, norm(X_g.T @ y, ord=2))
            alpha_max /= n_samples

        alphas = alpha_max * np.geomspace(1, eps, n_alphas,
                                          dtype=X.dtype)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    coefs = np.zeros((n_features, n_alphas), order='F', dtype=X.dtype)
    thetas = np.zeros((n_alphas, n_samples), dtype=X.dtype)
    dual_gaps = np.zeros(n_alphas)

    if return_n_iter:
        n_iters = np.zeros(n_alphas, dtype=int)

    if is_sparse:
        X_dense = np.empty([1, 1], order='F', dtype=X.data.dtype)
        X_data = X.data
        X_indptr = X.indptr
        X_indices = X.indices
    else:
        X_dense = X
        X_data = np.empty([1], dtype=X.dtype)
        X_indices = np.empty([1], dtype=np.int32)
        X_indptr = np.empty([1], dtype=np.int32)

    if pb == GRPLASSO:
        lc_grp = np.zeros(n_groups, dtype=X_dense.dtype)
        for g in range(n_groups):
            X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
            lc_grp[g] = norm(X_g, ord=2)

    else:
        # TODO harmonize names
        norms_X_col = np.zeros(n_features, dtype=X_dense.dtype)
        compute_norms_X_col(
            is_sparse, norms_X_col, n_samples, X_dense, X_data,
            X_indices, X_indptr, X_sparse_scaling)

    # do not skip alphas[0], it is not always alpha_max
    for t in range(n_alphas):
        alpha = alphas[t]
        if verbose:
            to_print = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
            print("#" * len(to_print))
            print(to_print)
            print("#" * len(to_print))
        if t > 0:
            w = coefs[:, t - 1].copy()
            theta = thetas[t - 1].copy()
            p0 = max(len(np.where(w != 0)[0]), 1)
        else:
            if coef_init is not None:
                w = coef_init.copy()
                p0 = max((w != 0.).sum(), p0)
                # y - Xw for Lasso, Xw for Logreg:
                Xw = np.zeros(n_samples, dtype=X.dtype)
                compute_Xw(
                    is_sparse, pb, Xw, w, y, X_sparse_scaling.any(), X_dense,
                    X_data, X_indices, X_indptr, X_sparse_scaling)
            else:
                w = np.zeros(n_features, dtype=X.dtype)
                Xw = y.copy() if pb == LASSO else np.zeros(n_samples, X.dtype)

            if pb == LASSO:
                theta = Xw / np.linalg.norm(X.T.dot(Xw), ord=np.inf)
            else:
                theta = y / (1 + np .exp(y * Xw)) / alpha
                theta /= np.linalg.norm(X.T.dot(theta), ord=np.inf)

        # celer modifies w, Xw, and theta in place:
        if pb == GRPLASSO:  # TODO this if else scheme is complicated
            dual_gaps[t] = group_lasso(
                is_sparse, X, grp_indices, grp_ptr, X_data, X_indices,
                X_indptr, X_sparse_scaling, y, alpha, w, Xw, theta, lc_grp, tol, max_epochs, gap_freq)  # TODO max_iter
            coefs[:, t], thetas[t] = w, theta
        elif solver == "celer":
            sol = celer(
                is_sparse, pb,
                X_dense, X_data, X_indices, X_indptr, X_sparse_scaling, y,
                alpha, w, Xw, theta, norms_X_col,
                max_iter=max_iter, gap_freq=gap_freq, max_epochs=max_epochs,
                p0=p0, verbose=verbose, verbose_inner=verbose_inner,
                use_accel=1, tol=tol, prune=prune, positive=positive)
        else:  # pb == LOGREG and use_PN
            sol = newton_celer(
                is_sparse, X_dense, X_data, X_indices, X_indptr, y, alpha, w,
                max_iter, verbose, verbose_inner, tol, prune, p0, True, K=6,
                growth=2, blitz_sc=False)

        coefs[:, t], thetas[t], dual_gaps[t] = sol[0], sol[1], sol[2][-1]
        if return_n_iter:
            n_iters[t] = len(sol[2])

        if dual_gaps[t] > tol:
            warnings.warn(
                'Objective did not converge. Increasing `tol` may make the' +
                ' solver faster without affecting the results much.' +
                ' Fitting with very small alpha may cause precision problems.',
                ConvergenceWarning)

    results = alphas, coefs, dual_gaps
    if return_thetas:
        results += (thetas,)
    if return_n_iter:
        results += (n_iters,)

    return results


def _grp_converter(groups, n_features):
    if isinstance(groups, int):
        grp_size = groups
        if n_features % grp_size != 0:
            raise ValueError("n_features %d is not a multitple of the desired"
                             "group size %d" % (n_features, grp_size))
        n_groups = n_features // grp_size
        grp_ptr = grp_size * np.arange(n_groups + 1)
        grp_indices = np.arange(n_features)
    elif isinstance(groups, list) and isinstance(groups[0], int):
        grp_indices = np.arange(n_features).astype(np.int32)
        grp_ptr = np.cumsum(np.hstack([[0], groups]))
    elif isinstance(groups, list) and isinstance(groups[0], list):
        grp_sizes = np.array([len(l) for l in groups])
        grp_ptr = np.cumsum(np.hstack([[0], grp_sizes]))
        grp_indices = np.array([idx for grp in groups for idx in grp])
    else:
        raise ValueError("Unsupported group format.")
    return grp_ptr.astype(np.int32), grp_indices.astype(np.int32)


# TODO put this in logreg_path with solver variable
def PN_solver(X, y, alpha, w_init, max_iter, verbose=False,
              verbose_inner=False, tol=1e-4, prune=True, p0=10,
              use_accel=True, K=6, growth=2, blitz_sc=False):
    is_sparse = sparse.issparse(X)
    w = w_init.copy()
    if is_sparse:
        X_dense = np.empty([2, 2], order='F')
        X_indices = X.indices.astype(np.int32)
        X_indptr = X.indptr.astype(np.int32)
        X_data = X.data
    else:
        X_dense = X
        X_indices = np.empty([1], dtype=np.int32)
        X_indptr = np.empty([1], dtype=np.int32)
        X_data = np.empty([1])

    return newton_celer(
        is_sparse, X_dense, X_data, X_indices, X_indptr, y, alpha, w,
        max_iter, verbose, verbose_inner, tol, prune, p0, use_accel, K,
        growth=growth, blitz_sc=blitz_sc)


def mtl_path(
        X, Y, eps=1e-2, n_alphas=100, alphas=None, max_iter=100, gap_freq=10,
        max_epochs=50000, p0=10, verbose=False, verbose_inner=False, tol=1e-6,
        prune=True, use_accel=True, return_thetas=False, K=6,
        coef_init=None):
    X = check_array(X, "csc", dtype=[
        np.float64, np.float32], order="F", copy=False)
    # TODO check Y is fortran too
    n_samples, n_features = X.shape
    n_tasks = Y.shape[1]
    if alphas is None:
        alpha_max = np.max(norm(X.T @ Y, ord=2, axis=1)) / n_samples
        alphas = alpha_max * \
            np.geomspace(1, eps, n_alphas, dtype=X.dtype)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    if coef_init is None:
        coefs = np.zeros((n_features, n_tasks, n_alphas), order="F",
                         dtype=X.dtype)
    else:
        coefs = np.swapaxes(coef_init, 0, 1).copy('F')
    thetas = np.zeros((n_alphas, n_samples, n_tasks), dtype=X.dtype)
    gaps = np.zeros(n_alphas)

    norms_X_col = np.linalg.norm(X, axis=0)
    Y = np.asfortranarray(Y)
    R = Y.copy(order='F')
    theta = np.zeros_like(R, order='F')

    # do not skip alphas[0], it is not always alpha_max
    for t in range(n_alphas):
        if verbose:
            print("#" * 60)
            print("##### Computing %dth alpha" % (t + 1))
            print("#" * 60)
        if t > 0:
            W = coefs[:, :, t - 1].copy()
            p_t = max(len(np.where(W[:, 0] != 0)[0]), p0)
        else:
            W = coefs[:, :, t].copy()
            p_t = 10

        alpha = alphas[t]

        sol = celer_mtl(
            X, Y, alpha, W, R, theta, norms_X_col, p0=p_t, tol=tol,
            prune=prune, max_iter=max_iter, max_epochs=max_epochs,
            verbose=verbose_inner, use_accel=use_accel, gap_freq=gap_freq,
            K=K)

        coefs[:, :, t], thetas[t], gaps[t] = sol[0], sol[1], sol[2]

    coefs = np.swapaxes(coefs, 0, 1).copy('F')

    if return_thetas:
        return alphas, coefs, gaps, thetas

    return alphas, coefs, gaps
