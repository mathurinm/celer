# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np

from scipy import sparse

from .sparse import celer_sparse
from .dense import celer_dense


def celer(X, y, alpha, beta_init=None, max_iter=100, gap_freq=10,
          max_epochs=50000, p0=10, verbose=1, verbose_inner=0,
          tol=1e-6, prune=1):
    """
    Compute the Lasso solution with the Celer algorithm.

    The minimized objective function is::

        ||y - X beta||_2^2 / (2 * n_samples) + alpha * ||beta||_1

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data or column
        sparse format (CSC) to avoid unnecessary memory duplication.

    y : array-like, shape (n_samples,)
        Observation vector.

    alpha : float
        Value of the Lasso regularization parameter.

    beta_init : array-like, shape (n_features,), optional
        Initial value for the coefficients vector.

    max_iter : int, optional
        Maximum number of outer loop (working set definition) iterations.

    max_epochs : int, optional
        Maximum number of epochs for the coordinate descent solver called on
        the subproblems.

    gap_freq : int, optional
        Number of epochs between every gap computation in the inner solver.

    p0 : int, optional
        Size of the first working set.

    verbose : (0, 1), optional
        Verbosity level of the outer loop.

    verbose_inner : (0, 1), optional
        Verbosity level of the inner solver.

    tol : float, optional
        Optimization tolerance: the solver stops when the duality gap goes
        below ``tol`` or the maximum number of iteration is reached.

    prune : (0, 1), optional
        Whether or not to use pruning when growing the working sets.

    Returns
    -------
    beta : array, shape (n_features,)
        Estimated coefficient vector.

    theta: array, shape (n_samples,)
        Dual point (potentially accelerated) when the solver exits.

    gaps : array
        Duality gap at each outer loop iteration.

    times : array
        Time elapsed since entering the solver, at each outer loop iteration.
    """
    data_is_sparse = sparse.issparse(X)
    if not data_is_sparse:
        if not np.isfortran(X):
            X = np.asfortranarray(X)
    else:
        if X.getformat() != 'csc' or not X.has_sorted_indices:
            raise TypeError("Sparse X must be in column sparse format with "
                            "sorted indices.")
    # cython function only accepts float64 for X and y:
    if X.dtype != 'float64':
        X = X.astype(np.float64)
    if y.dtype != 'float64':
        y = y.astype(np.float64)

    n_features = X.shape[1]
    if beta_init is None:
        beta_init = np.zeros(n_features)

    if data_is_sparse:
        sol = celer_sparse(X.data, X.indices, X.indptr, y, alpha,
                           beta_init, max_iter=max_iter, gap_freq=gap_freq,
                           max_epochs=max_epochs, p0=p0,
                           verbose=verbose,
                           verbose_inner=verbose_inner,
                           use_accel=1, tol=tol, prune=prune)
    else:
        sol = celer_dense(X, y, alpha,
                          beta_init, max_iter=max_iter, gap_freq=gap_freq,
                          max_epochs=max_epochs, p0=p0,
                          verbose=verbose,
                          verbose_inner=verbose_inner,
                          use_accel=1, tol=tol, prune=prune)
    return sol
