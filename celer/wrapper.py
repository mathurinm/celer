# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np

from .homotopy import celer_path


def celer(X, y, alpha, w_init=None, max_iter=100, gap_freq=10,
          max_epochs=50000, p0=10, verbose=1, verbose_inner=0,
          tol=1e-6, prune=0, positive=False, return_n_iter=False):
    """
    Compute the Lasso solution with the Celer algorithm.

    The minimized objective function is::

        ||y - X w||_2^2 / (2 * n_samples) + alpha * ||w||_1

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data or column
        sparse format (CSC) to avoid unnecessary memory duplication.

    y : array-like, shape (n_samples,)
        Observation vector.

    alpha : float
        Value of the Lasso regularization parameter.

    w_init : array-like, shape (n_features,), optional
        Initial value for the coefficients vector.

    max_iter : int, optional
        Maximum number of outer loop (working set definition) iterations.

    gap_freq : int, optional
        Number of epochs between every gap computation in the inner solver.

    max_epochs : int, optional
        Maximum number of epochs for the coordinate descent solver called on
        the subproblems.

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

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    return_n_iter : bool, optional (default=False)
        Whether or not to return the number of iterations (ie the number of
        subproblems solved).

    Returns
    -------
    w : array, shape (n_features,)
        Estimated coefficient vector.

    theta : array, shape (n_samples,)
        Dual point (potentially accelerated) when the solver exits.

    gap : float
        Final duality gap.

    n_iter : int
        Number of iterations (subproblems solved). Returned only if
        return_n_iter is True.
    """

    results = celer_path(
        X, y, alphas=np.array([alpha]), coef_init=w_init,
        max_iter=max_iter, gap_freq=gap_freq,
        max_epochs=max_epochs, p0=p0, verbose=verbose,
        verbose_inner=verbose_inner, tol=tol, prune=prune, return_thetas=True,
        return_n_iter=return_n_iter, positive=positive)

    w = results[1].T[0]
    gap = results[2][0]
    theta = results[3][0]

    res = (w, theta, gap)
    if return_n_iter:
        n_iter = results[4][0]
        res += (n_iter,)

    return res
