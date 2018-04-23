# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import time
import numpy as np

from .wrapper import celer


def celer_path(X, y, eps=1e-3, n_alphas=100, alphas=None, max_iter=20,
               gap_freq=10, max_epochs=50000, p0=10, verbose=0,
               verbose_inner=0, tol=1e-6, prune=0, return_thetas=False):
    """Compute Lasso path with Celer as inner solver.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data or column
        sparse format (CSC) to avoid unnecessary memory duplication.

    y : ndarray, shape (n_samples,)
        Target values

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min = 1e-3 * alpha_max``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    gap_freq : int, optional
        Number of coordinate descent epochs between each duality gap
        computations.

    max_epochs : int, optional
        Maximum number of CD epochs on each subproblem.

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

    return_thetas : bool, optional
        If True, dual variables along the path are returned.

    Returns
    -------
    alpha : array, shape (n,_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        Duality gaps returned by the solver along the path.

    thetas : array, shape (n_alphas, n_samples)
        The dual variables along the path.
        (Is returned only when ``return_thetas`` is set to True).
    """
    n_samples, n_features = X.shape
    if alphas is None:
        alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
        alphas = alpha_max * np.logspace(0, np.log10(eps), n_alphas)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    coefs = np.zeros((n_features, n_alphas), order='F')  # sklearn API
    thetas = np.zeros((n_alphas, n_samples))
    dual_gaps = np.zeros(n_alphas)
    all_times = np.zeros(n_alphas)

    # do not skip alphas[0], it is not always alpha_max
    for t in range(n_alphas):
        if verbose:
            print("#" * 60)
            print(" ##### Computing %dth alpha" % (t + 1))
            print("#" * 60)
        if t > 0:
            beta_init = coefs[:, t - 1].copy()
            p_t = max(len(np.where(beta_init != 0)[0]), 1)
        else:
            beta_init = coefs[:, t].copy()
            p_t = 10

        alpha = alphas[t]
        t0 = time.time()
        sol = celer(X, y, alpha,
                    beta_init, max_iter=max_iter, gap_freq=gap_freq,
                    max_epochs=max_epochs, p0=p_t,
                    verbose=verbose, verbose_inner=verbose_inner,
                    tol=tol, prune=prune)

        all_times[t] = time.time() - t0
        coefs[:, t], thetas[t], dual_gaps[t] = sol[0], sol[1], sol[2][-1]

    if return_thetas:
        return alphas, coefs, dual_gaps, thetas
    else:
        return alphas, coefs, dual_gaps
