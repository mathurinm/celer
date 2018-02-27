import time
import numpy as np
from .wrapper import celer


def celer_path(X, y, alphas=None, max_iter=20, gap_freq=10,
               max_epochs_inner=50000, p0=10, verbose=1, verbose_inner=1,
               tol=1e-6, prune=0):
    """Compute Lasso path with Celer as inner solver.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data or column
        sparse format (CSC) to avoid unnecessary memory duplication.

    y : ndarray, shape (n_samples,)
        Target values

    alphas : ndarray, shape (n_alphas,)
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    max_iter : int, optional
        The maximum number of iterations

    gap_freq : int, optional
        Number of coordinate descent epochs between each duality gap
        computations.

    max_epochs_inner : int, optional
        Maximum number of CD epochs on the subproblem.

    p0 : int, optional
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    verbose_inner : bool or integer
        Amount of verbosity in the inner solver.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : 0 | 1
        Whether or not to use pruning when growing working sets.

    Returns
    -------
    betas : array, shape (n_alphas, n_features)
        Coefficients along the path.

    thetas : array, shape (n_alphas, n_samples)
        The dual variables along the path.

    final_gaps : array, shape (n_alphas,)
        Duality gaps returned by the solver along the path.
    """
    n_alphas = len(alphas)
    n_samples, n_features = X.shape
    assert alphas[0] > alphas[-1]  # alphas must be given in decreasing order

    betas = np.zeros((n_alphas, n_features))
    thetas = np.zeros((n_alphas, n_samples))
    final_gaps = np.zeros(n_alphas)
    all_times = np.zeros(n_alphas)

    # skip alpha_max and use decreasing alphas
    thetas[0] = y / alphas[0]  # don't forget to set this one
    for t in range(1, n_alphas):
        if verbose:
            print("#" * 60)
            print(" ##### Computing %dth alpha" % (t + 1))
            print("#" * 60)
        if t > 1:
            beta_init = betas[t - 1].copy()
            p_t = max(len(np.where(beta_init != 0)[0]), 1)
        else:
            beta_init = betas[t]
            p_t = 10

        alpha = alphas[t]
        t0 = time.time()
        sol = celer(X, y, alpha,
                    beta_init, max_iter=max_iter, gap_freq=gap_freq,
                    max_epochs_inner=max_epochs_inner, p0=p_t,
                    verbose=verbose, verbose_inner=verbose_inner,
                    tol=tol, prune=prune)

        all_times[t] = time.time() - t0
        betas[t], thetas[t], final_gaps[t] = sol[0], sol[1], sol[3][-1]
        if final_gaps[t] > tol:
            print("-----WARNING: solver did not converge, t = %d" % t)
            print("gap = %.1e , tol = %.1e" % (final_gaps[t], tol))
    return betas, thetas, final_gaps
