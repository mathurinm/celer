import time
import numpy as np
from scipy import sparse
from .sparse import celer_sparse


def celer_path(X, y, alphas, max_iter=20, gap_freq=50,
               max_epochs_inner=50000, p0=10, verbose=1, verbose_inner=1,
               use_accel=0, tol=1e-6, safe=0):

    """Compute Lasso path with Celer as inner solver on sparse X"""
    n_alphas = len(alphas)
    n_samples, n_features = X.shape
    assert alphas[0] > alphas[-1]  # alphas must be given in decreasing order

    data_is_sparse = sparse.issparse(X)
    if not data_is_sparse:
        raise ValueError("not implemented")
        if not np.isfortran(X):
            X = np.asfortranarray(X)
    else:
        if X.getformat() != 'csc' or not X.has_sorted_indices:
            raise ValueError("Give X as csc matrix with sorted indices")

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
            min_ws_size = max(len(np.where(beta_init != 0)[0]), 1)
        else:
            beta_init = betas[t]
            min_ws_size = 10

        alpha = alphas[t]
        t0 = time.time()
        sol = celer_sparse(X.data, X.indices, X.indptr, y, alpha,
                           beta_init, max_iter=max_iter, gap_freq=gap_freq,
                           max_epochs_inner=max_epochs_inner, p0=min_ws_size,
                           verbose=verbose,
                           verbose_inner=verbose_inner,
                           use_accel=use_accel, tol=tol, safe=safe)

        all_times[t] = time.time() - t0
        betas[t], thetas[t], final_gaps[t] = sol[0], sol[1], sol[3][-1]
        if final_gaps[t] > tol:
            print("-------------WARNING: did not converge, t = %d" % t)
            print("gap = %.1e , tol = %.1e" % (final_gaps[t], tol))
    return betas, thetas
