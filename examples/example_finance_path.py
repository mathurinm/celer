import numpy as np
from scipy import sparse
import time
from celer import celer_path
from download_preprocess_finance import download_preprocess_finance


if __name__ == "__main__":
    print("Loading data...")
    try:
        X = sparse.load_npz("./data/finance_data_preprocessed.npz")
        y = np.load("./data/finance_target_preprocessed.npy")
    except FileNotFoundError:
        download_preprocess_finance()
        X = sparse.load_npz("./data/finance_data_preprocessed.npz")
        y = np.load("./data/finance_target_preprocessed.npy")

    print("Starting path computation...")
    alpha_max = np.max(np.abs(X.T.dot(y)))

    fine = False  # fine or coarse grid
    n_alphas = 100 if fine else 10
    alphas = alpha_max * np.logspace(0, -2, n_alphas)

    gap_freq = 10
    safe = 1
    verbose = 0
    verbose_inner = 0
    # just do the timing:
    # If you want to draw the Fig, store the times in np.array [len(tols), 3]
    for tol in [1e-2, 1e-4, 1e-6, 1e-8]:
        t0 = time.time()
        res = celer_path(X, y, alphas, max_iter=100, gap_freq=gap_freq,
                         max_epochs_inner=50000, p0=100, verbose=verbose,
                         verbose_inner=verbose_inner,
                         use_accel=1, tol=tol, safe=safe)
        print('Celer:', time.time() - t0)

        betas, thetas = res
