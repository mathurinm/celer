"""
=======================================================
Lasso path computation on Leukemia dataset
=======================================================
The example runs the Celer algorithm.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn.datasets import fetch_mldata
from celer import celer_path
from celer.plot_utils import plot_path_hist

print(__doc__)

print("Loading data...")
dataset = fetch_mldata("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = dataset.target.astype(float)
n_samples = len(y)

y -= np.mean(y)
y /= np.linalg.norm(y)

print("Starting path computation...")
alpha_max = np.max(np.abs(X.T.dot(y)))

fine = True  # fine or coarse grid
n_alphas = 100 if fine else 10
alphas = alpha_max * np.logspace(0, -2, n_alphas)

gap_freq = 10
safe = 1
verbose = 0
verbose_inner = 0
# just do the timing:
# If you want to draw the Fig, store the times in np.array [len(tols), 3]
tols = [1e-2, 1e-4, 1e-6, 1e-8]
results = np.zeros([2, len(tols)])
for tol_ix, tol in enumerate(tols):
    t0 = time.time()
    res = celer_path(X, y, alphas, max_iter=100, gap_freq=gap_freq,
                     max_epochs_inner=50000, p0=100, verbose=verbose,
                     verbose_inner=verbose_inner,
                     use_accel=1, tol=tol, safe=safe)
    results[0, tol_ix] = time.time() - t0
    print('Celer time: %.2f s' % results[0, tol_ix])
    # if you need to recover the results uncomment the next line:
    betas, thetas = res

    t0 = time.time()
    # _, coefs, dual_gaps
    _, coefs, dual_gaps = lasso_path(X, y, tol=tol, alphas=alphas / n_samples)
    results[1, tol_ix] = time.time() - t0
    coefs = coefs.T

labels = [r"\sc{CELER}", "scikit-learn"]
figsize = (7, 4)
fig = plot_path_hist(results, labels, tols, figsize, ylim=None)
plt.show()
