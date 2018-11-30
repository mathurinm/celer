"""
=======================================================
Lasso path computation on Leukemia dataset
=======================================================

The example runs the Celer algorithm on the Leukemia
dataset which is a dense dataset.

Running time is compared with the scikit-learn implementation.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path
from sklearn.datasets import fetch_openml

from celer import celer_path
from celer.plot_utils import plot_path_hist

print(__doc__)

print("Loading data...")
dataset = fetch_openml("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = 2 * ((dataset.target != "AML") - 0.5)
n_samples = len(y)

y -= np.mean(y)
y /= np.std(y)

print("Starting path computation...")
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples

fine = True  # fine or coarse grid
n_alphas = 100 if fine else 10
alphas = alpha_max * np.logspace(0, -2, n_alphas)

gap_freq = 10
prune = 1
verbose = 0
verbose_inner = 0
tols = [1e-2, 1e-4, 1e-6, 1e-8]
results = np.zeros([2, len(tols)])
for tol_ix, tol in enumerate(tols):
    t0 = time.time()
    res = celer_path(X, y, alphas=alphas, max_iter=100, gap_freq=gap_freq,
                     p0=100, verbose=verbose, verbose_inner=verbose_inner,
                     tol=tol, prune=prune, return_thetas=True)
    results[0, tol_ix] = time.time() - t0
    print('Celer time: %.2f s' % results[0, tol_ix])
    _, coefs, gaps, thetas = res

    t0 = time.time()
    _, coefs, dual_gaps = lasso_path(X, y, tol=tol, alphas=alphas)
    results[1, tol_ix] = time.time() - t0
    coefs = coefs.T

labels = [r"\sc{Celer}", "scikit-learn"]
figsize = (7.1, 4.3)
fig = plot_path_hist(results, labels, tols, figsize, ylim=None)
plt.show()
