import time

import numpy as np
import matplotlib.pyplot as plt

from celer.lasso_fast import celer
from sklearn.datasets import fetch_mldata


print("Loading data...")
dataset = fetch_mldata("leukemia")
X = np.asfortranarray(dataset.data.astype(float))
y = dataset.target.astype(float)
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

X_data = np.array([1.])
X_indptr = np.array([1], dtype=np.int32)
X_indices = np.array([1], dtype=np.int32)
X_mean = np.zeros(X.shape[1])
# tols = [1e-2, 1e-4, 1e-6, 1e-8]
tols = np.geomspace(1e-2, 1e-14, 13)
results = np.zeros([2, len(tols)])
all_WS = []
all_screens = []
for tol_ix, tol in enumerate(tols):
    t0 = time.time()
    res = celer(
        False, X, X_data, X_indices, X_indptr, X_mean, y,
        alphas[-1], np.zeros(X.shape[1]), 100, 50000, gap_freq,
        p0=100, verbose=verbose, verbose_inner=verbose_inner,
        tol=tol, prune=prune, return_ws_size=True)
    results[0, tol_ix] = time.time() - t0
    print('Celer time: %.2f s' % results[0, tol_ix])
    _, coefs, gaps, thetas, ws_size, screens = res
    all_WS.append(ws_size)
    all_screens.append(screens)

print(all_WS)
print(all_screens)
