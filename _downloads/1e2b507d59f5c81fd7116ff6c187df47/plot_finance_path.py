"""
=======================================================
Lasso path computation on Finance/log1p dataset
=======================================================

The example runs the Celer algorithm on the Finance dataset
which is a large sparse dataset.

Running time is not compared with the scikit-learn
implementation as it makes the example too long to run.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from celer import celer_path
from celer.datasets import fetch_libsvm

print(__doc__)

print("*** Warning: this example may take more than 5 minutes to run ***")
X, y = fetch_libsvm('finance')
n_samples, n_features = X.shape
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
print("Dataset size: %d samples, %d features" % X.shape)

# construct grid of regularization parameters alpha
n_alphas = 11
alphas = alpha_max * np.geomspace(1, 0.05, n_alphas)

###############################################################################
# Run Celer on a grid of regularization parameters, for various tolerances:
tols = [1e-2, 1e-4, 1e-6]
results = np.zeros([1, len(tols)])
gaps = np.zeros((len(tols), len(alphas)))

print("Starting path computation...")
for tol_ix, tol in enumerate(tols):
    t0 = time.time()
    res = celer_path(X, y, 'lasso', alphas=alphas,
                     tol=tol, prune=True, verbose=1)
    results[0, tol_ix] = time.time() - t0
    _, coefs, gaps[tol_ix] = res


labels = [r"\sc{Celer}"]
figsize = (7, 4)

df = pd.DataFrame(results.T, columns=["Celer"])
df.index = tols
df.plot.bar(rot=0)
plt.xlabel("stopping tolerance")
plt.ylabel("path computation time (s)")
plt.tight_layout()
plt.show(block=False)

###############################################################################
# Measure the influence of regularization on the sparsity of the solutions:

fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
plt.bar(np.arange(n_alphas), (coefs != 0).sum(axis=0))
plt.title("Sparsity of solution along regularization path")
ax.set_ylabel(r"$||\hat w||_0$")
ax.set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
ax.set_yscale('log')
ax.set_xticks(np.arange(n_alphas)[::2])
ax.set_xticklabels(map(lambda x: "%.2f" % x, alphas[::2] / alphas[0]))
plt.show(block=False)


###############################################################################
# Check convergence guarantees: gap is inferior to tolerance

df = pd.DataFrame(gaps.T, columns=map(lambda x: r"tol=%.0e" % x, tols))
df.index = map(lambda x: "%.2f" % x, alphas / alphas[0])
ax = df.plot.bar(figsize=(12, 5))
ax.set_ylabel("duality gap reached")
ax.set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
ax.set_yscale('log')
ax.set_yticks(tols)
plt.tight_layout()
plt.show(block=False)
