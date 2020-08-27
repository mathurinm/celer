from celer.PN_logreg import newton_celer
import numpy as np
from numpy.linalg import norm

from celer import LogisticRegression
from celer.datasets import load_ml_uci

from celer.homotopy import celer_path, _sparse_and_dense


# dataset = "gisette_train"
# X, y = load_ml_uci(dataset)

np.random.seed(0)
X = np.asfortranarray(np.random.randn(20, 40))
y = np.sign(np.random.randn(X.shape[0]))
alpha_max = norm(X.T @ y, ord=np.inf) / 2
C = 50 / alpha_max


X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)
newton_celer(
    False, X_dense, X_data, X_indices, X_indptr, y, 1 /
    C, np.zeros(X.shape[1]),
    max_iter=1, tol=1e-2, p0=40, verbose=2, prune=1, blitz_sc=0, max_pn_iter=10)


# res = celer_path(X, y, "logreg", alphas=[
#                  1/C], use_PN=False, prune=True, verbose=2, tol=1e-6, return_thetas=True)

# theta = res[-1][0]
# w = res[1].ravel()
# clf = LogisticRegression(C=C, verbose=1
#                          fit_intercept=False, tol=1).fit(X, y)
