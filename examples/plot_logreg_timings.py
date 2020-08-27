from celer.PN_logreg import newton_celer
import numpy as np
from numpy.linalg import norm

from celer import LogisticRegression
from celer.datasets import load_ml_uci

from celer.homotopy import celer_path, _sparse_and_dense

from celer.utils.testing import build_dataset

from scipy import sparse
dataset = "gisette_train"
X, y = load_ml_uci(dataset)

# np.random.seed(0)
# X = np.asfortranarray(np.random.randn(5, 15))

y = np.sign(np.random.randn(X.shape[0]))
alpha_max = norm(X.T @ y, ord=np.inf) / 2
C = 10 / alpha_max

# X, y = build_dataset(
#     n_samples=100, n_features=100, sparse_X=True)


# _, coefs_c, gaps = celer_path(X, y, use_PN=True, pb="logreg",
#                               alphas=alphas, tol=1e-7, p0=1,
#                               verbose=2, max_iter)
X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)
w = np.zeros(X.shape[1])
newton_celer(
    sparse.issparse(X), X_dense, X_data, X_indices, X_indptr, y, 1 /
    C, w,
    max_iter=10, tol=1e-7, p0=40, verbose=1, prune=1, blitz_sc=0, max_pn_iter=10)


# res = celer_path(X, y, "logreg", alphas=[
#                  1/C], use_PN=False, prune=True, verbose=2, tol=1e-6, return_thetas=True)
# theta = res[-1][0]
# w = res[1].ravel()
# clf = LogisticRegression(C=C, verbose=1
#                          fit_intercept=False, tol=1).fit(X, y)
