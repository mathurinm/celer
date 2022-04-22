import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less
from sklearn.linear_model import lasso_path

from celer import celer_path
from celer.utils.testing import build_dataset


sparse_X, prune = False, 1
n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
tol = 1e-14
params = dict(eps=1e-3, n_alphas=10, tol=tol,
              alphas=[alpha_max / 1000., alpha_max / 1000., alpha_max / 10000.])

alphas1, coefs1, gaps1 = celer_path(
    X, y, "lasso", return_thetas=False, verbose=2, prune=prune,
    max_iter=30, **params)

alphas2, coefs2, _ = lasso_path(X, y, verbose=False, **params,
                                max_iter=10000)

assert_allclose(alphas1, alphas2)
assert_array_less(gaps1, tol * norm(y) ** 2 / len(y))
assert_allclose(coefs1, coefs2, rtol=1e-03, atol=1e-4)
