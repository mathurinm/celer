
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less
from sklearn.linear_model import (
    enet_path, ElasticNet as sk_ElasticNet, lasso_path)

from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


sparse_X, prune = False, 0
n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

tol = 1e-14
l1_ratio = 0.7
n_alphas = 10
params = dict(eps=1e-3, n_alphas=n_alphas, tol=tol, l1_ratio=l1_ratio)


alphas1, coefs1, gaps1 = celer_path(
    X, y, "lasso", return_thetas=False, verbose=0, prune=prune, **params, max_iter=30)


alphas2, coefs2, _ = enet_path(X, y, verbose=0, **params,
                               max_iter=10000)

assert_allclose()
