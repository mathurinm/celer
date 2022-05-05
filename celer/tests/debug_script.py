import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_allclose, assert_array_less, assert_equal)

from sklearn.linear_model import enet_path, ElasticNet as sk_ElasticNet, Lasso as sk_Lasso

from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features)
l1_ratio = 1.

params = {'max_iter': 10000, 'tol': 1e-14, 'fit_intercept': False}

alpha_max = norm(X.T@y, ord=np.inf) / n_samples
alpha = alpha_max / 100.

reg_enet = sk_ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **params).fit(X, y)

lmbda = alpha * l1_ratio
mu = alpha * (1 - l1_ratio)
X_tilde = np.vstack(
    (X, np.sqrt(n_samples*mu) * np.eye(n_features)))
y_tilde = np.hstack((y, np.zeros(n_features)))

reg_lasso = sk_Lasso(alpha=lmbda, **params)
reg_lasso.fit(X_tilde, y_tilde)

assert_allclose(reg_enet.coef_, reg_lasso.coef_)
