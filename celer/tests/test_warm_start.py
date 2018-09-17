
import numpy as np
import pytest
import time

from scipy import sparse

from celer.dropin_sklearn import Lasso


def build_dataset(n_samples=50, n_features=200, n_informative_features=10,
                  n_targets=1, sparse_X=False):
    """Build samples and observation for linear regression problem."""
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    if sparse_X:
        X = sparse.random(n_samples, n_features, density=0.5, format='csc',
                          random_state=random_state)
        X_test = sparse.random(n_samples, n_features, density=0.5,
                               format='csc', random_state=random_state)
    else:
        X = random_state.randn(n_samples, n_features)
        X_test = random_state.randn(n_samples, n_features)
    y = X.dot(w)
    y_test = X_test.dot(w)
    return X, y, X_test, y_test


@pytest.mark.parametrize("sparse_X", [False, True])
def test_warm_start(sparse_X):
    """Test Lasso path convergence."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)
    n_samples, n_features = X.shape
    alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
    n_alphas = 10
    alphas = alpha_max * np.logspace(0, -2, n_alphas)
    tol = 1e-6

    reg1 = Lasso(alpha=alphas[0], tol=tol, warm_start=True)
    reg2 = Lasso(alpha=alphas[0], tol=tol, warm_start=False)
    reg1.coef_ = np.zeros(n_features)

    reg1.fit(X, y)
    reg2.fit(X, y)

    ws_time_start = time.time()

    for k in range(1, len(alphas)):
        coef1 = reg1.coef_
        reg1 = Lasso(alpha=alphas[k], tol=tol, warm_start=True)
        reg1.coef_ = coef1

        reg1.fit(X, y)

    ws_time = time.time() - ws_time_start

    nws_time_start = time.time()

    for k in range(1, len(alphas)):
        reg2 = Lasso(alpha=alphas[k], tol=tol, warm_start=False)
        reg2.fit(X, y)

    nws_time = time.time() - nws_time_start

    assert ws_time < nws_time
