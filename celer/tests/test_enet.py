from itertools import product
import pytest

import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_allclose, assert_array_less, assert_equal)

from sklearn.linear_model import enet_path, ElasticNet as sk_ElasticNet

from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


def test_raise_errors_l1_ratio():
    with np.testing.assert_raises(ValueError):
        ElasticNet(l1_ratio=5.)

    with np.testing.assert_raises(NotImplementedError):
        ElasticNet(l1_ratio=0.)

    with np.testing.assert_raises(NotImplementedError):
        X, y = build_dataset(n_samples=30, n_features=50)
        y = np.sign(y)
        celer_path(X, y, 'logreg', l1_ratio=0.5)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_enet_lasso_equivalence(sparse_X):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    coef_lasso = Lasso().fit(X, y).coef_
    coef_enet = ElasticNet(l1_ratio=1.0).fit(X, y).coef_

    assert_allclose(coef_lasso, coef_enet)


@pytest.mark.parametrize("sparse_X, prune", [(False, 0), (False, 1)])
def test_celer_enet_sk_enet_equivalence(sparse_X, prune):
    """Test that celer_path matches sklearn enet_path."""

    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    tol = 1e-14
    l1_ratio = 0.7
    alpha_max = norm(X.T@y, ord=np.inf) / n_samples
    params = dict(eps=1e-3, alphas=[alpha_max / 100.], tol=tol, l1_ratio=l1_ratio)

    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", return_thetas=False, verbose=0, prune=prune,
        max_iter=30, **params)

    alphas2, coefs2, _ = enet_path(X, y, verbose=0, **params,
                                   max_iter=10000)

    assert_equal(alphas1, alphas2)
    assert_array_less(gaps1, tol * norm(y) ** 2 / n_samples)
    assert_allclose(coefs1, coefs2, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("sparse_X, prune", product([False], [0, 1]))
def test_celer_enet_sk_enet_equivalence_many(sparse_X, prune):
    """Test that celer_path matches sklearn enet_path."""

    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    tol = 1e-14
    l1_ratio = 0.7
    n_alphas = 20
    params = dict(eps=1e-3, n_alphas=n_alphas, tol=tol, l1_ratio=l1_ratio)

    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", return_thetas=False, verbose=0, prune=prune,
        max_iter=30, **params)

    alphas2, coefs2, _ = enet_path(X, y, verbose=0, **params,
                                   max_iter=10000)

    assert_allclose(alphas1, alphas2)
    assert_array_less(gaps1, tol * norm(y) ** 2 / n_samples)
    assert_allclose(coefs1, coefs2, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("sparse_X, fit_intercept, positive",
                         product([False], [False, True], [False, False]))
def test_celer_ElasticNet_vs_sk_ElasticNet(sparse_X, fit_intercept, positive):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    params = {'l1_ratio': 0.5, 'tol': 1e-14,
              'fit_intercept': fit_intercept, 'positive': positive}

    reg_celer = ElasticNet(**params).fit(X, y)
    reg_sk = sk_ElasticNet(**params).fit(X, y)

    assert_allclose(reg_celer.coef_, reg_sk.coef_, rtol=1e-3, atol=1e-3)
    if fit_intercept:
        assert_allclose(reg_celer.intercept_, reg_sk.intercept_)


@pytest.mark.parametrize("sparse_X, fit_intercept",
                         product([False], [False, True]))
def test_infinit_weights(sparse_X, fit_intercept):
    n_samples, n_features = 30, 100
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    np.random.seed(0)
    weights = abs(np.random.rand(n_features))
    n_inf = n_features // 5
    inf_indices = np.random.choice(n_features, size=n_inf, replace=False)
    weights[inf_indices] = np.inf

    reg = ElasticNet(l1_ratio=0.5, tol=1e-8,
                     fit_intercept=fit_intercept, weights=weights)
    reg.fit(X, y)

    assert_equal(reg.coef_[inf_indices], 0)


if __name__ == '__main__':
    pass
