from itertools import product
import pytest

import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_allclose, assert_array_less, assert_equal)

from sklearn.linear_model import (
    enet_path, ElasticNet as sk_ElasticNet, ElasticNetCV as sk_ElasticNetCV)

from celer import Lasso, ElasticNet, celer_path, ElasticNetCV
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


@pytest.mark.parametrize("sparse_X", (True, False))
def test_ElasticNet_Lasso_equivalence(sparse_X):
    n_samples, n_features = 50, 100
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)
    alpha_max = norm(X.T@y, ord=np.inf) / n_samples

    alpha = alpha_max / 100.
    coef_lasso = Lasso(alpha=alpha).fit(X, y).coef_
    coef_enet = ElasticNet(alpha=alpha, l1_ratio=1.0).fit(X, y).coef_

    assert_allclose(coef_lasso, coef_enet)

    np.random.seed(0)
    weights = abs(np.random.randn(n_features))
    alpha_max = norm(X.T@y / weights, ord=np.inf) / n_samples

    alpha = alpha_max / 100.
    coef_lasso = Lasso(alpha=alpha, weights=weights).fit(X, y).coef_
    coef_enet = ElasticNet(alpha=alpha, l1_ratio=1.0, weights=weights).fit(X, y).coef_

    assert_allclose(coef_lasso, coef_enet)


@pytest.mark.parametrize("prune", (0, 1))
def test_sk_enet_path_equivalence(prune):
    """Test that celer_path matches sklearn enet_path."""

    n_samples, n_features = 40, 80
    X, y = build_dataset(n_samples, n_features, sparse_X=False)

    tol = 1e-14
    l1_ratio = 0.7
    alpha_max = norm(X.T@y, ord=np.inf) / n_samples
    params = dict(eps=1e-3, tol=tol, l1_ratio=l1_ratio)

    # one alpha
    alpha = alpha_max / 100.
    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", alphas=[alpha],
        prune=prune, max_iter=30, **params)

    alphas2, coefs2, _ = enet_path(X, y, max_iter=10000,
                                   alphas=[alpha], **params)

    assert_equal(alphas1, alphas2)
    assert_array_less(gaps1, tol * norm(y) ** 2 / n_samples)
    assert_allclose(coefs1, coefs2, rtol=1e-3, atol=1e-4)

    # many alphas
    n_alphas = 20
    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", n_alphas=n_alphas,
        prune=prune, max_iter=30, **params)

    alphas2, coefs2, _ = enet_path(X, y, max_iter=10000,
                                   n_alphas=n_alphas, **params)

    assert_allclose(alphas1, alphas2)
    assert_array_less(gaps1, tol * norm(y) ** 2 / n_samples)
    assert_allclose(coefs1, coefs2, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("sparse_X, fit_intercept, positive",
                         product([False, True], [False, True], [False, True]))
def test_sk_ElasticNet_equivalence(sparse_X, fit_intercept, positive):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    params = {'l1_ratio': 0.5, 'tol': 1e-14,
              'fit_intercept': fit_intercept, 'positive': positive}

    reg_celer = ElasticNet(**params).fit(X, y)
    reg_sk = sk_ElasticNet(**params).fit(X, y)

    assert_allclose(reg_celer.coef_, reg_sk.coef_, rtol=1e-3, atol=1e-3)
    if fit_intercept:
        assert_allclose(reg_celer.intercept_, reg_sk.intercept_)


@pytest.mark.parametrize("sparse_X", (True, False))
def test_weighted_ElasticNet(sparse_X):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X)

    np.random.seed(0)
    weights = abs(np.random.randn(n_features))
    l1_ratio = .7

    params = {'max_iter': 10000, 'tol': 1e-14, 'fit_intercept': False}

    alpha_max = norm(X.T@y / weights, ord=np.inf) / n_samples
    alpha = alpha_max / 100.

    reg_enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **params).fit(X, y)

    lmbda = alpha * l1_ratio * n_samples / (n_samples + n_features)
    mu = alpha * (1 - l1_ratio)
    X_tilde = np.vstack(
        (X, np.sqrt(n_samples*mu) * np.eye(n_features)))
    y_tilde = np.hstack((y, np.zeros(n_features)))

    reg_lasso = Lasso(alpha=lmbda, **params)
    reg_lasso.fit(X_tilde, y_tilde)

    assert_allclose(reg_enet.coef_, reg_lasso.coef_, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("fit_intercept", (False, True))
def test_infinite_weights(fit_intercept):
    n_samples, n_features = 30, 100
    X, y = build_dataset(n_samples, n_features, sparse_X=False)

    np.random.seed(42)
    weights = abs(np.random.rand(n_features))
    n_inf = n_features // 5
    inf_indices = np.random.choice(n_features, size=n_inf, replace=False)
    weights[inf_indices] = np.inf

    reg = ElasticNet(l1_ratio=0.5, tol=1e-8,
                     fit_intercept=fit_intercept, weights=weights)
    reg.fit(X, y)

    assert_equal(reg.coef_[inf_indices], 0)


@pytest.mark.parametrize("fit_intercept", (False, True))
def test_ElasticNetCV(fit_intercept):
    n_samples, n_features = 30, 100
    X, y = build_dataset(n_samples, n_features, sparse_X=False)

    params = dict(l1_ratio=[0.7, 0.8, 0.5], eps=0.05, n_alphas=10, tol=1e-10, cv=2,
                  fit_intercept=fit_intercept, n_jobs=-1)

    clf = ElasticNetCV(**params)
    clf.fit(X, y)

    clf2 = sk_ElasticNetCV(**params, max_iter=10000)
    clf2.fit(X, y)

    assert_allclose(
        clf.mse_path_, clf2.mse_path_, rtol=1e-3, atol=1e-4)
    assert_allclose(clf.alpha_, clf2.alpha_)
    assert_allclose(clf.coef_, clf2.coef_, atol=1e-5)
    assert_allclose(clf.l1_ratio_, clf2.l1_ratio_, atol=1e-5)


if __name__ == '__main__':
    pass
