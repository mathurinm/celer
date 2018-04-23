# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np

from scipy import sparse
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import (LassoCV as sklearn_LassoCV,
                                  Lasso as sklearn_Lasso, lasso_path)

from celer import celer_path
from celer.dropin_sklearn import Lasso, LassoCV


def build_dataset(n_samples=50, n_features=200, n_informative_features=10,
                  n_targets=1, sparse_X=False):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """

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


def test_celer_path_dense():
    """Test Lasso path computation on dense data."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50)
    n_samples = X.shape[0]
    alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
    n_alphas = 5
    alphas = alpha_max * np.logspace(0, -2, n_alphas)

    tol = 1e-6
    alphas, coefs, gaps, thetas = celer_path(X, y, alphas=alphas, tol=tol,
                                             return_thetas=True, verbose=False,
                                             verbose_inner=False)
    np.testing.assert_array_less(gaps, tol)


def test_celer_path_sparse():
    """Test Lasso path computation on sparse data."""
    X, y, _, _ = build_dataset(n_samples=50, n_features=50, sparse_X=True)
    n_samples = X.shape[0]
    alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
    n_alphas = 10
    alphas = alpha_max * np.logspace(0, -2, n_alphas)

    tol = 1e-6
    alphas, coefs, gaps, thetas = celer_path(X, y, alphas=alphas, tol=tol,
                                             return_thetas=True, verbose=False,
                                             verbose_inner=False)
    np.testing.assert_array_less(gaps, tol)


def test_celer_path_vs_lasso_path():
    """Test that celer_path matches sklearn lasso_path."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50)

    params = dict(eps=1e-2, n_alphas=10, tol=1e-8)
    alphas1, coefs1, gaps1 = celer_path(
        X, y, return_thetas=False, verbose=False, **params)

    alphas2, coefs2, gaps2 = lasso_path(X, y, verbose=False, **params)

    np.testing.assert_allclose(alphas1, alphas2)
    np.testing.assert_allclose(coefs1, coefs2, rtol=1e-05, atol=1e-6)


def test_LassoCV_compatibility():
    """Test that our estimator is pluggable into sklearn's LassoCV."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50)
    params = dict(eps=1e-1, n_alphas=100, tol=1e-10, cv=2)

    clf = LassoCV(**params)
    clf.fit(X, y)

    clf2 = sklearn_LassoCV(**params)
    clf2.fit(X, y)

    np.testing.assert_allclose(clf.mse_path_, clf2.mse_path_,
                               rtol=1e-04)
    np.testing.assert_allclose(clf.alpha_, clf2.alpha_,
                               rtol=1e-05)
    np.testing.assert_allclose(clf.coef_, clf2.coef_,
                               rtol=1e-05)

    check_estimator(LassoCV)


def test_dropin_lasso():
    """Test that our Lasso class behaves as sklearn's Lasso."""
    X, y, _, _ = build_dataset(n_samples=20, n_features=15)

    alpha_max = np.linalg.norm(X.T.dot(y), ord=np.inf) / X.shape[0]
    alpha = alpha_max / 2.
    clf = Lasso(alpha=alpha)
    clf.fit(X, y)

    clf2 = sklearn_Lasso(alpha=alpha)
    clf2.fit(X, y)
    np.testing.assert_allclose(clf.coef_, clf2.coef_)

    check_estimator(Lasso)
