# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import warnings
from itertools import product

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less
import pytest

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (LassoCV as sklearn_LassoCV,
                                  Lasso as sklearn_Lasso, lasso_path)

from celer import celer_path
from celer.dropin_sklearn import Lasso, LassoCV
from celer.utils.testing import build_dataset


def test():
    assert False
    assert_array_less(1,2)


@pytest.mark.parametrize("sparse_X, alphas, pb",
                         product([False, True], [None, 1],
                                 ["lasso", "logreg"]))
def test_celer_path(sparse_X, alphas, pb):
    """Test Lasso path convergence."""
    X, y = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)
    tol = 1e-6
    if pb == "logreg":
        y = np.sign(y)
        tol_scaled = tol * len(y) * np.log(2)
    else:
        tol_scaled = tol * norm(y) ** 2 / len(y)
    n_samples = X.shape[0]
    if alphas is not None:
        alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
        n_alphas = 10
        alphas = alpha_max * np.logspace(0, -2, n_alphas)

    alphas, _, gaps, _, n_iters = celer_path(
        X, y, pb, alphas=alphas, tol=tol, return_thetas=True,
        verbose=1, return_n_iter=True)
    assert_array_less(gaps, tol_scaled)
    # hack because array_less wants strict inequality
    assert_array_less(0.99, n_iters)


def test_convergence_warning():
    X, y = build_dataset(n_samples=10, n_features=10)
    tol = - 1  # gap canot be negative, a convergence warning should be raised
    alpha_max = np.max(np.abs(X.T.dot(y))) / X.shape[0]
    clf = Lasso(alpha_max / 10, max_iter=1, max_epochs=100, tol=tol)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        clf.fit(X, y)
        assert len(w) >= 1
        assert issubclass(w[-1].category, ConvergenceWarning)


@pytest.mark.parametrize("sparse_X, prune", [(False, 0), (False, 1)])
def test_celer_path_vs_lasso_path(sparse_X, prune):
    """Test that celer_path matches sklearn lasso_path."""
    X, y = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)

    tol = 1e-14
    params = dict(eps=1e-3, n_alphas=10, tol=tol)
    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", return_thetas=False, verbose=1, prune=prune,
        max_iter=30, **params)

    alphas2, coefs2, _ = lasso_path(X, y, verbose=False, **params,
                                    max_iter=10000)

    assert_allclose(alphas1, alphas2)
    assert_array_less(gaps1, tol * norm(y) ** 2 / len(y))
    assert_allclose(coefs1, coefs2, rtol=1e-03, atol=1e-4)


@pytest.mark.parametrize("sparse_X, fit_intercept, positive",
                         product([False, True], [False, True], [False, True]))
def test_LassoCV(sparse_X, fit_intercept, positive):
    """Test that our LassoCV behaves like sklearn's LassoCV."""

    X, y = build_dataset(n_samples=20, n_features=30, sparse_X=sparse_X)
    params = dict(eps=0.05, n_alphas=10, tol=1e-10, cv=2,
                  fit_intercept=fit_intercept, positive=positive, n_jobs=-1)

    clf = LassoCV(**params)
    clf.fit(X, y)

    clf2 = sklearn_LassoCV(**params, max_iter=10000)
    clf2.fit(X, y)

    assert_allclose(
        clf.mse_path_, clf2.mse_path_, rtol=1e-3, atol=1e-4)
    assert_allclose(clf.alpha_, clf2.alpha_)
    assert_allclose(clf.coef_, clf2.coef_, atol=1e-5)

    # TODO this one is slow (3s * 8 tests). Pass an instance and increase tol
    # check_estimator(LassoCV)


@pytest.mark.parametrize("sparse_X, fit_intercept, positive",
                         product([False, True], [False, True], [False, True]))
def test_Lasso(sparse_X, fit_intercept, positive):
    """Test that our Lasso class behaves as sklearn's Lasso."""
    X, y = build_dataset(n_samples=20, n_features=30, sparse_X=sparse_X)
    if not positive:
        alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]
    else:
        alpha_max = X.T.dot(y).max() / X.shape[0]

    alpha = alpha_max / 2.
    params = dict(alpha=alpha, fit_intercept=fit_intercept, tol=1e-10,
                  positive=positive)
    clf = Lasso(**params)
    clf.fit(X, y)

    clf2 = sklearn_Lasso(**params)
    clf2.fit(X, y)
    assert_allclose(clf.coef_, clf2.coef_, rtol=1e-5)
    if fit_intercept:
        assert_allclose(clf.intercept_, clf2.intercept_)

    # TODO fix for sklearn 0.24, pass an instance instead (buffer type error)
    # check_estimator(Lasso)


@pytest.mark.parametrize("sparse_X, pb",
                         product([True, False], ["lasso", "logreg"]))
def test_celer_single_alpha(sparse_X, pb):
    X, y = build_dataset(n_samples=20, n_features=100, sparse_X=sparse_X)
    tol = 1e-6

    if pb == "logreg":
        y = np.sign(y)
        tol_scaled = tol * np.log(2) * len(y)
    else:
        tol_scaled = tol * norm(y) ** 2 / len(y)

    alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]
    _, _, gaps = celer_path(X, y, pb, alphas=[alpha_max / 10.], tol=tol)
    assert_array_less(gaps, tol_scaled)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_zero_column(sparse_X):
    X, y = build_dataset(n_samples=60, n_features=50, sparse_X=sparse_X)
    n_zero_columns = 20
    if sparse_X:
        X.data[:X.indptr[n_zero_columns]].fill(0.)
    else:
        X[:, :n_zero_columns].fill(0.)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]
    tol = 1e-6
    _, coefs, gaps = celer_path(
        X, y, "lasso", alphas=[alpha_max / 10.], tol=tol, p0=50, prune=0)
    w = coefs.T[0]
    assert_array_less(gaps, tol * norm(y) ** 2 / len(y))
    np.testing.assert_equal(w.shape[0], X.shape[1])


def test_warm_start():
    """Test Lasso path convergence."""
    X, y = build_dataset(
        n_samples=100, n_features=100, sparse_X=True)
    n_samples, n_features = X.shape
    alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
    n_alphas = 10
    alphas = alpha_max * np.logspace(0, -2, n_alphas)

    reg1 = Lasso(tol=1e-6, warm_start=True, p0=10)
    reg1.coef_ = np.zeros(n_features)

    for alpha in alphas:
        reg1.set_params(alpha=alpha)
        reg1.fit(X, y)
        # refitting with warm start should take less than 2 iters:
        reg1.fit(X, y)
        # hack because assert_array_less does strict comparison...
        assert_array_less(reg1.n_iter_, 2.01)


def test_weights():
    sparse_X = 1
    X, y = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)

    np.random.seed(0)
    weights = np.abs(np.random.randn(X.shape[1]))

    tol = 1e-14
    params = {'n_alphas': 10, 'tol': tol}
    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", weights=weights, verbose=1,
        **params)

    alphas2, coefs2, gaps2 = celer_path(
        X / weights[None, :], y, "lasso", **params)

    assert_allclose(alphas1, alphas2)
    assert_allclose(
        coefs1, coefs2 / weights[:, None], atol=1e-4, rtol=1e-3)
    assert_array_less(gaps1, tol * norm(y) ** 2 / len(y))
    assert_array_less(gaps2, tol * norm(y) ** 2 / len(y))

    alpha = 0.001
    clf1 = Lasso(alpha=alpha, weights=weights, fit_intercept=False).fit(X, y)
    clf2 = Lasso(alpha=alpha, fit_intercept=False).fit(X / weights, y)

    assert_allclose(clf1.coef_, clf2.coef_ / weights)

    # weights must be > 0
    clf1.weights[0] = 0.
    np.testing.assert_raises(ValueError, clf1.fit, X=X, y=y)


if __name__ == "__main__":
    pass
