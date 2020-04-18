# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import warnings
from itertools import product

import numpy as np
from numpy.linalg import norm

import pytest

from sklearn.linear_model._logistic import _logistic_regression_path
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import (LassoCV as sklearn_LassoCV,
                                  Lasso as sklearn_Lasso, lasso_path,
                                  LogisticRegression as sklearn_Logreg)

from celer import celer_path, celer
from celer.dropin_sklearn import Lasso, LassoCV, LogisticRegression
from celer.utils.testing import build_dataset


def test_logreg():
    X, y, _, _ = build_dataset(
        n_samples=100, n_features=100, sparse_X=True)
    y = np.sign(y)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
    alphas = alpha_max * np.geomspace(1, 1e-2, 10)

    tol = 1e-8
    coefs, Cs, n_iters = _logistic_regression_path(
        X, y, Cs=1. / alphas, fit_intercept=False, penalty='l1',
        solver='liblinear', tol=tol)

    _, coefs_c, gaps = celer_path(X, y, "logreg", alphas=alphas, tol=tol)

    np.testing.assert_array_less(gaps, tol)
    np.testing.assert_allclose(coefs != 0, coefs_c.T != 0)
    np.testing.assert_allclose(coefs, coefs_c.T, atol=1e-5, rtol=1e-3)


def test_dropin_logreg():
    np.random.seed(1409)
    check_estimator(LogisticRegression)
    X, y, _, _ = build_dataset(
        n_samples=100, n_features=100, sparse_X=True)
    y = np.sign(y)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
    C = 30. / alpha_max

    tol = 1e-8
    clf1 = LogisticRegression(C=C, tol=tol)
    clf1.fit(X, y)

    clf2 = sklearn_Logreg(
        C=C, penalty='l1', solver='liblinear', fit_intercept=False, tol=tol)
    clf2.fit(X, y)
    np.testing.assert_allclose(clf1.coef_, clf2.coef_, rtol=1e-3, atol=1e-5)

    # multinomial test:
    y = np.random.choice(4, len(y))
    clf3 = LogisticRegression(C=C, tol=tol)
    clf3.fit(X, y)

    clf4 = sklearn_Logreg(
        C=C, penalty='l1', solver='liblinear', fit_intercept=False, tol=tol)
    clf4.fit(X, y)
    np.testing.assert_allclose(clf3.coef_, clf4.coef_, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("sparse_X, alphas, pb",
                         product([False, True], [None, 1],
                                 ["lasso", "logreg"]))
def test_celer_path(sparse_X, alphas, pb):
    """Test Lasso path convergence."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)
    if pb == "logreg":
        y = np.sign(y)
    n_samples = X.shape[0]
    if alphas is not None:
        alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
        n_alphas = 10
        alphas = alpha_max * np.logspace(0, -2, n_alphas)

    tol = 1e-6
    alphas, coefs, gaps, thetas, n_iters = celer_path(
        X, y, pb, alphas=alphas, tol=tol, return_thetas=True,
        verbose=False, verbose_inner=False, return_n_iter=True)
    np.testing.assert_array_less(gaps, tol)
    # hack because array_less wants strict inequality
    np.testing.assert_array_less(0.99, n_iters)


def test_convergence_warning():
    X, y, _, _ = build_dataset(n_samples=10, n_features=10)
    tol = - 1  # gap canot be negative, a covnergence warning should be raised
    alpha_max = np.max(np.abs(X.T.dot(y))) / X.shape[0]
    clf = Lasso(alpha_max / 10, max_iter=1, max_epochs=100, tol=tol)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        clf.fit(X, y)
        assert len(w) == 1
        assert issubclass(w[-1].category, ConvergenceWarning)


@pytest.mark.parametrize("sparse_X, prune", [(False, 0), (False, 1)])
def test_celer_path_vs_lasso_path(sparse_X, prune):
    """Test that celer_path matches sklearn lasso_path."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)

    params = dict(eps=1e-2, n_alphas=10, tol=1e-14)
    alphas1, coefs1, gaps1 = celer_path(
        X, y, "lasso", return_thetas=False, verbose=1, prune=prune, **params)

    alphas2, coefs2, gaps2 = lasso_path(X, y, verbose=False, **params)

    np.testing.assert_allclose(alphas1, alphas2)
    np.testing.assert_allclose(coefs1, coefs2, rtol=1e-05, atol=1e-6)


@pytest.mark.parametrize("sparse_X, fit_intercept, positive",
                         product([False, True], [False, True], [False, True]))
def test_dropin_LassoCV(sparse_X, fit_intercept, positive):
    """Test that our LassoCV behaves like sklearn's LassoCV."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)
    params = dict(eps=1e-1, n_alphas=100, tol=1e-10, cv=2,
                  fit_intercept=fit_intercept, positive=positive)

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


@pytest.mark.parametrize("sparse_X, fit_intercept, positive",
                         product([False, True], [False, True], [False, True]))
def test_dropin_lasso(sparse_X, fit_intercept, positive):
    """Test that our Lasso class behaves as sklearn's Lasso."""
    X, y, _, _ = build_dataset(n_samples=20, n_features=30, sparse_X=sparse_X)
    if not positive:
        alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]
    else:
        alpha_max = X.T.dot(y).max() / X.shape[0]

    alpha = alpha_max / 2.
    params = dict(alpha=alpha, fit_intercept=fit_intercept, tol=1e-10,
                  normalize=True, positive=positive)
    clf = Lasso(**params)
    clf.fit(X, y)

    clf2 = sklearn_Lasso(**params)
    clf2.fit(X, y)
    np.testing.assert_allclose(clf.coef_, clf2.coef_, rtol=1e-5)
    if fit_intercept:
        np.testing.assert_allclose(clf.intercept_, clf2.intercept_)

    check_estimator(Lasso)


@pytest.mark.parametrize("sparse_X, pb",
                         product([True, False], ["lasso", "logreg"]))
def test_celer_single_alpha(sparse_X, pb):
    X, y, _, _ = build_dataset(n_samples=20, n_features=100, sparse_X=sparse_X)
    if pb == "logreg":
        y = np.sign(y)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]

    tol = 1e-6
    w, theta, gap = celer(X, y, pb, alpha_max / 10., tol=tol)
    np.testing.assert_array_less(gap, tol)
    np.testing.assert_equal(w.shape[0], X.shape[1])
    np.testing.assert_equal(theta.shape[0], X.shape[0])


@pytest.mark.parametrize("sparse_X", [True, False])
def test_zero_column(sparse_X):
    X, y, _, _ = build_dataset(n_samples=60, n_features=50, sparse_X=sparse_X)
    n_zero_columns = 20
    if sparse_X:
        X.data[:X.indptr[n_zero_columns]].fill(0.)
    else:
        X[:, :n_zero_columns].fill(0.)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]
    tol = 1e-6
    w, theta, gap = celer(X, y, "lasso", alpha_max / 10., tol=tol, p0=50,
                          prune=0, verbose=1, verbose_inner=1)
    np.testing.assert_array_less(gap, tol)
    np.testing.assert_equal(w.shape[0], X.shape[1])
    np.testing.assert_equal(theta.shape[0], X.shape[0])


def test_warm_start():
    """Test Lasso path convergence."""
    X, y, _, _ = build_dataset(
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
        np.testing.assert_array_less(reg1.n_iter_, 2.01)


if __name__ == "__main__":
    test_dropin_logreg()
