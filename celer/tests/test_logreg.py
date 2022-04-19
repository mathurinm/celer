# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause
import pytest
import numpy as np
from numpy.linalg import norm

from numpy.testing import assert_allclose, assert_array_less
from sklearn.linear_model._logistic import _logistic_regression_path
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression as sklearn_Logreg

from celer import celer_path
from celer.dropin_sklearn import LogisticRegression
from celer.utils.testing import build_dataset


@pytest.mark.parametrize("solver", ["celer", "celer-pn"])
def test_celer_path_logreg(solver):
    X, y = build_dataset(
        n_samples=60, n_features=100, sparse_X=True)
    y = np.sign(y)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
    alphas = alpha_max * np.geomspace(1, 1e-2, 10)

    tol = 1e-11
    coefs, Cs, n_iters = _logistic_regression_path(
        X, y, Cs=1. / alphas, fit_intercept=False, penalty='l1',
        solver='liblinear', tol=tol, max_iter=1000, random_state=0)

    _, coefs_c, gaps = celer_path(
        X, y, "logreg", alphas=alphas, tol=tol, verbose=0,
        use_PN=(solver == "celer-pn"))

    assert_array_less(gaps, tol * len(y) * np.log(2))
    assert_allclose(coefs != 0, coefs_c.T != 0)
    assert_allclose(coefs, coefs_c.T, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_binary(sparse_X):
    np.random.seed(1409)
    X, y = build_dataset(
        n_samples=30, n_features=60, sparse_X=sparse_X)
    y = np.sign(y)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
    C = 20. / alpha_max

    clf = LogisticRegression(C=-1)
    np.testing.assert_raises(ValueError, clf.fit, X, y)
    tol = 1e-8
    clf = LogisticRegression(C=C, tol=tol, verbose=0)
    clf.fit(X, y)

    clf_sk = sklearn_Logreg(
        C=C, penalty='l1', solver='liblinear', fit_intercept=False, tol=tol)
    clf_sk.fit(X, y)
    assert_allclose(clf.coef_, clf_sk.coef_, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_multinomial(sparse_X):
    np.random.seed(1409)
    X, y = build_dataset(
        n_samples=30, n_features=60, sparse_X=sparse_X)
    y = np.random.choice(4, len(y))
    tol = 1e-8
    clf = LogisticRegression(C=1, tol=tol, verbose=0)
    clf.fit(X, y)

    clf_sk = sklearn_Logreg(
        C=1, penalty='l1', solver='liblinear', fit_intercept=False, tol=tol)
    clf_sk.fit(X, y)
    assert_allclose(clf.coef_, clf_sk.coef_, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("solver", ["celer-pn"])
def test_check_estimator(solver):
    # sklearn fits on unnormalized data for which there are convergence issues
    # fix with increased tolerance:
    clf = LogisticRegression(C=1, solver=solver, tol=0.1)
    check_estimator(clf)
