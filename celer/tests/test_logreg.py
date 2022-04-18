# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause
import pytest
import numpy as np
from numpy.linalg import norm

from numpy.testing import assert_allclose, assert_array_less, assert_array_equal
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


def test_infinite_weights():
    np.random.seed(1)
    n_samples, n_features = 50, 100
    X, y = build_dataset(n_samples, n_features)
    y = np.sign(y)

    weights = np.ones(n_features)
    n_inf_index = n_features // 10
    arr_inf_index = np.random.randint(0, n_features+1, size=n_inf_index)
    weights[arr_inf_index] = np.inf

    alpha_max = norm(X.T @ y / weights, ord=np.inf) / n_samples

    tol = 1e-5
    _, coefs, dual_gaps = celer_path(X, y, pb="logreg",
                                     alphas=[alpha_max / 100.], weights=weights, tol=tol)

    # assert convergence
    atol = tol * 0.5 * norm(y) ** 2
    assert(dual_gaps[0] <= atol)

    # coef with inf weight should be set to 0
    assert_array_equal(coefs[arr_inf_index], 0)


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
