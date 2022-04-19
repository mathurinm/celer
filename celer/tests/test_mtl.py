import pytest
import itertools

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_allclose, assert_array_less, assert_array_equal

from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import MultiTaskLassoCV as sklearn_MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso as sklearn_MultiTaskLasso
from sklearn.linear_model import lasso_path

from celer import (Lasso, GroupLasso, GroupLassoCV, MultiTaskLasso,
                   MultiTaskLassoCV)
from celer.homotopy import celer_path, mtl_path, _grp_converter
from celer.group_fast import dnorm_grp
from celer.utils.testing import build_dataset


@pytest.mark.parametrize("sparse_X, fit_intercept",
                         itertools.product([0, 1], [0, 1]))
def test_GroupLasso_Lasso_equivalence(sparse_X, fit_intercept):
    """Check that GroupLasso with groups of size 1 gives Lasso."""
    n_features = 1000
    X, y = build_dataset(
        n_samples=100, n_features=n_features, sparse_X=sparse_X)
    alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
    alpha = alpha_max / 10
    clf = Lasso(alpha, tol=1e-12, fit_intercept=fit_intercept,
                verbose=0)
    clf.fit(X, y)
    # take groups of size 1:
    clf1 = GroupLasso(alpha=alpha, groups=1, tol=1e-12,
                      fit_intercept=fit_intercept, verbose=0)
    clf1.fit(X, y)

    np.testing.assert_allclose(clf1.coef_, clf.coef_, atol=1e-6)
    np.testing.assert_allclose(clf1.intercept_, clf.intercept_, rtol=1e-4)


def test_GroupLasso_MultitaskLasso_equivalence():
    "GroupLasso and MultitaskLasso equivalence."""
    n_samples, n_features = 30, 50
    X_, Y_ = build_dataset(n_samples, n_features, n_targets=3)
    y = Y_.reshape(-1, order='F')
    X = np.zeros([3 * n_samples, 3 * n_features], order='F')

    # block filling new design
    for i in range(3):
        X[i * n_samples:(i + 1) * n_samples, i *
          n_features:(i + 1) * n_features] = X_

    grp_indices = np.arange(
        3 * n_features).reshape(3, -1).reshape(-1, order='F').astype(np.int32)
    grp_ptr = 3 * np.arange(n_features + 1).astype(np.int32)

    alpha_max = np.max(norm(X_.T @ Y_, axis=1)) / len(Y_)

    X_data = np.empty([1], dtype=X.dtype)
    X_indices = np.empty([1], dtype=np.int32)
    X_indptr = np.empty([1], dtype=np.int32)
    weights = np.ones(len(grp_ptr) - 1)

    other = dnorm_grp(
        False, y, grp_ptr, grp_indices, X, X_data,
        X_indices, X_indptr, X_data, weights, len(grp_ptr) - 1,
        np.zeros(1, dtype=np.int32), False)
    np.testing.assert_allclose(alpha_max, other / len(Y_))

    alpha = alpha_max / 10
    clf = MultiTaskLasso(alpha, fit_intercept=False, tol=1e-8, verbose=0)
    clf.fit(X_, Y_)

    groups = [grp.tolist() for grp in grp_indices.reshape(50, 3)]
    clf1 = GroupLasso(alpha=alpha / 3, groups=groups,
                      fit_intercept=False, tol=1e-8, verbose=0)
    clf1.fit(X, y)

    np.testing.assert_allclose(clf1.coef_, clf.coef_.reshape(-1), atol=1e-4)


def test_convert_groups():
    n_features = 6
    grp_ptr, grp_indices = _grp_converter(3, n_features)
    np.testing.assert_equal(grp_ptr, [0, 3, 6])
    np.testing.assert_equal(grp_indices, [0, 1, 2, 3, 4, 5])

    grp_ptr, grp_indices = _grp_converter([1, 3, 2], 6)
    np.testing.assert_equal(grp_ptr, [0, 1, 4, 6])
    np.testing.assert_equal(grp_indices, [0, 1, 2, 3, 4, 5])

    groups = [[0, 2, 5], [1, 3], [4]]
    grp_ptr, grp_indices = _grp_converter(groups, 6)
    np.testing.assert_equal(grp_ptr, [0, 3, 5, 6])
    np.testing.assert_equal(grp_indices, [0, 2, 5, 1, 3, 4])


def test_mtl_path():
    X, Y = build_dataset(n_targets=3)
    tol = 1e-12
    params = dict(eps=0.01, tol=tol, n_alphas=10)
    alphas, coefs, gaps = mtl_path(X, Y, **params)
    np.testing.assert_array_less(gaps, tol * norm(Y) ** 2 / len(Y))

    sk_alphas, sk_coefs, sk_gaps = lasso_path(X, Y, **params, max_iter=10000)
    np.testing.assert_array_less(sk_gaps, tol * np.linalg.norm(Y, 'fro')**2)
    np.testing.assert_array_almost_equal(coefs, sk_coefs, decimal=5)
    np.testing.assert_allclose(alphas, sk_alphas)


def test_MultiTaskLassoCV():
    """Test that our MultitaskLassoCV behaves like sklearn's."""
    X, y = build_dataset(n_samples=30, n_features=50, n_targets=3)

    params = dict(eps=1e-2, n_alphas=10, tol=1e-12, cv=2, n_jobs=1,
                  fit_intercept=False, verbose=0)

    clf = MultiTaskLassoCV(**params)
    clf.fit(X, y)

    clf2 = sklearn_MultiTaskLassoCV(**params)
    clf2.max_iter = 10000  # increase max_iter bc of low tol
    clf2.fit(X, y)

    np.testing.assert_allclose(clf.mse_path_, clf2.mse_path_,
                               atol=1e-4, rtol=1e-04)
    np.testing.assert_allclose(clf.alpha_, clf2.alpha_,
                               atol=1e-4, rtol=1e-04)
    np.testing.assert_allclose(clf.coef_, clf2.coef_,
                               atol=1e-4, rtol=1e-04)

    # check_estimator tests float32 so using tol < 1e-7 causes precision
    # issues
    # we don't support sample_weights for MTL
    # clf.tol = 1e-5
    # check_estimator(clf)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_MultiTaskLasso(fit_intercept):
    """Test that our MultiTaskLasso behaves as sklearn's."""
    X, Y = build_dataset(n_samples=20, n_features=30, n_targets=10)
    alpha_max = np.max(norm(X.T.dot(Y), axis=1)) / X.shape[0]

    alpha = alpha_max / 2.
    params = dict(alpha=alpha, fit_intercept=fit_intercept, tol=1e-10)
    clf = MultiTaskLasso(**params)
    clf.verbose = 2
    clf.fit(X, Y)

    clf2 = sklearn_MultiTaskLasso(**params)
    clf2.fit(X, Y)
    np.testing.assert_allclose(clf.coef_, clf2.coef_, rtol=1e-5)
    if fit_intercept:
        np.testing.assert_allclose(clf.intercept_, clf2.intercept_)

    # we don't support sample_weights for MTL
    # clf.tol = 1e-7
    # check_estimator(clf)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_group_lasso_path(sparse_X):
    n_features = 50
    X, y = build_dataset(
        n_samples=11, n_features=n_features, sparse_X=sparse_X)

    alphas, coefs, gaps = celer_path(
        X, y, "grouplasso", groups=5, eps=1e-2, n_alphas=10, tol=1e-8)
    tol = 1e-8
    np.testing.assert_array_less(gaps, tol * norm(y) ** 2 / len(y))


@pytest.mark.parametrize("sparse_X", [True, False])
def test_GroupLasso(sparse_X):
    n_features = 50
    X, y = build_dataset(
        n_samples=11, n_features=n_features, sparse_X=sparse_X)

    tol = 1e-8
    clf = GroupLasso(alpha=0.8, groups=10, tol=tol)
    clf.fit(X, y)
    np.testing.assert_array_less(clf.dual_gap_, tol * norm(y) ** 2 / len(y))

    clf.tol = 1e-6
    clf.groups = 1  # unsatisfying but sklearn will fit out of 5 features
    check_estimator(clf)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_GroupLassoCV(sparse_X):
    n_features = 50
    X, y = build_dataset(
        n_samples=11, n_features=n_features, sparse_X=sparse_X)

    tol = 1e-8
    clf = GroupLassoCV(groups=10, tol=tol)
    clf.fit(X, y)
    np.testing.assert_array_less(clf.dual_gap_, tol * norm(y) ** 2 / len(y))

    clf.tol = 1e-6
    clf.groups = 1  # unsatisfying but sklearn will fit with 5 features
    check_estimator(clf)


def test_weights_group_lasso():
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=True)

    groups = 5
    n_groups = n_features // groups
    np.random.seed(0)
    weights = np.abs(np.random.randn(n_groups))

    tol = 1e-14
    params = {'n_alphas': 10, 'tol': tol, 'verbose': 1}
    augmented_weights = np.repeat(weights, groups)

    alphas1, coefs1, gaps1 = celer_path(
        X, y, "grouplasso", groups=groups, weights=weights,
        eps=1e-2, **params)
    alphas2, coefs2, gaps2 = celer_path(
        X.multiply(1 / augmented_weights[None, :]), y, "grouplasso",
        groups=groups, eps=1e-2, **params)

    assert_allclose(alphas1, alphas2)
    assert_allclose(
        coefs1, coefs2 / augmented_weights[:, None], rtol=1e-3)
    assert_array_less(gaps1, tol * norm(y) ** 2 / len(y))
    assert_array_less(gaps2, tol * norm(y) ** 2 / len(y))


def test_check_weights():
    X, y = build_dataset(30, 42)
    weights = np.ones(X.shape[1] // 7)
    weights[0] = 0
    clf = GroupLasso(weights=weights, groups=7)  # groups of size 7
    # weights must be > 0
    np.testing.assert_raises(ValueError, clf.fit, X=X, y=y)
    # len(weights) must be equal to number of groups (6 here)
    clf.weights = np.ones(8)
    np.testing.assert_raises(ValueError, clf.fit, X=X, y=y)


def test_infinite_weights_group():
    n_samples, n_features = 50, 100
    X, y = build_dataset(n_samples, n_features)

    np.random.seed(1)
    group_size = 5
    weights = np.abs(np.random.randn(n_features // group_size))
    n_inf = 3
    inf_indices = np.random.choice(
        n_features // group_size, size=n_inf, replace=False)
    weights[inf_indices] = np.inf
    alpha_max = np.max(
        norm((X.T @ y).reshape(-1, group_size), 2, axis=1)
    ) / n_samples

    clf = GroupLasso(
        alpha=alpha_max / 100., weights=weights, groups=group_size, tol=1e-8
    ).fit(X, y)

    assert_array_less(clf.dual_gap_, clf.tol * norm(y) ** 2 / 2)
    assert_array_equal(
        norm(clf.coef_.reshape(-1, group_size), axis=1)[inf_indices], 0)


if __name__ == "__main__":
    pass
