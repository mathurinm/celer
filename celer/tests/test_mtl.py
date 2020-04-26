import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import MultiTaskLassoCV as sklearn_MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso as sklearn_MultiTaskLasso
from sklearn.linear_model import lasso_path

from celer import Lasso, MultiTaskLasso, MultiTaskLassoCV
from celer.homotopy import celer_path, mtl_path, _grp_converter
from celer.group_lasso_fast import group_lasso, dscal_grplasso
from celer.utils.testing import build_dataset


def test_group_lasso_lasso():
    # check that group Lasso with groups of size 1 gives Lasso
    n_features = 200
    X, y = build_dataset(
        n_samples=100, n_features=n_features, sparse_X=False)[:2]
    # take groups of size 1:
    X = np.asfortranarray(X)
    grp_indices = np.arange(n_features).astype(np.int32)
    grp_ptr = np.arange(n_features + 1).astype(np.int32)
    n_samples = len(y)

    X_data = np.empty([1], dtype=X.dtype)
    X_indices = np.empty([1], dtype=np.int32)
    X_indptr = np.empty([1], dtype=np.int32)

    alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
    alpha = alpha_max / 10
    tol = 1e-4
    theta = np.zeros(n_samples)
    w = np.zeros(n_features)
    group_lasso(
        False, X, grp_indices, grp_ptr, X_data,
        X_indices, X_indptr, X_data, y, alpha,
        w, y.copy(), theta,
        norm(X, axis=0) ** 2, tol, 1000, 10, verbose=True)

    clf = Lasso(alpha, fit_intercept=False)
    clf.fit(X, y)

    np.testing.assert_allclose(w, clf.coef_)


def test_group_lasso_multitask():
    "Group Lasso and Multitask Lasso equivalence."""
    n_samples, n_features = 30, 50
    X_, Y_ = build_dataset(n_samples, n_features,
                           n_informative_features=n_features, n_targets=3)[:2]
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
    other = dscal_grplasso(
        False, y, grp_ptr, grp_indices, X, X_data,
        X_indices, X_indptr, X_data, False)
    np.testing.assert_allclose(alpha_max, other / len(Y_))

    alpha = alpha_max / 10
    clf = MultiTaskLasso(alpha, fit_intercept=False, tol=1e-8)
    clf.fit(X_, Y_)

    W_mtl = clf.coef_.T
    tol = 1e-8
    theta = np.zeros_like(y)
    w = np.zeros_like(X[0])
    group_lasso(
        False, X, grp_indices, grp_ptr, X_data, X_indices,
        X_indptr, X_data, y, alpha / 3,
        w, y.copy(), theta,
        norm(X_, axis=0) ** 2, tol, 100, 10, verbose=True)
    W_grp = w.reshape(n_features, 3, order='F')

    np.testing.assert_allclose(W_grp, W_mtl, atol=1e-6)


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


def test_mtl():
    X, Y, _, _ = build_dataset(n_targets=10)
    tol = 1e-8
    alphas, coefs, gaps = mtl_path(X, Y, eps=1e-2, tol=tol)
    np.testing.assert_array_less(gaps, tol)

    sk_alphas, sk_coefs, sk_gaps = lasso_path(X, Y, eps=1e-2, tol=tol)
    np.testing.assert_array_less(sk_gaps, tol * np.linalg.norm(Y, 'fro')**2)
    np.testing.assert_array_almost_equal(coefs, sk_coefs, decimal=6)
    np.testing.assert_allclose(alphas, sk_alphas)


def test_dropin_MultiTaskLassoCV():
    """Test that our LassoCV behaves like sklearn's LassoCV."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, n_targets=3)
    params = dict(eps=1e-1, n_alphas=100, tol=1e-10, cv=2, n_jobs=2,
                  fit_intercept=False, verbose=True)

    clf = MultiTaskLassoCV(**params)
    clf.fit(X, y)

    clf2 = sklearn_MultiTaskLassoCV(**params)
    clf2.fit(X, y)

    np.testing.assert_allclose(clf.mse_path_, clf2.mse_path_,
                               rtol=1e-04)
    np.testing.assert_allclose(clf.alpha_, clf2.alpha_,
                               rtol=1e-05)
    np.testing.assert_allclose(clf.coef_, clf2.coef_,
                               rtol=1e-05)

    check_estimator(MultiTaskLassoCV)


def test_dropin_MultiTaskLasso():
    """Test that our MultiTaskLasso class behaves as sklearn's."""
    X, Y, _, _ = build_dataset(n_samples=20, n_features=30, n_targets=10)
    alpha_max = np.max(norm(X.T.dot(Y), axis=1)) / X.shape[0]

    alpha = alpha_max / 2.
    params = dict(alpha=alpha, fit_intercept=False, tol=1e-10,
                  normalize=True)
    clf = MultiTaskLasso(**params)
    clf.fit(X, Y)

    clf2 = sklearn_MultiTaskLasso(**params)
    clf2.fit(X, Y)
    np.testing.assert_allclose(clf.coef_, clf2.coef_, rtol=1e-5)
    # if fit_intercept:
    #     np.testing.assert_allclose(clf.intercept_, clf2.intercept_)

    check_estimator(MultiTaskLasso)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_group_lasso_path(sparse_X):
    n_features = 50
    X, y = build_dataset(
        n_samples=11, n_features=n_features, sparse_X=sparse_X,
        n_informative_features=n_features)[:2]

    alphas, coefs, gaps = celer_path(
        X, y, "grouplasso", groups=5, eps=1e-2, n_alphas=10, tol=1e-8)
    tol = 1e-8
    np.testing.assert_array_less(gaps, tol)


if __name__ == "__main__":
    n_features = 6
    X, y = build_dataset(
        n_samples=50, n_features=n_features, sparse_X=True,
        n_informative_features=n_features)[:2]

    groups = [[0, 2, 5], [1, 3], [4]]
    alphas, coefs, gaps = celer_path(
        X, y, "grouplasso", groups=groups, eps=1e-1, n_alphas=10, tol=1e-8)

    # X_dense = X
    # X_data = np.empty([1], dtype=X.dtype)
    # X_indices = np.empty([1], dtype=np.int32)
    # X_indptr = np.empty([1], dtype=np.int32)

    # X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)
    # groups = 1
    # grp_ptr, grp_indices = _grp_converter(groups, X.shape[1])
    # n_groups = len(grp_ptr) - 1
    # lc_grp = np.zeros(n_groups, dtype=X_dense.dtype)
    # for g in range(n_groups):
    #     X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
    #     lc_grp[g] = norm(X_g, ord=2)

    # # alpha = alphas[8]
    # alpha = 0.01
    # is_sparse = False
    # w = np.zeros(n_features)
    # R = y.copy()
    # theta = y.copy() / norm(X.T @ y, ord=np.inf)
    # tol = 0.01
    # max_epochs = 5000
    # gap_freq = 2

    # group_lasso(
    #     is_sparse, X, grp_indices, grp_ptr, X_data, X_indices,
    #     X_indptr, X_sparse_scaling, y, alpha, w, R, theta, lc_grp ** 2, tol,
    #     max_epochs, gap_freq, verbose=True)
