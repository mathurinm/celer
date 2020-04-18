import numpy as np
from numpy.linalg import norm

import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import MultiTaskLassoCV as sklearn_MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso as sklearn_MultiTaskLasso
from sklearn.linear_model import lasso_path

from celer.utils.testing import build_dataset
from celer.homotopy import mtl_path
from celer import MultiTaskLasso, MultiTaskLassoCV


def test_mtl():
    X, Y, _, _ = build_dataset(n_targets=10)
    tol = 1e-8
    alphas, coefs, gaps = mtl_path(X, Y, eps=1e-2, tol=tol)
    np.testing.assert_array_less(gaps, tol)

    sk_alphas, sk_coefs, sk_gaps = lasso_path(X, Y, eps=1e-2, tol=tol)
    np.testing.assert_array_less(sk_gaps, tol * np.linalg.norm(Y, 'fro')**2)
    np.testing.assert_array_almost_equal(coefs, sk_coefs, decimal=6)
    np.testing.assert_allclose(alphas, sk_alphas * len(X))


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_dropin_MultiTaskLassoCV(fit_intercept):
    """Test that our LassoCV behaves like sklearn's LassoCV."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, n_targets=3)
    params = dict(eps=1e-1, n_alphas=100, tol=1e-10, cv=2,
                  fit_intercept=fit_intercept)

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


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_dropin_MultiTaskLasso(fit_intercept):
    """Test that our Lasso class behaves as sklearn's Lasso."""
    X, y, _, _ = build_dataset(n_samples=20, n_features=30, n_targets=10)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / X.shape[0]

    alpha = alpha_max / 2.
    params = dict(alpha=alpha, fit_intercept=fit_intercept, tol=1e-10,
                  normalize=True)
    clf = MultiTaskLasso(**params)
    clf.fit(X, y)

    clf2 = sklearn_MultiTaskLasso(**params)
    clf2.fit(X, y)
    np.testing.assert_allclose(clf.coef_, clf2.coef_, rtol=1e-5)
    if fit_intercept:
        np.testing.assert_allclose(clf.intercept_, clf2.intercept_)

    check_estimator(MultiTaskLasso)

if __name__ == "__main__":
    test_dropin_MultiTaskLassoCV(fit_intercept=False)
