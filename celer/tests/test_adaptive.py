# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
from numpy.linalg import norm
# from sklearn.utils.estimator_checks import check_estimator

from celer.dropin_sklearn import Lasso, AdaptiveLasso  # AdaptiveLassoCV
from celer.utils.testing import build_dataset


# def test_adaptive_lasso_class():
# check_estimator(AdaptiveLasso())
# check_estimator(AdaptiveLassoCV())


def test_adaptive_lasso():
    sparse_X = False
    X, y = build_dataset(
        n_samples=60, n_features=100, sparse_X=sparse_X)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / len(y)
    alpha = alpha_max / 10
    tol = 1e-10
    n_reweightings = 5
    adalasso = AdaptiveLasso(alpha=alpha, n_reweightings=n_reweightings,
                             fit_intercept=False, tol=tol)
    adalasso.fit(X, y)

    lasso = Lasso(alpha=alpha, warm_start=False,
                  weights=None, fit_intercept=False, tol=tol)
    reweights = np.ones(X.shape[1])
    for _ in range(n_reweightings):
        lasso.weights = reweights
        lasso.fit(X, y)
        reweights = 1. / np.abs(lasso.coef_)

    np.testing.assert_allclose(lasso.coef_, adalasso.coef_)
