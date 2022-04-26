import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import enet_path, ElasticNet as sk_ElasticNet

from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


def test_raise_errors_l1_ratio():
    with np.testing.assert_raises(ValueError):
        ElasticNet(l1_ratio=5.)

    with np.testing.assert_raises(NotImplementedError):
        X, y = build_dataset(n_samples=30, n_features=50)
        y = np.sign(y)
        celer_path(X, y, 'logreg', l1_ratio=0.5)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_enet_lasso_equivalence(sparse_X):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    coef_lasso = Lasso().fit(X, y).coef_
    coef_enet = ElasticNet(l1_ratio=1.0).fit(X, y).coef_

    assert_allclose(coef_lasso, coef_enet)


@pytest.mark.parametrize("sparse_X", [True, False])
def test_celer_enet_sk_enet_equivalence(sparse_X):
    n_samples, n_features = 30, 50
    X, y = build_dataset(n_samples, n_features, sparse_X=sparse_X)

    params = {'X': X, 'y': y, 'n_alphas': 10, 'l1_ratio': 0.5}
    # alphas_sk, coefs_sk, dual_gaps_sk = enet_path(**params)

    # alphas_celer, coefs_celer, dual_gaps_celer = celer_path(pb='lasso', **params)

    # assert_allclose(alphas_sk, alphas_celer)
