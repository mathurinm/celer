import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import enet_path, ElasticNet as sklearn_ElasticNet

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

    reg_1 = Lasso()
    reg_1.fit(X, y)

    reg_2 = ElasticNet(l1_ratio=1.0)
    reg_2.fit(X, y)

    assert_allclose(reg_1.coef_, reg_2.coef_)
