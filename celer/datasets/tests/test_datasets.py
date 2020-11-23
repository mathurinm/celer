import numpy as np
from numpy.linalg import norm

from celer.datasets import make_correlated_data


def test_correlated():
    X, y, w_true = make_correlated_data(snr=0)
    np.testing.assert_allclose(y, X @ w_true)

    snr = 5
    w_true = np.ones(50)
    X, y, _ = make_correlated_data(n_features=w_true.shape[0], snr=5)
    np.testing.assert_allclose(snr, norm(X @ w_true), norm(y - X @ w_true))

# # def test_climate():
#     # X, y = fetch_climate()
#     # np.testing.assert_equal(X.shape[0], y.shape[0])
