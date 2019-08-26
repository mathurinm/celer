import numpy as np

from celer.utils.testing import build_dataset
from celer.homotopy import mtl_path


def test_mtl():
    X, Y, _, _ = build_dataset(n_targets=10)
    tol = 1e-6
    alphas, coefs, gaps = mtl_path(X, Y)
    np.testing.assert_array_less(gaps, tol)
