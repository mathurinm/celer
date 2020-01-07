import pytest

import numpy as np

from celer.homotopy import celer_path
from celer.utils.testing import build_dataset


@pytest.mark.parametrize("sparse_X", [False, True])
def test_logreg_path(sparse_X):
    """Test Lasso path convergence."""
    X, y, _, _ = build_dataset(n_samples=30, n_features=50, sparse_X=sparse_X)
    y = np.sign(y)
    solver = "celer"

    tol = 1e-6
    alphas, coefs, gaps, thetas = celer_path(
        X, y, "logreg", solver, tol=tol, return_thetas=True,
        verbose=True, verbose_inner=False)
    np.testing.assert_array_less(gaps, tol)
