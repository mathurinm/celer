import numpy as np

from scipy import sparse

from celer import celer_path


def build_dataset(n_samples=50, n_features=200, n_informative_features=10,
                  n_targets=1):
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


def test_celer_path_dense():

    X, y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=1)

    alpha_max = np.max(np.abs(X.T.dot(y)))
    n_alphas = 10
    alphas = alpha_max * np.logspace(0, -2, n_alphas)

    tol = 1e-6
    betas, thetas, gaps = celer_path(X, y, alphas=alphas, tol=tol, verbose=1)
    np.testing.assert_array_less(gaps, tol)


def test_celer_path_sparse():

    X, y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=1)
    X = sparse.csc_matrix(X)
    alpha_max = np.max(np.abs(X.T.dot(y)))
    n_alphas = 10
    alphas = alpha_max * np.logspace(0, -2, n_alphas)

    tol = 1e-6
    betas, thetas, gaps = celer_path(X, y, alphas=alphas, tol=tol, verbose=1)
    np.testing.assert_array_less(gaps, tol)
