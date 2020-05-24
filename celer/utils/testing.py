import numpy as np

from scipy import sparse


def build_dataset(n_samples=50, n_features=200, n_targets=1, sparse_X=False):
    """Build samples and observation for linear regression problem."""
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)

    if sparse_X:
        X = sparse.random(n_samples, n_features, density=0.5, format='csc',
                          random_state=random_state)

    else:
        X = np.asfortranarray(random_state.randn(n_samples, n_features))

    y = X.dot(w)
    return X, y
