import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet as ElasticNet_sklearn, enet_path
from sklearn.utils import check_array

from celer import Lasso, ElasticNet, celer_path
from celer.cython_utils import dnorm_enet
from celer.utils.testing import build_dataset


n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features)


X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                order='F', copy=False, accept_large_sparse=False)
y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                ensure_2d=False)


# params = {'X': X, 'y': y, 'n_alphas': 2, 'l1_ratio': 0.5}
l1_ratio = 0.5
alpha_max = norm(X.T @ y, ord=np.inf) / (n_samples * l1_ratio)
# alpha = alpha_max / 10.

alphas = [alpha_max / 100.]

alphas, coefs, gaps = celer_path(
    pb='lasso', l1_ratio=l1_ratio, alphas=alphas, verbose=2,
    X=X, y=y, p0=50, max_epochs=500, )


alphas_sk, coefs_sk, gaps_sk = enet_path(
    alphas=alphas, l1_ratio=l1_ratio, X=X, y=y)


print(abs(coefs - coefs_sk).max(axis=0))

# use coefs of celer_path
w = coefs[:, 0]
alpha = alphas[0]

theta = (X @ w - y) / n_samples


kwargs_dnorm = {
    'is_sparse': 0, 'theta': theta, 'w': w, 'X': X,
    # this params doesn't matter (X is not sparse)
    'X_data': np.empty([1], dtype=X.dtype),
    'X_indices': np.empty([1], dtype=np.int32),
    'X_indptr': np.empty([1], dtype=np.int32),
    'skip': np.zeros(X.shape[1], dtype=np.int32),
    'X_mean': np.zeros(n_features, dtype=X.dtype),
    'weights': np.ones(n_features, dtype=float),
    'center': 0,
    'positive': 0,
    'alpha': alpha, 'l1_ratio': l1_ratio,
}

dnorm = dnorm_enet(*kwargs_dnorm.values())
# dnorm = norm(X.T @ theta + alpha * (1 - l1_ratio) * w, ord=np.inf)
print(dnorm, alpha * l1_ratio)  # should be equal, it is

p_obj = norm(y - X @ w) ** 2 / (2 * n_samples) + alpha * l1_ratio * \
    norm(w, 1) + alpha * (1 - l1_ratio) * norm(w)**2 / 2

d_obj = (norm(y)**2 / (2 * n_samples)
         - norm(y + n_samples * theta) ** 2 / (2 * n_samples)
         - 0.5 * alpha * (1 - l1_ratio) * norm(w) ** 2)
print(p_obj - d_obj)  # should be equal, it is
