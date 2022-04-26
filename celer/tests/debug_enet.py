import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet as ElasticNet_sklearn, enet_path
from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features)


# params = {'X': X, 'y': y, 'n_alphas': 2, 'l1_ratio': 0.5}
l1_ratio = 0.1
alpha_max = norm(X.T @ y, ord=np.inf) / (n_samples * l1_ratio)
# alpha = alpha_max / 10.

alphas = [alpha_max / 100.]

alphas, coefs, gaps = celer_path(
    pb='lasso', l1_ratio=l1_ratio, alphas=alphas, verbose=2,
    X=X, y=y, p0=50, max_epochs=100, max_iter=1, tol=-1000)


alphas_sk, coefs_sk, gaps_sk = enet_path(
    alphas=alphas, l1_ratio=l1_ratio, X=X, y=y,)


print(abs(coefs - coefs_sk).max(axis=0))

w = coefs_sk[:, 0]  # also fails with coefs[:, 0]
alpha = alphas[0]

theta = (X @ w - y) / n_samples
dnorm = norm(X.T @ theta + alpha * (1 - l1_ratio) * w, ord=np.inf)
print(dnorm, alpha * l1_ratio)  # should be equal
p_obj = norm(y - X @ w) ** 2 / (2 * n_samples) + alpha * l1_ratio * \
    norm(w, 1) + alpha * (1 - l1_ratio) * norm(w)**2 / 2

print(p_obj)

d_obj = (norm(y)**2 / (2 * n_samples)
         - norm(y + n_samples * theta) ** 2 / (2 * n_samples)
         - alpha * (1 - l1_ratio) * norm(w) ** 2) / 2
print(d_obj)  # should be equal to d_obj, but is not
