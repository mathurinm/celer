import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features)


# params = {'X': X, 'y': y, 'n_alphas': 2, 'l1_ratio': 0.5}
l1_ratio = 0.5
alpha_max = norm(X.T @ y, ord=np.inf) / (n_samples * l1_ratio)
alpha = alpha_max / 10

alphas, coefs, gaps = celer_path(
    pb='lasso', l1_ratio=l1_ratio, alphas=[alpha], verbose=2,
    X=X, y=y, p0=50, max_epochs=100, max_iter=1)

w = coefs[:, 0]

clf = ElasticNet_sklearn(fit_intercept=False, alpha=alpha, l1_ratio=l1_ratio).fit(X, y)
w_star = clf.coef_
