import numpy as np

from celer import Lasso, ElasticNet, celer_path
from celer.utils.testing import build_dataset


n_samples, n_features = 30, 50
X, y = build_dataset(n_samples, n_features)

params = {'X': X, 'y': y, 'n_alphas': 10, 'l1_ratio': 0.5}


alphas, coefs, gaps = celer_path(pb='lasso', **params)
