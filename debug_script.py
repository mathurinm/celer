import numpy as np

from celer.utils.testing import build_dataset
from celer.homotopy import celer_path


n_features = 50
sparse_X = True

X, y = build_dataset(
    n_samples=11, n_features=n_features, sparse_X=sparse_X)

alphas = np.array([
    # 1.73792429, 1.04185824,
    0.6245776 ,
    0.37442444,
    0.22446155,
    # 0.13456117, 0.0806673 , 0.04835878, 0.02899032, 0.01737924
])

alphas, coefs, gaps = celer_path(
    X, y, "grouplasso", groups=5, alphas=alphas, prune=False,
    # n_alphas=10,
    tol=1e-10, verbose=2,
)