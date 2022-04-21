from celer import Lasso
import numpy as np
from numpy.linalg import norm
from benchopt.datasets.simulated import make_correlated_data

X, y, _ = make_correlated_data(random_state=0)
alpha_max = norm(X.T @ y, ord=np.inf) / len(y)

# this should converge in 1 iteration:
clf = Lasso(alpha=alpha_max, verbose=2, fit_intercept=False).fit(X, y)


clf = Lasso(alpha=alpha_max / 20, verbose=2,
            fit_intercept=False, tol=1e-10).fit(X, y)
