"""Check whether ``C`` is the only reason behind the warning.

If yes, no matter how the dummy data is generated,
for ``C >= 1e4 / alpha_max``, the warning is raised.

It's not the case.
"""

import numpy as np
from numpy.linalg import norm
from celer.dropin_sklearn import LogisticRegression


# params
C_fac = 1e4
tol = 1e-4

n_samples, n_features = np.random.randint(low=1, high=100, size=2)
min_bound, max_bound = 0, 2

np.random.seed(0)
X = np.random.random(size=(n_samples, n_features))
y = np.random.randint(low=min_bound, high=max_bound, size=n_samples)

alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
C = C_fac / alpha_max

clf = LogisticRegression(C=C, tol=tol, verbose=0)
clf.fit(X, y)
