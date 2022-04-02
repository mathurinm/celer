"""For debbuging purposes."""

import numpy as np
from numpy.linalg import norm
from sklearn.utils.estimator_checks import check_estimator

from celer.dropin_sklearn import LogisticRegression
from celer.utils.testing import build_dataset


np.random.seed(1409)
X, y = build_dataset(
    n_samples=30, n_features=60, sparse_X=True)
y = np.sign(y)
alpha_max = norm(X.T.dot(y), ord=np.inf) / 2
C = 20. / alpha_max

tol = 1e-4
clf1 = LogisticRegression(C=C, tol=tol, verbose=0)

generator = check_estimator(clf1, generate_only=True)
generator = list(generator)

for i, (estimator, check_estimator) in enumerate(generator[37:]):
    check_estimator(estimator)
