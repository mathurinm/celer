"""ConvergenceWarning in celer/tests/test_logreg.py.

ConvergenceWarning is due to ``sklearn.utils.estimator_checks.check_estimator``.
The followings functions in ``check_estimator`` sources code: 
https://github.com/scikit-learn/scikit-learn/blob/582fa30a3/sklearn/utils/estimator_checks.py#L514
raises the warning:
    - ``check_fit_idempotent``
    - ``check_fit_check_is_fitted``
    - ``check_n_features_in``

Code inspired by the latter functions implementation.
"""

import numpy as np
from celer.dropin_sklearn import LogisticRegression


# params
n_samples, n_features = 100, 2
mu, std = 100, 1
min_bound, max_bound = 0, 2
tol = 1e-4
C = 1.

np.random.seed(0)
X = np.random.normal(loc=mu, scale=std, size=(n_samples, n_features))
y = np.random.randint(low=min_bound, high=max_bound, size=n_samples)

clf = LogisticRegression(C=C, tol=tol, verbose=0)
clf.fit(X, y)
