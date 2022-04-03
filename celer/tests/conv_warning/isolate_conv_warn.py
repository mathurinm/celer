import numpy as np
from numpy.linalg import norm

from celer.dropin_sklearn import LogisticRegression
from celer.tests.conv_warning.dumped_data import DICT_DATA


check_props = DICT_DATA['check_fit_check_is_fitted']

X = check_props['X']
y = check_props['y']
C = check_props['C']
tol = check_props['tol']

alpha_max = norm(X.T @ y, ord=np.inf)
current_alpha = 1. / C
# print(f"ratio current_alpha/alpha_max: {current_alpha / alpha_max}")

# init with alpha_max instead of current_alpha
clf = LogisticRegression(C=C, tol=tol, verbose=1, max_iter=200)
clf.fit(X, y)


# with a standard normal
n_samples = 100
C = 6.
tol = 1e-4

alpha_max = norm(X.T @ y, ord=np.inf)
current_alpha = 1. / C
# print(f"ratio current_alpha/alpha_max: {current_alpha / alpha_max}")

X = np.random.randn(100, 2)
y = np.random.randint(0, 2, size=100)

clf2 = LogisticRegression(C=C, tol=tol, verbose=0)
clf2.fit(X, y)
