import numpy as np
from sympy import lowergamma
from celer import LogisticRegression
from sklearn.linear_model import LogisticRegression as sk_LR

from celer.tests.conv_warning.logs.dumped_data import DICT_DATA

data = DICT_DATA['check_fit_idempotent']
X = data["X"]  # data centered around 100
y = data["y"]
C = data["C"]

y = 2 * y - 1
C_min = 2 / np.max(np.abs(X.T @ y))

# C is very high (higher is more difficult):
print(f"C / Cmin {C / C_min:.2e}")
# to get the warning:
clf = LogisticRegression(C=C, verbose=0).fit(X, y)

# a lower C gets no warning :
LogisticRegression(C=50 * C_min, verbose=1).fit(X, y)

###############################################################################
# regular celer converges but takes >= 10_000 epochs:
clf = LogisticRegression(C=C, solver="celer", verbose=1, tol=1e-10).fit(X, y)

# liblinear does not seem to
clf_sk = sk_LR(C=C, fit_intercept=False, penalty="l1",
               solver="liblinear", max_iter=10000, tol=1e-8).fit(X, y)


# finally, centering the columns yields convergence in very few iterations
X_c = X - X.mean(axis=0)

y = 2 * y - 1
C_min = 2 / np.max(np.abs(X_c.T @ y))
clf = LogisticRegression(C=C, verbose=2,
                         solver="celer-pn", tol=1e-10).fit(X_c, y)
