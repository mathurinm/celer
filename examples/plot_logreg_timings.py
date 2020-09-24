"""
===============================================================
Use LogisticRegression class with Celer and Prox-Newton solvers
===============================================================
"""

import numpy as np
from scipy import sparse
from numpy.linalg import norm
from sklearn import linear_model

from celer import LogisticRegression
from celer.datasets import fetch_ml_uci

dataset = "gisette_train"
X, y = fetch_ml_uci(dataset)

C_min = 2 / norm(X.T @ y, ord=np.inf)
C = 5 * C_min
clf = LogisticRegression(C=C, verbose=1, solver="celer-pn", tol=1e0).fit(X, y)
w_celer = clf.coef_.ravel()

clf = linear_model.LogisticRegression(
    C=C, solver="liblinear", penalty='l1', fit_intercept=False).fit(X, y)
w_lib = clf.coef_.ravel()
