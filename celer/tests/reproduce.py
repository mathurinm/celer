import libsvmdata
from celer import Lasso
import numpy as np
from numpy.linalg import norm


X, y = libsvmdata.fetch_libsvm("rcv1.binary")
alpha_max = norm(X.T @ y, ord=np.inf) / len(y)

m = 5
clf = Lasso(alpha_max*m, fit_intercept=False)
clf.fit(X, y)

assert norm(clf.coef_) == 0
assert clf.n_iter_ == 1
