import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm
from scipy import sparse

# from celer import LogisticRegression
from celer.datasets import load_ml_uci, load_libsvm
from celer.PN_logreg import newton_celer
from celer.homotopy import celer_path, _sparse_and_dense

dataset = "gisette_train"
# X, y = load_ml_uci(dataset)
# X = sparse.load_npz("X.npz")
# y = np.load("y.npy")

X, y = load_libsvm("rcv1_train")
# np.random.seed(0)
# X = np.asfortranarray(np.random.randn(5, 15))

alpha_max = norm(X.T @ y, ord=np.inf) / 2
C = 1000 / alpha_max
X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)
w = np.zeros(X.shape[1])
newton_celer(
    sparse.issparse(X), X_dense, X_data, X_indices, X_indptr, y, 1 /
    C, w,
    max_iter=40, tol=1e0, p0=40, verbose=1, prune=1, blitz_sc=0, max_pn_iter=10)

clf = LogisticRegression(C=C, solver="liblinear",
                         penalty='l1', fit_intercept=False).fit(X, y)
w_lib = clf.coef_.ravel()

# res = celer_path(X, y, "logreg", alphas=[
#                  1/C], use_PN=False, prune=True, verbose=2, tol=1e-6, return_thetas=True)
# theta = res[-1][0]
# w = res[1].ravel()
# clf = LogisticRegression(C=C, verbose=1
#                          fit_intercept=False, tol=1).fit(X, y)
