"""
==============================================================
Run MultitaskLassoCV and compare performance with scikit-learn
==============================================================

The example runs the MultitaskLassoCV scikit-learn like estimator.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from celer import MultiTaskLassoCV
from numpy.linalg import norm
from sklearn.utils import check_random_state
from sklearn import linear_model

rng = check_random_state(0)

###############################################################################
# Generate some 2D coefficients with sine waves with random frequency and phase
n_samples, n_features, n_tasks = 100, 500, 20
n_relevant_features = 20
support = rng.choice(n_features, n_relevant_features, replace=False)
coef = np.zeros((n_tasks, n_features))
times = np.linspace(0, 2 * np.pi, n_tasks)
for k in support:
    coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))


X = rng.randn(n_samples, n_features)
Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
Y /= norm(Y, ord='fro')


###############################################################################
# Fit with sklearn and celer, using the same API
params = dict(tol=1e-6, cv=4, n_jobs=-1)
t0 = time.perf_counter()
clf = MultiTaskLassoCV(**params).fit(X, Y)
t_celer = time.perf_counter() - t0

t0 = time.perf_counter()
clf_sklearn = linear_model.MultiTaskLassoCV(**params).fit(X, Y)
t_sklearn = time.perf_counter() - t0

###############################################################################
# Celer is faster
print("Time for celer  : %.2f s" % t_celer)
print("Time for sklearn: %.2f s" % t_sklearn)

###############################################################################
# Both packages find the same solution
print("Celer's optimal regularizer  : %s" % clf.alpha_)
print("Sklearn's optimal regularizer: %s" % clf_sklearn.alpha_)

print("Relative norm difference between optimal coefs: %.2f %%" %
      (100 * norm(clf.coef_ - clf_sklearn.coef_) / norm(clf.coef_)))

###############################################################################
fig, axarr = plt.subplots(2, 1, constrained_layout=True)
axarr[0].spy(clf.coef_, aspect="auto")
axarr[0].xaxis.tick_bottom()
axarr[0].set_title("celer")
axarr[0].set_ylabel("tasks")
axarr[1].spy(clf_sklearn.coef_, aspect="auto")
axarr[1].xaxis.tick_bottom()
axarr[1].set_title("sklearn")
plt.suptitle("Sparsity patterns")
plt.ylabel("tasks")
plt.xlabel("features")
plt.show(block=False)
