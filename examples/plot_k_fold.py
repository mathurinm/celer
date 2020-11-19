"""
Experiment to check how LassoCV's optimal alpha depend on K of K-fold.
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from celer import LassoCV
from celer.datasets import make_correlated_data
from celer.plot_utils import configure_plt

configure_plt()

n_samples, n_features, nnz = 500, 500, 100
X_, y_, _ = make_correlated_data(
    n_samples, n_features, nnz=nnz, rho=0.5, snr=5)


X, X_val, y, y_val = train_test_split(X_, y_, train_size=0.8, random_state=0)

plt.figure(figsize=(6, 6), constrained_layout=True)
best_alphas = []
Ks = [2, 3, 4, 5, 10]
for K in Ks:
    clf = LassoCV(cv=K, n_jobs=-1, verbose=True).fit(X, y)
    best_alphas.append(clf.alpha_)
    plt.semilogx(clf.alphas_, clf.mse_path_.mean(
        axis=1), label=r"$K = %d$" % K)
plt.legend()
plt.ylabel("Average MSE on test folds")
plt.xlabel(r"$\alpha$")


plt.figure()
plt.scatter(Ks, best_alphas)
plt.show(block=False)
