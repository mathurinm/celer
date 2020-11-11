"""
============================================================
Run AdaptiveLasso to illustrate its smaller coefficient bias
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import toeplitz
from sklearn.utils import check_random_state
from numpy.linalg import norm

from celer import Lasso, LassoCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()

# Generating X and y data

n_samples, n_features = 200, 100
rng = check_random_state(0)
X = rng.multivariate_normal(size=n_samples, mean=np.zeros(n_features),
                            cov=toeplitz(0.7 ** np.arange(n_features)))


# Create true regression coefficients

w_true = np.zeros(n_features)
size_supp = 20
w_true[::n_features // size_supp] = (-1) ** np.arange(size_supp)
noise = rng.randn(n_samples)
y = X @ w_true
y += noise / norm(noise) * 0.5 * norm(y)


# Fit an adapted AdaptiveLasso clf

clf = LassoCV(verbose=0, tol=1e-10)
clf.fit(X, y)

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
ax.semilogx(clf.alphas_, clf.mse_path_, ':')
ax.semilogx(clf.alphas_, clf.mse_path_.mean(axis=-1), 'k',
            label='Average across the folds', linewidth=2)
ax.axvline(clf.alpha_, linestyle='--', color='k',
           label='alpha: CV estimate')

ax.legend()

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Mean square error')
plt.show(block=False)

model = Lasso(alpha=clf.alpha_, warm_start=True)
model.coef_ = clf.coef_.copy()  # it is important to do a copy here
for _ in range(5):
    model.weights = np.zeros(n_features)
    c = model.coef_
    model.weights[c != 0] = 1 / np.abs(c[c != 0])
    model.fit(X, y)


fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
m, s, _ = ax.stem(w_true, label=r"true coef",
                  use_line_collection=True)
m, s, _ = ax.stem(clf.coef_, label=r"LassoCV coef",
                  markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')
m, s, _ = ax.stem(model.coef_, label=r"AdaptiveLasso coef",
                  markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='k')
ax.set_xlabel("feature index")
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)
plt.legend(ncol=3, loc='upper center')
plt.show(block=False)
