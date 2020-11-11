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

from celer import AdaptiveLassoCV, LassoCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()

# Generating X and y data

n_samples, n_features = 100, 160
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

lasso = LassoCV().fit(X, y)
adaptive_lasso = AdaptiveLassoCV().fit(X, y)

fig, axarr = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

for i, model in enumerate([lasso, adaptive_lasso]):
    ax = axarr[i]
    ax.semilogx(model.alphas_, model.mse_path_, ':')
    ax.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
                label='Average across the folds', linewidth=2)
    ax.axvline(model.alpha_, linestyle='--', color='k',
               label='alpha: CV estimate')

    ax.legend()
    ax.set_title(model.__class__.__name__)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Mean square error')
plt.show(block=False)


fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
m, s, _ = ax.stem(w_true, label=r"true coef",
                  use_line_collection=True)
m, s, _ = ax.stem(lasso.coef_, label=r"LassoCV coef",
                  markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')
m, s, _ = ax.stem(adaptive_lasso.coef_, label=r"AdaptiveLassoCV coef",
                  markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='k')
ax.set_xlabel("feature index")
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)
plt.legend(ncol=3, loc='upper center')
plt.show(block=False)
