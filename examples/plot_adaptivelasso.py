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

n_samples, n_features = 100, 50
rng = check_random_state(0)
X = rng.multivariate_normal(size=n_samples, mean=np.zeros(n_features),
                            cov=toeplitz(0.5 ** np.arange(n_features)))


# Create true regression coefficients with 3 groups of 5 non-zero values

w_true = np.zeros(n_features)
w_true[::3] = (-1) ** np.arange(n_features // 3 + 1)
noise = rng.randn(n_samples)
y = X @ w_true
y += noise / norm(noise) * 0.5 * norm(y)


# Fit an adapted AdaptiveLasso clf
alpha_max = np.max(np.abs(X.T @ y)) / n_samples
clf = LassoCV(verbose=1, tol=1e-10)
clf.fit(X, y)

plt.figure()
plt.semilogx(clf.alphas_, clf.mse_path_, ':')
plt.semilogx(clf.alphas_, clf.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
plt.axvline(clf.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square error')
plt.axis('tight')
plt.show(block=False)

model = Lasso(alpha=clf.alpha_, warm_start=True)
model.coef_ = clf.coef_
for _ in range(100):
    model.weights = np.zeros(n_features)
    c = model.coef_
    model.weights[c != 0] = 1 / np.abs(c[c != 0])
    model.fit(X, y)
    print(model.coef_[model.coef_ != 0])


fig = plt.figure(figsize=(10, 4))
m, s, _ = plt.stem(w_true, label=r"true regression coefficients",
                   use_line_collection=True)
m, s, _ = plt.stem(clf.coef_, label=r"LassoCV coefficients",
                   markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')
m, s, _ = plt.stem(model.coef_, label=r"AdaptiveLasso coefficients",
                   markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='k')
plt.xlabel("feature index")
plt.legend()
plt.show(block=False)
