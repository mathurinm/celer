"""
===============================================================
Run GroupLasso and GroupLasso CV for structured sparse recovery
===============================================================

The example runs the GroupLasso scikit-learn like estimators.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

from celer import GroupLasso, GroupLassoCV
from celer.plot_utils import configure_plt

print(__doc__)
configure_plt()

# Generating X and y data

n_samples, n_features = 30, 50
rng = check_random_state(0)
X = rng.randn(n_samples, n_features)


# Create true regression coefficients with 3 groups of 5 non-zero values

w_true = np.zeros(n_features)
w_true[:5] = 1
w_true[20:25] = -2
w_true[40:45] = 1
y = X @ w_true + rng.randn(n_samples)


# Fit an adapted GroupLasso model

groups = 5  # groups are contiguous and of size 5
# irregular groups are also supported,
clf = GroupLasso(groups=groups, alpha=1.1)
clf.fit(X, y)

###############################################################################
# Display results

fig = plt.figure(figsize=(10, 4))
m, s, _ = plt.stem(w_true, label=r"true regression coefficients",
                   use_line_collection=True)
m, s, _ = plt.stem(clf.coef_, label=r"estimated regression coefficients",
                   markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')
plt.xlabel("feature index")
plt.legend()
plt.show(block=False)


###############################################################################
# Get optimal alpha for prediction by cross validation
model = GroupLassoCV(groups=groups)
model.fit(X, y)

print("Estimated regularization parameter alpha: %s" % model.alpha_)

fig = plt.figure(figsize=(11, 4.5))
plt.semilogx(model.alphas_, model.mse_path_, ':')
plt.semilogx(model.alphas_, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
plt.axvline(model.alpha_, linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel(r'$\alpha$')
plt.ylabel('Mean square error')
plt.show(block=False)


###############################################################################
# Show optimal regression vector for prediction, obtained by cross validation

fig = plt.figure(figsize=(10, 4))
m, s, _ = plt.stem(w_true, label=r"true regression coefficients",
                   use_line_collection=True)
m, s, _ = plt.stem(model.coef_, label=r"CV-estimated regression coefficients",
                   markerfmt='x', use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')
plt.xlabel("feature index")
plt.legend()
plt.show(block=False)
